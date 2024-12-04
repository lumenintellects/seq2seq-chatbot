import glob
import os
import time
import spacy
from common import PATH_WORKSPACE_ROOT, csv_filename, pt_filename, pkl_filename, get_setting_training_loop_continue, get_setting_next_subset_continue
from common import log_filename, get_setting_analyze_sequences, get_setting_training_subset_size, get_setting_debug_mode
from common import Encoder, Decoder, Seq2Seq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from torch.utils.data import Subset
import logging
from sklearn.model_selection import train_test_split

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

LOG_BASE_FILENAME = "seq2seq_training"
WORKING_FOLDER = 'dataset'

NON_TRAIN_DATA_PROPORTION = 0.2
RANDOM_SEED = 42

PATIENCE_LEVEL = 5 # Number of epochs to wait for improvement before early stopping
TORCH_THREAD_COUNT = 10

N_PROCESS_VALUE = 8
BATCH_SIZE = 500000
TRAINING_SUBSET_SIZE = get_setting_training_subset_size()
LOSS_THRESHOLD = 1.0

# Tokenizer using spaCy with multithreading
def spacy_tokenizer_pipe(texts, nlp, n_process=4):
    """
    Tokenizes a list of texts using SpaCy's nlp.pipe for multithreaded tokenization.

    Parameters:
        texts (list): List of text strings to tokenize.
        nlp: SpaCy language model.
        n_process (int): Number of processes for parallel processing.

    Returns:
        list: List of tokenized texts.
    """
    logger.info(f"Tokenizing {len(texts)} texts using {n_process} processes...")
    tokenized_texts = []
    for doc in nlp.pipe(texts, n_process=n_process):
        tokenized_texts.append([token.text.lower() for token in doc if not token.is_space])
    return tokenized_texts

# Tokenizer using spaCy
def spacy_tokenizer(text):
    return [token.text.lower() for token in nlp(text) if not token.is_space]

# Tokenize the input and output texts using spaCy
def yield_tokens_spacy(texts):
    for text in texts:
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string. Got: {text}")
        yield spacy_tokenizer(text)

# Build vocabulary from spaCy tokens
def build_vocab(tokens_iterable, specials=["<unk>", "<pad>", "<bos>", "<eos>"]):
    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    for tokens in tokens_iterable:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

# Process text into sequences of indices with SpaCy pipeline
def process_text_spacy_pipe(texts, vocab, nlp, n_process=4):
    """
    Tokenizes a list of texts using SpaCy's nlp.pipe for multithreaded processing
    and converts them to sequences of indices.

    Parameters:
        texts (list): List of text strings to process.
        vocab (dict): Vocabulary mapping tokens to indices.
        nlp: SpaCy language model.
        n_process (int): Number of processes for parallel processing.

    Returns:
        list: List of sequences of indices.
    """
    logger.info(f"Processing {len(texts)} texts using {n_process} processes...")
    tokenized_sequences = []
    for doc in nlp.pipe(texts, n_process=n_process):
        tokens = ["<bos>"] + [token.text.lower() for token in doc if not token.is_space] + ["<eos>"]
        tokenized_sequences.append([vocab.get(token, vocab["<unk>"]) for token in tokens])
    return tokenized_sequences

# Process text into sequences of indices
def process_text_spacy(text, vocab):
    tokens = ["<bos>"] + spacy_tokenizer(text) + ["<eos>"]
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

def analyze_sequences(sequences):
    sequence_lengths = [len(seq) for seq in sequences]
    max_length = max(sequence_lengths)
    mean_length = sum(sequence_lengths) / len(sequence_lengths)
    median_length = sorted(sequence_lengths)[len(sequence_lengths) // 2]

    # Percentiles
    import numpy as np
    percentile_95 = np.percentile(sequence_lengths, 95)

    logger.info(f"Max Length: {max_length}")
    logger.info(f"Mean Length: {mean_length}")
    logger.info(f"Median Length: {median_length}")
    logger.info(f"95th Percentile: {percentile_95}")

# Explicitly pad sequences to the global maximum length
def pad_to_length(sequences, max_length, padding_value):
    """
    Pads all sequences to a specified maximum length.

    Parameters:
        sequences (list of lists): Sequences to pad.
        max_length (int): Desired maximum length.
        padding_value (int): Padding value.

    Returns:
        torch.Tensor: Tensor of padded sequences.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            seq = seq[:max_length]  # Truncate if longer than max_length
        else:
            seq = seq + [padding_value] * (max_length - len(seq))  # Pad if shorter
        padded_sequences.append(seq)
    return torch.tensor(padded_sequences, dtype=torch.int64)

# ==========================

if __name__ == "__main__":

    log_start_time = time.strftime('%Y%m%d_%H%M%S')
    path_log = os.path.join(WORKING_FOLDER, log_filename(f"{LOG_BASE_FILENAME}_{log_start_time}"))

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum log level (DEBUG, INFO, WARNING, etc.)
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format with timestamps
        handlers=[
            logging.FileHandler(path_log),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )

    logger = logging.getLogger(__name__)

    logger.info("Running main script...")

    # Load spaCy language model
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded.")

    # Set the current working directory
    os.chdir(PATH_WORKSPACE_ROOT)
    logger.info("Current Working Directory:", os.getcwd())

    # ==========================

    WORKING_FOLDER = 'dataset'
    base_filename = 'ubuntu_dialogue_corpus_196_input_output_pairs'
    path_input_csv = os.path.join(WORKING_FOLDER, csv_filename(base_filename))
    path_vocab_pkl = os.path.join(WORKING_FOLDER, pkl_filename(f"{base_filename}_vocab"))
    path_input_sequences = os.path.join(WORKING_FOLDER, pt_filename(f"{base_filename}_input_sequences"))
    path_output_sequences = os.path.join(WORKING_FOLDER, pt_filename(f"{base_filename}_output_sequences"))
    path_input_sequences_padded_batch_pattern = os.path.join(WORKING_FOLDER, pt_filename(f"{base_filename}_input_sequences_padded_batch_*"))
    path_output_sequences_padded_batch_pattern = os.path.join(WORKING_FOLDER, pt_filename(f"{base_filename}_output_sequences_padded_batch_*"))

    # Define the save path
    path_model = os.path.join(PATH_WORKSPACE_ROOT, "seq2seq_model.pth")

    # ==========================

    # Check for existing vocabulary
    if os.path.exists(path_vocab_pkl):
        logger.info("Vocabulary file found. Loading vocabulary...")
        with open(path_vocab_pkl, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        logger.info(f"Vocabulary loaded. Size: {len(vocab)}")
    else:
        logger.error("Vocabulary file not found. Unable to proceed, exiting...")
        exit()

    # Check for previous serialized padded input sequences matching batch file name pattern
    if len(glob.glob(path_input_sequences_padded_batch_pattern)) > 0:
        logger.info("Serialized padded input sequences found.")
    else:
        logger.error("Serialized padded input sequences not found. Unable to proceed, exiting...")
        exit()

    padding_value = vocab["<pad>"]

    # Check for previous serialized output sequences
    if os.path.exists(path_output_sequences):
        logger.info("Serialized output sequences found.")
    else:
        logger.error("Serialized output sequences not found. Unable to proceed, exiting...")
        exit()

    input_sequences_padded = torch.cat([torch.load(file, weights_only=True) for file in glob.glob(path_input_sequences_padded_batch_pattern)], dim=0)
    logger.info("Loaded input sequences from files.")

    output_sequences_padded = torch.cat([torch.load(file, weights_only=True) for file in glob.glob(path_output_sequences_padded_batch_pattern)], dim=0)
    logger.info("Loaded output sequences from file.")

    # Analyze sequences
    if get_setting_analyze_sequences():
        logger.info("Analyzing input and output sequences...")
        analyze_sequences(input_sequences_padded)
        analyze_sequences(output_sequences_padded)

        logger.info("Input shape:", input_sequences_padded.shape)
        logger.info("Output shape:", output_sequences_padded.shape)

    # ==========================

    # Combine input and output sequences into a single list of pairs
    combined_sequences = list(zip(input_sequences_padded, output_sequences_padded))
    combined_indices = list(range(len(combined_sequences)))

    # ==========================
    # Instantiate Seq2Seq Model

    logger.info("Initializing Seq2Seq model...")

    # Hyperparameters
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    EMB_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 64
    num_epochs = 10

    # Check GPU Availability
    logger.info("Checking GPU availability...")
    logger.info(f"Is CUDA available: {torch.cuda.is_available()}")  # Should print True
    logger.info(f"Device ID: {torch.cuda.current_device()}")  # Should print the device ID (e.g., 0)
    logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")  # Should print the name of the GPU

    # Initialize encoder, decoder, and seq2seq model
    logger.info("Initializing encoder, decoder, and seq2seq model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=padding_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # If model state exists, load it
    if os.path.exists(path_model):
        logger.info("Loading model state...")
        model.load_state_dict(torch.load(path_model, weights_only=True))
        logger.info("Model state loaded.")
    else:
        logger.info("Model state not found. Initializing new model.")

# ==========================

    # Training Loop
    logger.info("Training...")

    best_val_loss = float("inf")  # Initialize best validation loss as infinity
    no_improvement_epochs = 0  # Counter for epochs without improvement

    torch.set_num_threads(TORCH_THREAD_COUNT)
    logger.info(f"Number of Torch Threads is set to {torch.get_num_threads()}")

    continue_training = True
    while continue_training:

        # Create a subset of the dataset for training
        logger.info(f"Creating a subset of the dataset for training with size: {TRAINING_SUBSET_SIZE}")
        subset_indices = np.random.choice(combined_indices, size=TRAINING_SUBSET_SIZE, replace=False)
        subset_combined_sequences = [combined_sequences[i] for i in subset_indices]

        # Split data into training and temporary (validation + test)
        subset_train_data, subset_non_train_data = train_test_split(subset_combined_sequences, test_size=NON_TRAIN_DATA_PROPORTION, random_state=RANDOM_SEED)

        # Further split the non-training dataset into validation and test sets
        subset_val_data, subset_test_data = train_test_split(subset_non_train_data, test_size=0.5, random_state=RANDOM_SEED)

        # Unzip training, validation, and test datasets
        subset_train_inputs, subset_train_outputs = zip(*subset_train_data)
        subset_val_inputs, subset_val_outputs = zip(*subset_val_data)
        subset_test_inputs, subset_test_outputs = zip(*subset_test_data)

        # Convert back to torch tensors
        subset_train_inputs = torch.stack(subset_train_inputs)
        subset_train_outputs = torch.stack(subset_train_outputs)
        subset_val_inputs = torch.stack(subset_val_inputs)
        subset_val_outputs = torch.stack(subset_val_outputs)
        subset_test_inputs = torch.stack(subset_test_inputs)
        subset_test_outputs = torch.stack(subset_test_outputs)

        # Create DataLoaders
        subset_train_dataset = TensorDataset(subset_train_inputs, subset_train_outputs)
        subset_train_loader = DataLoader(subset_train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

        subset_val_dataset = TensorDataset(subset_val_inputs, subset_val_outputs)
        subset_val_loader = DataLoader(subset_val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

        subset_test_dataset = TensorDataset(subset_test_inputs, subset_test_outputs)
        subset_test_loader = DataLoader(subset_test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

        # Epoch-wise training
        iteration_start_time = time.time()
        loss_history = []
        val_loss_history = []
        epoch_number = 0

        while get_setting_training_loop_continue():

            # Training Phase
            model.train()
            epoch_loss = 0
            for src, trg in subset_train_loader:
                src, trg = src.to(device), trg.to(device)
                optimizer.zero_grad()

                outputs = model(src, trg[:, :-1])
                outputs = outputs.reshape(-1, outputs.shape[-1])
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(outputs, trg)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_number += 1
            train_loss = epoch_loss / len(subset_train_loader)
            loss_history.append(train_loss)

            # Validation Phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, trg in subset_val_loader:
                    src, trg = src.to(device), trg.to(device)

                    outputs = model(src, trg[:, :-1])
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    trg = trg[:, 1:].reshape(-1)

                    loss = criterion(outputs, trg)
                    val_loss += loss.item()

            val_loss /= len(subset_val_loader)
            val_loss_history.append(val_loss)
            logger.info(f"Epoch {epoch_number}, Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f} (vs. current best of {best_val_loss:.3f})")

            # Early Stopping Check
            has_val_loss_improved = val_loss < best_val_loss
            if has_val_loss_improved:
                best_val_loss = val_loss

                torch.save(model.state_dict(), path_model)
                logger.info("Model state saved.")

                no_improvement_epochs = 0 # reset counter
            else:
              no_improvement_epochs += 1

              out_of_patience = no_improvement_epochs >= PATIENCE_LEVEL
              if out_of_patience:
                  logger.info(f"Early stopping triggered after {PATIENCE_LEVEL} epochs with no improvement.")
                  continue_training = get_setting_next_subset_continue()
                  break

        iteration_end_time = time.time()
        iteration_time_training_loop = iteration_end_time - iteration_start_time
        iteration_time_average_per_epoch = iteration_time_training_loop / epoch_number
        logger.info(f"Training loop completed in {iteration_time_training_loop:.2f} seconds.")
        logger.info(f"Average time per epoch: {iteration_time_average_per_epoch:.2f} seconds.")

        # Stop outer loop if no continuation
        if not get_setting_next_subset_continue():
            break
