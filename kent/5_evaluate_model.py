import glob
import os
import time
from common import PATH_WORKSPACE_ROOT, csv_filename, pt_filename, pkl_filename
from common import log_filename, get_setting_training_subset_size
from common import Encoder, Decoder, Seq2Seq
import torch
import torch.nn as nn
import pickle
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

LOG_BASE_FILENAME = "seq2seq_model_evaluation"
WORKING_FOLDER = 'dataset'

NON_TRAIN_DATA_PROPORTION = 0.2
RANDOM_SEED = 42

PATIENCE_LEVEL = 5 # Number of epochs to wait for improvement before early stopping
TORCH_THREAD_COUNT = 10

N_PROCESS_VALUE = 8
BATCH_SIZE = 500000
TRAINING_SUBSET_SIZE = get_setting_training_subset_size()
LOSS_THRESHOLD = 1.0

# ==========================

def evaluate_model(model, dataloader, criterion, vocab, device):
    """
    Evaluate the model on the test set and compute metrics.
    
    Args:
        model: The trained seq2seq model.
        dataloader: DataLoader for the test set.
        criterion: Loss function.
        vocab: Vocabulary mapping.
        device: The device (CPU/GPU) to use.

    Returns:
        test_loss: Average loss on the test set.
        bleu_score: BLEU score for generated predictions.
    """
    model.eval()
    test_loss = 0
    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            # Generate predictions
            outputs = model(src, trg[:, :-1])  # Exclude <eos> from target
            outputs = outputs.argmax(dim=-1)  # Get the predicted token indices

            # Compute loss
            outputs_reshaped = outputs.reshape(-1, outputs.shape[-1])
            trg_reshaped = trg[:, 1:].reshape(-1)  # Exclude <bos>
            loss = criterion(outputs_reshaped, trg_reshaped)
            test_loss += loss.item()

            # Decode predictions and references
            for i in range(src.size(0)):
                reference = [idx_to_token(vocab, trg[i].tolist())]
                hypothesis = idx_to_token(vocab, outputs[i].tolist())

                all_references.append(reference)  # Reference needs to be a list of list
                all_hypotheses.append(hypothesis)

    test_loss /= len(dataloader)
    bleu_score = corpus_bleu(all_references, all_hypotheses)

    return test_loss, bleu_score


def idx_to_token(vocab, indices):
    """
    Convert a list of indices to tokens using the vocabulary.

    Args:
        vocab: Vocabulary mapping indices to tokens.
        indices: List of indices to convert.

    Returns:
        tokens: Decoded token list as a single string.
    """
    token_list = [key for key, value in vocab.items() if value in indices]
    return token_list

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
        logger.error("Trained model state not found. Exiting...")
        exit()

# ==========================

    # Combine input and output sequences into a single list of pairs
    combined_sequences = list(zip(input_sequences_padded, output_sequences_padded))

    # Split data into test set only
    _, _, test_data = train_test_split(combined_sequences, test_size=NON_TRAIN_DATA_PROPORTION, random_state=RANDOM_SEED)

    # Unpack test set
    test_inputs, test_outputs = zip(*test_data)
    test_inputs = torch.stack(test_inputs)
    test_outputs = torch.stack(test_outputs)

    # Create test DataLoader
    test_dataset = TensorDataset(test_inputs, test_outputs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    # Evaluate the model
    logger.info("Evaluating the model on the test set...")
    test_loss, bleu_score = evaluate_model(model, test_loader, criterion, vocab, device)
    logger.info(f"Test Loss: {test_loss:.3f}")
    logger.info(f"BLEU Score: {bleu_score:.3f}")
