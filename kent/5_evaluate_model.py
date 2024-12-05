import glob
import os
import random
import time
from common import PATH_WORKSPACE_ROOT, get_path_input_output_pairs
from common import get_setting_evaluation_loop_continue
from common import Encoder, Decoder, Seq2Seq
from common import PATH_WORKSPACE_ROOT, get_path_log, get_setting_evaluation_reload_model_in_loop, get_path_vocab
from common import get_path_input_sequences, get_path_output_sequences
from common import get_path_input_sequences_padded_batch_pattern, get_path_output_sequences_padded_batch_pattern
from common import get_path_model, get_setting_evaluation_subset_size
import torch
import torch.nn as nn
import pickle
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

DATASET_NAME = 'ubuntu_dialogue_corpus_000'
LOG_BASE_FILENAME = "5_evaluate_model"

MODEL_NAME = 'seq2seq'
MODEL_VERSION = '1.0'

TEST_DATA_PROPORTION = 0.2
RANDOM_SEED = 42

SUBSET_SIZE = get_setting_evaluation_subset_size()  # Number of sequences in each subset
TEST_SIZE = 0.2  # Proportion of the subset to use for testing
BATCH_SIZE = 64  # Adjust as needed

# ==========================

def load_latest_model_state(model, path_model, logger):
    """
    Load the latest saved model state if available.

    Args:
        model: The model instance to load the state into.
        path_model: Path to the saved model state.
        logger: Logger instance for logging status messages.

    Returns:
        bool: True if the model state was loaded successfully, False otherwise.
    """
    if os.path.exists(path_model):
        logger.info("Loading model state...")
        # place in try-catch block to handle exceptions
        try:
            model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu'), weights_only=True))
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            return False
        logger.info("Model state loaded.")
        return True
    else:
        logger.error("Trained model state not found.")
        return False

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
            logits = model(src, trg[:, :-1])  # Exclude <eos> from target
            
            # Use logits for loss computation
            logits_reshaped = logits.reshape(-1, logits.shape[-1])
            trg_reshaped = trg[:, 1:].reshape(-1)  # Exclude <bos>
            loss = criterion(logits_reshaped, trg_reshaped)
            test_loss += loss.item()

            # Use argmax for token prediction (for BLEU)
            predictions = logits.argmax(dim=-1)  # Get the predicted token indices

            # Decode predictions and references
            for i in range(src.size(0)):
                reference = [idx_to_token(vocab, trg[i].tolist())]
                hypothesis = idx_to_token(vocab, predictions[i].tolist())

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
    path_log = get_path_log(LOG_BASE_FILENAME, DATASET_NAME, log_start_time)

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
    logger.info(f"Current Working Directory: {os.getcwd()}")

    # ==========================

    path_input_csv = get_path_input_output_pairs(DATASET_NAME)
    path_vocab_pkl = get_path_vocab(DATASET_NAME)
    path_input_sequences = get_path_input_sequences(DATASET_NAME)
    path_output_sequences = get_path_output_sequences(DATASET_NAME)
    path_input_sequences_padded_batch_pattern = get_path_input_sequences_padded_batch_pattern(DATASET_NAME)
    path_output_sequences_padded_batch_pattern = get_path_output_sequences_padded_batch_pattern(DATASET_NAME)

    # Define the save path
    path_model = get_path_model(MODEL_NAME, MODEL_VERSION)

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

    # Combine input and output sequences into a single list of pairs
    combined_sequences = list(zip(input_sequences_padded, output_sequences_padded))

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

    # Load the latest model state
    if not load_latest_model_state(model, path_model, logger):
        logger.error("Unable to load model state. Exiting...")
        exit()

# ==========================

    # Initialize accumulators for overall scores
    cumulative_loss = 0
    cumulative_bleu = 0

    # Loop through the subsets
    sample_subset_counter = 0
    while get_setting_evaluation_loop_continue():

        sample_subset_counter += 1

        # Step 1: Randomly sample a subset of sequences
        subset_combined_sequences = random.sample(combined_sequences, SUBSET_SIZE)
        
        # Step 2: Split the subset into train and test
        _, subset_test_data = train_test_split(subset_combined_sequences, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        
        # Step 3: Unpack test data
        subset_test_inputs, subset_test_outputs = zip(*subset_test_data)
        subset_test_inputs = torch.stack(subset_test_inputs)
        subset_test_outputs = torch.stack(subset_test_outputs)
        
        # Create test DataLoader
        subset_test_dataset = TensorDataset(subset_test_inputs, subset_test_outputs)
        subset_test_loader = DataLoader(subset_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        if get_setting_evaluation_reload_model_in_loop():
            # Reload the model state
            if not load_latest_model_state(model, path_model, logger):
                logger.warning("Unable to load model state. Continuing with the current model state...")

        # Step 4: Evaluate the model on the subset
        logger.info(f"Evaluating subset {sample_subset_counter}...")
        subset_loss, subset_bleu = evaluate_model(model, subset_test_loader, criterion, vocab, device)
        
        # Accumulate the results
        cumulative_loss += subset_loss
        cumulative_bleu += subset_bleu
        
        logger.info(f"Subset {sample_subset_counter} - Loss: {subset_loss:.3f}, BLEU: {subset_bleu:.3f}")

    if sample_subset_counter == 0:
        logger.warning("No subsets evaluated. Exiting...")
        exit()

    # Step 5: Compute overall scores
    average_loss = cumulative_loss / sample_subset_counter
    average_bleu = cumulative_bleu / sample_subset_counter

    logger.info(f"Overall Evaluation - Average Loss: {average_loss:.3f}, Average BLEU: {average_bleu:.3f}")
