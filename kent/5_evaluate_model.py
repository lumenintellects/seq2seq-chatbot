import glob
import os
import random
import time
import sentencepiece as sp
from common import PATH_WORKSPACE_ROOT, TupleDataset, get_path_input_output_pairs, get_path_sentencepiece_model
from common import get_setting_evaluation_loop_continue
from common import Encoder, Decoder, Seq2Seq
from common import PATH_WORKSPACE_ROOT, get_path_log, get_setting_evaluation_reload_model_in_loop, get_path_vocab
from common import get_path_input_sequences_padded_batch_pattern, get_path_output_sequences_padded_batch_pattern
from common import get_path_model, get_setting_evaluation_subset_size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from nltk.translate.bleu_score import corpus_bleu

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

DATASET_NAME = 'ubuntu_dialogue_corpus_000'
LOG_BASE_FILENAME = "5_evaluate_model"

MODEL_NAME = 'seq2seq'
MODEL_VERSION = '2.0'

TEST_DATA_PROPORTION = 0.2
RANDOM_SEED = 42

SUBSET_SIZE = get_setting_evaluation_subset_size()  # Number of sequences in each subset
TEST_SIZE = 0.2  # Proportion of the subset to use for testing
BATCH_SIZE = 64  # Adjust as needed

# ==========================

path_input_csv = get_path_input_output_pairs(DATASET_NAME)
path_vocab_pkl = get_path_vocab(DATASET_NAME)
path_input_sequences_padded_batch_pattern = get_path_input_sequences_padded_batch_pattern(DATASET_NAME)
path_output_sequences_padded_batch_pattern = get_path_output_sequences_padded_batch_pattern(DATASET_NAME)

path_sentencepiece_model = get_path_sentencepiece_model(DATASET_NAME)

path_model = get_path_model(MODEL_NAME, MODEL_VERSION)

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
        vocab: Model vocabulary (SentencePiece model for decoding).
        device: The device (CPU/GPU) to use.

    Returns:
        test_loss: Average loss on the test set.
        bleu_score: BLEU score for generated predictions.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    references = []
    hypotheses = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            # Remove the <eos> token from the target sequences
            trg_input = trg[:, :-1]
            trg_target = trg[:, 1:].reshape(-1)  # Targets for loss calculation

            # Forward pass through the model
            outputs = model(src, trg_input)
            outputs = outputs.reshape(-1, outputs.shape[-1])

            # Compute loss
            loss = criterion(outputs, trg_target)
            total_loss += loss.item()

            # Generate predictions
            predicted_ids = outputs.argmax(dim=1).view(trg_input.shape[0], -1)

            # Convert predictions and targets to text for BLEU score calculation
            for i in range(trg_input.size(0)):  # Batch size loop
                predicted_tokens = [vocab.id_to_piece(idx.item()) for idx in predicted_ids[i] if idx != vocab.pad_id()]
                reference_tokens = [vocab.id_to_piece(idx.item()) for idx in trg[i, 1:] if idx != vocab.pad_id()]

                hypotheses.append(predicted_tokens)
                references.append([reference_tokens])  # BLEU requires a list of references per sentence

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses)

    # Compute average loss
    test_loss = total_loss / len(dataloader)

    return test_loss, bleu_score

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

    # Load SentencePiece model
    if os.path.exists(path_sentencepiece_model):
        sp_model = sp.SentencePieceProcessor(model_file=path_sentencepiece_model)
        logger.info(f"Loaded SentencePiece model from {path_sentencepiece_model}.")
    else:
        logger.error("SentencePiece model file not found. Exiting...")
        exit()

    # ==========================

    matching_files_input = glob.glob(path_input_sequences_padded_batch_pattern)
    if len(matching_files_input) == 0:
        logger.error("No matching input files found. Unable to proceed, exiting...")
        exit()

    matching_files_output = glob.glob(path_output_sequences_padded_batch_pattern)
    if len(matching_files_output) == 0:
        logger.error("No matching output files found. Unable to proceed, exiting...")
        exit()

    # ==========================
    # Instantiate Seq2Seq Model

    logger.info("Initializing Seq2Seq model...")

    # Hyperparameters
    INPUT_DIM = sp_model.get_piece_size()
    OUTPUT_DIM = sp_model.get_piece_size()
    EMB_DIM = 128
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 32

    # Check GPU Availability
    logger.info("Checking GPU availability...")
    logger.info(f"Is CUDA available: {torch.cuda.is_available()}")  # Should print True
    logger.info(f"Device ID: {torch.cuda.current_device()}")  # Should print the device ID (e.g., 0)
    logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")  # Should print the name of the GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize encoder, decoder, and seq2seq model
    logger.info("Initializing encoder, decoder, and seq2seq model...")

    # Initialize encoder
    encoder = Encoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
    logger.info("Encoder initialized with input dimension of {INPUT_DIM}, embedding dimension of {EMB_DIM}, hidden dimension of {HIDDEN_DIM}, {N_LAYERS} layers, and dropout of {DROPOUT}.")

    # Initialize decoder
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
    logger.info("Decoder initialized with output dimension of {OUTPUT_DIM}, embedding dimension of {EMB_DIM}, hidden dimension of {HIDDEN_DIM}, {N_LAYERS} layers, and dropout of {DROPOUT}.")  

    # Initialize Seq2Seq model
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Define Loss Function and Optimizer
    pad_id = sp_model.pad_id()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # If model state exists, load it
    if os.path.exists(path_model):
        logger.info(f"Loading model state: {path_model}")
        model.load_state_dict(torch.load(path_model, weights_only=True))
        logger.info("Model state loaded.")
    else:
        logger.info("Model state not found. Initializing new model.")

# ==========================

    # Initialize accumulators for overall scores
    cumulative_loss = 0
    cumulative_bleu = 0

    # Loop through the subsets
    sample_subset_counter = 0
    while get_setting_evaluation_loop_continue():

        sample_subset_counter += 1

        # Select a random batch
        input_batch_file = random.choice(matching_files_input)
        logger.info(f"Selected input batch file: {input_batch_file}")
        output_batch_file = random.choice(matching_files_output)
        logger.info(f"Selected output batch file: {output_batch_file}")

        input_sequences = torch.load(input_batch_file)
        output_sequences = torch.load(output_batch_file)

        # Sample subsets for training and validation
        len_input = len(input_sequences)
        logger.info(f"Number of input sequences: {len_input}")
        len_output = len(output_sequences)
        logger.info(f"Number of output sequences: {len_output}")
        len_population = min(len_input, len_output) # Ensure equal number of samples
        logger.info(f"Number of samples to train and validate: {len_population}")

        TEST_SUBSET_SIZE = 100
        # Ensure test subset size is within bounds
        if TEST_SUBSET_SIZE > len_population:
            logger.error(f"Test subset size ({TEST_SUBSET_SIZE}) exceeds population size ({len_population}). Exiting...")
            exit()

        sample_indices = random.sample(range(len_population), TEST_SUBSET_SIZE)
        len_sampled = len(sample_indices)

        logger.info(f"Number of samples selected for testing: {TEST_SUBSET_SIZE}.")

        test_input = input_sequences[sample_indices[:TEST_SUBSET_SIZE]]
        test_output = output_sequences[sample_indices[:TEST_SUBSET_SIZE]]

        test_loader = DataLoader(TupleDataset(test_input, test_output), batch_size=BATCH_SIZE, shuffle=True)

        if get_setting_evaluation_reload_model_in_loop():
            # Reload the model state
            if not load_latest_model_state(model, path_model, logger):
                logger.warning("Unable to load model state. Continuing with the current model state...")

        # Step 4: Evaluate the model on the subset
        logger.info(f"Evaluating subset {sample_subset_counter}...")
        subset_loss, subset_bleu = evaluate_model(model, test_loader, criterion, sp_model, device)

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
