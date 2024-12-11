import glob
import os
import random
import time
import sentencepiece as sp
from common import PATH_WORKSPACE_ROOT, TupleDataset, get_path_sentencepiece_model, get_setting_training_loop_continue, get_setting_next_subset_continue
from common import get_setting_analyze_sequences, get_setting_training_subset_size
from common import Encoder, Decoder, Seq2Seq
from common import PATH_WORKSPACE_ROOT, get_path_log, get_path_input_output_pairs, get_path_vocab
from common import get_path_input_sequences, get_path_output_sequences
from common import get_path_input_sequences_padded_batch_pattern, get_path_output_sequences_padded_batch_pattern
from common import get_path_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from sklearn.model_selection import train_test_split

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

LOG_BASE_FILENAME = "4_train_model"
DATASET_NAME = 'ubuntu_dialogue_corpus_000'
MODEL_NAME = 'seq2seq'
MODEL_VERSION = '2.0'

VAL_DATA_PROPORTION = 0.2
RANDOM_SEED = 42
PATIENCE_LEVEL = 2 # Number of epochs to wait for improvement before early stopping
TORCH_THREAD_COUNT = 10
TRAINING_SUBSET_SIZE = get_setting_training_subset_size()
LOSS_THRESHOLD = 1.0

# ==========================

path_input_csv = get_path_input_output_pairs(DATASET_NAME)
path_input_sequences = get_path_input_sequences(DATASET_NAME)
path_output_sequences = get_path_output_sequences(DATASET_NAME)
path_input_sequences_padded_batch_pattern = get_path_input_sequences_padded_batch_pattern(DATASET_NAME)
path_output_sequences_padded_batch_pattern = get_path_output_sequences_padded_batch_pattern(DATASET_NAME)

path_sentencepiece_model = get_path_sentencepiece_model(DATASET_NAME)

# Define the save path
path_model = get_path_model(MODEL_NAME, MODEL_VERSION)

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
    BATCH_SIZE = 16

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

    # Training Loop
    logger.info("Training...")

    best_val_loss = float("inf")  # Initialize best validation loss as infinity
    no_improvement_epochs = 0  # Counter for epochs without improvement

    torch.set_num_threads(TORCH_THREAD_COUNT)
    logger.info(f"Number of Torch Threads is set to {torch.get_num_threads()}")

    continue_training = True
    while continue_training:

# ==========================

        # Select a random batch
        input_batch_file = random.choice(matching_files_input)
        output_batch_file = random.choice(matching_files_output)

        input_sequences = torch.load(input_batch_file)
        output_sequences = torch.load(output_batch_file)

        # Sample subsets for training and validation
        len_input = len(input_sequences)
        len_output = len(output_sequences)
        total_samples = min(len_input, len_output) # Ensure equal number of samples

        indices = random.sample(range(total_samples), TRAINING_SUBSET_SIZE)
        train_size = int(TRAINING_SUBSET_SIZE * (1 - VAL_DATA_PROPORTION))

        train_input = input_sequences[indices[:train_size]]
        train_output = output_sequences[indices[:train_size]]
        val_input = input_sequences[indices[train_size:]]
        val_output = output_sequences[indices[train_size:]]

        train_loader = DataLoader(TupleDataset(train_input, train_output), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TupleDataset(val_input, val_output), batch_size=BATCH_SIZE)

# ==========================

        # Epoch-wise training
        iteration_start_time = time.time()
        loss_history = []
        val_loss_history = []
        epoch_number = 0

        while get_setting_training_loop_continue():

            # Training Phase
            model.train()
            epoch_loss = 0
            for src, trg in train_loader:
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
            train_loss = epoch_loss / len(train_loader)
            loss_history.append(train_loss)

            # Validation Phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, trg in val_loader:
                    src, trg = src.to(device), trg.to(device)

                    outputs = model(src, trg[:, :-1])
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    trg = trg[:, 1:].reshape(-1)

                    loss = criterion(outputs, trg)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_loss_history.append(val_loss)
            logger.info(f"Epoch {epoch_number}, Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f} (vs. current best of {best_val_loss:.3f})")

            # Early Stopping Check
            has_val_loss_improved = val_loss < best_val_loss
            if has_val_loss_improved:
                best_val_loss = val_loss

                torch.save(model.state_dict(), path_model)
                logger.info("Model state saved.")

                no_improvement_epochs = 0 # reset counter
                scheduler.step(val_loss) # adjust learning rate if needed
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
