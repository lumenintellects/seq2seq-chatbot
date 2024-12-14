import glob
import os
import time
import sentencepiece as sp
import re
from common import PATH_WORKSPACE_ROOT, SPACE, VOCAB_BOS, VOCAB_EOS, get_path_sentencepiece_model, initialize_seq2seq_with_attention
from common import PATH_WORKSPACE_ROOT, get_path_log
from common import get_path_input_sequences_padded_batch_pattern, get_path_output_sequences_padded_batch_pattern
from common import get_path_model
import torch
import torch.nn as nn
import logging

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

DATASET_NAME = 'ubuntu_dialogue_corpus_000'
LOG_BASE_FILENAME = "6_chatbot_interface_with_attention"

MODEL_NAME = 'seq2seq_attention'
MODEL_VERSION = '2.0'

RANDOM_SEED = 42

CHATBOT_COMMAND_EXIT = "exit"

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

def generate_response_beam_search(model, input_text, sp_model, device, max_length=50, beam_width=3):
    """
    Generate a response using beam search decoding.

    Args:
        model: Trained Seq2Seq model with attention.
        input_text: User input text string.
        sp_model: SentencePieceProcessor for tokenization.
        device: The device (CPU/GPU) to use.
        max_length: Maximum length of the generated response.
        beam_width: Number of beams to keep at each step.

    Returns:
        response: Generated response as a string.
    """
    model.eval()
    with torch.no_grad():
        # Tokenize and convert input to indices
        tokens = [VOCAB_BOS] + input_text.split() + [VOCAB_EOS]
        input_indices = [sp_model.piece_to_id(token) for token in tokens]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

        # Encode input and transform hidden state
        encoder_outputs, hidden = model.encoder(input_tensor)
        print(f"Raw Hidden Shape (Encoder): {hidden.shape}")
        hidden = model._transform_hidden(hidden)  # Apply transformation
        print(f"Transformed Hidden Shape: {hidden.shape}")

        # Initialize beams: (sequence, score, hidden, encoder_outputs)
        beams = [([sp_model.bos_id()], 0.0, hidden, encoder_outputs)]

        for _ in range(max_length):
            all_candidates = []
            for seq, score, hidden, encoder_outputs in beams:
                trg_tensor = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)
                predictions, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
                log_probs = torch.log_softmax(predictions.squeeze(0), dim=-1)
                top_tokens = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    next_token = top_tokens.indices[i].item()
                    next_score = score + top_tokens.values[i].item()
                    all_candidates.append((seq + [next_token], next_score, hidden, encoder_outputs))

            # Prune to the top beam_width candidates
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Stop if all beams end with <eos>
            if all(seq[-1] == sp_model.eos_id() for seq, _, _, _ in beams):
                break

        # Select the best beam (highest score)
        best_sequence = beams[0][0]
        response_tokens = [sp_model.id_to_piece(idx) for idx in best_sequence if idx not in {sp_model.bos_id(), sp_model.eos_id()}]
        trimmed_response_tokens = [re.sub(r"[▁\s]", "", token) for token in response_tokens]

        return SPACE.join(trimmed_response_tokens)

def chatbot_interface(model, vocab, device):
    """
    Start a simple chatbot interface.

    Args:
        model: Trained Seq2Seq model.
        vocab: Vocabulary mapping tokens to indices.
        device: The device (CPU/GPU) to use.
    """
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        # Get user input
        user_input = input("You: ")
        logger.info(f"User Input: {user_input}")

        # Check for exit command
        if user_input.lower() == CHATBOT_COMMAND_EXIT:
            print("Chatbot: Goodbye!")
            break

        # Generate response
        preprocessed_input = preprocess_input(user_input)
        logger.info(f"Preprocessed Input: {preprocessed_input}")
        response = generate_response_beam_search(model, preprocessed_input, vocab, device)
        logger.info(f"Chatbot: {response}")

def preprocess_input(text):
    """
    Preprocess the input text by converting to lowercase and removing special characters.

    Args:
        text: Input text string.

    Returns:
        text: Preprocessed text.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())  # Remove special chars
    return text

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

    path_input_sequences_padded_batch_pattern = get_path_input_sequences_padded_batch_pattern(DATASET_NAME)
    path_output_sequences_padded_batch_pattern = get_path_output_sequences_padded_batch_pattern(DATASET_NAME)
    path_sentencepiece_model = get_path_sentencepiece_model(DATASET_NAME)

    # Define the save path
    path_model = get_path_model(MODEL_NAME, MODEL_VERSION)

    # ==========================

    # Load SentencePiece model
    if os.path.exists(path_sentencepiece_model):
        sp_model = sp.SentencePieceProcessor(model_file=path_sentencepiece_model)
        logger.info(f"Loaded SentencePiece model from {path_sentencepiece_model}.")
    else:
        logger.error("SentencePiece model file not found. Exiting...")
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

    # Check GPU Availability
    logger.info("Checking GPU availability...")
    logger.info(f"Is CUDA available: {torch.cuda.is_available()}")  # Should print True
    logger.info(f"Device ID: {torch.cuda.current_device()}")  # Should print the device ID (e.g., 0)
    logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")  # Should print the name of the GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Hyperparameters
    INPUT_DIM = sp_model.get_piece_size()
    OUTPUT_DIM = sp_model.get_piece_size()
    EMB_DIM = 128
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 32

    # Initialize Seq2Seq model with attention
    model_with_attention, criterion = initialize_seq2seq_with_attention(
        sp_model=sp_model,
        device=device,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )
    logger.info("Seq2Seq model with attention initialized.")

    # If model state exists, load it
    if os.path.exists(path_model):
        logger.info(f"Loading model state: {path_model}")
        model_with_attention.load_state_dict(torch.load(path_model, weights_only=True))
        logger.info("Model state loaded.")
    else:
        logger.info("Model state not found. Initializing new model.")

    # Define Loss Function and Optimizer
    pad_id = sp_model.pad_id()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model_with_attention.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# ==========================

    # Launch chatbot interface
    chatbot_interface(model_with_attention, sp_model, device)
