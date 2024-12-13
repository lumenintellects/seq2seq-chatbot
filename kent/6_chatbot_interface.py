import glob
import os
import time
import sentencepiece as sp
import re
from common import FILE_MODE_READ_BINARY, PATH_WORKSPACE_ROOT, SPACE, VOCAB_BOS, VOCAB_EOS, VOCAB_PAD, VOCAB_UNK, get_path_input_output_pairs, get_path_sentencepiece_model, initialize_seq2seq
from common import PATH_WORKSPACE_ROOT, get_path_log, get_path_vocab
from common import get_path_input_sequences, get_path_output_sequences
from common import get_path_input_sequences_padded_batch_pattern, get_path_output_sequences_padded_batch_pattern
from common import get_path_model, get_setting_evaluation_subset_size
import torch
import torch.nn as nn
import logging

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

DATASET_NAME = 'ubuntu_dialogue_corpus_000'
LOG_BASE_FILENAME = "6_chatbot_interface"

MODEL_NAME = 'seq2seq'
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

def generate_response(model, input_text, vocab, device, max_length=50):
    model.eval()
    with torch.no_grad():
        # Tokenize and convert input to indices
        tokens = [VOCAB_BOS] + input_text.split() + [VOCAB_EOS]
        input_indices = [vocab.get(token, vocab[VOCAB_UNK]) for token in tokens]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

        # Encode the input
        _, hidden = model.encoder(input_tensor)

        # Adjust hidden state to match decoder's n_layers
        if hidden.size(0) == 1:  # If the encoder is single-layer
            hidden = hidden.repeat(model.decoder.rnn.num_layers, 1, 1)  # Repeat for n_layers

        # Prepare decoder input (<bos> token)
        trg_idx = vocab["<bos>"]
        trg_tensor = torch.tensor([[trg_idx]], dtype=torch.long).to(device)  # (1, 1)

        # Generate response
        response_tokens = []
        for _ in range(max_length):
            # Call the decoder
            predictions, hidden = model.decoder(trg_tensor, hidden)

            # Get the predicted token (argmax over vocabulary)
            trg_idx = predictions.squeeze(0).argmax(1).item()  # Squeeze batch size, get token index
            trg_tensor = torch.tensor([[trg_idx]], dtype=torch.long).to(device)  # Prepare input for next step

            # Stop if <eos> is generated
            if trg_idx == vocab["<eos>"]:
                break

            response_tokens.append(trg_idx)

        # Convert indices back to tokens
        idx_to_word = {idx: token for token, idx in vocab.items()}
        response = " ".join([idx_to_word[idx] for idx in response_tokens])

    return response

def generate_response_beam_search(model, input_text, vocab, device, max_length=50, beam_width=3):
    """
    Generate a response using beam search decoding.

    Args:
        model: Trained Seq2Seq model.
        input_text: User input text string.
        vocab: Vocabulary mapping tokens to indices.
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
        input_indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

        # Encode the input
        _, hidden = model.encoder(input_tensor)

        # Adjust hidden state to match decoder's n_layers
        if hidden.size(0) == 1:  # If the encoder is single-layer
            hidden = hidden.repeat(model.decoder.rnn.num_layers, 1, 1)  # Repeat for n_layers

        # Initialize beams: Each beam is a tuple (sequence, score, hidden)
        beams = [([vocab[VOCAB_BOS]], 0.0, hidden)]  # Start with the <bos> token and score 0.0

        for _ in range(max_length):
            all_candidates = []  # To store all beam expansions

            for seq, score, hidden in beams:
                # Prepare decoder input for the last token in the sequence
                trg_tensor = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)  # (1, 1)

                # Decode the next token
                predictions, hidden = model.decoder(trg_tensor, hidden)

                # Get log probabilities
                log_probs = torch.log_softmax(predictions.squeeze(0), dim=-1)  # (1, vocab_size)

                # Expand each beam with top beam_width tokens
                top_tokens = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    next_token = top_tokens.indices[0, i].item()
                    next_score = score + top_tokens.values[0, i].item()  # Accumulate log-probabilities
                    all_candidates.append((seq + [next_token], next_score, hidden))

            # Prune to the top beam_width candidates
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Stop if all beams end with <eos>
            if all(seq[-1] == vocab[VOCAB_EOS] for seq, _, _ in beams):
                break

        # Select the best beam (highest score)
        best_sequence = beams[0][0]  # Sequence with the highest score

        # Convert indices back to tokens
        idx_to_word = {idx: token for token, idx in vocab.items()}
        response_tokens = [idx_to_word[idx] for idx in best_sequence if idx not in {vocab["<bos>"], vocab["<eos>"]}]

        return SPACE.join(response_tokens)

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

    # Initialize Seq2Seq model
    model, criterion = initialize_seq2seq(
        sp_model=sp_model,
        device=device,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )

    # If model state exists, load it
    if os.path.exists(path_model):
        logger.info(f"Loading model state: {path_model}")
        model.load_state_dict(torch.load(path_model, weights_only=True))
        logger.info("Model state loaded.")
    else:
        logger.info("Model state not found. Initializing new model.")

    # Define Loss Function and Optimizer
    pad_id = sp_model.pad_id()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================

    # Launch chatbot interface
    chatbot_interface(model, sp_model, device)
