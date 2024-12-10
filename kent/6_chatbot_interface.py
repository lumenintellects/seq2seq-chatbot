import glob
import os
import time
import re
from common import PATH_WORKSPACE_ROOT, get_path_input_output_pairs
from common import Encoder, Decoder, Seq2Seq
from common import PATH_WORKSPACE_ROOT, get_path_log, get_path_vocab
from common import get_path_input_sequences, get_path_output_sequences
from common import get_path_input_sequences_padded_batch_pattern, get_path_output_sequences_padded_batch_pattern
from common import get_path_model, get_setting_evaluation_subset_size
import torch
import torch.nn as nn
import pickle
import logging

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

DATASET_NAME = 'ubuntu_dialogue_corpus_000'
LOG_BASE_FILENAME = "6_chatbot_interface"

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

def generate_response(model, input_text, vocab, device, max_length=50):
    """
    Generate a response for the given input_text using the Seq2Seq model.

    Args:
        model: Trained Seq2Seq model.
        input_text: User input text string.
        vocab: Vocabulary mapping tokens to indices.
        device: The device (CPU/GPU) to use.
        max_length: Maximum length of the generated response.

    Returns:
        response: Generated response as a string.
    """
    model.eval()
    with torch.no_grad():
        # Tokenize and convert input to indices
        tokens = ["<bos>"] + input_text.split() + ["<eos>"]
        input_indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

        # Encode the input
        encoder_outputs, hidden = model.encoder(input_tensor)

        # Prepare decoder input (<bos> token)
        trg_idx = vocab["<bos>"]
        trg_tensor = torch.tensor([trg_idx], dtype=torch.long).unsqueeze(0).to(device)

        # Generate response
        response_tokens = []
        for _ in range(max_length):
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
            trg_idx = output.argmax(2).item()  # Get the predicted token index
            trg_tensor = torch.tensor([[trg_idx]], dtype=torch.long).to(device)

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
        tokens = ["<bos>"] + input_text.split() + ["<eos>"]
        input_indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

        # Encode the input
        encoder_outputs, hidden = model.encoder(input_tensor)

        # Initialize beams
        beams = [([vocab["<bos>"]], 0.0, hidden)]  # List of (sequence, score, hidden state)

        for _ in range(max_length):
            all_candidates = []
            for seq, score, hidden in beams:
                # Get the last token in the sequence
                trg_tensor = torch.tensor([seq[-1]], dtype=torch.long).unsqueeze(0).to(device)

                # Decode next token
                output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
                log_probs = torch.log_softmax(output, dim=-1).squeeze(0)

                # Expand each beam with top beam_width tokens
                top_tokens = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    next_token = top_tokens.indices[i].item()
                    next_score = score + top_tokens.values[i].item()
                    all_candidates.append((seq + [next_token], next_score, hidden))

            # Prune to the top beam_width candidates
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Stop if all beams end with <eos>
            if all(seq[-1] == vocab["<eos>"] for seq, _, _ in beams):
                break

        # Select the best beam (highest score)
        best_sequence = beams[0][0]

        # Convert indices back to tokens
        idx_to_word = {idx: token for token, idx in vocab.items()}
        response_tokens = [idx_to_word[idx] for idx in best_sequence if idx not in {vocab["<bos>"], vocab["<eos>"]}]

        return " ".join(response_tokens)

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
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Generate response
        preprocessed_input = preprocess_input(user_input)
        logger.info(f"Preprocessed Input: {preprocessed_input}")
        response = generate_response(model, preprocessed_input, vocab, device)
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

    # Launch chatbot interface
    chatbot_interface(model, vocab, device)
