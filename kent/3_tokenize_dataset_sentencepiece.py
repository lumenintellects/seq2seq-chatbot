import os
import time
import logging
import pandas as pd
import sentencepiece as sp
from common import COL_INPUT, COL_OUTPUT, EMPTY_STRING, FILE_MODE_APPEND, FILE_MODE_WRITE, FOLDER_DATASET, PATH_WORKSPACE_ROOT, VOCAB_BOS, VOCAB_EOS, VOCAB_PAD, VOCAB_UNK, get_path_input_sequences, get_path_input_sequences_padded_batch_pattern, get_path_log, get_path_input_output_pairs, get_path_output_sequences, get_path_output_sequences_padded_batch_pattern, get_path_sentencepiece_combined_text, get_path_sentencepiece_model

DATASET_NAME = 'ubuntu_dialogue_corpus_000'
LOG_BASE_FILENAME = "3_tokenize_dataset_sentencepiece"

VOCAB_SIZE_DEFAULT = 32000
CHAR_COVERAGE_DEFAULT = 0.995

N_PROCESS_VALUE = 10
BATCH_SIZE = 500000
TRAINING_SUBSET_SIZE = 200
SETTING_ANALYZE_SEQUENCES = False
LOSS_THRESHOLD = 1.0

# ==========================

def train_sentencepiece(df, combined_text_path, model_prefix, vocab_size=32000, character_coverage=1.0):
    """
    Train a SentencePiece model on a dataset.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'input' and 'output' columns.
        combined_text_path (str): Path to save the combined input-output text file for training.
        model_prefix (str): Prefix for the output SentencePiece model files.
        vocab_size (int, optional): Size of the vocabulary for SentencePiece. Defaults to 32000.
        character_coverage (float, optional): Coverage of the character set for SentencePiece. Defaults to 1.0.
    
    Returns:
        None
    """
    # Combine input and output into one file for training
    df[COL_INPUT].to_csv(combined_text_path, index=False, header=False, mode=FILE_MODE_WRITE)
    df[COL_OUTPUT].to_csv(combined_text_path, index=False, header=False, mode=FILE_MODE_APPEND)

    # Train the SentencePiece model
    sp.SentencePieceTrainer.train(
        input=combined_text_path, 
        model_prefix=model_prefix, 
        vocab_size=vocab_size, 
        character_coverage=character_coverage
    )

def sentencepiece_tokenizer(texts, sentencepiece_model):
    """
    Tokenizes a list of texts using a SentencePiece model.

    Parameters:
        texts (list): List of text strings to tokenize.
        sentencepiece_model: Loaded SentencePiece model.

    Returns:
        list: List of tokenized texts (as integer IDs).
    """
    logger.info(f"Tokenizing {len(texts)} texts using SentencePiece...")
    return [sentencepiece_model.encode(text, out_type=int) for text in texts]

def process_text(texts, sentencepiece_model):
    """
    Tokenizes a list of texts and converts them to sequences of indices using SentencePiece.

    Parameters:
        texts (list): List of text strings to process.
        sentencepiece_model: Loaded SentencePiece model.

    Returns:
        list: List of sequences of indices.
    """
    logger.info(f"Processing {len(texts)} texts with SentencePiece...")
    return [sentencepiece_model.encode(text, out_type=int) for text in texts]

# ==========================

if __name__ == "__main__":

    # Set the current working directory
    os.chdir(PATH_WORKSPACE_ROOT)

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
    logger.info(f"Current Working Directory: {os.getcwd()}")

    # ==========================

    path_input_output_pairs = get_path_input_output_pairs(DATASET_NAME)
    SENTENCEPIECE_MODEL_NAME = DATASET_NAME
    path_sentencepiece_model = get_path_sentencepiece_model(SENTENCEPIECE_MODEL_NAME)
    path_combined_text = get_path_sentencepiece_combined_text(DATASET_NAME)
    path_input_sequences = get_path_input_sequences(DATASET_NAME)
    path_output_sequences = get_path_output_sequences(DATASET_NAME)
    path_input_sequences_padded_batch_pattern = get_path_input_sequences_padded_batch_pattern(DATASET_NAME)
    path_output_sequences_padded_batch_pattern = get_path_output_sequences_padded_batch_pattern(DATASET_NAME)

    # ==========================

    # Load the dataset
    df = pd.read_csv(path_input_output_pairs)
    logger.info(f"Loaded csv into dataframe: {path_input_output_pairs}")

    # Replace NaN in COL_INPUT and COL_OUTPUT columns
    df[COL_INPUT] = df[COL_INPUT].fillna(EMPTY_STRING)
    df[COL_OUTPUT] = df[COL_OUTPUT].fillna(EMPTY_STRING)
    logger.info("NaN replaced with empty strings.")

    # check for existing SentencePiece model
    if os.path.exists(path_sentencepiece_model):
        logger.info(f"Found existing SentencePiece model: {path_sentencepiece_model}")
        model = sp.SentencePieceProcessor(model_file=path_sentencepiece_model)
        logger.info(f"Loaded SentencePiece model.")
    else:
        logger.info(f"No existing SentencePiece model found. Training new model...")
        dir_model_name = os.path.join(FOLDER_DATASET, SENTENCEPIECE_MODEL_NAME)
        train_sentencepiece(
            df, path_combined_text, dir_model_name,
            vocab_size=VOCAB_SIZE_DEFAULT, character_coverage=CHAR_COVERAGE_DEFAULT)
        logger.info(f"SentencePiece model training complete. Model files saved to: {path_sentencepiece_model}")

    #vocab = {VOCAB_PAD: sp.pad_id(), VOCAB_UNK: sp.unk_id(), VOCAB_BOS: sp.bos_id(), VOCAB_EOS: sp.eos_id()}
    #padding_value = vocab[VOCAB_PAD]

    #input_sequences = process_text(df[COL_INPUT].tolist(), sp)
    #output_sequences = process_text(df[COL_OUTPUT].tolist(), sp)
