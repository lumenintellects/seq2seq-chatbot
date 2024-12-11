import glob
import os
import time
import logging
import pandas as pd
import sentencepiece as sp
from common import COL_INPUT, COL_OUTPUT, EMPTY_STRING, FILE_MODE_APPEND, FILE_MODE_WRITE, FOLDER_DATASET, PATH_WORKSPACE_ROOT, VOCAB_BOS, VOCAB_EOS, VOCAB_PAD, VOCAB_UNK, get_base_filename_sentencepiece_model, get_path_input_sequences, get_path_input_sequences_padded_batch, get_path_input_sequences_padded_batch_pattern, get_path_log, get_path_input_output_pairs, get_path_output_sequences, get_path_output_sequences_padded_batch, get_path_output_sequences_padded_batch_pattern, get_path_sentencepiece_combined_text, get_path_sentencepiece_model
import torch

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
        sp_model = sp.SentencePieceProcessor(model_file=path_sentencepiece_model)
        logger.info(f"Loaded SentencePiece model.")
    else:
        logger.info(f"No existing SentencePiece model found. Training new model...")
        dir_model_name = os.path.join(FOLDER_DATASET, get_base_filename_sentencepiece_model(SENTENCEPIECE_MODEL_NAME))
        train_sentencepiece(
            df, path_combined_text, dir_model_name,
            vocab_size=VOCAB_SIZE_DEFAULT, character_coverage=CHAR_COVERAGE_DEFAULT)
        logger.info(f"SentencePiece model training complete. Model files saved to: {path_sentencepiece_model}")

    vocab = {VOCAB_PAD: sp_model.pad_id(), VOCAB_UNK: sp_model.unk_id(), VOCAB_BOS: sp_model.bos_id(), VOCAB_EOS: sp_model.eos_id()}
    padding_value = vocab[VOCAB_PAD]

# ==========================

    # Check for previous serialized input sequences
    if os.path.exists(path_input_sequences):
        logger.info("Serialized input sequences found.")
    else:
        logger.info("Serialized input sequences not found, generating input sequences...")

        # Tokenize and convert to sequences
        logger.info("Tokenizing and converting to input sequences...")

        input_texts = df[COL_INPUT].tolist()

        # Process input sequences in parallel
        time_input_sequences_start = time.time()
        time_input_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_input_sequences_start))
        logger.info(f"The time is: {time_input_sequences_start_hh_mm_ss}")
        input_sequences = process_text(input_texts, sp_model)
        logger.info(f"Processed {len(input_sequences)} input sequences.")

        input_lengths = [len(seq) for seq in input_sequences]
        input_max_length = max(input_lengths)
        logger.info(f"Max input length: {input_max_length}")

        input_mean_length = sum(input_lengths) / len(input_lengths)
        logger.info(f"Mean input length: {input_mean_length}")

        input_median_length = sorted(input_lengths)[len(input_lengths) // 2]
        logger.info(f"Median input length: {input_median_length}")

        input_percentile_95 = np.percentile(input_lengths, 95)
        logger.info(f"95th percentile input length: {input_percentile_95}")

        # truncate input sequences longer than the 95th percentile
        logger.info("Truncating input sequences longer than the 95th percentile...")
        input_max_length = int(input_percentile_95)
        input_sequences = [seq[:input_max_length] for seq in input_sequences]
        logger.info(f"Truncated input sequences to {input_max_length} tokens.")

        torch.save(input_sequences, path_input_sequences)
        time_input_sequences_end = time.time()
        logger.info(f"Input sequences completed in {time_input_sequences_end - time_input_sequences_start} seconds.")

    # Check for previous serialized padded input sequences matching batch file name pattern
    if len(glob.glob(path_input_sequences_padded_batch_pattern)) > 0:
        logger.info("Serialized padded input sequences found.")
    else:
        logger.info("Serialized padded input sequences not found, padding input sequences...")
        input_sequences = torch.load(path_input_sequences, weights_only=True)

        time_pad_input_sequences_start = time.time()
        time_pad_input_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_pad_input_sequences_start))
        logger.info(f"The time is: {time_pad_input_sequences_start_hh_mm_ss}")

        # Process sequences in batches to avoid memory issues
        for i in range(0, len(input_sequences), BATCH_SIZE):
            batch = input_sequences[i:i + BATCH_SIZE]
            logger.info(f"Padding sequences in batch {i // BATCH_SIZE} to {input_max_length}")
            padded_batch = pad_to_length(batch, input_max_length, padding_value)  # Use explicit padding

            # Examine the padded batch
            logger.info(f"Batch {i // BATCH_SIZE} shape: {padded_batch.shape}")

            batch_file_path = get_path_input_sequences_padded_batch(DATASET_NAME, i // BATCH_SIZE)
            torch.save(padded_batch, batch_file_path)
            logger.info(f"Saved batch {i // BATCH_SIZE} to {batch_file_path}")

        time_pad_input_sequences_end = time.time()
        logger.info(f"Padding input sequences completed in {time_pad_input_sequences_end - time_pad_input_sequences_start} seconds.")

        logger.info("Exiting program.")
        exit()

# ==========================

    # Check for previous serialized output sequences
    if os.path.exists(path_output_sequences):
        logger.info("Serialized output sequences found.")
    else:
        logger.info("Serialized output sequences not found, generating output sequences...")

        # Tokenize and convert to sequences
        logger.info("Tokenizing and converting to output sequences...")

        output_texts = df[COL_OUTPUT].tolist()

        # Process output sequences in parallel
        time_output_sequences_start = time.time()
        time_output_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_output_sequences_start))
        logger.info(f"The time is: {time_output_sequences_start_hh_mm_ss}")
        output_sequences = process_text(output_texts, sp_model)
        logger.info(f"Processed {len(output_sequences)} output sequences.")

        output_lengths = [len(seq) for seq in output_sequences]
        output_max_length = max(output_lengths)
        logger.info(f"Max output length: {output_max_length}")

        output_mean_length = sum(output_lengths) / len(output_lengths)
        logger.info(f"Mean output length: {output_mean_length}")

        output_median_length = sorted(output_lengths)[len(output_lengths) // 2]
        logger.info(f"Median output length: {output_median_length}")

        output_percentile_95 = np.percentile(output_lengths, 95)
        logger.info(f"95th percentile output length: {output_percentile_95}")

        # truncate output sequences longer than the 95th percentile
        logger.info("Truncating output sequences longer than the 95th percentile...")
        output_max_length = int(output_percentile_95)
        output_sequences = [seq[:output_max_length] for seq in output_sequences]
        logger.info(f"Truncated output sequences to {output_max_length} tokens.")

        torch.save(output_sequences, path_output_sequences)
        time_output_sequences_end = time.time()
        logger.info(f"output sequences completed in {time_output_sequences_end - time_output_sequences_start} seconds.")

    # Check for previous serialized padded output sequences matching batch file name pattern
    if len(glob.glob(path_output_sequences_padded_batch_pattern)) > 0:
        logger.info("Serialized padded output sequences found.")
    else:
        logger.info("Serialized padded output sequences not found, padding output sequences...")
        output_sequences = torch.load(path_output_sequences, weights_only=True)

        time_pad_output_sequences_start = time.time()
        time_pad_output_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_pad_output_sequences_start))
        logger.info(f"The time is: {time_pad_output_sequences_start_hh_mm_ss}")

        # Process sequences in batches to avoid memory issues
        for i in range(0, len(output_sequences), BATCH_SIZE):
            batch = output_sequences[i:i + BATCH_SIZE]
            logger.info(f"Padding sequences in batch {i // BATCH_SIZE} to {output_max_length}")
            padded_batch = pad_to_length(batch, output_max_length, padding_value)  # Use explicit padding

            # Examine the padded batch
            logger.info(f"Batch {i // BATCH_SIZE} shape: {padded_batch.shape}")

            batch_file_path = get_path_output_sequences_padded_batch(DATASET_NAME, i // BATCH_SIZE)
            torch.save(padded_batch, batch_file_path)
            logger.info(f"Saved batch {i // BATCH_SIZE} to {batch_file_path}")

        time_pad_output_sequences_end = time.time()
        logger.info(f"Padding output sequences completed in {time_pad_output_sequences_end - time_pad_output_sequences_start} seconds.")

        logger.info("Exiting program.")
        exit()

    input_sequences_padded = torch.cat([torch.load(file, weights_only=True) for file in glob.glob(path_input_sequences_padded_batch_pattern)], dim=0)
    logger.info("Loaded input sequences from files.")

    output_sequences_padded = torch.cat([torch.load(file, weights_only=True) for file in glob.glob(path_output_sequences_padded_batch_pattern)], dim=0)
    logger.info("Loaded output sequences from file.")

    # Analyze sequences
    if SETTING_ANALYZE_SEQUENCES:
        logger.info("Analyzing input and output sequences...")
        analyze_sequences(input_sequences_padded)
        analyze_sequences(output_sequences_padded)

        logger.info(f"Input shape: {input_sequences_padded.shape}")
        logger.info(f"Output shape: {output_sequences_padded.shape}")
