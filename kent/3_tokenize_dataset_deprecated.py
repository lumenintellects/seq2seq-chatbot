import glob
import os
import time
import logging
import pandas as pd
import spacy
from common import PATH_WORKSPACE_ROOT, get_path_log, get_path_input_output_pairs, get_path_vocab
from common import get_path_input_sequences, get_path_output_sequences
from common import get_path_input_sequences_padded_batch, get_path_output_sequences_padded_batch
from common import get_path_input_sequences_padded_batch_pattern, get_path_output_sequences_padded_batch_pattern
import torch
import pickle
import numpy as np

DATASET_NAME = 'ubuntu_dialogue_corpus_000'
LOG_BASE_FILENAME = "3_tokenize_dataset"

N_PROCESS_VALUE = 10
BATCH_SIZE = 500000
TRAINING_SUBSET_SIZE = 200
SETTING_ANALYZE_SEQUENCES = False
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
        tokenized_texts.append([token.text for token in doc if not token.is_space])
    return tokenized_texts

# Build vocabulary from spaCy tokens
def build_vocab(tokens_iterable, specials=["<unk>", "<pad>", "<bos>", "<eos>"]):
    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    for tokens in tokens_iterable:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

# Prune vocabulary by token frequency
def build_pruned_vocab(tokens_iterable, min_freq=2, specials=["<unk>", "<pad>", "<bos>", "<eos>"]):
    """
    Prunes tokens from the vocabulary that appear less than `min_freq`.

    Parameters:
        tokens_iterable (list of lists): Iterable of tokenized texts.
        min_freq (int): Minimum frequency for tokens to be kept.
        specials (list): Special tokens that should always be kept.

    Returns:
        dict: Pruned vocabulary mapping tokens to indices.
        dict: Token frequencies for reference.
    """
    # Count token frequencies
    token_freq = {}
    for tokens in tokens_iterable:
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1

    # Build pruned vocabulary
    vocab = {special: i for i, special in enumerate(specials)}
    for token, freq in token_freq.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    return vocab, token_freq

# Log token frequency distribution
def log_token_frequency_stats(token_freq):
    """
    Logs statistics about token frequency distribution.

    Parameters:
        token_freq (dict): Dictionary of token frequencies.
    """
    frequencies = list(token_freq.values())
    mean_freq = np.mean(frequencies)
    median_freq = np.median(frequencies)
    max_freq = np.max(frequencies)
    percentile_95 = np.percentile(frequencies, 95)

    logger.info(f"Token Frequency Stats: Mean={mean_freq}, Median={median_freq}, Max={max_freq}, 95th Percentile={percentile_95}")
    logger.info(f"Total Unique Tokens: {len(token_freq)}")

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
        tokens = ["<bos>"] + [token.text for token in doc if not token.is_space] + ["<eos>"]
        tokenized_sequences.append([vocab.get(token, vocab["<unk>"]) for token in tokens])
    return tokenized_sequences

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

    # Load spaCy language model
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded.")

    # ==========================

    path_input_output_pairs = get_path_input_output_pairs(DATASET_NAME)
    path_vocab = get_path_vocab(DATASET_NAME)
    path_input_sequences = get_path_input_sequences(DATASET_NAME)
    path_output_sequences = get_path_output_sequences(DATASET_NAME)
    path_input_sequences_padded_batch_pattern = get_path_input_sequences_padded_batch_pattern(DATASET_NAME)
    path_output_sequences_padded_batch_pattern = get_path_output_sequences_padded_batch_pattern(DATASET_NAME)

    # Define the save path
    path_model = os.path.join(PATH_WORKSPACE_ROOT, "seq2seq_model.pth")

    # ==========================

    # Load the dataset
    df = pd.read_csv(path_input_output_pairs)
    logger.info(f"Loaded csv into dataframe: {path_input_output_pairs}")

    # Replace NaN in 'input' and 'output' columns
    df['input'] = df['input'].fillna("")
    df['output'] = df['output'].fillna("")
    logger.info("NaN replaced with empty strings.")

    # Check for existing vocabulary
    if os.path.exists(path_vocab):
        logger.info("Vocabulary file found. Loading vocabulary...")
        with open(path_vocab, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        logger.info(f"Vocabulary loaded. Size: {len(vocab)}")
    else:
        logger.info("Vocabulary file not found. Generating vocabulary...")

        # Tokenize using SpaCy's multithreading
        logger.info("Tokenizing input and output texts...")
        time_input_sequences_start = time.time()
        time_input_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_input_sequences_start))
        logger.info(f"The time is: {time_input_sequences_start_hh_mm_ss}")
        texts_combined = df['input'].tolist() + df['output'].tolist()
        combined_tokens = spacy_tokenizer_pipe(texts_combined, nlp, n_process=N_PROCESS_VALUE)
        time_input_sequences_end = time.time()
        logger.info(f"Tokenization completed in {time_input_sequences_end - time_input_sequences_start:.2f} seconds.")

        # Build pruned vocabulary based on frequency
        logger.info("Building pruned vocabulary...")
        time_build_pruned_vocab_start = time.time()
        vocab, token_freq = build_pruned_vocab(combined_tokens, min_freq=2)  # Use MIN_FREQ=2 as default
        time_build_pruned_vocab_end = time.time()
        logger.info(f"Pruned vocabulary built in {time_build_pruned_vocab_end - time_build_pruned_vocab_start:.2f} seconds.")
        logger.info(f"Pruned vocabulary size: {len(vocab)}")

        # Log statistics
        log_token_frequency_stats(token_freq)

        # Save the pruned vocabulary to file
        with open(path_vocab, "wb") as vocab_file:
            pickle.dump(vocab, vocab_file)
        logger.info("Pruned vocabulary saved to file.")

    padding_value = vocab["<pad>"]

    # Check for previous serialized input sequences
    if os.path.exists(path_input_sequences):
        logger.info("Serialized input sequences found.")
    else:
        logger.info("Serialized input sequences not found, generating input sequences...")

        # Tokenize and convert to sequences
        logger.info("Tokenizing and converting to input sequences...")

        input_texts = df['input'].tolist()

        # Process input sequences in parallel
        time_input_sequences_start = time.time()
        time_input_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_input_sequences_start))
        logger.info(f"The time is: {time_input_sequences_start_hh_mm_ss}")
        input_sequences = process_text_spacy_pipe(input_texts, vocab, nlp, n_process=N_PROCESS_VALUE)
        torch.save(input_sequences, path_input_sequences)
        time_input_sequences_end = time.time()
        logger.info(f"Input sequences completed in {time_input_sequences_end - time_input_sequences_start} seconds.")

    # Check for previous serialized padded input sequences matching batch file name pattern
    if len(glob.glob(path_input_sequences_padded_batch_pattern)) > 0:
        logger.info("Serialized padded input sequences found.")
    else:
        input_sequences = torch.load(path_input_sequences, weights_only=True)

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

        logger.info("Serialized padded input sequences not found, padding input sequences...")
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

    # Check for previous serialized output sequences
    if os.path.exists(path_output_sequences):
        logger.info("Serialized output sequences found.")
    else:
        logger.info("Serialized output sequences not found, generating output sequences...")

        # Tokenize and convert to sequences
        logger.info("Tokenizing and converting to output sequences...")

        output_texts = df['output'].tolist()

        # Process output sequences in parallel
        time_output_sequences_start = time.time()
        time_output_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_output_sequences_start))
        logger.info(f"The time is: {time_output_sequences_start_hh_mm_ss}")
        output_sequences = process_text_spacy_pipe(output_texts, vocab, nlp, n_process=N_PROCESS_VALUE)
        torch.save(output_sequences, path_output_sequences)
        time_output_sequences_end = time.time()
        logger.info(f"Output sequences completed in {time_output_sequences_end - time_output_sequences_start} seconds.")

    # Check for previous serialized padded output sequences matching batch file name pattern
    if len(glob.glob(path_output_sequences_padded_batch_pattern)) > 0:
        logger.info("Serialized padded output sequences found.")
    else:
        output_sequences = torch.load(path_output_sequences, weights_only=True)

        output_lengths = [len(seq) for seq in output_sequences]
        output_max_length = max(output_lengths)
        logger.info(f"Max output length: output_max_length")

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

        logger.info("Serialized padded output sequences not found, padding output sequences...")
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
