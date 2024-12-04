import glob
import os
import time
import pandas as pd
import spacy
from common import PATH_WORKSPACE_ROOT, FOLDER_DATASET, BASE_FILENAME_MODEL
from common import csv_filename, pt_filename, pkl_filename, pth_filename
import torch
import pickle
import numpy as np

N_PROCESS_VALUE = 8
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
    print(f"Tokenizing {len(texts)} texts using {n_process} processes...")
    tokenized_texts = []
    for doc in nlp.pipe(texts, n_process=n_process):
        tokenized_texts.append([token.text.lower() for token in doc if not token.is_space])
    return tokenized_texts

# Tokenizer using spaCy
def spacy_tokenizer(text):
    return [token.text.lower() for token in nlp(text) if not token.is_space]

# Tokenize the input and output texts using spaCy
def yield_tokens_spacy(texts):
    for text in texts:
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string. Got: {text}")
        yield spacy_tokenizer(text)

# Build vocabulary from spaCy tokens
def build_vocab(tokens_iterable, specials=["<unk>", "<pad>", "<bos>", "<eos>"]):
    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    for tokens in tokens_iterable:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

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
    print(f"Processing {len(texts)} texts using {n_process} processes...")
    tokenized_sequences = []
    for doc in nlp.pipe(texts, n_process=n_process):
        tokens = ["<bos>"] + [token.text.lower() for token in doc if not token.is_space] + ["<eos>"]
        tokenized_sequences.append([vocab.get(token, vocab["<unk>"]) for token in tokens])
    return tokenized_sequences

# Process text into sequences of indices
def process_text_spacy(text, vocab):
    tokens = ["<bos>"] + spacy_tokenizer(text) + ["<eos>"]
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

def analyze_sequences(sequences):
    sequence_lengths = [len(seq) for seq in sequences]
    max_length = max(sequence_lengths)
    mean_length = sum(sequence_lengths) / len(sequence_lengths)
    median_length = sorted(sequence_lengths)[len(sequence_lengths) // 2]

    # Percentiles
    import numpy as np
    percentile_95 = np.percentile(sequence_lengths, 95)

    print(f"Max Length: {max_length}")
    print(f"Mean Length: {mean_length}")
    print(f"Median Length: {median_length}")
    print(f"95th Percentile: {percentile_95}")

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
    print("Running main script...")

    # Load spaCy language model
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded.")

    # Set the current working directory
    os.chdir(PATH_WORKSPACE_ROOT)
    print("Current Working Directory:", os.getcwd())

    # ==========================

    base_filename = 'ubuntu_dialogue_corpus_196_input_output_pairs'
    path_input_csv = os.path.join(FOLDER_DATASET, csv_filename(base_filename))
    path_vocab_pkl = os.path.join(FOLDER_DATASET, pkl_filename(f"{base_filename}_vocab"))
    path_input_sequences = os.path.join(FOLDER_DATASET, pt_filename(f"{base_filename}_input_sequences"))
    path_output_sequences = os.path.join(FOLDER_DATASET, pt_filename(f"{base_filename}_output_sequences"))
    path_input_sequences_padded_batch_pattern = os.path.join(FOLDER_DATASET, pt_filename(f"{base_filename}_input_sequences_padded_batch_*"))
    path_output_sequences_padded_batch_pattern = os.path.join(FOLDER_DATASET, pt_filename(f"{base_filename}_output_sequences_padded_batch_*"))

    # Define the save path
    path_model = os.path.join(PATH_WORKSPACE_ROOT, pth_filename(BASE_FILENAME_MODEL))

    # ==========================

    # Load the dataset
    df = pd.read_csv(path_input_csv)
    print("Loaded csv into dataframe:", path_input_csv)

    # Replace NaN in 'input' and 'output' columns
    df['input'] = df['input'].fillna("")
    df['output'] = df['output'].fillna("")
    print("NaN replaced with empty strings.")

    # Check for existing vocabulary
    if os.path.exists(path_vocab_pkl):
        print("Vocabulary file found. Loading vocabulary...")
        with open(path_vocab_pkl, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        print(f"Vocabulary loaded. Size: {len(vocab)}")
    else:
        print("Vocabulary file not found. Generating vocabulary...")

        # Tokenize using SpaCy's multithreading
        print("Tokenizing input and output texts...")
        time_input_sequences_start = time.time()
        time_input_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_input_sequences_start))
        print("The time is:", time_input_sequences_start_hh_mm_ss)
        texts_combined = df['input'].tolist() + df['output'].tolist()
        combined_tokens = spacy_tokenizer_pipe(texts_combined, nlp, n_process=N_PROCESS_VALUE)
        time_input_sequences_end = time.time()
        print(f"Tokenization completed in {time_input_sequences_end - time_input_sequences_start:.2f} seconds.")

        # Build vocabulary from tokenized texts
        print("Building vocabulary...")
        time_build_vocab_start = time.time()
        time_build_vocab_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_build_vocab_start))
        print("The time is:", time_build_vocab_start_hh_mm_ss)
        vocab = build_vocab(combined_tokens)
        time_build_vocab_end = time.time()
        print(f"Vocabulary built in {time_build_vocab_end - time_build_vocab_start} seconds.")
        print(f"Vocabulary built. Size: {len(vocab)}")

        # Save the vocabulary to file
        with open(path_vocab_pkl, "wb") as vocab_file:
            pickle.dump(vocab, vocab_file)
        print("Vocabulary saved to file.")

    padding_value = vocab["<pad>"]

    # Check for previous serialized input sequences
    if os.path.exists(path_input_sequences):
        print("Serialized input sequences found.")
    else:
        print("Serialized input sequences not found, generating input sequences...")

        # Tokenize and convert to sequences
        print("Tokenizing and converting to input sequences...")

        input_texts = df['input'].tolist()

        # Process input sequences in parallel
        time_input_sequences_start = time.time()
        time_input_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_input_sequences_start))
        print("The time is:", time_input_sequences_start_hh_mm_ss)
        input_sequences = process_text_spacy_pipe(input_texts, vocab, nlp, n_process=N_PROCESS_VALUE)
        torch.save(input_sequences, path_input_sequences)
        time_input_sequences_end = time.time()
        print(f"Input sequences completed in {time_input_sequences_end - time_input_sequences_start} seconds.")

    # Check for previous serialized padded input sequences matching batch file name pattern
    if len(glob.glob(path_input_sequences_padded_batch_pattern)) > 0:
        print("Serialized padded input sequences found.")
    else:
        input_sequences = torch.load(path_input_sequences)

        input_lengths = [len(seq) for seq in input_sequences]
        input_max_length = max(input_lengths)
        print("Max input length:", input_max_length)

        input_mean_length = sum(input_lengths) / len(input_lengths)
        print("Mean input length:", input_mean_length)

        input_median_length = sorted(input_lengths)[len(input_lengths) // 2]
        print("Median input length:", input_median_length)

        input_percentile_95 = np.percentile(input_lengths, 95)
        print("95th percentile input length:", input_percentile_95)

        # truncate input sequences longer than the 95th percentile
        print("Truncating input sequences longer than the 95th percentile...")
        input_max_length = int(input_percentile_95)
        input_sequences = [seq[:input_max_length] for seq in input_sequences]

        print("Serialized padded input sequences not found, padding input sequences...")
        time_pad_input_sequences_start = time.time()
        time_pad_input_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_pad_input_sequences_start))
        print("The time is:", time_pad_input_sequences_start_hh_mm_ss)

        # Process sequences in batches to avoid memory issues
        for i in range(0, len(input_sequences), BATCH_SIZE):
            batch = input_sequences[i:i + BATCH_SIZE]
            print(f"Padding sequences in batch {i // BATCH_SIZE} to {input_max_length}")
            padded_batch = pad_to_length(batch, input_max_length, padding_value)  # Use explicit padding

            # Examine the padded batch
            print(f"Batch {i // BATCH_SIZE} shape:", padded_batch.shape)

            batch_file_name = pt_filename(f"{base_filename}_input_sequences_padded_batch_{i // BATCH_SIZE}")
            batch_file_path = os.path.join(FOLDER_DATASET, batch_file_name)
            torch.save(padded_batch, batch_file_path)
            print(f"Saved batch {i // BATCH_SIZE} to {batch_file_path}")

        time_pad_input_sequences_end = time.time()
        print(f"Padding input sequences completed in {time_pad_input_sequences_end - time_pad_input_sequences_start} seconds.")

    # Check for previous serialized output sequences
    if os.path.exists(path_output_sequences):
        print("Serialized output sequences found.")
    else:
        print("Serialized output sequences not found, generating output sequences...")

        # Tokenize and convert to sequences
        print("Tokenizing and converting to output sequences...")

        output_texts = df['output'].tolist()

        # Process output sequences in parallel
        time_output_sequences_start = time.time()
        time_output_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_output_sequences_start))
        print("The time is:", time_output_sequences_start_hh_mm_ss)
        output_sequences = process_text_spacy_pipe(output_texts, vocab, nlp, n_process=N_PROCESS_VALUE)
        torch.save(output_sequences, path_output_sequences)
        time_output_sequences_end = time.time()
        print(f"Output sequences completed in {time_output_sequences_end - time_output_sequences_start} seconds.")

    # Check for previous serialized padded output sequences matching batch file name pattern
    if len(glob.glob(path_output_sequences_padded_batch_pattern)) > 0:
        print("Serialized padded output sequences found.")
    else:
        output_sequences = torch.load(path_output_sequences)

        output_lengths = [len(seq) for seq in output_sequences]
        output_max_length = max(output_lengths)
        print("Max output length:", output_max_length)

        output_mean_length = sum(output_lengths) / len(output_lengths)
        print("Mean output length:", output_mean_length)

        output_median_length = sorted(output_lengths)[len(output_lengths) // 2]
        print("Median output length:", output_median_length)

        output_percentile_95 = np.percentile(output_lengths, 95)
        print("95th percentile input length:", output_percentile_95)

        # truncate output sequences longer than the 95th percentile
        print("Truncating output sequences longer than the 95th percentile...")
        output_max_length = int(output_percentile_95)
        output_sequences = [seq[:output_max_length] for seq in output_sequences]

        print("Serialized padded output sequences not found, padding output sequences...")
        time_pad_output_sequences_start = time.time()
        time_pad_output_sequences_start_hh_mm_ss = time.strftime('%H:%M:%S', time.localtime(time_pad_output_sequences_start))
        print("The time is:", time_pad_output_sequences_start_hh_mm_ss)

        # Process sequences in batches to avoid memory issues
        for i in range(0, len(output_sequences), BATCH_SIZE):
            batch = output_sequences[i:i + BATCH_SIZE]
            print(f"Padding sequences in batch {i // BATCH_SIZE} to {output_max_length}")
            padded_batch = pad_to_length(batch, output_max_length, padding_value)  # Use explicit padding

            # Examine the padded batch
            print(f"Batch {i // BATCH_SIZE} shape:", padded_batch.shape)

            batch_file_name = pt_filename(f"{base_filename}_output_sequences_padded_batch_{i // BATCH_SIZE}")
            batch_file_path = os.path.join(FOLDER_DATASET, batch_file_name)
            torch.save(padded_batch, batch_file_path)
            print(f"Saved batch {i // BATCH_SIZE} to {batch_file_path}")

        time_pad_output_sequences_end = time.time()
        print(f"Padding output sequences completed in {time_pad_output_sequences_end - time_pad_output_sequences_start} seconds.")
        print("Exiting program.")
        exit()

    input_sequences_padded = torch.cat([torch.load(file, weights_only=True) for file in glob.glob(path_input_sequences_padded_batch_pattern)], dim=0)
    print("Loaded input sequences from files.")

    output_sequences_padded = torch.cat([torch.load(file, weights_only=True) for file in glob.glob(path_output_sequences_padded_batch_pattern)], dim=0)
    print("Loaded output sequences from file.")

    # Analyze sequences
    if SETTING_ANALYZE_SEQUENCES:
        print("Analyzing input and output sequences...")
        analyze_sequences(input_sequences_padded)
        analyze_sequences(output_sequences_padded)

        print("Input shape:", input_sequences_padded.shape)
        print("Output shape:", output_sequences_padded.shape)
