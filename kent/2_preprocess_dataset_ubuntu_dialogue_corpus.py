import os
import time
import logging
import pandas as pd
import numpy as np
from common import PATH_WORKSPACE_ROOT, log_filename, clean_text, csv_filename
from concurrent.futures import ProcessPoolExecutor

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)
LOG_BASE_FILENAME = "2_preprocess_dataset_ubuntu_dialogue_corpus"
LOG_FOLDER = 'dataset'

N_CHUNKS = 8
LENGTH_PERCENTILE = 95  # Percentile for filtering sequence lengths

# ==========================

folder_dataset = 'dataset'
base_filename = 'ubuntu_dialogue_corpus_000'
path_input_csv = os.path.join(folder_dataset, csv_filename(base_filename))
path_output_csv = os.path.join(folder_dataset, csv_filename(base_filename + '_input_output_pairs'))

# ==========================

def process_dialog_lines(dialog_lines):
    """
    Process dialog lines to create input-output pairs.

    Parameters:
        dialog_lines (list): A list of tuples containing dialog lines, where each tuple is (sender, message).

    Returns:
        list: A list of tuples containing input-output pairs.
    """
    input_output_pairs = []
    current_input = ""
    current_output = ""
    current_sender = None

    for sender, message in dialog_lines:
        # If sender changes, finalize the previous pair
        if sender != current_sender:
            if current_input and current_output:
                # Append the input-output pair
                current_input_clean = clean_text(current_input)
                current_output_clean = clean_text(current_output)
                input_output_pairs.append((current_input_clean, current_output_clean))
            # Reset input to the last accumulated output, start new output
            current_input = current_output
            current_output = message
            current_sender = sender
        else:
            # Accumulate messages for the same sender
            current_output += f" {message}"

    # Handle the last accumulated input-output pair
    if current_input and current_output:
        input_output_pairs.append((clean_text(current_input), clean_text(current_output)))

    return input_output_pairs

def create_input_output_pairs_for_chunk(df_chunk, max_length):
    """
    Create input-output pairs for a chunk of the DataFrame, filtering pairs based on sequence length.

    Parameters:
        df_chunk (pd.DataFrame): A chunk of the dialogue dataset.
        max_length (int): Maximum allowed length for input or output sequences.

    Returns:
        list: A list of input-output pairs for the chunk.
    """
    dialog_lines = []
    input_output_pairs = []
    previous_dialogue_id = None

    for _, row in df_chunk.iterrows():
        current_dialogue_id = row['dialogueID']
        current_dialog_sender = row['from']
        current_text = row['text']

        if previous_dialogue_id is None:
            dialog_lines = []
        elif current_dialogue_id != previous_dialogue_id:
            # Process the previous dialogue
            dialog_input_output_pairs = process_dialog_lines(dialog_lines)
            # Filter out overlong pairs
            filtered_pairs = [
                pair for pair in dialog_input_output_pairs
                if len(pair[0].split()) <= max_length and len(pair[1].split()) <= max_length
            ]
            input_output_pairs.extend(filtered_pairs)
            dialog_lines = []  # Reset for the next dialogue

        # Append dialog line to dialog_lines
        dialog_lines.append((current_dialog_sender, current_text))
        previous_dialogue_id = current_dialogue_id

    # Process the last dialogue
    dialog_input_output_pairs = process_dialog_lines(dialog_lines)
    # Filter out overlong pairs
    filtered_pairs = [
        pair for pair in dialog_input_output_pairs
        if len(pair[0].split()) <= max_length and len(pair[1].split()) <= max_length
    ]
    input_output_pairs.extend(filtered_pairs)

    return input_output_pairs

def process_chunk(args):
    """
    Process a chunk of the DataFrame to generate input-output pairs.

    Parameters:
        args (tuple): (df_chunk, max_length)

    Returns:
        pd.DataFrame: A DataFrame containing input-output pairs for the chunk.
    """
    df_chunk, max_length = args
    pairs = create_input_output_pairs_for_chunk(df_chunk, max_length)
    return pd.DataFrame(pairs, columns=['input', 'output'])

# ==========================

if __name__ == "__main__":

    log_start_time = time.strftime('%Y%m%d_%H%M%S')
    path_log = os.path.join(LOG_FOLDER, log_filename(f"{LOG_BASE_FILENAME}_{log_start_time}"))

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

    # Load the dataset
    df = pd.read_csv(path_input_csv)

    # Drop unnecessary columns: 'folder', 'date'
    df = df.drop(columns=['folder', 'date'])

    # Filter rows where 'text' is not null
    df = df[df['text'].notnull()]

    row_count = df.shape[0]
    logger.info(f"Dataset Loaded: {row_count} rows")

    # Compute maximum allowed length based on 95th percentile
    all_texts = pd.concat([df['text']]).str.split().map(len)
    max_length = int(np.percentile(all_texts, LENGTH_PERCENTILE))
    logger.info(f"{LENGTH_PERCENTILE}th Percentile Length: {max_length}")

    # Split the DataFrame into chunks
    df_chunks = np.array_split(df, N_CHUNKS)
    logger.info(f"Dataset Split into {N_CHUNKS} Chunks")

    # Use ProcessPoolExecutor for parallel processing
    logger.info("Processing chunks in parallel...")
    start_time = pd.Timestamp.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Start Time: {start_time_str}")

    with ProcessPoolExecutor() as executor:
        chunk_results = list(executor.map(process_chunk, [(chunk, max_length) for chunk in df_chunks]))

    # Combine results from all chunks
    pairs_df = pd.concat(chunk_results, ignore_index=True)

    end_time = pd.Timestamp.now()
    duration_in_seconds = (end_time - start_time).total_seconds()
    duration_in_hour_minute_seconds = pd.to_datetime(duration_in_seconds, unit='s').strftime('%H:%M:%S')
    logger.info(f"Input-Output Pairs Generated in: {duration_in_hour_minute_seconds}")

    # Get the stats on the generated pairs
    filtered_pairs_count = pairs_df.shape[0]
    logger.info(f"Input-Output Pairs Generated: {filtered_pairs_count}")

    max_input_length = pairs_df['input'].str.split().map(len).max()
    logger.info(f"Max Input Length: {max_input_length}")

    mean_input_length = pairs_df['input'].str.split().map(len).mean()
    logger.info(f"Mean Input Length: {mean_input_length:.2f}")

    median_input_length = pairs_df['input'].str.split().map(len).median()
    logger.info(f"Median Input Length: {median_input_length}")

    percentile_95_input_length = np.percentile(pairs_df['input'].str.split().map(len), 95)
    logger.info(f"95th Percentile Input Length: {percentile_95_input_length}")

    max_output_length = pairs_df['output'].str.split().map(len).max()
    logger.info(f"Max Output Length: {max_output_length}")

    mean_output_length = pairs_df['output'].str.split().map(len).mean()
    logger.info(f"Mean Output Length: {mean_output_length:.2f}")

    median_output_length = pairs_df['output'].str.split().map(len).median()
    logger.info(f"Median Output Length: {median_output_length}")

    percentile_95_output_length = np.percentile(pairs_df['output'].str.split().map(len), 95)
    logger.info(f"95th Percentile Output Length: {percentile_95_output_length}")

    max_input_output_length = max(percentile_95_input_length, percentile_95_output_length)
    # filter out pairs with lengths greater than the 95th percentile
    filtered_pairs_df = pairs_df[
        (pairs_df['input'].str.split().map(len) <= max_input_output_length) &
        (pairs_df['output'].str.split().map(len) <= max_input_output_length)
    ]

    # Save the preprocessed data
    filtered_pairs_df.to_csv(path_output_csv, index=False)

    filtered_pairs_count = filtered_pairs_df.shape[0]
    logger.info(f"{filtered_pairs_count} Input-Output Pairs Saved to {path_output_csv}")

    # Example preview
    logger.info("Input-Output Pairs Example:")
    logger.info(pairs_df.head())
