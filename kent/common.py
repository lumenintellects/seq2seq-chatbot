import json
import os
import pandas as pd
import torch.nn as nn

PATH_WORKSPACE_ROOT = r'.' # Set the workspace root path here
FOLDER_DATASET = 'dataset' # Set the name of the folder containing the datasets here

EXTENSION_CSV = '.csv'
EXTENSION_PT = '.pt'
EXTENSION_PKL = '.pkl'
EXTENSION_JSON = '.json'
EXTENSION_LOG = '.log'
EXTENSION_PTH = '.pth'

MODE_READONLY = 'r'

BASE_FILENAME_SETTINGS = 'settings' # Set the base filename for the settings JSON file here
BASE_FILENAME_MODEL = 'seq2seq_model'

# Define the settings keys inside the settings JSON file
SETTING_ENABLE_LOGGING = 'enableLogging'
SETTING_TRAINING_LOOP_CONTINUE = 'trainingLoopContinue'
SETTING_NEXT_SUBSET_CONTINUE = 'nextSubsetContinue'
SETTING_ANALYZE_SEQUENCES = 'analyzeSequences'
SETTING_DEBUG_MODE = 'debugMode'
SETTING_TRAINING_SUBSET_SIZE = 'trainingSubsetSize'

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch_size, src_len, emb_dim)
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, src_len, hidden_dim), hidden: (n_layers, batch_size, hidden_dim)
        return hidden  # Return only the hidden state

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden):
        # trg shape: (batch_size, trg_len)
        # hidden shape: (n_layers, batch_size, hidden_dim)
        embedded = self.dropout(self.embedding(trg))  # (batch_size, trg_len, emb_dim)
        outputs, hidden = self.rnn(embedded, hidden)  # (batch_size, trg_len, hidden_dim), hidden: (n_layers, batch_size, hidden_dim)
        predictions = self.fc_out(outputs).float()  # Ensure predictions are float32
        return predictions, hidden

# Define the Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        # src: (batch_size, src_len), trg: (batch_size, trg_len)
        hidden = self.encoder(src)  # Get the context vector
        outputs, _ = self.decoder(trg, hidden)  # Decode based on the context vector
        return outputs

def get_setting(setting_name):
    """
    Get the value of a setting from the settings file.

    Parameters:
        setting_name (str): The name of the setting.

    Returns:
        str: The value of the setting.
    """
    filename_settings = json_filename(BASE_FILENAME_SETTINGS)
    relative_path_settings = os.path.join(FOLDER_DATASET, filename_settings)
    path_settings = os.path.join(PATH_WORKSPACE_ROOT, relative_path_settings)

    # Load the settings JSON file
    with open(path_settings, MODE_READONLY) as file:
        settings = json.load(file)
        return settings[setting_name]

def get_setting_training_subset_size():
    """
    Get the value of the setting 'trainingSubsetSize' from the settings file.

    Returns:
        int: The value of the setting 'trainingSubsetSize'.
    """
    return get_setting(SETTING_TRAINING_SUBSET_SIZE)

def get_setting_debug_mode():
    """
    Get the value of the setting 'debugMode' from the settings file.

    Returns:
        bool: The value of the setting 'debugMode'.
    """
    return get_setting(SETTING_DEBUG_MODE)

def get_setting_analyze_sequences():
    """
    Get the value of the setting 'analyzeSequences' from the settings file.

    Returns:
        bool: The value of the setting 'analyzeSequences'.
    """
    return get_setting(SETTING_ANALYZE_SEQUENCES)

def get_setting_enable_logging():
    """
    Get the value of the setting 'enableLogging' from the settings file.

    Returns:
        bool: The value of the setting 'enableLogging'.
    """
    return get_setting(SETTING_ENABLE_LOGGING)

def get_setting_training_loop_continue():
    """
    Get the value of the setting 'trainingLoopContinue' from the settings file.

    Returns:
        bool: The value of the setting 'trainingLoopContinue'.
    """
    return get_setting(SETTING_TRAINING_LOOP_CONTINUE)

def get_setting_next_subset_continue():
    """
    Get the value of the setting 'nextSubsetContinue' from the settings file.

    Returns:
        bool: The value of the setting 'nextSubsetContinue'.
    """
    return get_setting(SETTING_NEXT_SUBSET_CONTINUE)

def extract_rows(input_csv, output_csv, num_rows):
    """
    Extract the first X rows from a CSV file and save to a new file.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the output CSV file.
        num_rows (int): Number of rows to extract.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(input_csv)

        # Extract the first num_rows rows
        sampled_df = df.head(num_rows)

        # Save the extracted rows to a new CSV file
        sampled_df.to_csv(output_csv, index=False)

        print(f"Successfully extracted {num_rows} rows to {output_csv}")
    except Exception as e:
        print(f"Error: {e}")

# for each row in the input csv, evaluate each cell against the filter_predicate and write to output csv
def filter_csv(input_csv, output_csv, filter_predicate):
    try:
        # Load the CSV file
        df = pd.read_csv(input_csv)

        # Filter the dataset
        filtered_df = df[df.apply(filter_predicate, axis=1)]

        # Save the filtered dataset to a new CSV file
        filtered_df.to_csv(output_csv, index=False)

        print(f"Successfully filtered dataset to {output_csv}")
    except Exception as e:
        print(f"Error: {e}")

def clean_text(text):
    """
    Clean the text by:
    1. removing leading/trailing spaces
    2. converting to lowercase.
    3. remove excess whitespace chars

    Parameters:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    return " ".join(text.strip().lower().split())

def csv_filename(name):
    """
    Create a CSV filename with the given name.

    Parameters:
        name (str): The name of the CSV file.

    Returns:
        str: The full path to the CSV file.
    """
    return f"{name}{EXTENSION_CSV}"

def pt_filename(name):
    """
    Create a PT filename with the given name.

    Parameters:
        name (str): The name of the PT file.

    Returns:
        str: The full path to the PT file.
    """
    return f"{name}{EXTENSION_PT}"

def pkl_filename(name):
    """
    Create a PKL filename with the given name.

    Parameters:
        name (str): The name of the PKL file.

    Returns:
        str: The full path to the PKL file.
    """
    return f"{name}{EXTENSION_PKL}"

def json_filename(name):
    """
    Create a JSON filename with the given name.

    Parameters:
        name (str): The name of the JSON file.

    Returns:
        str: The full path to the JSON file.
    """
    return f"{name}{EXTENSION_JSON}"

def log_filename(name):
    """
    Create a log filename with the given name.

    Parameters:
        name (str): The name of the log file.

    Returns:
        str: The full path to the log file.
    """
    return f"{name}{EXTENSION_LOG}"

def pth_filename(name):
    """
    Create a PTH filename with the given name.

    Parameters:
        name (str): The name of the PTH file.

    Returns:
        str: The full path to the PTH file.
    """
    return f"{name}{EXTENSION_PTH}"
