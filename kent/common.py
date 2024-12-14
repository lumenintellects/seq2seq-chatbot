import json
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

PATH_WORKSPACE_ROOT = r'D:\git\github\seq2seq-chatbot\kent' # Set the workspace root path here
FOLDER_DATASET = 'dataset' # Set the name of the folder containing the datasets here
FOLDER_LOG = FOLDER_DATASET # Set the name of the folder containing the log files here

COL_INPUT = 'input'
COL_OUTPUT = 'output'

EMPTY_STRING = ''
SPACE = ' '

FILE_MODE_WRITE = 'w'
FILE_MODE_READ = 'r'
FILE_MODE_APPEND = 'a'
FILE_MODE_READ_BINARY = 'rb'
FILE_MODE_WRITE_BINARY = 'wb'

VOCAB_PAD = '<pad>' # Padding token
VOCAB_UNK = '<unk>' # Unknown token
VOCAB_BOS = '<bos>' # Beginning of sentence
VOCAB_EOS = '<eos>' # End of sentence

EXTENSION_TXT = '.txt'
EXTENSION_CSV = '.csv'
EXTENSION_PT = '.pt'
EXTENSION_PKL = '.pkl'
EXTENSION_JSON = '.json'
EXTENSION_LOG = '.log'
EXTENSION_PTH = '.pth'
EXTENSION_MODEL = '.model'

FILE_NAME_DELIMITER = '_'

def to_model_filename(name):
    """
    Create a model filename with the given name.

    Parameters:
        name (str): The name of the model file.

    Returns:
        str: The full path to the model file.
    """
    return f"{name}{EXTENSION_MODEL}"

def to_txt_filename(name):
    """
    Create a TXT filename with the given name.

    Parameters:
        name (str): The name of the TXT file.

    Returns:
        str: The full path to the TXT file.
    """
    return f"{name}{EXTENSION_TXT}"

def to_csv_filename(name):
    """
    Create a CSV filename with the given name.

    Parameters:
        name (str): The name of the CSV file.

    Returns:
        str: The full path to the CSV file.
    """
    return f"{name}{EXTENSION_CSV}"

def to_pt_filename(name):
    """
    Create a PT filename with the given name.

    Parameters:
        name (str): The name of the PT file.

    Returns:
        str: The full path to the PT file.
    """
    return f"{name}{EXTENSION_PT}"

def to_pkl_filename(name):
    """
    Create a PKL filename with the given name.

    Parameters:
        name (str): The name of the PKL file.

    Returns:
        str: The full path to the PKL file.
    """
    return f"{name}{EXTENSION_PKL}"

def to_json_filename(name):
    """
    Create a JSON filename with the given name.

    Parameters:
        name (str): The name of the JSON file.

    Returns:
        str: The full path to the JSON file.
    """
    return f"{name}{EXTENSION_JSON}"

def to_log_filename(name):
    """
    Create a log filename with the given name.

    Parameters:
        name (str): The name of the log file.

    Returns:
        str: The full path to the log file.
    """
    return f"{name}{EXTENSION_LOG}"

def to_pth_filename(name):
    """
    Create a PTH filename with the given name.

    Parameters:
        name (str): The name of the PTH file.

    Returns:
        str: The full path to the PTH file.
    """
    return f"{name}{EXTENSION_PTH}"

def compose_filename(base, name_tokens):
    """
    Compose a filename with the given base and name tokens.

    Parameters:
        base (str): The base filename.
        name_tokens (list): A list of name tokens.

    Returns:
        str: The composed filename.
    """
    return FILE_NAME_DELIMITER.join([base] + name_tokens)

NAME_TOKEN_OUTLIERS = 'outliers'
NAME_TOKEN_INPUT_OUTPUT_PAIRS = 'input_output_pairs'
NAME_TOKEN_INPUT = 'input'
NAME_TOKEN_OUTPUT = 'output'
NAME_TOKEN_VOCAB = 'vocab'
NAME_TOKEN_SEQ = 'seq'
NAME_TOKEN_PADDED = 'padded'
NAME_TOKEN_BATCH = 'batch'
NAME_TOKEN_INPUT_OUTPUT_PAIRS_COMBINED = 'input_output_pairs_combined'
NAME_TOKEN_SENTENCEPIECE = 'sentencepiece'

def get_base_filename_sentencepiece_model(dataset_name):
    """
    Get the base filename for the SentencePiece model file.

    Parameters:
        dataset_name (str): The name of the dataset.

    Returns:
        str: The base filename for the SentencePiece model file.
    """
    return compose_filename(dataset_name, [NAME_TOKEN_SENTENCEPIECE])

def get_path_sentencepiece_model(dataset_name):
    """
    Get the path to the SentencePiece model file.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path to the SentencePiece model file.
    """
    base_filename = get_base_filename_sentencepiece_model(dataset_name)
    model_filename = to_model_filename(base_filename)
    return os.path.join(FOLDER_DATASET, model_filename)

def get_path_sentencepiece_combined_text(dataset_name):
    """
    Get the path to the combined text file for SentencePiece training.

    Parameters:
        dataset_name (str): The name of the dataset.

    Returns:
        str: The path to the combined text file.
    """
    base_filename = compose_filename(dataset_name, [NAME_TOKEN_INPUT_OUTPUT_PAIRS_COMBINED])
    txt_filename = to_txt_filename(base_filename)
    return os.path.join(FOLDER_DATASET, txt_filename)

def get_path_log(base, dataset_name, timestamp_token):
    """
    Get the path to the log file.

    Parameters:
        base (str): The base filename.
        timestamp_token (str): The timestamp token.

    Returns:
        str: The path to the log file.
    """
    base_filename = compose_filename(base, [dataset_name, timestamp_token])
    log_filename = to_log_filename(base_filename)
    return os.path.join(FOLDER_LOG, log_filename)

def get_path_source_csv(base):
    """
    Get the path to the source CSV file.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path to the source CSV file.
    """

    base_filename = compose_filename(base, [])
    csv_filename = to_csv_filename(base_filename)
    return os.path.join(FOLDER_DATASET, csv_filename)

def get_path_outliers(base):
    """
    Get the path to the outliers CSV file.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path to the outliers CSV file.
    """
    base_filename = compose_filename(base, [NAME_TOKEN_OUTLIERS])
    csv_filename = to_csv_filename(base_filename)
    return os.path.join(FOLDER_DATASET, csv_filename)

def get_path_input_output_pairs(base):
    """
    Get the path to the input-output pairs file.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path to the input-output pairs file.
    """
    base_filename = compose_filename(base, [NAME_TOKEN_INPUT_OUTPUT_PAIRS])
    csv_filename = to_csv_filename(base_filename)
    return os.path.join(FOLDER_DATASET, csv_filename)

def get_path_input_sequences(base):
    """
    Get the path to the input sequences file.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path to the input sequences file.
    """
    base_filename = compose_filename(base, [NAME_TOKEN_INPUT, NAME_TOKEN_SEQ])
    pt_filename = to_pt_filename(base_filename)
    return os.path.join(FOLDER_DATASET, pt_filename)

def get_path_output_sequences(base):
    """
    Get the path to the output sequences file.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path to the output sequences file.
    """
    base_filename = compose_filename(base, [NAME_TOKEN_OUTPUT, NAME_TOKEN_SEQ])
    pt_filename = to_pt_filename(base_filename)
    return os.path.join(FOLDER_DATASET, pt_filename)

def get_path_vocab(base):
    """
    Get the path to the vocabulary file.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path to the vocabulary file.
    """
    base_filename = compose_filename(base, [NAME_TOKEN_VOCAB])
    pkl_filename = to_pkl_filename(base_filename)
    return os.path.join(FOLDER_DATASET, pkl_filename)

def get_path_input_sequences_padded_batch(base, batch_number):
    """
    Get the path to the input sequences padded batch file.

    Parameters:
        base (str): The base filename.
        batch_number (int): The batch number.

    Returns:
        str: The path to the input sequences padded batch file.
    """
    base_filename = compose_filename(base, [NAME_TOKEN_INPUT, NAME_TOKEN_SEQ, NAME_TOKEN_PADDED, NAME_TOKEN_BATCH, str(batch_number)])
    pt_filename = to_pt_filename(base_filename)
    return os.path.join(FOLDER_DATASET, pt_filename)

def get_path_output_sequences_padded_batch(base, batch_number):
    """
    Get the path to the output sequences padded batch file.

    Parameters:
        base (str): The base filename.
        batch_number (int): The batch number.

    Returns:
        str: The path to the output sequences padded batch file.
    """
    base_filename = compose_filename(base, [NAME_TOKEN_OUTPUT, NAME_TOKEN_SEQ, NAME_TOKEN_PADDED, NAME_TOKEN_BATCH, str(batch_number)])
    pt_filename = to_pt_filename(base_filename)
    return os.path.join(FOLDER_DATASET, pt_filename)

def get_path_input_sequences_padded_batch_pattern(base):
    """
    Get the path pattern for the input sequences padded batch files.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path pattern for the input sequences padded batch files.
    """
    path_pattern = compose_filename(base, [NAME_TOKEN_INPUT, NAME_TOKEN_SEQ, NAME_TOKEN_PADDED, NAME_TOKEN_BATCH])
    return os.path.join(FOLDER_DATASET, f"{path_pattern}{FILE_NAME_DELIMITER}*.pt")

def get_path_output_sequences_padded_batch_pattern(base):
    """
    Get the path pattern for the output sequences padded batch files.

    Parameters:
        base (str): The base filename.

    Returns:
        str: The path pattern for the output sequences padded batch files.
    """
    path_pattern = compose_filename(base, [NAME_TOKEN_OUTPUT, NAME_TOKEN_SEQ, NAME_TOKEN_PADDED, NAME_TOKEN_BATCH])
    return os.path.join(FOLDER_DATASET, f"{path_pattern}{FILE_NAME_DELIMITER}*.pt")

def get_path_model(base, version):
    """
    Get the path to the model file.

    Parameters:
        base (str): The base filename.
        version (str): The version of the model.

    Returns:
        str: The path to the model file.
    """
    base_filename = compose_filename(base, [version])
    pth_filename = to_pth_filename(base_filename)
    return os.path.join(FOLDER_DATASET, pth_filename)

class TupleDataset(Dataset):
    """
    A custom dataset to yield tuples of input and output sequences.
    """
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

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
SETTING_EVALUATION_SUBSET_SIZE = 'evaluationSubsetSize'
SETTING_EVALUATION_LOOP_CONTINUE = 'evaluationLoopContinue'
SETTING_EVALUATION_RELOAD_MODEL_IN_LOOP = 'evaluationReloadModelInLoop'

# ==========================

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

def initialize_seq2seq(sp_model, device, emb_dim=128, hidden_dim=256, n_layers=2, dropout=0.5):
    """
    Initializes a Seq2Seq model (without attention).

    Args:
        sp_model (sentencepiece.SentencePieceProcessor): Preloaded SentencePiece model.
        device (torch.device): Device to use for model computation.
        emb_dim (int): Dimension of embedding layer.
        hidden_dim (int): Dimension of hidden layers in the encoder and decoder.
        n_layers (int): Number of layers in encoder and decoder.
        dropout (float): Dropout rate for encoder and decoder.

    Returns:
        model (Seq2Seq): Initialized Seq2Seq model.
        criterion (nn.CrossEntropyLoss): Loss function for training or evaluation.
    """
    # Define dimensions
    input_dim = sp_model.get_piece_size()
    output_dim = sp_model.get_piece_size()
    pad_id = sp_model.pad_id()

    # Initialize encoder
    encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout).to(device)

    # Initialize decoder
    decoder = Decoder(output_dim, emb_dim, hidden_dim, n_layers, dropout).to(device)

    # Initialize Seq2Seq model
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    return model, criterion

# ==========================

class Attention(nn.Module):
    """
    Attention mechanism to compute attention weights.
    """

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        # Adjust Linear layer input dimension to match concatenated dimensions
        self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Compute attention weights.
        Args:
            hidden: [batch_size, decoder_hidden_dim] - Decoder's last hidden state.
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim * 2] - All encoder outputs.
            mask: [batch_size, src_len] - Optional padding mask.
        Returns:
            attention: [batch_size, src_len] - Normalized attention scores.
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state `src_len` times to match encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, decoder_hidden_dim]

        # Concatenate hidden state with encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, decoder_hidden_dim]

        # Project energy down to a single attention score per time step
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        # Apply mask if provided (optional)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # Normalize scores
        return F.softmax(attention, dim=1)  # [batch_size, src_len]

class BidirectionalEncoderWithAttention(nn.Module):
    """
    Bidirectional Encoder with attention mechanism.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        """
        Initialize the encoder.

        Args:
            input_dim (int): Size of the input vocabulary.
            emb_dim (int): Dimensionality of word embeddings.
            hidden_dim (int): Hidden dimension for GRU.
            n_layers (int): Number of GRU layers.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim

        # Embedding layer to convert tokens to embeddings
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # Bidirectional GRU
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass through the encoder.

        Args:
            src (Tensor): Source sequence tensor with shape [batch_size, src_len].

        Returns:
            outputs (Tensor): Encoder outputs for attention, shape [batch_size, src_len, hidden_dim * 2].
            hidden (Tensor): Hidden state, shape [n_layers * 2, batch_size, hidden_dim].
        """
        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]

        # Pass through GRU
        outputs, hidden = self.rnn(embedded)  # 
        # outputs: [batch_size, src_len, hidden_dim * 2] (bidirectional outputs)
        # hidden: [n_layers * 2, batch_size, hidden_dim] (stacked forward & backward hidden states)

        return outputs, hidden

class DecoderWithAttention(nn.Module):
    """
    Decoder with attention mechanism.
    """

    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention, encoder_hidden_dim):
        """
        Initialize the decoder.

        Args:
            output_dim (int): Size of the output vocabulary.
            emb_dim (int): Dimensionality of word embeddings.
            hidden_dim (int): Hidden dimension for GRU.
            n_layers (int): Number of GRU layers.
            dropout (float): Dropout rate.
            attention (nn.Module): Attention mechanism.
            encoder_hidden_dim (int): Hidden dimension of the encoder.
        """
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attention = attention

        # Embedding layer for target sequences
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # GRU layer
        self.rnn = nn.GRU(
            emb_dim + encoder_hidden_dim * 2,  # Input is embedding + attention context
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True
        )

        # Fully connected layer to generate predictions
        self.fc_out = nn.Linear(hidden_dim + encoder_hidden_dim * 2 + emb_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        """
        Forward pass through the decoder.

        Args:
            input (Tensor): Current input token indices [batch_size].
            hidden (Tensor): Previous hidden state [n_layers, batch_size, hidden_dim].
            encoder_outputs (Tensor): Encoder outputs [batch_size, src_len, encoder_hidden_dim * 2].

        Returns:
            prediction (Tensor): Predicted token scores [batch_size, output_dim].
            hidden (Tensor): Updated hidden state [n_layers, batch_size, hidden_dim].
        """
        # Input shape: [batch_size]
        input = input.view(-1, 1)  # Ensure proper shape [batch_size, 1]

        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        print(f"Embedded Shape After Fix: {embedded.shape}")

        # Compute attention weights and context
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # Add time dimension: [batch_size, 1, src_len]
        context = torch.bmm(a, encoder_outputs)  # [batch_size, 1, encoder_hidden_dim * 2]
        print(f"Context Shape: {context.shape}")

        # Concatenate context with embedded input
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + encoder_hidden_dim * 2]
        print(f"RNN Input Shape: {rnn_input.shape}")
        print(f"Hidden Shape: {hidden.shape}")

        # Pass through GRU
        output, hidden = self.rnn(rnn_input, hidden)  # output: [batch_size, 1, hidden_dim]

        # Concatenate output, context, and embedded input for prediction
        output = output.squeeze(1)  # [batch_size, hidden_dim]
        context = context.squeeze(1)  # [batch_size, encoder_hidden_dim * 2]
        embedded = embedded.squeeze(1)  # [batch_size, emb_dim]
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch_size, output_dim]

        return prediction, hidden

# Seq2Seq model with Attention
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        Initialize the Seq2Seq model with attention.

        Args:
            encoder (nn.Module): The encoder module.
            decoder (nn.Module): The decoder module with attention.
            device (torch.device): The device to run the model on (CPU/GPU).
        """
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source input sequence [batch_size, src_len]
            trg: Target input sequence [batch_size, trg_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            outputs: Predictions [batch_size, trg_len, output_dim]
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.output_dim

        # Initialize tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)

        # Prepare hidden state for the decoder (handle bidirectionality)
        print(f"Original Hidden Shape: {hidden.shape}")  # Should be [n_layers * 2, batch_size, hidden_dim]
        transformed_hidden = self._transform_hidden(hidden)
        print(f"Transformed Hidden Shape: {transformed_hidden.shape}")  # Should be [n_layers, batch_size, hidden_dim]

        # First input to the decoder is the <bos> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Decode the next token
            output, transformed_hidden = self.decoder(input, transformed_hidden, encoder_outputs)

            # Store prediction
            outputs[:, t, :] = output

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs

    def _transform_hidden(self, hidden):
        """
        Transforms bidirectional encoder hidden state to match decoder requirements.
        Args:
            hidden: [n_layers * 2, batch_size, hidden_dim]
        Returns:
            hidden: [n_layers, batch_size, hidden_dim]
        """
        # Split hidden state into forward and backward components
        forward_hidden = hidden[0:hidden.size(0):2]  # Select even indices
        backward_hidden = hidden[1:hidden.size(0):2]  # Select odd indices

        # Sum the forward and backward states
        transformed_hidden = forward_hidden + backward_hidden  # [n_layers, batch_size, hidden_dim]

        return transformed_hidden

import torch
from torch import nn
from common import Attention, BidirectionalEncoderWithAttention, DecoderWithAttention, Seq2SeqWithAttention

def initialize_seq2seq_with_attention(sp_model, device, emb_dim=128, hidden_dim=256, n_layers=2, dropout=0.5):
    """
    Initializes a Seq2Seq model with attention.

    Args:
        sp_model (sentencepiece.SentencePieceProcessor): Preloaded SentencePiece model.
        device (torch.device): Device to use for model computation.
        emb_dim (int): Dimension of embedding layer.
        hidden_dim (int): Dimension of hidden layers in the encoder and decoder.
        n_layers (int): Number of layers in encoder and decoder.
        dropout (float): Dropout rate for encoder and decoder.

    Returns:
        model (Seq2SeqWithAttention): Initialized Seq2Seq model with attention mechanism.
        criterion (nn.CrossEntropyLoss): Loss function for training or evaluation.
    """
    # Define dimensions
    input_dim = sp_model.get_piece_size()
    output_dim = sp_model.get_piece_size()
    pad_id = sp_model.pad_id()

    # Initialize components
    attention = Attention(encoder_hidden_dim=hidden_dim, decoder_hidden_dim=hidden_dim).to(device)
    encoder = BidirectionalEncoderWithAttention(input_dim, emb_dim, hidden_dim, n_layers, dropout).to(device)
    decoder = DecoderWithAttention(
        output_dim=output_dim,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        attention=attention,
        encoder_hidden_dim=hidden_dim
    ).to(device)

    # Initialize Seq2Seq model
    model = Seq2SeqWithAttention(encoder=encoder, decoder=decoder, device=device).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    return model, criterion

# ==========================

def get_setting(setting_name):
    """
    Get the value of a setting from the settings file.

    Parameters:
        setting_name (str): The name of the setting.

    Returns:
        str: The value of the setting.
    """
    filename_settings = to_json_filename(BASE_FILENAME_SETTINGS)
    relative_path_settings = os.path.join(filename_settings)
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

def get_setting_evaluation_subset_size():
    """
    Get the value of the setting 'evaluationSubsetSize' from the settings file.

    Returns:
        int: The value of the setting 'evaluationSubsetSize'.
    """
    return get_setting(SETTING_EVALUATION_SUBSET_SIZE)

def get_setting_evaluation_loop_continue():
    """
    Get the value of the setting 'evaluationLoopContinue' from the settings file.

    Returns:
        bool: The value of the setting 'evaluationLoopContinue'.
    """
    return get_setting(SETTING_EVALUATION_LOOP_CONTINUE)

def get_setting_evaluation_reload_model_in_loop():
    """
    Get the value of the setting 'evaluationReloadModelInLoop' from the settings file.

    Returns:
        bool: The value of the setting 'evaluationReloadModelInLoop'.
    """
    return get_setting(SETTING_EVALUATION_RELOAD_MODEL_IN_LOOP)

def get_setting_evaluation_subset_size():
    """
    Get the value of the setting 'evaluationSubsetSize' from the settings file.

    Returns:
        int: The value of the setting 'evaluationSubsetSize'.
    """
    return get_setting(SETTING_EVALUATION_SUBSET_SIZE)

def get_setting_evaluation_loop_continue():
    """
    Get the value of the setting 'evaluationLoopContinue' from the settings file.

    Returns:
        bool: The value of the setting 'evaluationLoopContinue'.
    """
    return get_setting(SETTING_EVALUATION_LOOP_CONTINUE)

def get_setting_evaluation_reload_model_in_loop():
    """
    Get the value of the setting 'evaluationReloadModelInLoop' from the settings file.

    Returns:
        bool: The value of the setting 'evaluationReloadModelInLoop'.
    """
    return get_setting(SETTING_EVALUATION_RELOAD_MODEL_IN_LOOP)

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
