import glob
import os
import time
import spacy
from common import PATH_WORKSPACE_ROOT, get_setting_training_loop_continue, get_setting_next_subset_continue
from common import get_setting_training_subset_size
from common import PATH_WORKSPACE_ROOT, get_path_log, get_path_input_output_pairs, get_path_vocab
from common import get_path_input_sequences, get_path_output_sequences
from common import get_path_input_sequences_padded_batch_pattern, get_path_output_sequences_padded_batch_pattern
from common import get_path_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import torch.nn.functional as F

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

LOG_BASE_FILENAME = "4_train_model_with_attention"
DATASET_NAME = 'ubuntu_dialogue_corpus_000'
MODEL_NAME = 'seq2seq_attention'
MODEL_VERSION = '1.0'

path_input_csv = get_path_input_output_pairs(DATASET_NAME)
path_vocab_pkl = get_path_vocab(DATASET_NAME)
path_input_sequences = get_path_input_sequences(DATASET_NAME)
path_output_sequences = get_path_output_sequences(DATASET_NAME)
path_input_sequences_padded_batch_pattern = get_path_input_sequences_padded_batch_pattern(DATASET_NAME)
path_output_sequences_padded_batch_pattern = get_path_output_sequences_padded_batch_pattern(DATASET_NAME)

# Define the save path
path_model = get_path_model(MODEL_NAME, MODEL_VERSION)

# ==========================

NON_TRAIN_DATA_PROPORTION = 0.2
RANDOM_SEED = 42

PATIENCE_LEVEL = 5 # Number of epochs to wait for improvement before early stopping
TORCH_THREAD_COUNT = 10

TRAINING_SUBSET_SIZE = get_setting_training_subset_size()
LOSS_THRESHOLD = 1.0

# ==========================

def parallel_split(dataset, test_size, random_state):
    """
    Parallelized version of train_test_split for large datasets.
    """
    # Define the function to run train_test_split in parallel
    def split_in_chunks(chunk):
        return train_test_split(chunk, test_size=test_size, random_state=random_state)

    # Split dataset into chunks for parallel processing
    num_chunks = 4  # Number of threads/processes
    chunk_size = len(dataset) // num_chunks
    chunks = [dataset[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

    # Apply train_test_split in parallel
    results = Parallel(n_jobs=num_chunks)(delayed(split_in_chunks)(chunk) for chunk in chunks)

    # Combine results
    train_data = sum([r[0] for r in results], [])
    test_data = sum([r[1] for r in results], [])
    return train_data, test_data

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
        input = input.unsqueeze(1)  # Add a time dimension [batch_size, 1]

        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]

        # Compute attention weights and context
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(a, encoder_outputs)  # [batch_size, 1, encoder_hidden_dim * 2]

        # Concatenate context with embedded input
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + encoder_hidden_dim * 2]

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
        hidden = self._transform_hidden(hidden)

        # First input to the decoder is the <bos> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Decode the next token
            output, hidden = self.decoder(input, hidden, encoder_outputs)

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
        # Sum forward and backward hidden states for each layer
        hidden = hidden.view(self.encoder.n_layers, 2, hidden.size(1), hidden.size(2))
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        return hidden

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

    # Load spaCy language model
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded.")

    # Set the current working directory
    os.chdir(PATH_WORKSPACE_ROOT)
    logger.info(f"Current Working Directory: {os.getcwd()}")

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

    matching_files_input = glob.glob(path_input_sequences_padded_batch_pattern)
    if len(matching_files_input) == 0:
        logger.error("No matching input files found. Unable to proceed, exiting...")
        exit()
    elif len(matching_files_input) == 1:
        input_sequences_padded = torch.load(matching_files_input[0], weights_only=True)
        logger.info("Loaded input sequences from file.")
    else:
        input_sequences_padded = torch.cat([torch.load(file, weights_only=True) for file in matching_files_input], dim=0)
        logger.info("Loaded input sequences from files.")

    matching_files_output = glob.glob(path_output_sequences_padded_batch_pattern)
    if len(matching_files_output) == 0:
        logger.error("No matching output files found. Unable to proceed, exiting...")
        exit()
    elif len(matching_files_output) == 1:
        output_sequences_padded = torch.load(matching_files_output[0], weights_only=True)
        logger.info("Loaded output sequences from file.")
    else:
        output_sequences_padded = torch.cat([torch.load(file, weights_only=True) for file in matching_files_output], dim=0)
        logger.info("Loaded output sequences from file.")

    # ==========================

    # Combine input and output sequences into a single list of pairs
    combined_sequences = list(zip(input_sequences_padded, output_sequences_padded))
    combined_indices = list(range(len(combined_sequences)))

    # ==========================
    # Instantiate Seq2Seq Model

    logger.info("Initializing Seq2Seq model...")

    # Hyperparameters
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    EMB_DIM = 256
    ENCODER_HIDDEN_DIM = 512
    DECODER_HIDDEN_DIM = ENCODER_HIDDEN_DIM # using the same hidden dimension for encoder and decoder
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 8

    # Check GPU Availability
    logger.info("Checking GPU availability...")
    logger.info(f"Is CUDA available: {torch.cuda.is_available()}")  # Should print True
    logger.info(f"Device ID: {torch.cuda.current_device()}")  # Should print the device ID (e.g., 0)
    logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")  # Should print the name of the GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize encoder, decoder, and seq2seq model
    logger.info("Initializing encoder, decoder, and seq2seq model...")

    # Initialize attention mechanism
    attention = Attention(encoder_hidden_dim=ENCODER_HIDDEN_DIM, decoder_hidden_dim=DECODER_HIDDEN_DIM).to(device)
    logger.info("Attention mechanism initialized with encoder hidden dimension of {ENCODER_HIDDEN_DIM} and decoder hidden dimension of {DECODER_HIDDEN_DIM}.")

    # Initialize encoder
    encoder_with_attention = BidirectionalEncoderWithAttention(
        INPUT_DIM, EMB_DIM, ENCODER_HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
    logger.info(f"Encoder initialized with input dimension of {INPUT_DIM}, embedding dimension of {EMB_DIM}, hidden dimension of {ENCODER_HIDDEN_DIM}, {N_LAYERS} layers, and dropout of {DROPOUT}.")

    # Initialize decoder with attention
    decoder_with_attention = DecoderWithAttention(
        output_dim=OUTPUT_DIM,
        emb_dim=EMB_DIM,
        hidden_dim=DECODER_HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        attention=attention,
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,  # Pass this to support bidirectional encoding
    ).to(device)
    logger.info(f"Decoder initialized with output dimension of {OUTPUT_DIM}, embedding dimension of {EMB_DIM}, hidden dimension of {DECODER_HIDDEN_DIM}, {N_LAYERS} layers, and dropout of {DROPOUT}.") 

    # Initialize Seq2Seq model with attention
    model_with_attention = Seq2SeqWithAttention(encoder=encoder_with_attention, decoder=decoder_with_attention, device=device).to(device)
    logger.info("Seq2Seq model with attention initialized.")

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=padding_value)
    optimizer = torch.optim.Adam(model_with_attention.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # If model state exists, load it
    if os.path.exists(path_model):
        logger.info("Loading model state...")
        model_with_attention.load_state_dict(torch.load(path_model, weights_only=True))
        logger.info("Model state loaded.")
    else:
        logger.info("Model state not found. Initializing new model.")

# ==========================

    # Training Loop
    logger.info("Training...")

    best_val_loss = float("inf")  # Initialize best validation loss as infinity
    no_improvement_epochs = 0  # Counter for epochs without improvement

    torch.set_num_threads(TORCH_THREAD_COUNT)
    logger.info(f"Number of Torch Threads is set to {torch.get_num_threads()}")

    continue_training = True
    while continue_training:

# ==========================

        # Create a subset of the dataset for training
        logger.info(f"Creating a subset of the dataset for training with size: {TRAINING_SUBSET_SIZE}")
        subset_indices = np.random.choice(combined_indices, size=TRAINING_SUBSET_SIZE, replace=False)
        subset_combined_sequences = [combined_sequences[i] for i in subset_indices]

        logger.info("Splitting data into train, test, and val sets...")

        data_splitting_start_time = time.time()

        # Split data into training and temporary (validation + test)
        # Parallelized dataset splitting
        subset_train_data, subset_non_train_data = parallel_split(subset_combined_sequences, test_size=0.2, random_state=RANDOM_SEED)

        # Further split the non-training dataset into validation and test sets
        subset_val_data, subset_test_data = parallel_split(subset_non_train_data, test_size=0.5, random_state=RANDOM_SEED)

        # Unzip training, validation, and test datasets
        subset_train_inputs, subset_train_outputs = zip(*subset_train_data)
        subset_val_inputs, subset_val_outputs = zip(*subset_val_data)
        subset_test_inputs, subset_test_outputs = zip(*subset_test_data)

        # Convert back to torch tensors
        subset_train_inputs = torch.stack(subset_train_inputs)
        subset_train_outputs = torch.stack(subset_train_outputs)
        subset_val_inputs = torch.stack(subset_val_inputs)
        subset_val_outputs = torch.stack(subset_val_outputs)
        subset_test_inputs = torch.stack(subset_test_inputs)
        subset_test_outputs = torch.stack(subset_test_outputs)

        # Create DataLoaders
        subset_train_dataset = TensorDataset(subset_train_inputs, subset_train_outputs)
        subset_train_loader = DataLoader(subset_train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

        subset_val_dataset = TensorDataset(subset_val_inputs, subset_val_outputs)
        subset_val_loader = DataLoader(subset_val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

        subset_test_dataset = TensorDataset(subset_test_inputs, subset_test_outputs)
        subset_test_loader = DataLoader(subset_test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

        data_splitting_end_time = time.time()
        data_splitting_duration_seconds = data_splitting_end_time - data_splitting_start_time
        data_splitting_duration_hours_minutes_seconds = time.strftime('%H:%M:%S', time.gmtime(data_splitting_duration_seconds))
        logger.info(f"Data splitting completed in {data_splitting_duration_hours_minutes_seconds}")

# ==========================

        # Epoch-wise training
        iteration_start_time = time.time()
        loss_history = []
        val_loss_history = []
        epoch_number = 0

        while get_setting_training_loop_continue():

            # Training Phase
            model_with_attention.train()
            epoch_loss = 0
            for src, trg in subset_train_loader:
                src, trg = src.to(device), trg.to(device)
                optimizer.zero_grad()

                outputs = model_with_attention(src, trg[:, :-1])
                outputs = outputs.reshape(-1, outputs.shape[-1])
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(outputs, trg)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_number += 1
            train_loss = epoch_loss / len(subset_train_loader)
            loss_history.append(train_loss)

            # Validation Phase
            model_with_attention.eval()
            val_loss = 0
            with torch.no_grad():
                for src, trg in subset_val_loader:
                    src, trg = src.to(device), trg.to(device)

                    outputs = model_with_attention(src, trg[:, :-1])
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    trg = trg[:, 1:].reshape(-1)

                    loss = criterion(outputs, trg)
                    val_loss += loss.item()

            val_loss /= len(subset_val_loader)
            val_loss_history.append(val_loss)
            logger.info(f"Epoch {epoch_number}, Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f} (vs. current best of {best_val_loss:.3f})")

            # Early Stopping Check
            has_val_loss_improved = val_loss < best_val_loss
            if has_val_loss_improved:
                best_val_loss = val_loss

                torch.save(model_with_attention.state_dict(), path_model)
                logger.info("Model state saved.")

                no_improvement_epochs = 0 # reset counter
                scheduler.step(val_loss) # adjust learning rate if needed
            else:
              no_improvement_epochs += 1

              out_of_patience = no_improvement_epochs >= PATIENCE_LEVEL
              if out_of_patience:
                  logger.info(f"Early stopping triggered after {PATIENCE_LEVEL} epochs with no improvement.")
                  continue_training = get_setting_next_subset_continue()
                  break

        iteration_end_time = time.time()
        iteration_time_training_loop = iteration_end_time - iteration_start_time
        iteration_time_average_per_epoch = iteration_time_training_loop / epoch_number
        logger.info(f"Training loop completed in {iteration_time_training_loop:.2f} seconds.")
        logger.info(f"Average time per epoch: {iteration_time_average_per_epoch:.2f} seconds.")

        # Stop outer loop if no continuation
        if not get_setting_next_subset_continue():
            break
