import csv
import os
import time
import sentencepiece as sp
import re
import glob
import torch
import torch.nn as nn
import logging

from chatbot_interface_with_attention import generate_response_beam_search
from common import (
    PATH_WORKSPACE_ROOT,
    SPACE,
    VOCAB_BOS,
    VOCAB_EOS,
    get_path_sentencepiece_model,
    initialize_seq2seq_with_attention,
    get_path_log,
    get_path_model,
)

# Set the current working directory
os.chdir(PATH_WORKSPACE_ROOT)

DATASET_NAME = "ubuntu_dialogue_corpus_000"
LOG_BASE_FILENAME = "6_chatbot_interface_with_attention"
MODEL_NAME = "seq2seq_attention"
MODEL_VERSION = "2.0"
INPUT_CSV = "questions.csv"
OUTPUT_CSV = "responses.csv"

# Logging setup
log_start_time = time.strftime("%Y%m%d_%H%M%S")
path_log = get_path_log(LOG_BASE_FILENAME, DATASET_NAME, log_start_time)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(path_log),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ==========================
# Main Logic

if __name__ == "__main__":
    # Load SentencePiece model
    path_sentencepiece_model = get_path_sentencepiece_model(DATASET_NAME)
    if os.path.exists(path_sentencepiece_model):
        sp_model = sp.SentencePieceProcessor(model_file=path_sentencepiece_model)
        logger.info(f"Loaded SentencePiece model from {path_sentencepiece_model}.")
    else:
        logger.error("SentencePiece model file not found. Exiting...")
        exit()

    # Load the model
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    path_model = get_path_model(MODEL_NAME, MODEL_VERSION)
    model_with_attention, _ = initialize_seq2seq_with_attention(
        sp_model=sp_model,
        device=device,
        emb_dim=128,
        hidden_dim=256,
        n_layers=2,
        dropout=0.5,
    )

    if os.path.exists(path_model):
        logger.info(f"Loading model state: {path_model}")
        model_with_attention.load_state_dict(torch.load(path_model, map_location=device))
        logger.info("Model state loaded.")
    else:
        logger.error("Model state file not found. Exiting...")
        exit()

    # Function to generate a response
    def generate_response(input_text):
        return generate_response_beam_search(
            model_with_attention, input_text, sp_model, device
        )

    # Process the CSV file
    if os.path.exists(INPUT_CSV):
        with open(INPUT_CSV, "r") as infile, open(OUTPUT_CSV, "w", newline="") as outfile:
            reader = csv.DictReader(infile)
            writer = csv.writer(outfile)
            writer.writerow(["Response"])  # Single column for responses

            for row in reader:
                question = row.get("Question", "")
                if question:
                    response = generate_response(question)
                    writer.writerow([response])
        logger.info(f"Responses written to {OUTPUT_CSV}")
    else:
        logger.error(f"Input CSV file {INPUT_CSV} not found. Exiting...")
        exit()
