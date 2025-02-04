# Seq2Seq Chatbot Project

## Overview

This project implements a chatbot using a **Seq2Seq (Sequence-to-Sequence)** neural network model. The chatbot leverages a trained SentencePiece tokenizer for text preprocessing and supports advanced features like sequence padding and batching. The project includes the end-to-end Generative Chatbot Development pipeline from initial text corpus inspection to the final chatbot interface.

## Code Submission Structure

The code submission is organized as follows:

```shell
 CSCK507_EMA_GroupC_Code/
--flow_diagrams/
----scriptset_common_library.svg
----chatbot_scriptset_workflow.svg
----ubuntu_dialogue_corpus_preprocessing.svg
----dataset_tokenization_using_sentencepiece.svg
----seq2seq_model_training.svg
----seq2seq_model_evaluation.svg
----chatbot_interface.svg
--dataset/
--1_load_and_inspect_dataset.py
--2_preprocess_dataset_ubuntu_dialogue_corpus.py
--3_tokenize_dataset.py
--4_train_model.py
--4_train_model_with_attention.py
--5_evaluate_model.py
--5_evaluate_model_with_attention.py
--6_chatbot_interface.py
--6_chatbot_interface_with_attention.py
--common.py
--settings.json
--README.md
```

## Scripts

### `CSCK507_EMA_GroupC_Code/`

- Root directory containing all the code submission files.

### `dataset/`

- Contains dataset files, SentencePiece models, persisted script artifacts, and log files generated by the scripts.

### `1_load_and_inspect_dataset.py`

- Analyzes the dataset to compute statistics like sequence length distributions, percentiles, and outliers.

### `2_preprocess_dataset_ubuntu_dialogue_corpus.py`

- Converts raw dialogue data into cleaned input-output pairs.
- Filters sequences based on length (e.g., 95th percentile).

### `3_tokenize_dataset.py`

- Uses **SentencePiece** for tokenization.
- Trains a SentencePiece model if one does not exist.
- Saves tokenized sequences and applies padding for uniform length.

### `4_train_model.py` and `4_train_model_with_attention.py`

- Implements a training pipeline for the Seq2Seq model.
- Supports early stopping based on validation loss.
- Includes optional continuation settings for large datasets.

### `5_evaluate_model.py` and `5_evaluate_model_with_attention.py`

- Evaluates the trained model on test data.
- Computes metrics such as BLEU score and average loss.

### `6_chatbot_interface.py` and `6_chatbot_interface_with_attention.py`

- Provides a command-line interface for the chatbot.
- Accepts user inputs and generates responses using beam search decoding.

### `common.py`

- Shared utility functions and classes for handling files, datasets, and model initialization.
- Defines the Seq2Seq architecture and includes optional attention mechanisms.

### `settings.json`

- Configuration settings for the scripts, to control script settings and behaviors

## Installation

### Prerequisites

- Python 3.8 or later
- Required libraries: `torch`, `sentencepiece`, `pandas`, `numpy`, `nltk`

### Setup

1. Install dependencies:

2. Set the workspace root:
   - Update the `PATH_WORKSPACE_ROOT` variable in `common.py` to point to your working directory.

3. Create and place your dataset in the `dataset/` folder.

4. Adjust settings in `settings.json` as needed.

## Usage

### 1. Dataset Analysis

Run the dataset analysis script to understand the data properties:

```bash
python 1_data_analysis.py
```

### 2. Preprocessing

Clean and preprocess the dataset:

```bash
python 2_preprocess_dataset.py
```

### 3. Tokenization

Tokenize the dataset using SentencePiece:

```bash
python 3_tokenize_dataset.py
```

### 4. Training

Train the Seq2Seq model:

```bash
python 4_train_model.py
```

### 5. Evaluation

Evaluate the trained model:

```bash
python 5_evaluate_model.py
```

### 6. Chatbot Interface

Launch the chatbot interface:

```bash
python 6_chatbot_interface.py
```

## Key Features

- **Custom Tokenization**: Uses SentencePiece for subword-level tokenization.
- **Sequence Padding and Batching**: Handles variable-length sequences with padding.
- **Seq2Seq Model**: Implements a GRU-based Seq2Seq architecture.
- **Evaluation Metrics**: Includes BLEU score and cross-entropy loss for model evaluation.

## Contributions

This project was developed as part of a group assignment for CSCK507 October 2024 B.

Team members:

|Name|Role|
|---|---|
|Choo, Chin Fong|Report|
|Dogan, Ismail|Reporting|
|Geoffrey, Kent|Script Development, Result Analysis, Meeting Organizer/Scribe|
|Morgan, Chris|Testing and Video Presentation, Meeting Organizer and Facilitator|
|Pitertsev, Artem|Comparative Model Development, Git Repo Setup and Reporting|

## Acknowledgements

- **PyTorch**: Framework for deep learning.
- **SentencePiece**: Subword tokenization tool.
- **BLEU Score**: Evaluation metric for text generation.
