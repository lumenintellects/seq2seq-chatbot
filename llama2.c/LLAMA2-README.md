AI Model Training and Evaluation Pipeline

This project focuses on training and sampling from a custom AI model for natural language processing tasks. It includes preprocessing data, tokenization, model training, and sampling outputs. The pipeline is designed to be modular and efficient, enabling customization and experimentation with various configurations.
Project Structure

├── evaluate.py # Evaluation script for the trained model

├── Makefile # Build and execution automation commands

├── model.py # Defines the architecture of the custom model

├── preprocess.py # Data preprocessing script

├── preprocess.sh # Shell script for preprocessing setup

├── sample.py # Script for generating text using the trained model

├── tinystories.py # Dataset-related utilities for text processing

├── tokenizer.py # Tokenizer logic for input data

├── train.py # Main script for training the model

├── train.sh # Shell script for training automation

Installation and Setup

    Clone the Repository
    Clone this repository to your local machine:

git clone <repository_url>
cd <repository_name>

Install Dependencies
Ensure that Python 3.8+ and pip are installed on your system. Install required dependencies:

    pip install -r requirements.txt

    Prepare Dataset
    Place your dataset in the appropriate directory or update paths in preprocess.py.

Usage

1. Preprocessing the Data

Prepare the dataset by tokenizing and formatting it:

bash preprocess.sh

2. Training the Model

Train the model using the following commands:

bash train.sh

or execute the training script directly:

python train.py \
 --vocab_source=custom \
 --vocab_size=4096 \
 --batch_size=128 \
 --gradient_accumulation_steps=1 \
 --max_iters=100000 \
 --eval_interval=1000 \
 --device=mps \
 --dtype=float32 \
 --compile=False

3. Generating Outputs

Sample outputs from the trained model:

python sample.py --checkpoint=out/ckpt.pt --start="Your question or prompt"

4. Evaluation

Evaluate the performance of the model using:

python evaluate.py --checkpoint=out/model.bin

Key Scripts
train.py

The core training script for the model. It supports configurations for vocabulary size, batch size, and device type.
model.py

Defines the architecture of the AI model, including layers, activation functions, and training logic.
tokenizer.py

Implements the tokenization of input data for compatibility with the model.
sample.py

Generates outputs from the trained model based on a given starting prompt.
evaluate.py

Provides tools to evaluate the model's accuracy and performance.
Makefile

Automates tasks such as running scripts and managing the build pipeline:

make run # Train and evaluate the model

Configuration Options

The train.py script includes multiple configuration options:

    --vocab_size: Size of the vocabulary.
    --batch_size: Number of samples per batch.
    --device: Device to run the training (cpu, cuda, or mps).
    --dtype: Data type for computations (e.g., float32).

Example Workflow

    Preprocess the data:

bash preprocess.sh

Train the model:

bash train.sh

Sample text:

    python sample.py --checkpoint=out/ckpt.pt --start="Hello, world"

File Details
evaluate.py

Evaluates the model on test data. It calculates metrics like accuracy and loss.
Makefile

Streamlines the execution of common tasks with simple commands.
preprocess.py

Cleans and tokenizes the raw dataset into a format suitable for training.
train.sh

Runs the train.py script with pre-defined settings.