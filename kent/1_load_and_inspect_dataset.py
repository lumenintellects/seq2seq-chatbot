import os
import time
import logging
import pandas as pd
import numpy as np
from common import log_filename
from common import PATH_WORKSPACE_ROOT, csv_filename  # Import from common.py

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)
LOG_BASE_FILENAME = "1_load_and_inspect_dataset"
LOG_FOLDER = 'dataset'

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

# ==========================

folder_dataset = 'dataset'
base_filename = 'ubuntu_dialogue_corpus_000'
path_input_csv = os.path.join(folder_dataset, csv_filename(base_filename))

# ==========================

# Load the dataset
df = pd.read_csv(path_input_csv)

# Inspect the dataset
logger.info("First 5 rows:")
logger.info(df.head())

logger.info("\nDataset Info:")
logger.info(df.info())

# Calculate unique values for dialogueID
unique_value_count_dialogue_id = df['dialogueID'].nunique()
logger.info(f"\nUnique dialogueID count: {unique_value_count_dialogue_id}")
logger.info(df['dialogueID'].unique())

# ==========================

# Check for extreme outliers in `text` column
logger.info("\nAnalyzing lengths of `text` values...")

# Replace NaN in `text` with an empty string (if necessary)
if df['text'].isnull().any():
    logger.info("\nReplacing NaN values in `text` column with empty strings...")
    df['text'] = df['text'].fillna("")

# Calculate text lengths
df['text_length'] = df['text'].str.split().map(len)

# Summary statistics
logger.info("\nText Length Statistics:")
logger.info(df['text_length'].describe())

# Define extreme outliers as texts longer than the 95th percentile
percentile_95 = np.percentile(df['text_length'], 95)
logger.info(f"\n95th Percentile Length: {percentile_95}")

# Identify outliers
outliers = df[df['text_length'] > percentile_95]
logger.info(f"\nNumber of Outliers: {outliers.shape[0]}")

# Display a sample of outlier entries
logger.info("\nSample Outliers:")
logger.info(outliers[['dialogueID', 'from', 'text', 'text_length']].head())

# ==========================

# Save outlier entries to a separate file (optional)
path_outliers_csv = os.path.join(folder_dataset, csv_filename(base_filename + '_text_outliers'))
outliers.to_csv(path_outliers_csv, index=False)
logger.info(f"\nOutliers saved to: {path_outliers_csv}")
