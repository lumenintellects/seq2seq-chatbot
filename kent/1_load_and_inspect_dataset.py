import os
import pandas as pd
import numpy as np
from common import PATH_WORKSPACE_ROOT, csv_filename  # Import from common.py

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

# ==========================

folder_dataset = 'dataset'
base_filename = 'ubuntu_dialogue_corpus_000'
path_input_csv = os.path.join(folder_dataset, csv_filename(base_filename))

# ==========================

# Load the dataset
df = pd.read_csv(path_input_csv)

# Inspect the dataset
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Calculate unique values for dialogueID
unique_value_count_dialogue_id = df['dialogueID'].nunique()
print(f"\nUnique dialogueID count: {unique_value_count_dialogue_id}")
print(df['dialogueID'].unique())

# ==========================

# Check for extreme outliers in `text` column
print("\nAnalyzing lengths of `text` values...")

# Replace NaN in `text` with an empty string (if necessary)
if df['text'].isnull().any():
    print("\nReplacing NaN values in `text` column with empty strings...")
    df['text'] = df['text'].fillna("")

# Calculate text lengths
df['text_length'] = df['text'].str.split().map(len)

# Summary statistics
print("\nText Length Statistics:")
print(df['text_length'].describe())

# Define extreme outliers as texts longer than the 95th percentile
percentile_95 = np.percentile(df['text_length'], 95)
print(f"\n95th Percentile Length: {percentile_95}")

# Identify outliers
outliers = df[df['text_length'] > percentile_95]
print(f"\nNumber of Outliers: {outliers.shape[0]}")

# Display a sample of outlier entries
print("\nSample Outliers:")
print(outliers[['dialogueID', 'from', 'text', 'text_length']].head())

# ==========================

# Save outlier entries to a separate file (optional)
path_outliers_csv = os.path.join(folder_dataset, csv_filename(base_filename + '_text_outliers'))
outliers.to_csv(path_outliers_csv, index=False)
print(f"\nOutliers saved to: {path_outliers_csv}")
