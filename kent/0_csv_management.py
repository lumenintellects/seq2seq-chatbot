import os
from os import listdir
from os.path import isfile
from common import PATH_WORKSPACE_ROOT, FOLDER_DATASET
import pandas as pd

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

target_csv_folder_name = 'ubuntu_dialogue_corpus'
target_csv_folder_path = f'{FOLDER_DATASET}/{target_csv_folder_name}'

# List all CSV files in the target folder
csv_files = [os.path.join(FOLDER_DATASET, target_csv_folder_name, f) for f in listdir(target_csv_folder_path) if isfile(f'{target_csv_folder_path}/{f}') and f.endswith('.csv')]  
print(csv_files)

# Combine all CSV files into a single DataFrame
df_combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_csv_filename = f'{target_csv_folder_name}_combined.csv'
df_combined.to_csv(f'{FOLDER_DATASET}/{output_csv_filename}', index=False)
print(f"Combined CSV file saved to: {FOLDER_DATASET}/{output_csv_filename}")
