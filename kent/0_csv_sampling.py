import os
from common import PATH_WORKSPACE_ROOT, get_path_source_csv, extract_rows
import pandas as pd

# Set the current working directory using the constant from common.py
os.chdir(PATH_WORKSPACE_ROOT)

SOURCE_DATASET_NAME = 'ubuntu_dialogue_corpus_000'

path_input_csv = get_path_source_csv(SOURCE_DATASET_NAME)

TARGET_SAMPLE_SIZE = 1000
TARGET_DATASET_NAME = 'ubuntu_dialogue_corpus_sample'
path_output_csv = get_path_source_csv(TARGET_DATASET_NAME)

extract_rows(path_input_csv, path_output_csv, TARGET_SAMPLE_SIZE)
