import os
import glob
import logging
import json
import pandas as pd
import zipfile
import subprocess
import random
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tinystories import download, get_tokenizer_model_path
import numpy as np
from tqdm import tqdm
from tokenizer import Tokenizer
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_shard(args, vocab_size: int, dataset_type: str, split: str):
    """Process a single shard of data."""
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    
    with open(shard, "r") as f:
        data = json.load(f)
    
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        if dataset_type == "tinystories":
            text = example["story"]
        elif dataset_type in ["ubuntu", "wikiqa"]:
            text = f"Q: {example['question']}\nA: {example['answer']}"
        else:
            text = example["text"]
            
        text = text.strip()
        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # Use same naming convention as tinystories
    if vocab_size == 0:
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        bin_dir = os.path.join(os.path.dirname(shard), f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    logger.info(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize_dataset(data_dir: Path, vocab_size: int, dataset_type: str):
    """Pretokenize entire dataset using process pool."""
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    
    if not shard_filenames:
        logger.error(f"No shards found in {data_dir}")
        return
    
    # Determine split based on directory name
    split = "test" if "test" in str(data_dir) else "train"
    
    fun = partial(process_shard, vocab_size=vocab_size, dataset_type=dataset_type, split=split)
    with ProcessPoolExecutor() as executor:
        list(tqdm(
            executor.map(fun, enumerate(shard_filenames)),
            total=len(shard_filenames),
            desc=f"Processing {dataset_type} {split} shards"
        ))


def prepare_ubuntu_shards(base_dir: Path) -> Path:
    """Download and prepare Ubuntu dataset shards."""
    ubuntu_dir = base_dir / "Ubuntu_all_data"
    shards_dir = ubuntu_dir
    shards_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if shards already exist
    existing_shards = list(shards_dir.glob("*.json"))
    if existing_shards:
        logger.info(f"Found {len(existing_shards)} existing Ubuntu shards, skipping preparation")
        return shards_dir
        
    logger.info("Downloading and processing Ubuntu dataset...")
    try:
        # Download to data/ directory
        archive_path = base_dir / "ubuntu-dialogue-corpus.zip"
        ubuntu_corpus_dir = base_dir / "Ubuntu-dialogue-corpus"
        
        if not archive_path.exists():
            subprocess.run([
                "kaggle", "datasets", "download",
                "rtatman/ubuntu-dialogue-corpus",
                "-p", str(base_dir)
            ], check=True)
        
        # Extract from data/ to Ubuntu-dialogue-corpus
        if not ubuntu_corpus_dir.exists():
            logger.info("Extracting ubuntu-dialogue-corpus.zip...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
        
        # Process all CSV files in the directory
        dialogue_files = list(ubuntu_corpus_dir.glob("dialogueText*.csv"))
        
        shard_size = 10000
        shard_count = 0
        current_dialogues = []
        
        logger.info("Reading and processing Ubuntu dialogues...")
        for dialogue_file in dialogue_files:
            for chunk in pd.read_csv(dialogue_file, 
                                     chunksize=100000, 
                                     usecols=['folder', 'dialogueID', 'from', 'to', 'text'],
                                     dtype={'folder': str, 'dialogueID': str}):
                
                # Clean the text column - replace NaN with empty string
                chunk['text'] = chunk['text'].fillna('').astype(str)
                
                # Group by dialogue ID
                grouped = chunk.groupby(['folder', 'dialogueID'], sort=False)
                
                for (folder, dialogue_id), group in grouped:
                    # Convert group to list of dicts for easier processing
                    messages = group.to_dict('records')
                    if len(messages) < 2:  # Skip single message dialogues
                        continue
                    
                    # Build request-response pairs based on 'from' and 'to' fields
                    current_request = []
                    current_from = None
                    
                    for msg in messages:
                        if current_from is None:
                            current_from = msg['from']
                            current_request.append(msg['text'])
                        elif msg['from'] != current_from:
                            # Found a response from different user
                            current_dialogues.append({
                                "question": "\n".join(current_request).strip(),
                                "answer": msg['text'].strip(),
                                "features": ["Dialogue"]
                            })
                            # Reset for next pair
                            current_request = []
                            current_from = None
                        else:
                            # Same user continues speaking
                            current_request.append(msg['text'])
                    
                    if len(current_dialogues) >= shard_size:
                        shard_path = shards_dir / f"data{shard_count:02d}.json"
                        with open(shard_path, 'w') as f:
                            json.dump(current_dialogues, f)
                        logger.info(f"Saved Ubuntu shard {shard_count} with {len(current_dialogues)} dialogues")
                        current_dialogues = []
                        shard_count += 1
        
        # Save final shard
        if current_dialogues:
            shard_path = shards_dir / f"data{shard_count:02d}.json"
            with open(shard_path, 'w') as f:
                json.dump(current_dialogues, f)
            logger.info(f"Saved final Ubuntu shard {shard_count} with {len(current_dialogues)} dialogues")
        
    except Exception as e:
        logger.error(f"Error processing Ubuntu dataset: {str(e)}")
        return None
    
    return shards_dir


def prepare_wikiqa_shards(base_dir: Path) -> Path:
    """Download and prepare WikiQA dataset shards."""
    wikiqa_dir = base_dir / "WikiQA_all_data"
    shards_dir = wikiqa_dir
    shards_dir.mkdir(exist_ok=True, parents=True)
    
    if not list(shards_dir.glob("*.json")):
        # Download and extract WikiQA files to data/
        archive_path = base_dir / "WikiQACorpus.zip"
        wikiqa_corpus_dir = base_dir / "WikiQACorpus"
        
        if not archive_path.exists():
            logger.error("WikiQACorpus.zip not found in data/. Please download manually from Microsoft.")
            return None
            
        if not wikiqa_corpus_dir.exists():
            logger.info("Extracting WikiQACorpus.zip...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
        
        train_file = wikiqa_corpus_dir / "WikiQA-train.tsv"
        dev_file = wikiqa_corpus_dir / "WikiQA-dev.tsv"
        test_file = wikiqa_corpus_dir / "WikiQA-test.tsv"
        
        files_exist = all(f.exists() for f in [train_file, dev_file, test_file])
        if not files_exist:
            logger.error("WikiQA files not found. Please download manually from Microsoft.")
            return None
        
        try:
            shard_size = 1000
            shard_count = 0
            current_qa_pairs = []
            
            for file_path in [train_file, dev_file, test_file]:
                split = file_path.stem.split('-')[1]
                logger.info(f"Processing WikiQA {split} split")
                
                df = pd.read_csv(file_path, sep='\t')
                current_question = None
                current_answers = []
                current_labels = []
                
                for _, row in df.iterrows():
                    if current_question != row['Question']:
                        if current_question and current_answers:
                            answers_with_labels = []
                            for ans, label in zip(current_answers, current_labels):
                                prefix = "Correct answer: " if label == 1 else "Incorrect answer: "
                                answers_with_labels.append(f"{prefix}{ans}")
                                
                            current_qa_pairs.append({
                                "question": current_question,
                                "answer": " ".join(answers_with_labels),
                                "features": ["QA"]
                            })
                            
                            if len(current_qa_pairs) >= shard_size:
                                shard_path = shards_dir / f"data{shard_count:02d}.json"
                                with open(shard_path, 'w') as f:
                                    json.dump(current_qa_pairs, f)
                                logger.info(f"Saved WikiQA shard {shard_count} with {len(current_qa_pairs)} QA pairs")
                                current_qa_pairs = []
                                shard_count += 1
                        
                        current_question = row['Question']
                        current_answers = []
                        current_labels = []
                    
                    current_answers.append(row['Sentence'])
                    current_labels.append(row['Label'])
            
            # Save final shard
            if current_qa_pairs:
                shard_path = shards_dir / f"data{shard_count:02d}.json"
                with open(shard_path, 'w') as f:
                    json.dump(current_qa_pairs, f)
                logger.info(f"Saved final WikiQA shard {shard_count} with {len(current_qa_pairs)} QA pairs")
        
        except Exception as e:
            logger.error(f"Error processing WikiQA dataset: {str(e)}")
            return None
    
    return shards_dir


def datasets_test(base_dir: Path, vocab_size: int = 0):
    """Test run one example from each dataset."""
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    
    # Test TinyStories
    tinystories_dir = base_dir / "TinyStories_all_data"
    tinystory_file = next(tinystories_dir.glob("*.json"))
    with open(tinystory_file) as f:
        story_data = json.load(f)
        story = story_data[0]["story"]
        logger.info("\n=== TinyStories Example ===")
        logger.info(story)
    
    # Test Ubuntu
    ubuntu_dir = base_dir / "Ubuntu_all_data"
    ubuntu_file = next(ubuntu_dir.glob("*.json"))
    with open(ubuntu_file) as f:
        dialogue_data = json.load(f)
        dialogue = dialogue_data[0]
        logger.info("\n=== Ubuntu Dialogue Example ===")
        logger.info(f"Q: {dialogue['question']}")
        logger.info(f"A: {dialogue['answer']}")
    
    # Test WikiQA
    wikiqa_dir = base_dir / "WikiQA_all_data"
    wikiqa_file = next(wikiqa_dir.glob("*.json"))
    with open(wikiqa_file) as f:
        qa_data = json.load(f)
        qa = qa_data[0]
        logger.info("\n=== WikiQA Example ===")
        logger.info(f"Q: {qa['question']}")
        logger.info(f"A: {qa['answer']}")


def split_train_test(data_dir: Path, dataset_type: str, test_size: float = 0.2, seed: int = 42) -> Path:
    """Split data into train and test sets before tokenization."""
    random.seed(seed)
    test_dir = data_dir.parent / f"{dataset_type}_test_data"
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all json files
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    num_shards = len(shard_filenames)
    num_test = int(num_shards * test_size)
    
    # Randomly select test shards
    test_shards = random.sample(shard_filenames, num_test)
    
    # Move test files to test directory
    for shard in test_shards:
        src = Path(shard)
        dst = test_dir / src.name
        shutil.copy2(src, dst)
        logger.info(f"Copied {src.name} to {dataset_type} test set")
    
    return test_dir


def main():
    """CLI entry point similar to tinystories.py."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=0,
                       help="0 for Llama 2 tokenizer, or custom vocab size")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["all", "tinystories", "ubuntu", "wikiqa", "test"],
                       help="Which dataset(s) to process")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data to use for testing")
    args = parser.parse_args()
    
    base_dir = Path("data")
    
    # Process datasets
    if args.dataset in ["all", "tinystories"]:
        logger.info("Processing TinyStories dataset...")
        download()
        data_dir = base_dir / "TinyStories_all_data"
        test_dir = split_train_test(data_dir, "tinystories", args.test_size)
        pretokenize_dataset(data_dir, args.vocab_size, "tinystories")
        pretokenize_dataset(test_dir, args.vocab_size, "tinystories")
    
    if args.dataset in ["all", "ubuntu"]:
        logger.info("Processing Ubuntu dataset...")
        data_dir = prepare_ubuntu_shards(base_dir)
        if data_dir:
            test_dir = split_train_test(data_dir, "ubuntu", args.test_size)
            pretokenize_dataset(data_dir, args.vocab_size, "ubuntu")
            pretokenize_dataset(test_dir, args.vocab_size, "ubuntu")
    
    if args.dataset in ["all", "wikiqa"]:
        logger.info("Processing WikiQA dataset...")
        data_dir = prepare_wikiqa_shards(base_dir)
        if data_dir:
            test_dir = split_train_test(data_dir, "wikiqa", args.test_size)
            pretokenize_dataset(data_dir, args.vocab_size, "wikiqa")
            pretokenize_dataset(test_dir, args.vocab_size, "wikiqa")


if __name__ == "__main__":
    main()
