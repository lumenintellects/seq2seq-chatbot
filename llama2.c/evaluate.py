import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from tinystories import get_tokenizer_model_path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
from torch.nn import functional as F
from contextlib import nullcontext

# Download required NLTK data
nltk.download('punkt')


class Evaluator:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=torch.float32)
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gptconf = ModelArgs(**checkpoint['model_args'])
        self.model = Transformer(gptconf)
        
        # Clean up state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model.to(device)
        
        # Setup tokenizer
        vocab_source = checkpoint["config"].get("vocab_source", "llama2")
        vocab_size = gptconf.vocab_size
        query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
        tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
        self.tokenizer = Tokenizer(tokenizer_model=tokenizer_model)
        
        # Setup ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction()

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for a given text."""
        tokens = self.tokenizer.encode(text, bos=True, eos=False)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)[None, ...]
        
        with torch.no_grad():
            with self.ctx:
                logits = self.model(tokens)
                if self.model.last_loss is not None:
                    return torch.exp(self.model.last_loss).item()
        return float('inf')

    def generate_response(self, question: str, max_tokens: int = 100) -> str:
        """Generate response for a given question."""
        # Format prompt to enforce A: style responses
        prompt = f"Q: {question}\nA:"
        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)[None, ...]
        
        with torch.no_grad():
            with self.ctx:
                output = self.model.generate(tokens, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
                response = self.tokenizer.decode(output[0].tolist())
                
                # Extract only the answer part
                try:
                    # Split on A: and take everything after it until Q: if it exists
                    answer = response.split("A:")[1].split("Q:")[0].strip()
                    return answer  # Don't add A: prefix since it's already included
                except IndexError:
                    # If splitting fails, clean up the response and remove the prompt
                    answer = response.replace(prompt, "").strip()
                    if answer.startswith("A:"):  # Remove A: if it's already there
                        answer = answer[2:].strip()
                    return answer

    def calculate_metrics(self, generated: str, reference: str) -> dict[str, Any]:
        """Calculate BLEU and ROUGE scores."""
        # Tokenize sentences
        gen_tokens = nltk.word_tokenize(generated.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())
        
        # Calculate BLEU
        bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smooth.method1)
        
        # Calculate ROUGE
        rouge_scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'bleu': bleu_score,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        }

    def test_single_file(self, bin_file_path: str) -> pd.DataFrame:
        """Test evaluation on a single binary file."""
        results = []
        print(f"Testing evaluation on: {bin_file_path}")
        
        # Read binary file
        with open(bin_file_path, 'rb') as f:
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
        
        # Convert to tensor
        tokens = torch.from_numpy(tokens.astype(np.int64))
        
        # Find Q&A pairs by looking for special tokens
        text = self.tokenizer.decode(tokens.tolist())
        qa_pairs = text.split("Q: ")
        
        print(f"Found {len(qa_pairs)-1} Q&A pairs")
        
        # Process each Q&A pair
        for i, qa in enumerate(qa_pairs[1:], 1):  # Skip first empty split
            try:
                question, answer = qa.split("\nA: ")
                question = question.strip()
                reference = answer.strip()
                
                print(f"\nProcessing pair {i}:")
                print(f"Q: {question}")
                print(f"Reference A: {reference}")
                
                # Generate response
                generated = self.generate_response(question)
                print(f"Generated A: {generated}")
                
                # Calculate metrics
                metrics = self.calculate_metrics(generated, reference)
                perplexity = self.calculate_perplexity(generated)
                
                results.append({
                    'question': question,
                    'reference': reference,
                    'generated': generated,
                    'perplexity': perplexity,
                    **metrics
                })
                
                print(f"Metrics: BLEU={metrics['bleu']:.4f}, "
                      f"ROUGE-L={metrics['rougeL']:.4f}, "
                      f"Perplexity={perplexity:.4f}")
                
            except ValueError as e:
                print(f"Skipping malformed pair: {e}")
                continue
        
        return pd.DataFrame(results)

    def evaluate_test_data(self, test_dir: str) -> pd.DataFrame:
        """Evaluate test data using only one file for quick testing."""
        results = []
        bin_files = sorted(Path(test_dir).glob("*.bin"))
        
        if bin_files:
            print("Found preprocessed .bin files, using first file for testing...")
            # Take only the first bin file
            bin_file = bin_files[0]
            print(f"Testing file: {bin_file}")
            
            # Read binary file
            with open(bin_file, 'rb') as f:
                tokens = np.frombuffer(f.read(), dtype=np.uint16)
            
            # Convert to tensor
            tokens = torch.from_numpy(tokens.astype(np.int64))
            
            # Find Q&A pairs by looking for special tokens
            text = self.tokenizer.decode(tokens.tolist())
            qa_pairs = text.split("Q: ")
            
            print(f"Found {len(qa_pairs)-1} Q&A pairs in {bin_file}")
            
            # Process each Q&A pair
            for qa in tqdm(qa_pairs[1:], desc="Processing Q&A pairs"):  # Skip first empty split
                try:
                    question, answer = qa.split("\nA: ")
                    question = question.strip()
                    reference = answer.strip()
                    
                    # Generate response
                    generated = self.generate_response(question)
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(generated, reference)
                    perplexity = self.calculate_perplexity(generated)
                    
                    results.append({
                        'question': question,
                        'reference': reference,
                        'generated': generated,
                        'perplexity': perplexity,
                        **metrics
                    })
                except ValueError:
                    continue  # Skip malformed pairs
        else:
            print("No .bin files found!")
            return pd.DataFrame()
        
        return pd.DataFrame(results)


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize evaluator with proper device
    checkpoint_path = 'out/ckpt.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    evaluator = Evaluator(checkpoint_path, device=device)
    
    # Test single file first
    test_file = 'test_data/ubuntu_test_data/data06.bin'
    print(f"\nTesting single file: {test_file}")
    if os.path.exists(test_file):
        results_df = evaluator.test_single_file(test_file)
        print("\nTest Results Summary:")
        print(results_df.describe())
        
        # Save test results
        test_results_file = 'test_evaluation_results.parquet'
        results_df.to_parquet(test_results_file)
        print(f"\nTest results saved to {test_results_file}")
        
        user_input = input("\nContinue with full evaluation? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    # Continue with full evaluation if requested
    print("\nStarting full evaluation...")
    results_df = evaluator.evaluate_test_data('test_data/ubuntu_test_data')
    results_df.to_parquet('evaluation_results.parquet')
    
    metrics = {
        'Average BLEU': results_df['bleu'].mean(),
        'Average ROUGE-1': results_df['rouge1'].mean(),
        'Average ROUGE-2': results_df['rouge2'].mean(),
        'Average ROUGE-L': results_df['rougeL'].mean(),
        'Average Perplexity': results_df['perplexity'].mean()
    }
    
    print("\nAggregate Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
