import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from tokenizer import Tokenizer
from model import Transformer, ModelArgs
import json
import glob
import os
from tqdm import tqdm
import torch.nn.functional as F


def load_model(checkpoint_path, vocab_size):
    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_args = ModelArgs(
        dim=checkpoint['model_args']['dim'],
        n_layers=checkpoint['model_args']['n_layers'],
        n_heads=checkpoint['model_args']['n_heads'],
        n_kv_heads=checkpoint['model_args']['n_kv_heads'],
        vocab_size=vocab_size,
        multiple_of=checkpoint['model_args']['multiple_of'],
        max_seq_len=checkpoint['model_args']['max_seq_len']
    )
    model = Transformer(model_args)
    model.load_state_dict(checkpoint['model'])
    return model


def evaluate_model(model, tokenizer, test_dir, max_len=100):
    model.eval()
    bleu_scores = []
    perplexities = []
    
    # Load test files
    test_files = glob.glob(os.path.join(test_dir, "*.json"))
    
    for file in tqdm(test_files):
        with open(file, 'r') as f:
            stories = json.load(f)
        
        for story in stories:
            text = story["story"].strip()
            
            # Get reference tokens
            ref_tokens = tokenizer.encode(text, bos=True, eos=False)
            ref_text = text.split()
            
            # Generate prediction
            device = next(model.parameters()).device
            context = torch.tensor([[1]], dtype=torch.long, device=device)  # BOS token with batch dimension
            pred_tokens = []
            
            with torch.no_grad():
                for _ in range(max_len):
                    logits = model(context)
                    logits = logits[:, -1, :]  # get last token's logits
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1)
                    
                    if next_token[0].item() == 2:  # EOS token
                        break
                        
                    pred_tokens.append(next_token[0].item())
                    context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
            
            # Decode prediction
            pred_text = tokenizer.decode(pred_tokens).split()
            
            # Calculate BLEU
            bleu = sentence_bleu([ref_text], pred_text)
            bleu_scores.append(bleu)
            
            # Calculate perplexity
            with torch.no_grad():
                logits = model(torch.tensor(ref_tokens[:-1])[None, ...])
                loss = torch.nn.functional.cross_entropy(
                    logits[0], torch.tensor(ref_tokens[1:])
                )
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

    return {
        'bleu': np.mean(bleu_scores),
        'perplexity': np.mean(perplexities)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--vocab_source', type=str, default='llama2', choices=['llama2', 'custom'])
    args = parser.parse_args()
    
    # Load tokenizer
    if args.tokenizer is None:
        from tinystories import get_tokenizer_model_path
        if args.vocab_source == 'llama2':
            # For llama2, we pass vocab_size=0 to get the default tokenizer
            tokenizer_path = get_tokenizer_model_path(vocab_size=0)
        else:
            tokenizer_path = get_tokenizer_model_path(args.vocab_size)
        tokenizer = Tokenizer(tokenizer_path)
    
    # Load model
    model = load_model(args.checkpoint, args.vocab_size)
    
    # Evaluate
    test_dir = os.path.join('data', 'tinystories_test_data')
    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory not found: {test_dir}")
    metrics = evaluate_model(model, tokenizer, test_dir)
    
    print(f"BLEU Score: {metrics['bleu']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
