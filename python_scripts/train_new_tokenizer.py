"""
Tokenizer Training Script for Sequence Data
"""

import os
import random
import pickle
import multiprocessing
from itertools import repeat
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.empty_cache()

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Train a tokenizer on sequence data')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input text file containing sequences')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory for saving tokenizer and analysis outputs')
    parser.add_argument('--alphabet_file', type=str, required=True,
                      help='Path to .npy file containing domain-specific tokens')
    return parser.parse_args()

def get_length(txt: str, tokenizer) -> int:
    """
    Calculate the token length of a given text using the specified tokenizer.
    
    Args:
        txt (str): Input text to tokenize
        tokenizer: Hugging Face tokenizer instance
    
    Returns:
        int: Number of tokens in the text
    """
    encodings_dict = tokenizer(txt, truncation=False)
    return len(encodings_dict["input_ids"])

def get_length_handler(text: list, tokenizer) -> list:
    """
    Parallel processing handler for calculating token lengths of multiple texts.
    
    Args:
        text (list): List of input texts
        tokenizer: Hugging Face tokenizer instance
    
    Returns:
        list: List of token lengths
    """
    with multiprocessing.Pool(20) as p:
        return p.starmap(get_length, zip(text, repeat(tokenizer)))

class GPT2Dataset(Dataset):
    """
    PyTorch Dataset for GPT-2 training with sliding window tokenization.
    """
    
    def __init__(self, txt_list: list, tokenizer, max_length: int = 768):
        """
        Initialize the dataset with text processing and tokenization.
        
        Args:
            txt_list (list): List of input texts
            tokenizer: Hugging Face tokenizer instance
            max_length (int): Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.window_size = max_length
        self.stride = max_length // 16
        
        # Process only a subset of data for efficiency
        subset_portion = 12
        random.shuffle(txt_list)
        data = txt_list[:len(txt_list)//subset_portion]
        text = " ".join(['<|startoftext|>' + s + '<|endoftext|>' for s in data])
        
        self.encodings_dict = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_overflowing_tokens=True,
            return_length=True,
            stride=self.stride,
        )

    def __len__(self):
        return len(self.encodings_dict['input_ids'])

    def __getitem__(self, idx):
        return (torch.tensor([self.encodings_dict['input_ids'][idx]]),
                torch.tensor([self.encodings_dict['attention_mask'][idx]]))

def train_tokenizer(alphabet_file: str, tokenizer_dir: str):
    """
    Train or load a tokenizer for the specified dataset.
    
    Args:
        alphabet_file (str): Path to file containing domain-specific tokens
        tokenizer_dir (str): Directory to save/load the tokenizer
    
    Returns:
        tokenizer: Trained or loaded tokenizer instance
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print("Loaded pretrained tokenizer")
        return tokenizer
    except:
        print("Training new tokenizer")
        old_tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            bos_token='<|startoftext|>',
            eos_token='<|endoftext|>',
            pad_token='<|pad|>'
        )
        
        # Remove unwanted tokens except special tokens
        important_tokens = ["<|endoftext|>", "<|startoftext|>", "<|pad|>"]
        unwanted_words = list(set(old_tokenizer.vocab.keys()) - set(important_tokens))
        for word in unwanted_words:
            del old_tokenizer.vocab[word]
        
        # Add domain-specific tokens
        selfies_tokens = np.load(alphabet_file).tolist()
        old_tokenizer.add_tokens(selfies_tokens, special_tokens=False)
        
        print("Saving tokenizer")
        save_tokenizer(tokenizer_dir, old_tokenizer)
        return old_tokenizer

def save_tokenizer(output_dir: str, tokenizer):
    """
    Save the tokenizer to the specified directory.
    
    Args:
        output_dir (str): Directory to save the tokenizer
        tokenizer: Tokenizer instance to save
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)

def analyze_sequence_lengths(output_dir: str, tokenizer, data: list):
    """
    Analyze and visualize the sequence lengths of the dataset.
    
    Args:
        output_dir (str): Directory to save analysis outputs
        tokenizer: Tokenizer instance
        data (list): List of sequences
    """
    # Calculate sequence lengths
    data = ['<|startoftext|>' + txt + '<|endoftext|>' for txt in data]
    lens = get_length_handler(data, tokenizer)
    
    # Save lengths
    with open(os.path.join(output_dir, "lengths.pkl"), "wb") as fp:
        pickle.dump(lens, fp)
    
    # Print statistics
    print(f"Maximum sequence length: {max(lens)}")
    print(f"Average sequence length: {sum(lens)/len(lens):.2f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lens, bins=50)
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "lengths.png"))
    plt.close()

def main():
    """Main execution function"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_dir = os.path.join(args.output_dir, 'tokenizer')
    
    # Train or load tokenizer
    tokenizer = train_tokenizer(args.alphabet_file, tokenizer_dir)
    
    # Load and process data
    with open(args.input_file) as f:
        data = f.read().split("\n")
    
    # Analyze sequences
    analyze_sequence_lengths(args.output_dir, tokenizer, data)

if __name__ == "__main__":
    main()