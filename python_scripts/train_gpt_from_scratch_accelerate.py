"""
GPT-2 Model Training Script for Sequence Generation
"""

import os
import time
import datetime
import re
import argparse
from tqdm.auto import tqdm
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from accelerate import Accelerator

# Initialize environment
torch.cuda.empty_cache()
torch.manual_seed(42)

class GPT2Dataset(Dataset):
    """
    Custom Dataset class for GPT-2 training.
    
    Handles the conversion of text sequences to tokenized format suitable for GPT-2 training.
    Adds special tokens and handles padding/truncation.
    
    Args:
        txt_list (List[str]): List of input sequences
        tokenizer: HuggingFace tokenizer instance
        max_length (int): Maximum sequence length for padding/truncation
    """
    def __init__(self, txt_list, tokenizer, max_length=768):
        self.tokenizer = tokenizer
        self.corpus = [f'<|startoftext|>{s}<|endoftext|>' for s in txt_list]
        self.max_length = max_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        """
        Returns tokenized and formatted input for a single sequence.
        
        Returns:
            tuple: (input_ids, attention_mask) tensors
        """
        encodings_dict = self.tokenizer(
            self.corpus[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_length=True,
        )
        return (
            torch.tensor([encodings_dict['input_ids']]),
            torch.tensor([encodings_dict['attention_mask']])
        )

def get_dataloaders(dataset_path: str, params: dict, tokenizer_dir: str):
    """
    Prepares data loaders for training and validation.
    
    Args:
        dataset_path (str): Path to the dataset file
        params (dict): Training parameters
        tokenizer_dir (str): Directory containing/for tokenizer
        
    Returns:
        tuple: (tokenizer, train_dataloader, validation_dataloader)
    """
    # Load dataset
    with open(dataset_path) as f:
        data = f.read().split("\n")
    print(f"Dataset size: {len(data)}")

    # Load or train tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print("Loaded pretrained tokenizer")
    except:
        print("Training new tokenizer")
        tokenizer = create_new_tokenizer(dataset_path, tokenizer_dir)

    # Create dataset and split into train/val
    dataset = GPT2Dataset(data, tokenizer, max_length=params["window_size"])
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f'Training samples: {train_size:,}')
    print(f'Validation samples: {val_size:,}')

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=params["batch_size"]
    )
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=params["batch_size"]
    )

    return tokenizer, train_dataloader, validation_dataloader

def create_new_tokenizer(dataset_path: str, tokenizer_dir: str) -> AutoTokenizer:
    """
    Creates and saves a new tokenizer based on GPT-2 vocabulary.
    
    Args:
        dataset_path (str): Path to the dataset file
        tokenizer_dir (str): Directory to save the tokenizer
        
    Returns:
        AutoTokenizer: Configured tokenizer instance
    """
    old_tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        bos_token='<|startoftext|>',
        eos_token='<|endoftext|>',
        pad_token='<|pad|>'
    )
    
    # Keep only important tokens
    important_tokens = ["<|endoftext|>", "<|startoftext|>", "<|pad|>"]
    unwanted_words = list(set(old_tokenizer.vocab.keys()) - set(important_tokens))
    for word in unwanted_words:
        del old_tokenizer.vocab[word]

    # Add custom tokens
    dataset_base = os.path.splitext(os.path.basename(dataset_path))[0]
    alphabet_path = os.path.join(os.path.dirname(dataset_path), f'{dataset_base}_alphabet.npy')
    selfies_tokens = np.load(alphabet_path).tolist()
    old_tokenizer.add_tokens(selfies_tokens, special_tokens=False)

    # Save tokenizer
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)
    old_tokenizer.save_pretrained(tokenizer_dir)
    
    return old_tokenizer

def train_model(output_dir: str, model: GPT2LMHeadModel, accelerator: Accelerator, 
                tokenizer: AutoTokenizer, train_dataloader: DataLoader, 
                eval_dataloader: DataLoader, params: dict, 
                optimizer: AdamW, lr_scheduler):
    """
    Main training loop for the model.
    
    Args:
        output_dir (str): Directory for saving checkpoints
        model (GPT2LMHeadModel): The model to train
        accelerator (Accelerator): Accelerator instance for distributed training
        tokenizer (AutoTokenizer): Tokenizer instance
        train_dataloader (DataLoader): Training data loader
        eval_dataloader (DataLoader): Validation data loader
        params (dict): Training parameters
        optimizer (AdamW): Optimizer instance
        lr_scheduler: Learning rate scheduler
    """
    device = torch.device("cuda")
    print("Beginning training...")
    
    progress_bar = tqdm(range(len(train_dataloader)), 
                       disable=not accelerator.is_main_process)

    for epoch_i in range(params["load_epoch"] + 1, params["epochs"]):
        progress_bar.reset()
        model.train()
        
        # Training loop
        for batch in train_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            outputs = model(
                b_input_ids,
                labels=b_labels,
                attention_mask=b_masks,
                token_type_ids=None
            )
            
            try:
                accelerator.backward(outputs.loss)
            except Exception as error:
                print(f"Error in backward pass: {error}")
                torch.cuda.empty_cache()
                accelerator.backward(outputs.loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Validation loop
        model.eval()
        for batch in eval_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    attention_mask=b_masks,
                    labels=b_labels
                )
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics(
                    (predictions, b_labels)
                )

        # Save checkpoint
        accelerator.save_state(f"{output_dir}/checkpoint_{epoch_i}")

    print("Training complete!")

def generate(model: GPT2LMHeadModel, tokenizer: AutoTokenizer, params: dict):
    """
    Generates sequences using the trained model.
    
    Args:
        model (GPT2LMHeadModel): Trained model
        tokenizer (AutoTokenizer): Tokenizer instance
        params (dict): Generation parameters
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompt = "<|startoftext|>"
    
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    sample_outputs = model.generate(
        generated,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        max_length=params["context_length"],
        top_p=0.95,
        num_return_sequences=1
    )

    for i, sample_output in enumerate(sample_outputs):
        print(f"{i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")

def load_checkpoint(checkpoint_dir: str, accelerator: Accelerator) -> int:
    """
    Loads the latest checkpoint if available.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        accelerator (Accelerator): Accelerator instance
        
    Returns:
        int: Epoch number of loaded checkpoint (-1 if no checkpoint found)
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return -1

    def get_epoch_num(path):
        match = re.match(r"checkpoint_(\d+)", path)
        return int(match.group(1)) if match else 0

    checkpoints = [
        path for path in os.listdir(checkpoint_dir)
        if "checkpoint_" in path and os.path.isdir(os.path.join(checkpoint_dir, path))
    ]
    
    if not checkpoints:
        return -1

    latest_checkpoint = max(checkpoints, key=get_epoch_num)
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    accelerator.load_state(checkpoint_path)
    
    return get_epoch_num(latest_checkpoint)

def run_train(dataset_path: str, output_dir: str, params: dict):
    """
    Main training function that orchestrates the entire training process.
    
    Args:
        dataset_path (str): Path to the dataset file
        output_dir (str): Directory for outputs (checkpoints, tokenizer, etc.)
        params (dict): Training parameters
    """
    # Setup directories and accelerator
    checkpoint_dir = os.path.join(output_dir, "checkpointing")
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    accelerator = Accelerator(project_dir=output_dir)

    # Get data and model components
    tokenizer, train_dataloader, eval_dataloader = get_dataloaders(
        dataset_path, params, tokenizer_dir
    )

    # Initialize model
    configuration = GPT2Config.from_pretrained(
        'gpt2',
        output_hidden_states=False,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_ctx=params["context_length"],
        n_positions=params["context_length"],
        n_layer=params["n_layer"],
        n_embd=params["n_embd"],
        n_head=params["n_head"],
    )
    
    model = GPT2LMHeadModel(configuration)
    model.cuda()
    print(f"Model parameters: {model.num_parameters():,}")

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        eps=params["epsilon"]
    )
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params["warmup_steps"],
        num_training_steps=len(train_dataloader) * params["epochs"]
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    # Load checkpoint if available
    params["load_epoch"] = load_checkpoint(checkpoint_dir, accelerator)

    # Train model
    train_model(
        checkpoint_dir, model, accelerator, tokenizer,
        train_dataloader, eval_dataloader, params,
        optimizer, lr_scheduler
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GPT-2 model on custom dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the input dataset file (.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory for model outputs (checkpoints, tokenizer, etc.)')
    
    # Optional parameters that can override defaults
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--context_length', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    
    args = parser.parse_args()

    # Base parameters
    params = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "warmup_steps": 1e2,
        "epsilon": 1e-8,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "window_size": args.context_length,
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    run_train(args.dataset_path, args.output_dir, params)