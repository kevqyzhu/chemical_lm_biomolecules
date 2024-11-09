import os
import re
import argparse
import multiprocessing
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.cuda
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    GPT2Config,
)

from group_utils import convert_group_to_smi, create_FASTA_grammar
from primary_sequence import primary_sequence_handler

# Constants
SEED = 42
NUM_GENERATIONS = 10000
BATCH_SIZE = 10

# Model configuration defaults
BASE_MODEL_PARAMS = {
    "epochs": 300,
    "learning_rate": 5e-4,
    "warmup_steps": 1e2,
    "epsilon": 1e-8,
    "batch_size": 8,
    "n_layer": 12,
    "n_embd": 512,
    "n_head": 8,
}

CONTEXT_LENGTHS = {
    "single_chains_0_250_AA_95_sim_rdkit_0_2500_single_FASTA_group_selfies": 1464,
    "single_chains_0_250_AA_95_sim_rdkit_0_2500_group_selfies_primary_DS": 1464,
    "protein_drug_conjugates_single_FASTA_group_selfies": 1488,
    "unnatural_proteins_single_FASTA_group_selfies": 1504,
    "af2db_single_FASTA_group_selfies": 1568,
    "af2db_single_FASTA_group_selfies_small": 1568,
    "large_aa_group_selfies": 844,
}

class SequenceGenerator:
    def __init__(self, dataset: str, input_dir: Path, output_dir: Path, grammar):
        self.dataset = dataset
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.grammar = grammar
        self.params = self._get_model_params()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seed for reproducibility
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_model_params(self) -> Dict:
        """Get model parameters based on dataset name."""
        params = BASE_MODEL_PARAMS.copy()
        params["context_length"] = CONTEXT_LENGTHS.get(self.dataset)
        params["window_size"] = params["context_length"]
        
        if self.dataset == "af2db_single_FASTA_group_selfies":
            params.update({
                "epochs": 100,
                "batch_size": 6,
                "n_layer": 16,
            })
        
        return params

    def _load_model(self, epoch: int) -> tuple:
        """Load the model and tokenizer."""
        model_file = self.input_dir / f'checkpointing/checkpoint_{epoch}/pytorch_model.bin'
        tokenizer = AutoTokenizer.from_pretrained(self.input_dir / 'tokenizer/')
        
        configuration = GPT2Config.from_pretrained(
            'gpt2',
            output_hidden_states=False,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_ctx=self.params["context_length"],
            n_positions=self.params["context_length"],
            n_layer=self.params["n_layer"],
            n_embd=self.params["n_embd"],
            n_head=self.params["n_head"],
        )

        model = GPT2LMHeadModel(configuration)
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model, tokenizer

    def generate_sequences(self, model: GPT2LMHeadModel, tokenizer) -> List[str]:
        """Generate sequences using the trained model."""
        model.eval()
        prompt = "<|startoftext|>"
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        
        samples = []
        for _ in range(NUM_GENERATIONS // BATCH_SIZE):
            with torch.no_grad():
                outputs = model.generate(
                    generated,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    max_length=self.params["context_length"],
                    top_p=0.95,
                    num_return_sequences=BATCH_SIZE
                )
            
            samples.extend([tokenizer.decode(output, skip_special_tokens=True) 
                          for output in outputs])
        return samples

    def process_sequences(self, samples: List[str], epoch: int) -> None:
        """Process and save generated sequences."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw generated sequences
        with open(self.output_dir / f'generated_{epoch}.txt', 'w') as f:
            f.write('\n'.join(samples))

        # Convert to SMILES and process
        with multiprocessing.Pool() as pool:
            prot_conv = [x for x in pool.map(self._convert_group_parallel, samples) if x]
        
        good_backbones = [x for x in primary_sequence_handler(prot_conv) if x]
        success_rate = len(good_backbones) / len(prot_conv) if prot_conv else 0
        
        # Save results
        with open(self.output_dir / f'generated_smiles_{epoch}.txt', 'w') as f:
            f.write('\n'.join(prot_conv))
        
        with open(self.output_dir / f'generated_smiles_{epoch}_backbone.txt', 'w') as f:
            f.write(str(success_rate))
        
        print(f"Success rate: {success_rate}")

    def _convert_group_parallel(self, prot_smi: str) -> Optional[str]:
        """Convert group to SMILES."""
        return convert_group_to_smi(prot_smi, self.grammar)

    def generate_for_epoch(self, epoch: int) -> None:
        """Generate sequences for a specific epoch."""
        print(f"Generating for dataset: {self.dataset}, epoch: {epoch}")
        model, tokenizer = self._load_model(epoch)
        samples = self.generate_sequences(model, tokenizer)
        self.process_sequences(samples, epoch)

    def generate_all_checkpoints(self) -> None:
        """Generate sequences for all checkpoints."""
        checkpoint_dir = self.input_dir / "checkpointing"
        model_dirs = sorted(
            [d for d in os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir / d)],
            key=lambda x: int(re.match(r"checkpoint_(\d+)", x).group(1) if re.match(r"checkpoint_(\d+)", x) else 0)
        )
        
        last_epoch = int(model_dirs[-1].split("_")[-1])
        for epoch in range(last_epoch, 0, -10):
            output_files = [
                self.output_dir / f'generated_{suffix}_{epoch}.txt'
                for suffix in ['', 'smiles', 'smiles_backbone']
            ]
            if not all(f.exists() for f in output_files):
                self.generate_for_epoch(epoch)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate sequences from trained models')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (must match one in CONTEXT_LENGTHS)')
    parser.add_argument('--input-dir', type=Path, required=True,
                      help='Directory containing model checkpoints and tokenizer')
    parser.add_argument('--output-dir', type=Path, required=True,
                      help='Directory to save generated sequences')
    parser.add_argument('--epoch', type=int, required=True,
                      help='Epoch number to generate from')
    return parser.parse_args()

def main():
    args = parse_args()
    grammar = create_FASTA_grammar()
    
    generator = SequenceGenerator(
        dataset=args.dataset,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        grammar=grammar
    )
    generator.generate_for_epoch(args.epoch)

if __name__ == "__main__":
    main()