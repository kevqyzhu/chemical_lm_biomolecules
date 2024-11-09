"""
Script to convert SMILES to group SELFIES representations,
with optional regular SELFIES generation and length distribution analysis.
Usage: python convert_group_selfies.py --input path/to/input.txt --output path/to/output/dir [--max_samples N] [--generate_selfies]
"""

# Standard library imports
import multiprocessing
import argparse
from itertools import repeat
from io import BytesIO
from pathlib import Path

# Third-party imports
import selfies as sf
import pandas as pd
import random
from rdkit import Chem, DataStructs
from matplotlib import pyplot as plt
from PIL import Image

# Local imports
from group_utils import (
    amino_acid_smiles,
    remove_hydroxyl,
    get_num_heavy_smi,
    convert_smi_to_group
)
import group_selfies
from group_selfies import Group, GroupGrammar

# Configuration constants
DEFAULT_NUM_PROCESSES = 80

class ProteinConverter:
    def __init__(self, num_processes=DEFAULT_NUM_PROCESSES):
        self.num_processes = num_processes
        self.grammar = None

    def create_grammar(self):
        """Create and return a GroupGrammar instance for protein encoding."""
        constraints = group_selfies.get_preset_constraints("default")
        constraints.update({"S": 2, "S-1": 1, "S+1": 3})
        group_selfies.set_semantic_constraints(constraints)

        amino_acids = amino_acid_smiles()
        amino_acids = {key: remove_hydroxyl(amino_acids[key]) for key in amino_acids}

        self.grammar = GroupGrammar([
            Group(key, amino_acids[key], all_attachment=True) 
            for key in amino_acids
        ])
        return self.grammar

    def parallel_process(self, func, data):
        """Generic parallel processing function."""
        with multiprocessing.Pool(self.num_processes) as pool:
            return pool.map(func, data)

    def convert_smiles_to_selfies_group(self, prot_smi_list, generate_selfies=True):
        """Convert protein SMILES to group SELFIES and optionally regular SELFIES."""
        if self.grammar is None:
            self.create_grammar()

        print("Generating group SELFIES...")
        prot_conv = self.parallel_process(
            lambda x: convert_smi_to_group(x, self.grammar), 
            prot_smi_list
        )
        prot_conv = [p for p in prot_conv if p is not None]

        if not generate_selfies:
            return prot_conv

        print("Generating regular SELFIES...")
        selfies = self.parallel_process(smiles_to_selfies, prot_smi_list)
        selfies = [s for s in selfies if s is not None]

        return prot_conv, selfies

def smiles_to_selfies(smiles):
    """Convert SMILES to SELFIES with error handling."""
    try:
        selfies = sf.encoder(smiles)
        sf.decoder(selfies)  # Validate by attempting to decode
        return selfies
    except Exception as e:
        print(f"SELFIES conversion failed: {e}")
        return None

def get_len_selfies(s):
    """Get the length of a SELFIES string."""
    return len(list(sf.split_selfies(s)))

def plot_length_distributions(len_selfies, len_group_selfies, output_path):
    """Plot and save length distributions of SELFIES and group SELFIES."""
    plt.figure(figsize=(10, 6))
    plt.hist(len_selfies, alpha=0.5, label="SELFIES length", bins=50)
    plt.hist(len_group_selfies, alpha=0.5, label="Group SELFIES length", bins=50)
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.title("Length Distribution Comparison")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert protein SMILES to group SELFIES")
    parser.add_argument("--input", required=True, help="Path to input SMILES file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--max_samples", type=int, default=None, 
                      help="Maximum number of samples to process")
    parser.add_argument("--generate_selfies", action="store_true",
                      help="Generate regular SELFIES and length distribution plots")
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base name from input file
    name = Path(args.input).stem
    
    # Load data
    print(f"Loading data from {args.input}")
    with open(args.input, "r") as f:
        data = f.read().splitlines()
        if args.max_samples:
            data = data[:args.max_samples]
    print(f"Loaded {len(data)} SMILES strings")

    # Process data
    converter = ProteinConverter()
    if args.generate_selfies:
        prot_group_selfies, prot_selfies = converter.convert_smiles_to_selfies_group(
            data, 
            generate_selfies=True
        )
        
        # Analyze and plot lengths
        print("Calculating sequence lengths...")
        len_selfies = converter.parallel_process(get_len_selfies, prot_selfies)
        len_group_selfies = converter.parallel_process(get_len_selfies, prot_group_selfies)
        plot_length_distributions(
            len_selfies, 
            len_group_selfies, 
            output_dir / f"{name}_len_distribution.png"
        )
    else:
        prot_group_selfies = converter.convert_smiles_to_selfies_group(
            data, 
            generate_selfies=False
        )

    # Save results
    output_path = output_dir / f"{name}_group_selfies.txt"
    print(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        f.write("\n".join(prot_group_selfies))
    print("Done!")

if __name__ == "__main__":
    main()