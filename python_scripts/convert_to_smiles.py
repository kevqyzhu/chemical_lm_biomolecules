from rdkit import Chem
import os
import random
import multiprocessing
import pickle
from os import listdir
from os.path import isfile, join
import argparse

global CORES

def convert(pdb_file):
    pdb_name = pdb_file.split('/')[-1].split('.')[0]
    m = Chem.MolFromPDBFile(str(pdb_file), removeHs=False)
    if m is not None:
        out = Chem.MolToSmiles(m) + " " + pdb_name
        return out
    
def convert_handler(text):
    p = multiprocessing.Pool(CORES)
    return p.map(convert, text)

def convert_to_smiles(input_dir, output_file):
    pdb_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if isfile(join(input_dir, f))]

    smiles = convert_handler(pdb_files)
    smiles = [i for i in smiles if i is not None]

    with open(output_file, 'w') as f:
        for i in smiles:
            f.write(i + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Convert PDB files to SMILES format')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing PDB files')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Output file path for SMILES data')
    parser.add_argument('--cores', type=int, default=80,
                      help='Number of CPU cores to use (default: 80)')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    CORES = args.cores
    convert_to_smiles(args.input_dir, args.output_file)