import selfies as sf
import multiprocessing
import numpy as np
import argparse
import os


def get_alphabet_handler(selfies_list):
    num_workers = 80
    sf_lists = list(np.array_split(selfies_list, num_workers))
    p = multiprocessing.Pool(num_workers)
    return p.map(sf.get_alphabet_from_selfies, sf_lists)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process SELFIES and generate alphabet')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output files')
    parser.add_argument('--name', type=str, default="protein_drug_conjugates_af2db",
                       help='Base name of the files (default: protein_drug_conjugates_af2db)')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input file
    input_path = os.path.join(args.input_dir, f"{args.name}.txt")
    with open(input_path, "r") as f:
        selfies_list = f.read().split("\n")

    alphabet = sorted(set.union(*(get_alphabet_handler(selfies_list))))
    
    # Save to output directory
    output_path = os.path.join(args.output_dir, f"{args.name}_alphabet.npy")
    with open(output_path, 'wb') as f:
        np.save(f, np.array(list(alphabet)))