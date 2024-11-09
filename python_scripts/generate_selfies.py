import pandas as pd
import selfies as sf
import multiprocessing
import tqdm
import argparse
from itertools import repeat
from pathlib import Path
from typing import List, Optional

def filter_selfies(selfies: str, max_len: int) -> Optional[str]:
    """
    Filter SELFIES string based on maximum length.
    
    Args:
        selfies: SELFIES string to filter
        max_len: Maximum allowed length of SELFIES symbols
    
    Returns:
        Filtered SELFIES string if valid, None otherwise
    """
    try:
        symbols = list(sf.split_selfies(selfies))
        if len(symbols) > 0 and len(symbols) <= max_len:
            return selfies
    except:
        print("SELFIES split failed")
    return None

def smiles_to_selfies(smiles: str, max_len: int) -> Optional[str]:
    """
    Convert SMILES to SELFIES format with validation.
    
    Args:
        smiles: Input SMILES string
        max_len: Maximum allowed length of SELFIES symbols
    
    Returns:
        Validated SELFIES string if conversion successful, None otherwise
    """
    try:
        selfies = sf.encoder(smiles)
    except:
        print("SELFIES generation failed")
        return None
        
    try:
        # Validate by converting back to SMILES
        _ = sf.decoder(selfies)
    except:
        print("SELFIES decode failed")
        return None
        
    return filter_selfies(selfies, max_len)

def smiles_to_selfies_handler(text: List[str], max_len: int) -> List[Optional[str]]:
    """
    Handle batch conversion of SMILES to SELFIES using multiprocessing.
    
    Args:
        text: List of SMILES strings to convert
        max_len: Maximum allowed length of SELFIES symbols
    
    Returns:
        List of converted SELFIES strings (None for failed conversions)
    """
    with multiprocessing.Pool(N) as p:
        return p.starmap(
            smiles_to_selfies, 
            tqdm.tqdm(zip(text, repeat(max_len)), total=len(text))
        )

def generate_selfies(input_path: Path, output_path: Path, max_len: int) -> None:
    """
    Generate SELFIES from SMILES file and save results.
    
    Args:
        input_path: Path to input SMILES file
        output_path: Path to save output SELFIES file
        max_len: Maximum allowed length of SELFIES symbols
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read input SMILES
    with open(input_path, "r") as f:
        smiles_text = f.read().splitlines()[:-1]
    
    # Convert to SELFIES
    selfies = smiles_to_selfies_handler(smiles_text, max_len)
    selfies = [s for s in selfies if s is not None]
    
    # Save results
    with open(output_path, "w") as f:
        f.write("\n".join(selfies) + "\n")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert SMILES to SELFIES with length filtering")
    parser.add_argument('--input_path', type=str, required=True, 
                       help="Path to input SMILES file")
    parser.add_argument('--output_path', type=str, required=True, 
                       help="Path to save output SELFIES file")
    parser.add_argument('--max_len', type=int, required=True, 
                       help="Maximum length of SELFIES symbols")
    parser.add_argument('--num_procs', type=int, required=True, 
                       help="Number of processes to use")
    args = parser.parse_args()

    # Set global number of processes
    global N
    N = args.num_procs
    
    # Convert paths to Path objects
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    # Run conversion
    generate_selfies(input_path, output_path, args.max_len)