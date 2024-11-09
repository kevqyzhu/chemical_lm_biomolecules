#!/bin/bash

# ===== EDIT THESE VARIABLES =====
INPUT_FILE="proteins.txt"           # Path to input SMILES file
OUTPUT_DIR="data"                   # Directory for processed data
MAX_SAMPLES=2000                    # Maximum number of samples to process
MODE="group_selfies"               # Either 'selfies' or 'group_selfies'
NUM_PROCESSES=80                    # Number of processes for parallel computation
# ================================

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get base name of input file
BASE_NAME=$(basename "$INPUT_FILE" .txt)

# Convert SMILES to SELFIES/Group SELFIES
if [ "$MODE" = "selfies" ]; then
    echo "Converting SMILES to SELFIES..."
    python python_scripts/generate_selfies.py \
        --input_path "$INPUT_FILE" \
        --output_path "$OUTPUT_DIR/${BASE_NAME}_selfies.txt" \
        --max_len 250 \
        --num_procs "$NUM_PROCESSES"
else
    echo "Converting SMILES to Group SELFIES..."
    python python_scripts/convert_group_selfies.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR" \
        --max_samples "$MAX_SAMPLES" \
        --generate_selfies
fi

# Generate alphabet
echo "Generating alphabet..."
python python_scripts/get_alphabet.py \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --name "$BASE_NAME"

echo "Data preparation complete!" 