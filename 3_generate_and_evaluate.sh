#!/bin/bash

# ===== EDIT THESE VARIABLES =====
DATASET="protein_data"             # Name of your dataset
INPUT_DIR="models"                 # Directory containing trained model
OUTPUT_DIR="results"               # Directory for generation results
EPOCH=100                         # Which epoch to generate from
NUM_GENERATIONS=10000             # Number of sequences to generate
# ================================

# Load required modules (if using HPC)
module --force purge
module load StdEnv/2020 
module load gcc/9.3.0 
module load arrow/8
module load python/3.8.2
source ~/hf_trans/bin/activate

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate sequences
echo "Generating sequences..."
python python_scripts/generate_accelerate.py \
    --dataset "$DATASET" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epoch "$EPOCH"

# Calculate metrics
echo "Calculating metrics..."
python python_scripts/metrics.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --epoch "$EPOCH"

echo "Generation and evaluation complete!" 