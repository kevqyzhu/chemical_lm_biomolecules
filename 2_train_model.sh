#!/bin/bash

# ===== EDIT THESE VARIABLES =====
DATASET="protein_data"             # Name of your dataset
INPUT_DIR="data"                   # Directory containing processed data
OUTPUT_DIR="models"                # Directory for saving model
EPOCHS=500                         # Number of training epochs
BATCH_SIZE=4                       # Batch size for training
LEARNING_RATE=5e-4                 # Learning rate
CONTEXT_LENGTH=2048                # Maximum sequence length
N_LAYER=12                         # Number of transformer layers
N_EMBD=512                        # Embedding dimension
N_HEAD=8                          # Number of attention heads
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

# First train the tokenizer
echo "Training tokenizer..."
python python_scripts/train_new_tokenizer.py \
    --input_file "${INPUT_DIR}/${DATASET}.txt" \
    --output_dir "${OUTPUT_DIR}" \
    --alphabet_file "${INPUT_DIR}/${DATASET}_alphabet.npy"

# Then train the model
echo "Training model..."
python python_scripts/train_gpt_from_scratch_accelerate.py \
    --dataset_path "${INPUT_DIR}/${DATASET}.txt" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --context_length "$CONTEXT_LENGTH" \
    --n_layer "$N_LAYER" \
    --n_embd "$N_EMBD" \
    --n_head "$N_HEAD" 