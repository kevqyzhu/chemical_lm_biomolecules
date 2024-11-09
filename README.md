# Protein Sequence Generation Pipeline

This repository contains a pipeline for converting protein SMILES to SELFIES/group SELFIES representations, training a generative model, and evaluating the generated sequences.

## Directory Structure

```
.
├── python_scripts/
│   ├── convert_group_selfies.py
│   ├── generate_selfies.py
│   ├── get_alphabet.py
│   ├── train_gpt_from_scratch_accelerate.py
│   ├── train_new_tokenizer.py
│   ├── generate_accelerate.py
│   ├── metrics.py
│   ├── group_utils.py
│   └── primary_sequence.py
├── 1_prepare_data.sh
├── 2_train_model.sh
├── 3_generate_and_evaluate.sh
└── README.md
```

## Pipeline Overview

The pipeline consists of three main steps:

1. **Data Preparation**: Convert protein SMILES to SELFIES or group SELFIES representations
2. **Model Training**: Train a GPT-2 model on the processed sequences
3. **Generation & Evaluation**: Generate new sequences and evaluate their properties

## Requirements

### Environment Setup

The project uses conda for environment management. Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate protein-gen
```

The `environment.yml` file contains all necessary dependencies:

```yaml
name: protein-gen
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8.2
  - pip
  - pytorch
  - cudatoolkit
  - numpy
  - pandas
  - matplotlib
  - pillow
  - scipy
  - tqdm
  - git  # Required for installing group-selfies
  # Core dependencies via pip
  - pip:
    - rdkit
    - selfies
    - transformers
    - accelerate
    - datasets
    - git+https://github.com/aspuru-guzik-group/group-selfies.git
```

Note: The `group-selfies` package must be installed from source and is not available via PyPI. The environment file handles this by installing directly from the GitHub repository.

### Computing Environment

The scripts are designed for HPC environments with GPU support. Module loading commands in the scripts should be modified according to your specific environment:

```bash
module --force purge
module load StdEnv/2020 
module load gcc/9.3.0 
module load arrow/8
module load python/3.8.2
```

## Usage

### 1. Data Preparation

Edit the variables in `1_prepare_data.sh`:

```bash
INPUT_FILE="proteins.txt"           # Path to input SMILES file
OUTPUT_DIR="data"                   # Directory for processed data
MAX_SAMPLES=2000                    # Maximum number of samples to process
MODE="group_selfies"               # Either 'selfies' or 'group_selfies'
NUM_PROCESSES=80                    # Number of processes for parallel computation
```

Then run:

```bash
./1_prepare_data.sh
```

### 2. Model Training

Edit the variables in `2_train_model.sh`:

```bash
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
```

Then run:

```bash
./2_train_model.sh
```

### 3. Generation and Evaluation

Edit the variables in `3_generate_and_evaluate.sh`:

```bash
DATASET="protein_data"             # Name of your dataset
INPUT_DIR="models"                 # Directory containing trained model
OUTPUT_DIR="results"               # Directory for generation results
EPOCH=100                         # Which epoch to generate from
NUM_GENERATIONS=10000             # Number of sequences to generate
```

Then run:

```bash
./3_generate_and_evaluate.sh
```

## Output Files

### Data Preparation
- `{OUTPUT_DIR}/{BASE_NAME}_selfies.txt`: Converted SELFIES sequences
- `{OUTPUT_DIR}/{BASE_NAME}_alphabet.npy`: Vocabulary file for tokenizer

### Model Training
- `{OUTPUT_DIR}/tokenizer/`: Trained tokenizer files
- `{OUTPUT_DIR}/checkpointing/`: Model checkpoints
- `{OUTPUT_DIR}/lengths.png`: Sequence length distribution plot

### Generation
- `{OUTPUT_DIR}/generated_{EPOCH}.txt`: Raw generated sequences
- `{OUTPUT_DIR}/generated_smiles_{EPOCH}.txt`: Generated sequences converted to SMILES
- `{OUTPUT_DIR}/generated_smiles_{EPOCH}_backbone.txt`: Success rate statistics
- `{OUTPUT_DIR}/hist_*_{EPOCH}.png`: Various metric distribution plots

## Notes

- The pipeline assumes input SMILES strings represent protein structures
- For large datasets, adjust `MAX_SAMPLES` and `NUM_PROCESSES` according to available computational resources
- Model training parameters (`BATCH_SIZE`, `LEARNING_RATE`, etc.) may need tuning based on your specific dataset
- GPU availability is assumed for model training and generation
- Progress and error messages are printed to stdout

## Troubleshooting

Common issues:
1. **Memory errors during training**: Reduce `BATCH_SIZE` or `CONTEXT_LENGTH`
2. **GPU out of memory**: Reduce model size (`N_LAYER`, `N_EMBD`) or `BATCH_SIZE`
3. **Slow data processing**: Adjust `NUM_PROCESSES` based on available CPU cores
4. **Module load errors**: Modify module loading commands to match your HPC environment
