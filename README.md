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
│   ├── primary_sequence.py
│   ├── convert_to_smiles.py
│   └── build_modified_proteins_datasets.py
├── data/
│   ├── 50_150_amino_acids.csv.zip
│   ├── single_domain_antibodies.csv
│   ├── zinc.txt
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

## Data Processing Scripts

### Converting PDB to SMILES
The `convert_to_smiles.py` script converts protein PDB files to SMILES sequences. This is useful if you want to process your own PDB files. The script uses RDKit for the conversion.

To use your own PDB files:
1. Place your PDB files in a directory
2. Run the conversion script:
```bash
python convert_to_smiles.py --input_dir /path/to/pdb/files --output_file proteins.txt
```

### Protein Modification
The `build_modified_proteins_datasets.py` script attaches small molecules to protein backbones. It can:
- Attach modifications to specific residues (default: Lysine/K)
- Handle multiple modifications per protein
- Process proteins in parallel for efficiency

To modify proteins:
```bash
python build_modified_proteins_datasets.py \
    --input_file proteins.txt \
    --modifications_file modifications.csv \
    --output_file modified_proteins.txt \
    --sample_size 100 \
    --attachable_residue K
```

Note: Pre-processed datasets are already included in this repository. You only need to use these scripts if you want to generate your own data from custom PDB files or create new protein modifications.

## Requirements

### Environment Setup

The project uses conda for environment management. Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate clm-biomolecules
```

The `environment.yml` file contains all necessary dependencies:

```yaml
name: clm-biomolecules
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

## Troubleshooting

Common issues:
1. **Memory errors during training**: Reduce `BATCH_SIZE` or `CONTEXT_LENGTH`
2. **GPU out of memory**: Reduce model size (`N_LAYER`, `N_EMBD`) or `BATCH_SIZE`
