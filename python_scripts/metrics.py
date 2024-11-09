import random
import multiprocessing
from pathlib import Path
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from group_utils import (
    get_num_heavy_smi, 
    convert_group_to_smi, 
    create_FASTA_grammar, 
    get_mw_smi
)

# Initialize global grammar
GRAMMAR = create_FASTA_grammar()

def parallel_process(func, data, pool_size=20):
    """Generic parallel processing function."""
    with multiprocessing.Pool(pool_size) as pool:
        return pool.map(func, data)

def get_train_data(input_dir, dataset, sample_size=2000):
    """Load and process training data."""
    input_path = Path(input_dir) / f"{dataset}_group_selfies.txt"
    with open(input_path, "r") as f:
        group_selfies_data = f.read().splitlines()[:-1]
    
    random.shuffle(group_selfies_data)
    group_selfies_data = group_selfies_data[:sample_size]
    return parallel_process(
        lambda x: convert_group_to_smi(x, GRAMMAR), 
        group_selfies_data
    )

def plot_distribution_comparison(data_name, metric_name, gen_data, train_data, out_path):
    """Plot comparison of distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(data_name)
    
    ax1.hist(gen_data)
    ax2.hist(train_data)
    
    for ax, title in zip([ax1, ax2], ['Generated', 'Training']):
        ax.set_title(title)
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def get_dataset_samples(input_dir, dataset, epoch, sample_size=1000):
    """Get samples from generated and training data."""
    input_dir = Path(input_dir)
    with open(input_dir / f"{dataset}_group_selfies/generated_smiles_{epoch}.txt", "r") as f:
        generated_data = f.read().splitlines()[:-1]
    
    with open(input_dir / f"{dataset}_training_smiles.txt", "r") as f:
        train_data = f.read().splitlines()[:-1]
    
    random.shuffle(generated_data)
    random.shuffle(train_data)
    
    return (
        generated_data[:sample_size],
        train_data[:sample_size],
        train_data[-sample_size:]
    )

def calculate_metric(input_dir, output_dir, dataset, epoch, metric_func, metric_name):
    """Calculate and plot metric distributions."""
    gen_data, train_data1, train_data2 = get_dataset_samples(input_dir, dataset, epoch)
    
    metric_gen = parallel_process(metric_func, gen_data)
    metric_train1 = parallel_process(metric_func, train_data1)
    metric_train2 = parallel_process(metric_func, train_data2)
    
    print(f"Train-Train WD: {wasserstein_distance(metric_train1, metric_train2)}")
    print(f"Train-Gen WD: {wasserstein_distance(metric_train1, metric_gen)}")
    
    output_path = Path(output_dir) / f"{dataset}_group_selfies/hist_{metric_name.lower()}_{epoch}.png"
    plot_distribution_comparison(
        dataset, 
        metric_name,
        metric_gen,
        metric_train1,
        output_path
    )

def num_heavy_metric(input_dir, output_dir, dataset, epoch):
    """Calculate number of heavy atoms metric."""
    calculate_metric(input_dir, output_dir, dataset, epoch, get_num_heavy_smi, "Num Heavy Atoms")

def mol_weight_metric(input_dir, output_dir, dataset, epoch):
    """Calculate molecular weight metric."""
    calculate_metric(input_dir, output_dir, dataset, epoch, get_mw_smi, "Molecular Weight")