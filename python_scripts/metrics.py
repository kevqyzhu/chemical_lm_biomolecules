import random
import argparse
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
    
    # Create histograms with the same bins for both plots
    bins = 50
    range_min = min(min(gen_data), min(train_data))
    range_max = max(max(gen_data), max(train_data))
    
    ax1.hist(gen_data, bins=bins, range=(range_min, range_max))
    ax2.hist(train_data, bins=bins, range=(range_min, range_max))
    
    for ax, title in zip([ax1, ax2], ['Generated', 'Training']):
        ax.set_title(title)
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def calculate_metrics(gen_smiles, train_smiles, output_dir, epoch):
    """Calculate and plot various metrics."""
    metrics = {
        'Heavy Atoms': (get_num_heavy_smi, 'Number of Heavy Atoms'),
        'Molecular Weight': (get_mw_smi, 'Molecular Weight (Da)')
    }
    
    results = {}
    for metric_name, (metric_func, plot_label) in metrics.items():
        print(f"Calculating {metric_name}...")
        
        # Calculate metrics for both sets
        gen_data = parallel_process(metric_func, gen_smiles)
        train_data = parallel_process(metric_func, train_smiles)
        
        # Remove None values and handle empty results
        gen_data = [x for x in gen_data if x is not None]
        train_data = [x for x in train_data if x is not None]
        
        if not gen_data or not train_data:
            print(f"Warning: No valid data for {metric_name}")
            results[metric_name] = None
            continue
        
        # Calculate Wasserstein distance
        try:
            w_dist = wasserstein_distance(gen_data, train_data)
            results[metric_name] = w_dist
        except Exception as e:
            print(f"Error calculating Wasserstein distance for {metric_name}: {e}")
            results[metric_name] = None
            continue
        
        # Plot distribution
        try:
            plot_path = Path(output_dir) / f'hist_{metric_name.lower().replace(" ", "_")}_{epoch}.png'
            plot_distribution_comparison(
                metric_name,
                plot_label,
                gen_data,
                train_data,
                plot_path
            )
        except Exception as e:
            print(f"Error plotting distribution for {metric_name}: {e}")
    
    return results

def main():
    """Main function to run metrics calculation."""
    parser = argparse.ArgumentParser(description="Calculate metrics for generated sequences")
    parser.add_argument('--input_dir', type=str, required=True,
                      help="Directory containing model and data")
    parser.add_argument('--output_dir', type=str, required=True,
                      help="Directory for saving results")
    parser.add_argument('--dataset', type=str, required=True,
                      help="Dataset name")
    parser.add_argument('--epoch', type=int, required=True,
                      help="Epoch number to evaluate")
    args = parser.parse_args()

    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load generated sequences
        gen_file = output_dir / f'generated_smiles_{args.epoch}.txt'
        with open(gen_file, 'r') as f:
            gen_smiles = f.read().splitlines()
        print(f"Loaded {len(gen_smiles)} generated sequences")

        # Load and process training data
        train_smiles = get_train_data(input_dir, args.dataset)
        print(f"Loaded {len(train_smiles)} training sequences")

        # Calculate metrics
        print("Calculating metrics...")
        results = calculate_metrics(gen_smiles, train_smiles, output_dir, args.epoch)

        # Save metric results
        metrics_file = output_dir / f'metrics_{args.epoch}.txt'
        with open(metrics_file, 'w') as f:
            for metric, value in results.items():
                if value is not None:
                    f.write(f"{metric} Wasserstein distance: {value:.4f}\n")
                else:
                    f.write(f"{metric} Wasserstein distance: Failed to calculate\n")

        print("Metrics calculation complete!")
        
    except Exception as e:
        print(f"Error during metrics calculation: {e}")
        raise

if __name__ == "__main__":
    main()