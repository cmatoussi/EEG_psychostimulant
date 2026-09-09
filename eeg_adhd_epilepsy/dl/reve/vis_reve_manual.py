import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    print("REVE Plotting Script Starting...")
    csv_path = "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/finetune_results_reve.csv"
    output_image = "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/reve_finetune_results.png"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    # Manual CSV reading to avoid pandas issues if any
    results = {} # Condition -> {'AUC': [], 'BAcc': []}
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Check if header is long enough, otherwise handle raw rows
        for row in reader:
            if not row or len(row) < 5: continue
            
            fold_str = row[0]
            # Skip Average/StdDev rows
            if not fold_str.isdigit():
                continue
            
            cond = row[1]
            try:
                auc = float(row[2])
                bacc = float(row[4])
            except (ValueError, IndexError):
                continue
                
            if cond not in results:
                results[cond] = {'AUC': [], 'BAcc': []}
            results[cond]['AUC'].append(auc)
            results[cond]['BAcc'].append(bacc)

    if not results:
        print("No valid results found in CSV.")
        return

    conditions = sorted(results.keys())
    auc_means = [np.mean(results[c]['AUC']) for c in conditions]
    auc_stds = [np.std(results[c]['AUC']) for c in conditions]
    bacc_means = [np.mean(results[c]['BAcc']) for c in conditions]
    bacc_stds = [np.std(results[c]['BAcc']) for c in conditions]

    # Plotting
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax.bar(x - width/2, auc_means, width, yerr=auc_stds, label='Avg AUC', capsize=5, color='skyblue', alpha=0.8)
    ax.bar(x + width/2, bacc_means, width, yerr=bacc_stds, label='Avg Balanced Acc', capsize=5, color='salmon', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('REVE Fine-tuning Results by Condition (5-Fold Mean ± Std)', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='black', lw=1, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Successfully saved REVE plot to {output_image}")

if __name__ == "__main__":
    main()
