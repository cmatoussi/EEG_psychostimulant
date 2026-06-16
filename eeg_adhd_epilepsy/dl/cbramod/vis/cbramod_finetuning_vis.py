import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_cbramod_fine_tuning(csv_path, output_path):
    # 1. Load data
    df = pd.read_csv(csv_path)

    # 2. Extract Averages and Std
    avg_df = df[df['Fold'] == 'Average'].copy()
    std_df = df[df['Fold'] == 'Std'].copy()

    # Sort conditions
    conditions = sorted(avg_df['Condition'].unique())
    avg_df['Condition'] = pd.Categorical(avg_df['Condition'], categories=conditions, ordered=True)
    std_df['Condition'] = pd.Categorical(std_df['Condition'], categories=conditions, ordered=True)
    avg_df = avg_df.sort_values('Condition')
    std_df = std_df.sort_values('Condition')

    # Metrics to plot
    metrics = {
        'Val_AUC': 'Avg AUC',
        'Val_BAcc': 'Avg Balanced Accuracy',
        'Val_Acc': 'Avg Accuracy'
    }
    
    # 3. Plotting Setup
    n_conditions = len(conditions)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(26, 13))
    
    # Colors: Dark Red, Navy, Dark Green (User's preferred theme)
    colors = ['#8B0000', '#000080', '#006400']
    
    bar_width = 0.30
    index = np.arange(n_conditions)

    # 4. Create bars for each metric
    for i, (col, label) in enumerate(metrics.items()):
        means = avg_df[col].values
        stds = std_df[col].values
        
        pos = index + (i * bar_width)
        
        bars = ax.bar(pos, means, bar_width, yerr=stds, label=label, 
                      color=colors[i], capsize=5, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add labels on top of bars
        for k, bar in enumerate(bars):
            height = bar.get_height()
            err_offset = stds[k] if not np.isnan(stds[k]) else 0
            ax.text(bar.get_x() + bar.get_width() / 2, height + err_offset + 0.01, 
                    f'{height:.3f}', ha='center', va='bottom', fontsize=21, fontweight='bold')

    # 5. Aesthetics
    ax.set_xlabel('Condition', fontsize=24, fontweight='bold')
    ax.set_ylabel('Score', fontsize=24, fontweight='bold')
    ax.set_title('CBraMod Fine-Tuning Performance Across Conditions', fontsize=36, fontweight='bold', pad=30)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=28)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', ncol=1, fontsize=36, prop={'weight': 'bold'}, 
              frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    csv_file = "/home/mat/scratch/EEG_results/results/finetuned_param/cbramod/lp_ft/finetune_results_cbramod_lp_ft.csv"
    output_image = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/cbramod/vis/cbramod_finetuning_performance.png"
    
    visualize_cbramod_fine_tuning(csv_file, output_image)
