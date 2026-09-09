import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_reve_fine_tuning(csv_path, output_path):
    # 1. Load data
    df = pd.read_csv(csv_path)

    # 2. Extract Averages and StdDevs
    avg_df = df[df['Fold'] == 'Average'].copy()
    std_df = df[df['Fold'].isin(['Std', 'StdDev'])].copy()

    # Handle multiple entries by taking the mean
    avg_df = avg_df.groupby('Condition').mean(numeric_only=True).reset_index()
    std_df = std_df.groupby('Condition').mean(numeric_only=True).reset_index()

    # Sort conditions for consistent display
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
    
    # Colors: Dark Red, Dark Blue, and Dark Green
    colors = ['#8B0000', '#000080', '#006400'] # DarkRed, Navy, DarkGreen
    
    bar_width = 0.30
    index = np.arange(n_conditions)

    # 4. Create bars for each metric
    for i, (col, label) in enumerate(metrics.items()):
        means = avg_df[col].values
        stds = std_df[col].values
        
        pos = index + (i * bar_width)
        
        bars = ax.bar(pos, means, bar_width, yerr=stds, label=label, 
                      color=colors[i], capsize=5, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add values on top of bars
        for k, bar in enumerate(bars):
            height = bar.get_height()
            # Calculate offset to be above the error bar
            err_offset = stds[k] if not np.isnan(stds[k]) else 0
            ax.text(bar.get_x() + bar.get_width() / 2, height + err_offset + 0.01, 
                    f'{height:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')

    # 5. Aesthetics
    ax.set_xlabel('Condition', fontsize=24, fontweight='bold')
    ax.set_ylabel('Score', fontsize=24, fontweight='bold')
    ax.set_title('REVE large balanced Fine-Tuning Performance Across Conditions', fontsize=36, fontweight='bold', pad=30)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=28)
    ax.set_ylim(0, 1.0) # Scores are between 0 and 1
    ax.tick_params(axis='y', labelsize=24)
    ax.legend(loc='lower right', ncol=1, prop={'weight': 'bold', 'size': 36}, 
              frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    csv_file = "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/large/balanced/finetune_results_reve_large.csv"
    output_image = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve/vis/reve_large_balanced_performance.png"
    
    visualize_reve_fine_tuning(csv_file, output_image)
