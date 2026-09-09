import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_cbramod_improvement(without_ft_path, with_ft_path, output_path):
    # 1. Load data
    df_no = pd.read_csv(without_ft_path)
    df_wi = pd.read_csv(with_ft_path)

    # 2. Extract Averages and Std
    # Both use 'Average' and 'Std' for CBraMod
    avg_no = df_no[df_no['Fold'] == 'Average'].copy()
    std_no = df_no[df_no['Fold'] == 'Std'].copy()
    
    avg_wi = df_wi[df_wi['Fold'] == 'Average'].copy()
    std_wi = df_wi[df_wi['Fold'] == 'Std'].copy()

    # Metrics mapping
    metrics = {
        'Val_AUC': 'AUC',
        'Val_BAcc': 'Balanced Accuracy',
        'Val_Acc': 'Accuracy'
    }
    
    # Common conditions
    conditions = sorted(list(set(avg_no['Condition']) & set(avg_wi['Condition'])))
    
    # Filter and sort
    avg_no = avg_no[avg_no['Condition'].isin(conditions)].sort_values('Condition')
    std_no = std_no[std_no['Condition'].isin(conditions)].sort_values('Condition')
    avg_wi = avg_wi[avg_wi['Condition'].isin(conditions)].sort_values('Condition')
    std_wi = std_wi[std_wi['Condition'].isin(conditions)].sort_values('Condition')

    # 3. Plotting Setup
    fig, axes = plt.subplots(3, 1, figsize=(26, 22), sharex=True)
    
    colors_no = ['#E9967A', '#ADD8E6', '#90EE90'] # Light Salmon, Light Blue, Light Green
    colors_wi = ['#8B0000', '#000080', '#006400'] # Dark Red, Navy, Dark Green
    
    bar_width = 0.25
    index = np.arange(len(conditions))

    for i, (col, label) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Means and stds
        m_no = avg_no[col].values
        s_no = std_no[col].values
        m_wi = avg_wi[col].values
        s_wi = std_wi[col].values
        
        # Plot bars
        rects1 = ax.bar(index - bar_width/2, m_no, bar_width, yerr=s_no, label='Without Fine-Tuning', 
                        color=colors_no[i], capsize=5, alpha=0.9, edgecolor='black', linewidth=0.5)
        rects2 = ax.bar(index + bar_width/2, m_wi, bar_width, yerr=s_wi, label='With Fine-Tuning', 
                        color=colors_wi[i], capsize=5, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add labels on top
        def autolabel(rects, stds):
            for k, rect in enumerate(rects):
                height = rect.get_height()
                offset = stds[k] if not np.isnan(stds[k]) else 0
                ax.text(rect.get_x() + rect.get_width() / 2, height + offset + 0.01, 
                        f'{height:.3f}', ha='center', va='bottom', fontsize=21, fontweight='bold')

        autolabel(rects1, s_no)
        autolabel(rects2, s_wi)
        
        # Aesthetics for each subplot
        ax.set_ylabel(f'{label} Score', fontsize=24, fontweight='bold')
        ax.set_title(f'Comparison: {label} (Without vs. With Fine-Tuning) - CBraMod', fontsize=32, fontweight='bold', pad=30)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(loc='lower right', ncol=1, fontsize=36, prop={'weight': 'bold'}, 
                  frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')

    # Set x-ticks once at the bottom
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(conditions, rotation=45, ha='right', fontsize=28, fontweight='bold')
    axes[2].set_xlabel('EEG Condition', fontsize=24, fontweight='bold')

    plt.tight_layout()
    
    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Comparison visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    without_ft = "/home/mat/projects/EEG_psychostimulant/data/results/eval/cbramod_without_finetuning.csv"
    with_ft = "/home/mat/scratch/EEG_results/results/finetuned_param/cbramod/lp_ft/finetune_results_cbramod_lp_ft.csv"
    output_image = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/cbramod/vis/cbramod_ft_comparison.png"
    
    visualize_cbramod_improvement(without_ft, with_ft, output_image)
