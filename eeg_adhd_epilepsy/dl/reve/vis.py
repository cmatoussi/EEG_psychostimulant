import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--model", type=str, default="REVE")
    args = parser.parse_args()

    model_name = args.model
    if args.csv:
        csv_path = args.csv
        output_image = args.out or os.path.splitext(args.csv)[0] + "_plot.png"
    elif model_name == "REVE":
        csv_path = "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/finetune_results_reve.csv"
        output_image = "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/reve_finetune_results.png"
    else:
        csv_path = "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/cbramod/finetune_results_cbramod.csv"
        output_image = "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/cbramod/cbramod_finetune_results.png"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # 2. Load and Process Data
    # REVE CSV might have missing header columns. 
    # Read first line to check columns
    with open(csv_path, 'r') as f:
        header_line = f.readline().strip().split(',')
    
    print(f"Loading CSV from {csv_path}...")
    if len(header_line) < 6:
        print("Incomplete header detected in file. Specifying names manually...")
        column_names = ["Fold", "Condition", "Val_AUC", "Val_Acc", "Val_BAcc", "Val_F1"]
        df = pd.read_csv(csv_path, names=column_names, skiprows=1)
    else:
        df = pd.read_csv(csv_path)
    
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Filter out summary rows ('Average', 'Std') and drop duplicates
    # Handle both "Fold 1" string format and plain "1" integer format
    fold_numeric = df['Fold'].astype(str).str.replace(r'[Ff]old\s*', '', regex=True)
    df = df[pd.to_numeric(fold_numeric, errors='coerce').notna()].copy()
    print(f"Shape after filtering non-numeric folds: {df.shape}")

    df['Fold'] = pd.to_numeric(df['Fold'].astype(str).str.replace(r'[Ff]old\s*', '', regex=True), errors='coerce').astype(int)
    
    # Identify and drop EXACT duplicates (happens if job array restarted)
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} duplicate rows.")

    if len(df) == 0:
        print("ERROR: No valid data left to plot!")
        return

    # Group by Condition to get Mean and Std
    # We want Val_AUC and Val_BAcc
    metrics = ['Val_AUC', 'Val_BAcc']
    
    # Ensure columns are numeric
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate means and stds
    summary = df.groupby('Condition')[metrics].agg(['mean', 'std']).reset_index()
    
    # Flatten multi-index columns for easier plotting
    summary.columns = ['Condition', 'Val_AUC_mean', 'Val_AUC_std', 'Val_BAcc_mean', 'Val_BAcc_std']
    
    # Sort by condition name for consistency
    summary = summary.sort_values('Condition')

    # 3. Plotting
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Setting up the bar positions
    x = np.arange(len(summary['Condition']))
    width = 0.35  # width of the bars
    
    # Create the grouped bars
    rects1 = ax.bar(x - width/2, summary['Val_AUC_mean'], width, 
                    yerr=summary['Val_AUC_std'], label='Avg AUC', 
                    capsize=5, color='skyblue', alpha=0.8)
    rects2 = ax.bar(x + width/2, summary['Val_BAcc_mean'], width, 
                    yerr=summary['Val_BAcc_std'], label='Avg Balanced Acc', 
                    capsize=5, color='salmon', alpha=0.8)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} Fine-tuning Results by Condition (5-Fold Mean ± Std)', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Condition'], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.0) # Scores are between 0 and 1
    
    # Add optional dashed line at 0.5 (chance)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Function to add labels on top of bars, above the error bar caps
    def autolabel(rects, yerr_vals):
        for rect, err in zip(rects, yerr_vals):
            height = rect.get_height()
            err = err if not np.isnan(err) else 0
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height + err),
                        xytext=(0, 6),  # 6 points above the top of the error bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1, summary['Val_AUC_std'].fillna(0).values)
    autolabel(rects2, summary['Val_BAcc_std'].fillna(0).values)

    plt.tight_layout()
    
    # Save 
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Successfully saved plot to {output_image}")
    
    # Also show if in interactive env
    # plt.show()

if __name__ == "__main__":
    main()
