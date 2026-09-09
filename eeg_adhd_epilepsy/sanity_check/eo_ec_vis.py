import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set matplotlib backend to Agg for headless environments
import matplotlib
matplotlib.use('Agg')

def main():
    csv_path = "/home/mat/projects/EEG_psychostimulant/data/eo_ec_comaprison.csv"
    output_image = "/home/mat/projects/EEG_psychostimulant/data/results/eval/eo_ec_comparison_plot.png"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_image), exist_ok=True)

    # 1. Load data, skip the first empty line
    df = pd.read_csv(csv_path, skiprows=1)
    
    # Clean up column names (remove any leading/trailing spaces)
    df.columns = [c.strip() for c in df.columns]
    
    # Create a unique label for each configuration
    def format_label(row):
        size = row['Size'] if pd.notna(row['Size']) and row['Size'] != '—' else ""
        return f"{row['Model']} {size}\n({row['Representation']})"
    
    df['Label'] = df.apply(format_label, axis=1)

    # 2. Plotting
    plt.style.use('default') # Use default for a clean white background
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.grid(False) # Ensure grid is off
    
    x = np.arange(len(df))
    width = 0.35  # width of the bars
    
    # Values
    metric_ext = "balanced accuracy EEG Motor Movement"
    metric_our = "balanced accuracy our data"
    
    rects1 = ax.bar(x - width/2, df[metric_ext], width, label='EEG Motor Movement', color='darkblue', alpha=0.9)
    rects2 = ax.bar(x + width/2, df[metric_our], width, label='Our Data', color='darkred', alpha=0.9)
    
    # Add labels and title
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Comparison: EEG Motor Movement vs. Our Data', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Label'], fontsize=10)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    # Add numeric values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, weight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_image, dpi=300)
    print(f"Visualization saved to {output_image}")

if __name__ == "__main__":
    main()
