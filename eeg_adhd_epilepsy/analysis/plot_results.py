import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data from the user
data = [
    {"Model": "REVE Base", "Balanced Accuracy": 0.5883, "Accuracy": 0.6104, "F1": 0.5759},
    {"Model": "REVE Large", "Balanced Accuracy": 0.5729, "Accuracy": 0.6015, "F1": 0.5633},
    {"Model": "CBraMod", "Balanced Accuracy": 0.5657, "Accuracy": 0.6061, "F1": 0.5594},
]

df = pd.DataFrame(data)

# Set the style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plotting
ax = sns.barplot(x="Model", y="Balanced Accuracy", data=df, palette="viridis")

# Add labels and title
plt.title("Epilepsy Classification: Balanced Accuracy Comparison", fontsize=16)
plt.ylabel("Balanced Accuracy", fontsize=12)
plt.xlabel("Model Configuration", fontsize=12)
plt.ylim(0.5, 0.65) # Zoom in for better contrast

# Add values on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.4f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points',
                fontsize=11, weight='bold')

plt.tight_layout()

# Save the plot
output_path = "/home/mat/projects/EEG_psychostimulant/data/results/eval/model_performance_comparison.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
