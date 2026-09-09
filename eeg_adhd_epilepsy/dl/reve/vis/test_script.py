import pandas as pd
import numpy as np

csv_path = "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/cbramod/lp_ft/finetune_results_cbramod_lp_ft.csv"
df = pd.read_csv(csv_path)

avg_df = df[df['Fold'] == 'Average'].copy()
std_df = df[df['Fold'].isin(['Std', 'StdDev'])].copy()

avg_df = avg_df.groupby('Condition').mean(numeric_only=True).reset_index()
std_df = std_df.groupby('Condition').mean(numeric_only=True).reset_index()

conditions = sorted(avg_df['Condition'].unique())
avg_df['Condition'] = pd.Categorical(avg_df['Condition'], categories=conditions, ordered=True)
std_df['Condition'] = pd.Categorical(std_df['Condition'], categories=conditions, ordered=True)
avg_df = avg_df.sort_values('Condition')
std_df = std_df.sort_values('Condition')

print("avg_df shape:", avg_df.shape)
print("std_df shape:", std_df.shape)

for col in ['Val_AUC', 'Val_BAcc', 'Val_Acc']:
    means = avg_df[col].values
    stds = std_df[col].values
    print(f"Col: {col}")
    print("means shape:", means.shape)
    print("stds shape:", stds.shape)
