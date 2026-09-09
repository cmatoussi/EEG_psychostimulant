import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

def main():
    # 1. Load data from conditions.json
    json_path = Path("/home/mat/projects/EEG_psychostimulant/data/results/conditions.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    summary = data.get("SUMMARY", {})
    if not summary:
        print("Error: No SUMMARY data found in conditions.json")
        return

    # 2. Parse the SUMMARY strings: "count/total_sec/avg_min"
    parsed_data = []
    for condition, val_str in summary.items():
        parts = val_str.split('/')
        if len(parts) == 3:
            parsed_data.append({
                "Condition": condition,
                "Count": int(parts[0]),
                "AvgMinutes": float(parts[2])
            })
    
    df = pd.DataFrame(parsed_data).sort_values("Count", ascending=False)

    # Ensure output directory exists
    output_dir = Path("/home/mat/projects/EEG_psychostimulant/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create Plots
    # Plot 1: Subject Count per Condition
    plt.figure(figsize=(12, 6))
    plt.bar(df['Condition'], df['Count'], color='midnightblue', edgecolor='black')
    plt.title('Number of Subjects per EEG Condition', fontsize=14)
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Subject Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "subject_count_chart.png")
    print(f"Saved: {output_dir / 'subject_count_chart.png'}")

    # Plot 2: Average Recording Time per Condition
    plt.figure(figsize=(12, 6))
    df_sorted_time = df.sort_values("AvgMinutes", ascending=False)
    plt.bar(df_sorted_time['Condition'], df_sorted_time['AvgMinutes'], color='firebrick', edgecolor='black')
    plt.title('Average Recording Duration per Condition', fontsize=14)
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Duration (Minutes)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "avg_duration_chart.png")
    print(f"Saved: {output_dir / 'avg_duration_chart.png'}")

if __name__ == "__main__":
    main()
