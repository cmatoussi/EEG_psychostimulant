import os
import sys
import yaml
import pandas as pd
from pathlib import Path
import itertools
import argparse

def parse_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_binary_val(row, col):
    """Safely extract 1/0 binary value from a pandas row, handling strings/floats/NaNs."""
    val = row.get(col)
    if pd.notna(val):
        try:
            return int(float(val)) == 1
        except:
            pass
    return False

def check_comorbidity(row, condition):
    tsa = get_binary_val(row, 'TSA')
    tdah = get_binary_val(row, 'TDAH')
    
    if condition == 'none':
        return not tsa and not tdah
    elif condition == 'ASD':
        return tsa and not tdah
    elif condition == 'ADHD':
        return not tsa and tdah
    elif condition == 'ASD&ADHD':
        return tsa and tdah
    elif condition == 'ALL':
        return True
    return False

def check_medication(row, condition):
    asm_columns = ['LEV', 'LTG', 'LCS', 'CLB', 'CBZ', 'VPA', 'ETH', 'TPM', 'RUF', 'BRV', 'STP', 'OXZ', 'CBM']
    
    is_asm = False
    for col in asm_columns:
        if get_binary_val(row, col):
            is_asm = True
            break
                
    is_psycho = get_binary_val(row, 'Psychostimulant (y/n)')
            
    if condition == 'ASM':
        return is_asm and not is_psycho
    elif condition == 'no med':
        return not is_asm and not is_psycho
    elif condition == 'psychostimulant':
        return is_psycho and not is_asm
    elif condition == 'ASM+psychostimulant':
        return is_asm and is_psycho
    elif condition == 'ALL':
        return True
    return False

def check_age(row, condition):
    if condition == 'ALL':
        return True
        
    age_val = row.get('Age')
    if pd.isna(age_val):
        return False
        
    try:
        age = float(age_val)
    except:
        return False
        
    if condition == '0-4':
        return 0 <= age <= 4
    elif condition == '5-8':
        return 5 <= age <= 8
    elif condition == '9-12':
        return 9 <= age <= 12
    elif condition == '13-18':
        return 13 <= age <= 18
    return False

def generate_slurm_script(job, combo_name, subset_csv, output_dir, paths, log_dir):
    script_path = job['script_path']
    slurm_args = job.get('slurm_args', {})
    job_name = f"{job['name']}_{combo_name}"
    
    slurm_file = Path(output_dir) / "slurm_scripts" / f"{job_name}.slurm"
    
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/{job_name}_%j.out",
        f"#SBATCH --error={log_dir}/{job_name}_%j.err",
    ]
    
    for k, v in slurm_args.items():
        lines.append(f"#SBATCH --{k}={v}")
        
    lines.append("")
    lines.append("# Load required modules or environments here if needed")
    lines.append("# module load python/3.x")
    lines.append("")
    
    # Pass custom arguments depending on the job definition
    # Assuming standard is to pass --label-csv and --data-root
    command = f"python {script_path} --label-csv {subset_csv} --data-root {paths['data_root']}"
    lines.append(command)
    
    with open(slurm_file, 'w') as f:
        f.write("\n".join(lines) + "\n")
        
    return slurm_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments_config.yaml', help="Path to YAML config file")
    parser.add_argument('--dry-run', action='store_true', help="Generate subsets and slurm scripts but do not submit to cluster")
    args = parser.parse_args()

    try:
        config = parse_config(args.config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)
        
    paths = config.get('paths', {})
    if 'base_label_csv' not in paths:
        print("Error: 'base_label_csv' not defined in config paths.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(paths['base_label_csv'])
        print(f"Loaded {len(df)} total patients from {paths['base_label_csv']}")
    except Exception as e:
        print(f"Error reading CSV {paths['base_label_csv']}: {e}")
        sys.exit(1)
    
    groups = config.get('groups', {})
    sex_opts = groups.get('sex', ['ALL'])
    age_opts = groups.get('age', ['ALL'])
    com_opts = groups.get('comorbidities', ['ALL'])
    med_opts = groups.get('medication', ['ALL'])
    
    # Output dirs
    out_dir = Path(paths.get('output_dir', './orchestration_results'))
    cohorts_dir = out_dir / "cohorts"
    slurm_dir = out_dir / "slurm_scripts"
    logs_dir = out_dir / "logs"
    
    cohorts_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    active_jobs = [j for j in config.get('jobs', []) if j.get('run', True)]
    if not active_jobs:
        print("No active jobs defined in config. Exiting.")
        sys.exit(0)
        
    print(f"Active Jobs to generate: {[j['name'] for j in active_jobs]}")
    
    total_combinations = len(sex_opts) * len(age_opts) * len(com_opts) * len(med_opts)
    print(f"Generating combinations... (Total possible: {total_combinations})")
    
    generated_cohorts = 0
    jobs_submitted = 0
    
    for sex, age, com, med in itertools.product(sex_opts, age_opts, com_opts, med_opts):
        # 1. Filter DataFrame
        subset = df.copy()
        
        # Sex filter
        if sex != 'ALL':
            subset = subset[subset['Sex'].str.upper() == sex.upper()]
            
        # Age filter
        mask_age = subset.apply(lambda r: check_age(r, age), axis=1)
        subset = subset[mask_age]
        
        # Comorbidities filter
        mask_com = subset.apply(lambda r: check_comorbidity(r, com), axis=1)
        subset = subset[mask_com]
        
        # Medication filter
        mask_med = subset.apply(lambda r: check_medication(r, med), axis=1)
        subset = subset[mask_med]
        
        if len(subset) == 0:
            continue
            
        combo_name = f"sex-{sex}_age-{age}_com-{com.replace('&','_')}_med-{med.replace(' ','_')}"
        subset_csv = cohorts_dir / f"labels_{combo_name}.csv"
        subset.to_csv(subset_csv, index=False)
        generated_cohorts += 1
        
        print(f"Cohort '{combo_name}': {len(subset)} patients")
        
        # 2. Generate and Submit Slurm Jobs
        for job in active_jobs:
            slurm_file = generate_slurm_script(job, combo_name, subset_csv, out_dir, paths, logs_dir)
            jobs_submitted += 1
            if not args.dry_run:
                print(f"Submitting {slurm_file}...")
                os.system(f"sbatch {slurm_file}")
                
    print("\n" + "="*50)
    print("Orchestration Complete!")
    print(f"Valid Cohorts Found: {generated_cohorts}")
    print(f"Slurm Jobs Generated: {jobs_submitted}")
    if args.dry_run:
        print("\n[DRY RUN MODE]: No jobs were actually submitted to Slurm.")
        print(f"Check {slurm_dir} for the generated Slurm scripts.")
        print(f"Check {cohorts_dir} for the generated label CSVs.")

if __name__ == "__main__":
    main()
