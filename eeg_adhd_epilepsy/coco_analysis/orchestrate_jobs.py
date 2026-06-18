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
    elif condition in ['ASD & ADHD', 'ASD&ADHD']:
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
    elif condition in ['ASM & psychostimulant', 'ASM+psychostimulant']:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments_config.yaml', help="Path to YAML config file")
    parser.add_argument('--dry-run', action='store_true', help="Generate subsets and slurm scripts but do not submit to cluster")
    parser.add_argument('--combinations-file', default='', help="Path to JSON combinations file")
    parser.add_argument('--run-combo-id', type=int, default=None, help="Run a specific combination by ID")
    parser.add_argument('--run-combo-range', default=None, help="Run a range of combination IDs (e.g. 0-10)")
    args = parser.parse_args()

    # Locate config path
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to the script's directory
        config_path = Path(__file__).parent / args.config
        
    if not config_path.exists():
        print(f"Error: Config path {args.config} not found.")
        sys.exit(1)

    try:
        config = parse_config(config_path)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)
        
    paths = config.get('paths', {})
    if 'label_csv' not in paths:
        print("Error: 'label_csv' not defined in config paths.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(paths['label_csv'])
        print(f"Loaded {len(df)} total patients from {paths['label_csv']}")
    except Exception as e:
        print(f"Error reading CSV {paths['label_csv']}: {e}")
        sys.exit(1)
        
    exclude_subjects = config.get('exclude_subjects')
    if exclude_subjects:
        initial_len = len(df)
        exclude_mask = pd.Series(False, index=df.index)
        for r in exclude_subjects:
            if isinstance(r, (list, tuple)) and len(r) == 2:
                start, end = r
                study_ids = pd.to_numeric(df['Study ID'], errors='coerce')
                exclude_mask = exclude_mask | ((study_ids >= start) & (study_ids <= end))
        df = df[~exclude_mask].copy()
        print(f"Excluded {initial_len - len(df)} patients based on 'exclude_subjects' ranges. Remaining: {len(df)}")
    
    # Output dirs
    out_dir = Path(paths.get('output_dir', './results'))
    cohorts_dir = out_dir / "cohorts"
    slurm_dir = out_dir / "slurm_scripts"
    logs_dir = out_dir / "logs"
    
    cohorts_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.run_combo_id is not None or args.run_combo_range is not None:
        import json
        combos_path = Path(args.combinations_file) if args.combinations_file else Path()
        if not args.combinations_file or not combos_path.exists():
            combos_path = Path(__file__).parent / "analysis_combinations.json"
            
        try:
            with open(combos_path, 'r') as f:
                combos = json.load(f)
        except Exception as e:
            print(f"Error loading combinations file {combos_path}: {e}")
            sys.exit(1)
            
        selected_ids = []
        if args.run_combo_id is not None:
            selected_ids.append(args.run_combo_id)
        if args.run_combo_range is not None:
            parts = args.run_combo_range.split('-')
            start, end = int(parts[0]), int(parts[1])
            selected_ids.extend(range(start, end + 1))
            
        generated_cohorts = 0
        jobs_submitted = 0
        
        for cid in selected_ids:
            combo = next((c for c in combos if c['id'] == cid), None)
            if not combo:
                print(f"Warning: Combination ID {cid} not found.")
                continue
                
            cohort = combo['cohort']
            sex = cohort['sex']
            age = cohort['age']
            com = cohort['comorbidities']
            med = cohort['medication']
            
            subset = df.copy()
            if sex != 'ALL':
                subset = subset[subset['Sex'].str.upper() == sex.upper()]
            subset = subset[subset.apply(lambda r: check_age(r, age), axis=1)]
            subset = subset[subset.apply(lambda r: check_comorbidity(r, com), axis=1)]
            subset = subset[subset.apply(lambda r: check_medication(r, med), axis=1)]
            
            if len(subset) == 0:
                print(f"Combo {cid}: Cohort is empty. Skipping.")
                continue
                
            sex_clean = sex.replace(" ", "")
            age_clean = age.replace(" ", "")
            com_clean = com.replace(" & ", "-").replace(" ", "")
            med_clean = med.replace(" & ", "-").replace(" ", "")
            combo_name = f"sex-{sex_clean}_age-{age_clean}_com-{com_clean}_med-{med_clean}"
            
            subset_csv = cohorts_dir / f"labels_{combo_name}.csv"
            subset.to_csv(subset_csv, index=False)
            generated_cohorts += 1
            
            print(f"Combo {cid}: Cohort '{combo_name}': {len(subset)} patients")
            
            analysis = combo['analysis']
            atype = analysis['type']
            
            cmd_args = ""
            script_path = ""
            slurm_params = {
                "cpus-per-task": "16",
                "mem": "64G",
                "time": "12:00:00"
            }
            
            if atype == "dim_reduction":
                method = analysis['method']
                temp_config = {
                    "data_path": "/home/mat/scratch/signal_features/descriptors/combined/pooled_subject_features.csv",
                    "target_columns": "Epilepsy",
                    "analyses": [
                        {
                            "id": f"dim_reduction_{method.lower()}",
                            "enabled": True,
                            "mode": "dim_reduction",
                            "models": {
                                method.lower(): {
                                    "method": method,
                                    "n_components": 2
                                }
                            }
                        }
                    ]
                }
                temp_config_path = out_dir / f"temp_config_{cid}.yaml"
                with open(temp_config_path, 'w') as tf:
                    yaml.dump(temp_config, tf)
                    
                script_path = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/run_analysis.py"
                cmd_args = f"--config {temp_config_path} --analysis-id dim_reduction_{method.lower()} --group-id '{combo_name}' --output-dir {out_dir}/dimred/{method.lower()}/{combo_name}"
                
            elif atype == "handcrafted":
                unit = analysis['unit']
                reg_head = analysis['regression_head']
                if unit == "sensor":
                    feat_file = "/home/mat/scratch/signal_features/descriptors/combined/sensor_subject_features.csv"
                else:
                    feat_file = "/home/mat/scratch/signal_features/descriptors/combined/pooled_subject_features.csv"
                    
                temp_config = {
                    "data_path": feat_file,
                    "target_columns": "Epilepsy",
                    "analyses": [
                        {
                            "id": f"handcrafted_{unit}_{reg_head.lower()}",
                            "enabled": True,
                            "mode": "handcrafted",
                            "analysis_unit": "all" if unit != "sensor" else "sensor",
                            "spatial_units": "all",
                            "feature_names": "all",
                            "representation": "subject",
                            "balance": {
                                "strategy": "undersample"
                            },
                            "models": {
                                reg_head.lower(): {
                                    "method": reg_head,
                                    "class_weight": "balanced"
                                }
                            },
                            "metrics": ["accuracy", "roc_auc", "balanced_accuracy", "f1"],
                            "cv": {
                                "strategy": "group_kfold",
                                "n_splits": 5
                            }
                        }
                    ]
                }
                temp_config_path = out_dir / f"temp_config_{cid}.yaml"
                with open(temp_config_path, 'w') as tf:
                    yaml.dump(temp_config, tf)
                    
                script_path = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/run_analysis.py"
                cmd_args = f"--config {temp_config_path} --analysis-id handcrafted_{unit}_{reg_head.lower()} --group-id '{combo_name}' --output-dir {out_dir}/handcrafted/{unit}_{reg_head.lower()}/{combo_name}"
                
            elif atype == "embedding":
                model = analysis['model']
                if model == "reve":
                    script_path = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve/reve_extract.py"
                    if not Path(script_path).exists():
                        script_path = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve/embed/extract.py"
                elif model == "cbramod":
                    script_path = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/cbramod/make_embeddings.py"
                else:
                    script_path = f"/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/{model}/extract.py"
                    
                cmd_args = f"--label-csv {subset_csv} --data-root {paths['data_root']}"
                slurm_params["gres"] = "gpu:1"
                
            elif atype == "fine_tune":
                model = analysis['model']
                if model == "reve":
                    script_path = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve/fine_tune/reve_finetune_large.py"
                elif model == "cbramod":
                    script_path = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/cbramod/cbramod_fine_lp_ft.py"
                else:
                    script_path = f"/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/{model}/fine_tune.py"
                    
                cmd_args = f"--label-csv {subset_csv} --data-root {paths['data_root']}"
                slurm_params["gres"] = "gpu:2"
                
            job_name = f"combo_{cid}_{atype}_{combo_name}"
            slurm_file = slurm_dir / f"{job_name}.slurm"
            
            lines = [
                "#!/bin/bash",
                f"#SBATCH --job-name={job_name}",
                f"#SBATCH --output={logs_dir}/{job_name}_%j.out",
                f"#SBATCH --error={logs_dir}/{job_name}_%j.err",
            ]
            for k, v in slurm_params.items():
                lines.append(f"#SBATCH --{k}={v}")
                
            lines.append("")
            lines.append(f"python {script_path} {cmd_args}")
            
            with open(slurm_file, 'w') as f:
                f.write("\n".join(lines) + "\n")
                
            print(f"Generated SLURM script: {slurm_file}")
            jobs_submitted += 1
            if not args.dry_run:
                print(f"Submitting {slurm_file}...")
                os.system(f"sbatch {slurm_file}")
                
        print("\n" + "="*50)
        print("Orchestration Complete for requested combinations!")
        print(f"Valid Cohorts Found: {generated_cohorts}")
        print(f"Slurm Jobs Generated: {jobs_submitted}")
        
    else:
        # Fallback to original full Cartesian sweep if combo args are not supplied
        groups_opts = config.get('groups', {})
        sex_opts = groups_opts.get('sex', ['ALL'])
        age_opts = groups_opts.get('age', ['ALL'])
        com_opts = groups_opts.get('comorbidities', ['ALL'])
        med_opts = groups_opts.get('medication', ['ALL'])
        
        print("Generating all combinations from group dimensions...")
        
        generated_cohorts = 0
        jobs_submitted = 0
        
        for sex, age, com, med in itertools.product(sex_opts, age_opts, com_opts, med_opts):
            subset = df.copy()
            if sex != 'ALL':
                subset = subset[subset['Sex'].str.upper() == sex.upper()]
            mask_age = subset.apply(lambda r: check_age(r, age), axis=1)
            subset = subset[mask_age]
            mask_com = subset.apply(lambda r: check_comorbidity(r, com), axis=1)
            subset = subset[mask_com]
            mask_med = subset.apply(lambda r: check_medication(r, med), axis=1)
            subset = subset[mask_med]
            
            if len(subset) == 0:
                continue
                
            sex_clean = sex.replace(" ", "")
            age_clean = age.replace(" ", "")
            com_clean = com.replace(" & ", "-").replace(" ", "")
            med_clean = med.replace(" & ", "-").replace(" ", "")
            combo_name = f"sex-{sex_clean}_age-{age_clean}_com-{com_clean}_med-{med_clean}"
            
            subset_csv = cohorts_dir / f"labels_{combo_name}.csv"
            subset.to_csv(subset_csv, index=False)
            generated_cohorts += 1
            
            print(f"Cohort '{combo_name}': {len(subset)} patients")
            
            # Since there is no active jobs definition in this experiments_config, 
            # we just print the cohort generated.
            
        print("\n" + "="*50)
        print("Orchestration Complete!")
        print(f"Valid Cohorts Found: {generated_cohorts}")
        if args.dry_run:
            print("\n[DRY RUN MODE]")
            print(f"Check {cohorts_dir} for the generated label CSVs.")

if __name__ == "__main__":
    main()
