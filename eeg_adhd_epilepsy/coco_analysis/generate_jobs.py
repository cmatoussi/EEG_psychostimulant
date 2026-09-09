#!/usr/bin/env python3
"""
Generate SLURM scripts and YAML configs from analysis_combinations.csv.

Usage:
    python generate_jobs.py                        # generate all combos
    python generate_jobs.py --ids 18 19 20         # specific combo IDs
    python generate_jobs.py --type fine_tune       # only one analysis type
    python generate_jobs.py --dry-run              # print without writing

Edit the constants in the CONFIGURATION section below to adapt to your setup.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — only edit this section
# ─────────────────────────────────────────────────────────────────────────────

VENV            = "/home/mat/ep/bin/activate"
RUN_SCRIPT      = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/run_analysis.py"
PREPROC_ROOT    = "/home/mat/scratch/preproc/"
RESULTS_ROOT    = Path("/home/mat/scratch/results")
COMBINATIONS_CSV = Path("/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/analysis_combinations.csv")

# Path to the handcrafted feature CSV (set when available)
HANDCRAFTED_CSV = "/home/mat/scratch/results/handcrafted_features.csv"

SLURM_DIR   = RESULTS_ROOT / "slurm_scripts"
CONFIG_DIR  = RESULTS_ROOT / "configs"
LOG_DIR     = RESULTS_ROOT / "logs"
SUMMARY_CSV = RESULTS_ROOT / "results_summary.csv"

# 19-channel 10-20 layout used in every EEG job
CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]

# Models that cannot run with 19-channel data — skipped silently
SKIP_MODELS = {"eegpt"}

# ── per-model resource requirements ──────────────────────────────────────────
# gpu:        MIG slice (2g.20gb = 20 GB, 3g.40gb = 40 GB)
# batch_size: training batch size
# time:       wall-clock limit override (default GPU_TIME)
# extra_env:  extra shell exports added to the SLURM script
MODEL_REQS = {
    "cbramod":    {"gpu": "2g.20gb", "batch_size": 32, "lr": 1e-3},
    "reve":       {"gpu": "2g.20gb", "batch_size": 32, "extra_env": "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\nexport HF_HOME=/home/mat/.cache/huggingface"},
    "biot":       {"gpu": "2g.20gb", "batch_size": 32, "lr": 1e-2},
    "labram":     {"gpu": "2g.20gb", "batch_size": 8,  "time": "6:00:00"},
    "luna":       {"gpu": "2g.20gb", "batch_size": 32},
    "signaljepa": {"gpu": "2g.20gb", "batch_size": 32, "lr": 1e-5, "lora_alpha": 4},
    "bendr":      {"gpu": "2g.20gb", "batch_size": 2},
}

# ── SLURM resource defaults ───────────────────────────────────────────────────
GPU_TIME     = "3:00:00"   # fine_tune / embedding
CPU_TIME     = "1:00:00"   # handcrafted / dim_reduction
GPU_CPUS     = 8
CPU_CPUS     = 4
GPU_MEM      = "64G"
CPU_MEM      = "16G"

# ─────────────────────────────────────────────────────────────────────────────

CONDITION_MAP = {
    "EO":  ["EO_baseline"],
    "EC":  ["EC_baseline"],
    "N/A": None,
}


def sanitize(value: str) -> str:
    """Make a CSV field safe for use in filenames and shell strings."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", value).strip("-")


def group_id(row: dict) -> str:
    sex = sanitize(row["sex"])
    age = sanitize(row["age"])
    com = sanitize(row["comorbidities"])
    med = sanitize(row["medication"])
    return f"sex-{sex}_age-{age}_com-{com}_med-{med}"


def label_csv(gid: str) -> str:
    return str(RESULTS_ROOT / "cohorts" / f"labels_{gid}.csv")


def job_name(row: dict) -> str:
    combo_id  = row["id"]
    atype     = row["analysis_type"]
    model     = sanitize(row["method_or_model"])
    condition = sanitize(row["condition"])
    gid       = group_id(row)
    if atype in ("embedding", "fm_extract", "fine_tune", "frozen_fine_tune"):
        return f"combo_{combo_id}_{atype}_{model}_cond-{condition}_{gid}"
    return f"combo_{combo_id}_{atype}_{model}_{gid}"


# ─────────────────────────────────────────────────────────────────────────────
# YAML builders
# ─────────────────────────────────────────────────────────────────────────────

def _ch_names_yaml() -> str:
    lines = "\n".join(f"  - {ch}" for ch in CH_NAMES)
    return lines


def yaml_fine_tune(row: dict, analysis_id: str, batch_size: int, lr: float | None = None, lora_alpha: int = 16) -> str:
    conditions = CONDITION_MAP[row["condition"]]
    cond_yaml  = "\n".join(f"  - {c}" for c in conditions) if conditions else ""
    cond_block = f"  conditions:\n{cond_yaml}" if cond_yaml else ""
    model      = row["method_or_model"]
    lr_line    = f"\n    lr: {lr}" if lr is not None else ""
    return f"""\
analyses:
- class_weight: balanced
  cv:
    n_splits: 5
    strategy: group_kfold
  enabled: true
  id: {analysis_id}
  lora:
    alpha: {lora_alpha}
    dropout: 0.05
    r: 8
    target_modules: all-linear
  metrics:
  - accuracy
  - roc_auc
  - balanced_accuracy
  - f1
  mode: fm_lora
  model_key: {model}
  train_mode: lora
  trainer:
    batch_size: {batch_size}
    early_stopping_patience: 5
    max_epochs: 15{lr_line}
paths:
  data_root: {PREPROC_ROOT}
signal:
  ch_names:
{_ch_names_yaml()}
{cond_block}
  epoch_desc: base
  sfreq: 200.0
"""


def yaml_frozen_fine_tune(row: dict, analysis_id: str, batch_size: int) -> str:
    conditions = CONDITION_MAP[row["condition"]]
    cond_yaml  = "\n".join(f"  - {c}" for c in conditions) if conditions else ""
    cond_block = f"  conditions:\n{cond_yaml}" if cond_yaml else ""
    model      = row["method_or_model"]
    return f"""\
analyses:
- class_weight: balanced
  cv:
    n_splits: 5
    strategy: group_kfold
  enabled: true
  id: {analysis_id}
  metrics:
  - accuracy
  - roc_auc
  - balanced_accuracy
  - f1
  mode: fm_frozen
  model_key: {model}
  train_mode: frozen
  trainer:
    batch_size: {batch_size}
    early_stopping_patience: 5
    max_epochs: 15
paths:
  data_root: {PREPROC_ROOT}
signal:
  ch_names:
{_ch_names_yaml()}
{cond_block}
  epoch_desc: base
  sfreq: 200.0
"""


def yaml_embedding(row: dict, analysis_id: str) -> str:
    conditions = CONDITION_MAP[row["condition"]]
    cond_yaml  = "\n".join(f"  - {c}" for c in conditions) if conditions else ""
    cond_block = f"  conditions:\n{cond_yaml}" if cond_yaml else ""
    model      = row["method_or_model"]
    return f"""\
analyses:
- cv:
    n_splits: 5
    strategy: group_kfold
  enabled: true
  id: {analysis_id}
  metrics:
  - accuracy
  - roc_auc
  - balanced_accuracy
  - f1
  mode: fm_embed
  model_key: {model}
  models:
    LogisticRegression:
      C: 1.0
      max_iter: 1000
      method: LogisticRegression
paths:
  data_root: {PREPROC_ROOT}
signal:
  ch_names:
{_ch_names_yaml()}
{cond_block}
  epoch_desc: base
  sfreq: 200.0
"""


def yaml_fm_extract(row: dict, analysis_id: str) -> str:
    """Option B: extract + save BIDS embeddings, then run classical CV heads."""
    conditions = CONDITION_MAP[row["condition"]]
    cond_yaml  = "\n".join(f"  - {c}" for c in conditions) if conditions else ""
    cond_block = f"  conditions:\n{cond_yaml}" if cond_yaml else ""
    model      = row["method_or_model"]
    return f"""\
analysis_level: subject_level
analyses:
- cv:
    n_splits: 5
    strategy: group_kfold
  enabled: true
  id: {analysis_id}
  metrics:
  - accuracy
  - roc_auc
  - balanced_accuracy
  - f1
  mode: fm_extract
  model_key: {model}
  models:
    logreg:
      method: LogisticRegression
      max_iter: 1000
      class_weight: balanced
    rf:
      method: RandomForestClassifier
      n_estimators: 300
      class_weight: balanced
    hgb:
      method: HistGradientBoostingClassifier
    dummy:
      method: DummyClassifier
      strategy: stratified
    svm:
      method: SVC
      kernel: rbf
      probability: true
      class_weight: balanced
paths:
  data_root: {PREPROC_ROOT}
signal:
  ch_names:
{_ch_names_yaml()}
{cond_block}
  epoch_desc: base
  sfreq: 200.0
"""


def yaml_handcrafted(row: dict, analysis_id: str) -> str:
    method = row["method_or_model"]
    unit   = row["handcrafted_unit"]
    return f"""\
analyses:
- cv:
    n_splits: 5
    strategy: group_kfold
  enabled: true
  id: {analysis_id}
  analysis_unit: {unit}
  spatial_units: all
  data_path: {HANDCRAFTED_CSV}
  metrics:
  - accuracy
  - roc_auc
  - balanced_accuracy
  - f1
  mode: handcrafted
  models:
    {method}:
      method: {method}
paths:
  data_root: {PREPROC_ROOT}
"""


def yaml_dim_reduction(row: dict, analysis_id: str) -> str:
    method = row["method_or_model"]
    return f"""\
analyses:
- enabled: true
  id: {analysis_id}
  mode: dim_reduction
  models:
    {method}:
      method: {method}
      n_components: 2
paths:
  data_root: {PREPROC_ROOT}
"""


# ─────────────────────────────────────────────────────────────────────────────
# SLURM script builder
# ─────────────────────────────────────────────────────────────────────────────

def slurm_script(
    row: dict,
    jname: str,
    config_path: str,
    analysis_id: str,
    output_dir: str,
    *,
    gpu: str | None = None,
    extra_env: str = "",
    time_override: str | None = None,
    result_name: str = "results",
) -> str:
    gid      = group_id(row)
    condition = sanitize(row["condition"])
    lcsv     = label_csv(gid)
    log_base = str(LOG_DIR / f"{jname}_%j")

    if gpu:
        gres  = f"nvidia_h100_80gb_hbm3_{gpu}:1"
        time  = time_override or GPU_TIME
        cpus  = GPU_CPUS
        mem   = GPU_MEM
        gpu_line = f"#SBATCH --gres=gpu:{gres}"
    else:
        time  = CPU_TIME
        cpus  = CPU_CPUS
        mem   = CPU_MEM
        gpu_line = ""

    extra_env_line = f"\n{extra_env}" if extra_env else ""

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --output={log_base}.out
#SBATCH --error={log_base}.err
#SBATCH --cpus-per-task={cpus}
{gpu_line}
#SBATCH --mem={mem}
#SBATCH --time={time}

source {VENV}
export PYTORCH_ALLOC_CONF=expandable_segments:True{extra_env_line}
python {RUN_SCRIPT} \\
  --config {config_path} \\
  --analysis-id {analysis_id} \\
  --group-id '{gid}_cond-{condition}' \\
  --label-csv {lcsv} \\
  --output-dir {output_dir} \\
  --result-name {result_name} \\
  --summary-csv {SUMMARY_CSV}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(row: dict, dry_run: bool = False) -> str | None:
    combo_id  = row["id"]
    atype     = row["analysis_type"]
    model     = row["method_or_model"]
    condition = row["condition"]
    gid       = group_id(row)
    jname     = job_name(row)
    safe_model = sanitize(model)
    safe_cond  = sanitize(condition)

    if atype in ("embedding", "fm_extract", "fine_tune", "frozen_fine_tune") and model in SKIP_MODELS:
        print(f"  [skip] combo {combo_id} — {model} incompatible with 19-channel data")
        return None

    reqs          = MODEL_REQS.get(model, {})
    gpu           = reqs.get("gpu") if atype in ("embedding", "fm_extract", "fine_tune", "frozen_fine_tune") else None
    batch         = reqs.get("batch_size", 32)
    lr            = reqs.get("lr")
    lora_alpha    = reqs.get("lora_alpha", 16)
    extra_env     = reqs.get("extra_env", "")
    time_override = reqs.get("time") if atype in ("embedding", "fm_extract", "fine_tune", "frozen_fine_tune") else None

    # ── analysis id & output dir ──────────────────────────────────────────────
    if atype == "fine_tune":
        analysis_id = f"fm_lora_{safe_model}"
        output_dir  = str(RESULTS_ROOT / "fine_tune")
        result_name = f"results_{safe_model}_{safe_cond}"
    elif atype == "frozen_fine_tune":
        analysis_id = f"fm_frozen_{safe_model}"
        output_dir  = str(RESULTS_ROOT / "fine_tune")
        result_name = f"results_{safe_model}_{safe_cond}_frozen"
    elif atype == "embedding":
        analysis_id = f"fm_embed_{safe_model}"
        output_dir  = str(RESULTS_ROOT / "embedding" / safe_model / f"cond-{safe_cond}" / gid)
        result_name = "results"
    elif atype == "fm_extract":
        analysis_id = f"fm_extract_{safe_model}"
        # run_fm_extract treats output_dir as the embeddings ROOT; it appends
        # <model_key>/sub-XXXX/... and writes embeddings_performance_summary.csv here.
        output_dir  = str(RESULTS_ROOT / "embeddings")
        result_name = f"results_{safe_model}_{safe_cond}"
    elif atype == "handcrafted":
        unit        = row["handcrafted_unit"]
        analysis_id = f"handcrafted_{safe_model}_{unit}"
        output_dir  = str(RESULTS_ROOT / "handcrafted" / safe_model / unit / gid)
        result_name = "results"
    elif atype == "dim_reduction":
        analysis_id = f"dim_reduction_{safe_model}"
        output_dir  = str(RESULTS_ROOT / "dim_reduction" / safe_model / gid)
        result_name = "results"
    else:
        print(f"  [skip] combo {combo_id} — unknown analysis_type '{atype}'")
        return None

    config_path = str(CONFIG_DIR / f"combo_{combo_id}.yaml")

    # ── build YAML ────────────────────────────────────────────────────────────
    if atype == "fine_tune":
        yaml_content = yaml_fine_tune(row, analysis_id, batch, lr=lr, lora_alpha=lora_alpha)
    elif atype == "frozen_fine_tune":
        yaml_content = yaml_frozen_fine_tune(row, analysis_id, batch)
    elif atype == "embedding":
        yaml_content = yaml_embedding(row, analysis_id)
    elif atype == "fm_extract":
        yaml_content = yaml_fm_extract(row, analysis_id)
    elif atype == "handcrafted":
        yaml_content = yaml_handcrafted(row, analysis_id)
    else:
        yaml_content = yaml_dim_reduction(row, analysis_id)

    # ── build SLURM script ────────────────────────────────────────────────────
    slurm_content = slurm_script(
        row, jname, config_path, analysis_id, output_dir,
        gpu=gpu, extra_env=extra_env, time_override=time_override,
        result_name=result_name,
    )

    slurm_path = SLURM_DIR / f"{jname}.slurm"

    if dry_run:
        print(f"\n{'='*60}")
        print(f"combo {combo_id}: {atype} / {model} / cond={condition} / {gid}")
        print(f"  slurm → {slurm_path}")
        print(f"  config → {config_path}")
        print(f"  output → {output_dir}")
        if gpu:
            print(f"  gpu={gpu}  batch={batch}  time={GPU_TIME}  cpus={GPU_CPUS}")
        else:
            print(f"  no gpu  time={CPU_TIME}  cpus={CPU_CPUS}")
        return jname

    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    slurm_path.write_text(slurm_content)
    Path(config_path).write_text(yaml_content)
    print(f"  [ok] combo {combo_id}: {jname}")
    return jname


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM jobs from analysis_combinations.csv")
    parser.add_argument("--ids",     nargs="+", type=int, help="Only generate these combo IDs")
    parser.add_argument("--type",    choices=["fine_tune", "embedding", "fm_extract", "handcrafted", "dim_reduction"],
                        help="Only generate this analysis type")
    parser.add_argument("--model",   help="Only generate jobs for this model")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated without writing files")
    args = parser.parse_args()

    with open(COMBINATIONS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.ids:
        rows = [r for r in rows if int(r["id"]) in args.ids]
    if args.type:
        rows = [r for r in rows if r["analysis_type"] == args.type]
    if args.model:
        rows = [r for r in rows if r["method_or_model"] == args.model]

    print(f"Generating {len(rows)} job(s) ...")
    generated = 0
    for row in rows:
        result = generate(row, dry_run=args.dry_run)
        if result:
            generated += 1

    print(f"\nDone: {generated} job(s) {'would be ' if args.dry_run else ''}generated.")


if __name__ == "__main__":
    main()
