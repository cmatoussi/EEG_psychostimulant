# coco_analysis — EEG ADHD/epilepsy analysis scripts

Analysis drivers for epilepsy decoding on the merged ADHD/drug-resistant EEG
dataset, built on top of `coco_pipe`. Every script imports the shared library
**`run_analysis.py`** (same-directory import via `sys.path`), so the scripts are
kept **flat** in this directory rather than nested in packages.

> **Source confound (read first).** Controls come only from the ADHD study and
> ~1/3 of epilepsy cases only from the drug-resistant study, so "epilepsy vs not"
> is partly separable by recording provenance. Most drivers therefore support
> three tracks — all-sources, `adhd_only_source`, and `without_source_column` —
> and the honest numbers are the within-adhd-source ones.

## Core library
| file | role |
|---|---|
| `run_analysis.py` | Shared library + `--config` entrypoint. Data loaders (`load_eeg_epochs`, `load_precomputed_embeddings`, `load_dim_reduction_data`, `load_handcrafted_data`), label normalization, CV/head building, LoRA fine-tune path, post-hoc metrics (subject aggregation, `balanced_accuracy_optimal`), model-specific preprocessing (biot montage, labram/signaljepa windowing). |

## Frozen-embedding decoding (pre-extracted FM embeddings + classical heads)
| file | what it does |
|---|---|
| `run_embedding_cohorts.py` | Main embedding sweep. Cohort groups (all/sex/age/comorbidity) × 8 FMs × EO/EC × heads; `--aggregation {averaged_predictions, averaged_epochs, both}`; `--source` / `--label-csv` for confound tracks. |
| `run_source_sanity.py` | Confound probe: predict the **source study** (adhd vs drug_resistant) from embeddings; scopes all-subjects vs epilepsy-only. |
| `run_drug_resistance.py` | Predict **drug resistance** (`asm_resistant`) from subject-averaged embeddings; both-sources vs adhd-only. |
| `run_ec_vs_eo.py` | EC-vs-EO condition comparison from embeddings. |

## Handcrafted-feature analysis
| file | what it does |
|---|---|
| `run_handcrafted_cohorts.py` | Sensor-descriptor decoding: multivariate + per-sensor→MNE topomap + SFS; RF-Gini/L1/permutation importance; cohort groups incl. a **`drug`** (medication) group; epoch/subject levels. |

## Dimensionality reduction (PCA / UMAP / Isomap × {2,3,5,10,15})
| file | input |
|---|---|
| `run_dimred_cohorts.py` | **Handcrafted** features; MAD row-rejection + robust-scale/clip; per-cohort HTML reports. Exposes the shared `reduce_and_report` helper. |
| `run_dimred_embeddings.py` | **FM embeddings** (epoch subsampled + averaged-epoch), per model, EO/EC. Reuses `run_dimred_cohorts` helpers. |

## Fine-tuning (LoRA)
| file | what it does |
|---|---|
| `tune_lora.py` | **Ray Tune** LoRA hyperparameter search (random+ASHA or Optuna) over r/α/dropout/lr, Schedule-Free AdamW. Needs `module load arrow` for pyarrow; Ray state on `$SLURM_TMPDIR`. |
| `run_finetune_tuned.py` | Fine-tune with the Ray-Tune-selected config (5-fold CV); `--strategy {ft_only,lp_ft}` × `--level {epoch,subject}`. Reuses `tune_lora._prep_model_data`. |
| `run_lr_finder.py` | LR range test (Leslie-Smith) for the AdamW+warmup/cosine setup. |

## Job generation & orchestration
| file | what it does |
|---|---|
| `generate_analysis_combinations.py` | Enumerate cohort × analysis combos → `analysis_combinations.{json,csv}`. |
| `generate_jobs.py` | Emit SLURM scripts + YAML configs from the combinations CSV. |
| `orchestrate_jobs.py` | Submit/track jobs from the combinations file. |

## Statistics
| file | what it does |
|---|---|
| `run_permutation_tests.py` | Subject-level paired permutation tests for model/strategy comparisons → `permute_testing.csv`. |

## Conventions
- **CV:** `StratifiedGroupKFold(5)`, grouped by subject (no subject leakage), class-balanced.
- **Metrics:** accuracy, balanced_accuracy, balanced_accuracy_optimal (Youden-J), roc_auc, f1.
- **Env:** venv at `/home/mat/ep`; `PYTHONPATH=/home/mat/projects/coco-pipe`; SLURM `--account=def-kjerbi`.
