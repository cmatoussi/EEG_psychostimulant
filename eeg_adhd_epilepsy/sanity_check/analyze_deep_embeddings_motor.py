#!/usr/bin/env python3
"""
End-to-end Dimensionality Reduction Analysis for Deep Embeddings (ReVe/CBraMod).

Workflow:
1. Load Deep Embeddings (.npy) using temp_loader.py (auto-aligned to conditions).
2. Stack/Flatten data based on strategy (epoch_flat vs subject_flat).
3. Apply dimensionality reduction using coco-pipe (PCA, UMAP, PHATE, ISOMAP).
4. Score embedding separation using Logistic Regression (Grouped CV).
5. Generate interactive HTML reports.
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to sys.path to allow absolute imports when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from eeg_adhd_epilepsy.utils.config import results_dir
from coco_pipe.report.core import Report, Section, TableElement
from coco_pipe.viz import dim_reduction as viz

# Reuse core functions from existing analysis
from eeg_adhd_epilepsy.analysis.dimensionality_reduction import (
    compute_embeddings,
    compute_embedding_separation_score,
    build_condition_ranking_rows,
    add_reducer_report_elements,
    ALL_REDUCERS,
)
from eeg_adhd_epilepsy.analysis.utils import (
    apply_representation,
    build_meta_dict,
)
from eeg_adhd_epilepsy.io.patients import (
    clean_patients_df,
    load_raw_patients_df,
)

# New Deep Embedding loader
from eeg_adhd_epilepsy.dl.temp_loader import load_temp_dl_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_condition_section(
    condition: str,
    embeddings: Dict[str, Dict],
    labels: Optional[np.ndarray],
    meta_dict: Optional[Dict[str, np.ndarray]],
    ranking_rows: List[Dict[str, object]],
    loaded_subjects: int,
    loaded_epochs: int,
    samples_used: int,
    interactive: bool = True
) -> Section:
    """Create a report section for a specific condition containing all reducers."""
    section = Section(title=condition, icon="🧠")
    section.add_markdown(
        f"Condition-specific embeddings for {condition}. "
        f"Loaded subjects: {loaded_subjects}. "
        f"Loaded epochs: {loaded_epochs}. "
        f"Samples used after representation: {samples_used}."
    )
    
    ranking_df = pd.DataFrame(ranking_rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            by=["dimension", "cv_balanced_accuracy"],
            ascending=[True, False],
        ).reset_index(drop=True)
        section.add_element(
            TableElement(
                ranking_df[["reducer", "dimension", "cv_balanced_accuracy"]].round(4),
                title="Reducer Ranking",
            )
        )
    
    for name, results in embeddings.items():
        if results.get("embedding_2d") is not None or results.get("embedding_3d") is not None:
            add_reducer_report_elements(
                section, condition, name, results, labels, meta_dict, interactive
            )
    return section

def main():
    parser = argparse.ArgumentParser(description="Analyze Deep Embeddings (ReVe/CBraMod)")
    parser.add_argument("--embeddings_dir", type=str, default="/home/mat/scratch/motor_extracted_embeddings/", help="Path to extracted .npy files")
    parser.add_argument("--segments_root", type=str, required=False, default=None, help="Path to preproc directory (for segments.csv fallback)")
    parser.add_argument("--metadata", type=str, default=None, help="Path to clinical CSV")
    parser.add_argument("--model", type=str, choices=["reve", "cbramod"], required=True, help="Which model was used")
    parser.add_argument("--reve_size", type=str, choices=["base", "large"], default=None, help="REVE model size (base or large)")
    parser.add_argument("--pooling", type=str, choices=["with", "without"], default=None, help="Whether embeddings are pooled (with or without)")
    parser.add_argument("--desc", type=str, default="baseline", help="EEG stage descriptor (e.g., baseline, base)")
    parser.add_argument("--dataset_name", type=str, default="sanity_check", help="Name for the output report folder")
    parser.add_argument("--target_col", type=str, default="EO_EC", help="Clinical column to predict")
    parser.add_argument("--representation", type=str, default="epoch_flat", choices=["epoch_flat", "subject_flat"], help="How to aggregate segments")
    parser.add_argument("--interactive", action="store_true", default=True, help="Generate interactive plots")
    parser.add_argument("--reducers", nargs="+", default=ALL_REDUCERS, help="List of reducers to run")
    parser.add_argument("--conditions", nargs="+", default=None, help="Specific conditions to keep (e.g. eyes_open)")
    parser.add_argument("--min_overlap", type=float, default=0.8, help="Min overlap fraction for condition alignment")
    parser.add_argument("--all_layers", type=int, choices=[0, 1], default=0, help="Analyze all-layer embeddings (1) or base embeddings (0)")
    args = parser.parse_args()

    # 1. Setup outputs
    out_dir_name = args.dataset_name
    if args.model == "reve":
        if args.reve_size:
            out_dir_name += f"_{args.reve_size}"
        if args.pooling:
            out_dir_name += f"_{args.pooling}_pooling"
            
    out_dir = Path("/home/mat/projects/EEG_psychostimulant/data/results/sanity_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = Report(title=f"Deep Embeddings Analysis: {out_dir_name}")

    # 2. Metadata Processing
    raw_meta_df = load_raw_patients_df(Path(args.metadata)) if args.metadata else None
    if raw_meta_df is not None:
        clean_meta_df, _ = clean_patients_df(raw_meta_df)
    else:
        clean_meta_df = None

    # 3. Load Embeddings
    logger.info(f"Loading {args.model} embeddings from {args.embeddings_dir}")
    try:
        load_target_col = "condition" if args.target_col == "EO_EC" else args.target_col
        container = load_temp_dl_data(
            embeddings_root=Path(args.embeddings_dir),
            segments_root=Path(args.segments_root) if args.segments_root else None,
            model=args.model,
            desc=args.desc,
            metadata_df=clean_meta_df,
            subject_col="Study ID",
            target_col=load_target_col,
            conditions=args.conditions,
            min_overlap_fraction=args.min_overlap,
            drop_unassigned=True,
            model_size=args.reve_size,
            pooling=args.pooling,
            all_layers=(args.all_layers == 1)
        )
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return

    # Dynamically find conditions
    if "condition" not in container.coords:
        logger.error("No 'condition' coordinate found in merged data! Check load_temp_dl_data.")
        return
        
    if args.target_col == "EO_EC":
        loaded_conditions = ["EO_EC"]
    else:
        loaded_conditions = np.unique(container.coords["condition"])
        loaded_conditions = [c for c in loaded_conditions if pd.notna(c) and str(c).lower() != "nan" and c is not None]
    
    if not loaded_conditions:
        logger.warning(f"No valid conditions found in embeddings. Aborting.")
        return

    all_ranking_rows = []
    condition_summary_rows = []

    # 4. Iterate per condition
    for condition in loaded_conditions:
        logger.info(f"--- Analyzing condition: {condition} ---")
        
        # Filter DataContainer for this condition
        if args.target_col == "EO_EC":
            # 1. Identify rows for both conditions
            cond_mask = np.isin(np.asarray(container.coords["condition"]), ["eyes_open", "eyes_closed"])
            temp_container = container.isel(obs=cond_mask)
            
            # 2. Find subjects having BOTH conditions
            study_ids = np.asarray(temp_container.coords.get("Study ID", temp_container.ids)).astype(str)
            conditions_arr = np.asarray(temp_container.coords["condition"]).astype(str)
            
            subject_cond_map = {}
            for s, c in zip(study_ids, conditions_arr):
                if s not in subject_cond_map:
                    subject_cond_map[s] = set()
                subject_cond_map[s].add(c)
            
            subjects_with_both = {s for s, cs in subject_cond_map.items() if "eyes_open" in cs and "eyes_closed" in cs}
            
            # 3. Final mask: rows in ["eyes_open", "eyes_closed"] AND subject in subjects_with_both
            final_mask = np.array([
                (c in ["eyes_open", "eyes_closed"]) and (s in subjects_with_both)
                for s, c in zip(np.asarray(container.coords["Study ID"]).astype(str), np.asarray(container.coords["condition"]).astype(str))
            ])
            
            cond_container = container.isel(obs=final_mask)
            
            n_dropped = len(set(study_ids)) - len(subjects_with_both)
            if n_dropped > 0:
                logger.info(f"EO_EC analysis: dropped {n_dropped} subjects who did not have both EO and EC.")
        else:
            cond_mask = np.asarray(container.coords["condition"]) == condition
            cond_container = container.isel(obs=cond_mask)
        
        if len(cond_container.ids) == 0:
            logger.warning(f"No samples left for {condition}, skipping.")
            continue
            
        # Apply Representation (Segment Flat vs Subject Mean/Flat)
        logger.info(f"Applying representation: {args.representation}")
        try:
            if args.target_col == "EO_EC":
                # Create composite column so aggregation keeps EO and EC separate
                study_ids = np.asarray(cond_container.coords.get("Study ID", cond_container.ids)).astype(str)
                conds = np.asarray(cond_container.coords["condition"]).astype(str)
                cond_container.coords["Study ID_condition"] = np.array([f"{s}_{c}" for s, c in zip(study_ids, conds)])
                rep_container = apply_representation(
                    cond_container, 
                    representation=args.representation, 
                    study_id_col="Study ID_condition"
                )
            else:
                rep_container = apply_representation(
                    cond_container, 
                    representation=args.representation, 
                    study_id_col="Study ID"
                )
        except Exception as e:
            logger.error(f"Failed to apply representation {args.representation}: {e}")
            continue

        if args.target_col == "EO_EC":
            # Reconstruct condition and labels
            composite_keys = np.asarray(
                rep_container.coords.get("Study ID_condition", rep_container.ids)
            ).astype(str)
            reconstructed_conditions = np.array([k.split("_", 1)[1] for k in composite_keys])
            rep_container.coords["condition"] = reconstructed_conditions
            
            raw_labels = (np.array(reconstructed_conditions) == "eyes_open").astype(int)
            raw_groups = np.array([k.split("_", 1)[0] for k in composite_keys])
        else:
            raw_labels = np.asarray(rep_container.coords.get(args.target_col, [])).ravel()
            raw_groups = np.asarray(rep_container.coords.get("Study ID", rep_container.ids)).ravel()
        
        # Drop samples where the target condition is missing (NaN / 'nan')
        valid_mask = [i for i, v in enumerate(raw_labels) if pd.notna(v) and str(v).lower() != 'nan']
        
        if len(valid_mask) < 5:
            logger.warning(f"Too few valid labels ({len(valid_mask)}) to compute meaningful embeddings for {condition}.")
            continue
            
        rep_container = rep_container.isel(obs=valid_mask)
        X = rep_container.X
        
        if args.target_col == "EO_EC":
            cond_labels = np.asarray(rep_container.coords["condition"]).astype(str)
            labels = (np.array(cond_labels) == "eyes_open").astype(int)
            composite_keys_filtered = np.asarray(
                rep_container.coords.get("Study ID_condition", rep_container.ids)
            ).astype(str)
            groups = np.array([k.split("_", 1)[0] for k in composite_keys_filtered])
        else:
            labels = np.asarray(rep_container.coords.get(args.target_col, [])).ravel()
            groups = np.asarray(rep_container.coords.get("Study ID", rep_container.ids)).ravel()
        
        if pd.Index(labels).nunique() < 2:
            logger.warning(f"Condition '{condition}' has only 1 distinct label left after dropping NaNs. Skipping separation scoring.")
            continue

        logger.info(f"Condition '{condition}' final shape (after dropping NaNs): {X.shape}")

        # Compute Reductions
        embeddings_res = compute_embeddings(X, args.reducers)

        # Update scoring metadata
        loaded_subjects = pd.Index(np.asarray(cond_container.coords.get("Study ID", []))).nunique()
        loaded_epochs = len(cond_container.ids)
        samples_used = X.shape[0]

        ranking_rows = build_condition_ranking_rows(
            condition=condition,
            embeddings=embeddings_res,
            labels=labels,
            groups=groups,
            loaded_subjects=loaded_subjects,
            loaded_epochs=loaded_epochs,
            samples_used=samples_used,
        )
        all_ranking_rows.extend(ranking_rows)

        best_2d_score = None
        best_2d_reducer = None
        best_3d_score = None
        best_3d_reducer = None
        for r_row in ranking_rows:
            if "2D" in r_row.get("dimension", ""):
                if pd.notna(r_row.get("cv_balanced_accuracy")):
                    if best_2d_score is None or r_row["cv_balanced_accuracy"] > best_2d_score:
                        best_2d_score = r_row["cv_balanced_accuracy"]
                        best_2d_reducer = r_row["reducer"]
            elif "3D" in r_row.get("dimension", ""):
                if pd.notna(r_row.get("cv_balanced_accuracy")):
                    if best_3d_score is None or r_row["cv_balanced_accuracy"] > best_3d_score:
                        best_3d_score = r_row["cv_balanced_accuracy"]
                        best_3d_reducer = r_row["reducer"]

        condition_summary_rows.append({
            "condition": condition,
            "status": "Computed",
            "loaded_subjects": loaded_subjects,
            "loaded_epochs": loaded_epochs,
            "samples_used": samples_used,
            "best_2d_reducer": best_2d_reducer,
            "best_2d_score": best_2d_score,
            "best_3d_reducer": best_3d_reducer,
            "best_3d_score": best_3d_score,
        })

        # Build Interactive Metadata Dictionary
        meta_dict = build_meta_dict(rep_container)

        # Add section to report
        condition_section = create_condition_section(
            condition=condition,
            embeddings=embeddings_res,
            labels=labels,
            meta_dict=meta_dict,
            ranking_rows=ranking_rows,
            loaded_subjects=loaded_subjects,
            loaded_epochs=loaded_epochs,
            samples_used=samples_used,
            interactive=args.interactive
        )
        report.add_section(condition_section)

    # 5. Build Final Report Summary
    logger.info("Building final summary section and exporting report...")
    
    overview_df = pd.DataFrame(
        [
            {
                "dataset_name": args.dataset_name,
                "representation": args.representation,
                "target_col": args.target_col,
                "reducers": ", ".join(args.reducers),
                "conditions_tested": ", ".join(args.conditions) if args.conditions else "All available",
            }
        ]
    )
    overview_sec = Section("Overview", icon="📋")
    overview_sec.add_markdown(
        "This report ranks conditions by how well each reducer separates the target labels "
        "in 2D and 3D embeddings."
    )
    overview_sec.add_element(TableElement(overview_df, title="Run Configuration"))
    report.children.insert(0, overview_sec)

    condition_summary_df = pd.DataFrame(condition_summary_rows)
    summary_sec = Section("Condition Summary", icon="🧾")
    summary_sec.add_markdown(
        "Per-condition data availability and best reducer scores."
    )
    if not condition_summary_df.empty:
        summary_sec.add_element(
            TableElement(
                condition_summary_df[
                    [
                        "condition",
                        "status",
                        "loaded_subjects",
                        "loaded_epochs",
                        "samples_used",
                        "best_2d_reducer",
                        "best_2d_score",
                        "best_3d_reducer",
                        "best_3d_score",
                    ]
                ].round(4),
                title="Condition Summary",
            )
        )
    report.children.insert(1, summary_sec)

    ranking_df = pd.DataFrame(all_ranking_rows)
    if not ranking_df.empty and "cv_balanced_accuracy" in ranking_df.columns:
        ranking_sec = Section("Condition Ranking", icon="🏁")
        ranking_sec.add_markdown(
            "Cross-validated balanced accuracy on the low-dimensional embeddings. "
            "Higher is better."
        )
        ranking_sec.add_element(
            TableElement(
                ranking_df.sort_values(
                    by="cv_balanced_accuracy",
                    ascending=False,
                    na_position="last",
                ).reset_index(drop=True).round(4),
                title="Reducer x Condition Ranking",
            )
        )
        report.children.insert(2, ranking_sec)

    try:
        # Custom report naming (e.g., test_cbramod_last_layer_subject_flat_EO_EC_report.html)
        target_suffix = f"_{args.target_col}" if args.target_col else ""
        if args.conditions and len(args.conditions) > 1:
            # If specifically EO/EC
            if ("EO_baseline" in args.conditions and "EC_baseline" in args.conditions) or \
               ("eyes_open" in args.conditions and "eyes_closed" in args.conditions):
                target_suffix = "_EO_EC"
        
        report_filename = "report.html"
        # Standard report naming for motor sanity check
        if args.model == "reve":
            layer_suffix = "_all_layers" if args.all_layers else ""
            report_filename = f"motor_reve_{args.reve_size}_{args.pooling}_pooling{layer_suffix}_{args.representation}{target_suffix}_report.html"
        elif args.model == "cbramod":
            layer_suffix = "_all_layers" if args.all_layers else "_last_layer"
            report_filename = f"motor_cbramod{layer_suffix}_{args.representation}{target_suffix}_report.html"
        else: # Default naming if model is not reve or cbramod
            report_filename = f"motor_{args.model}_{args.representation}{target_suffix}_report.html"

        html_path = out_dir / report_filename
        report.save(html_path)
        logger.info(f"Report successfully saved to: {html_path}")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")

if __name__ == "__main__":
    main()
