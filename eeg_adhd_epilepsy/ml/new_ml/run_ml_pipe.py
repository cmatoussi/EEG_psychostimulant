#!/usr/bin/env python3
import argparse
import logging
import os
from copy import deepcopy
import inspect

import yaml
import pandas as pd
import numpy as np
import itertools
class MLPipeline:
    def __init__(self, X, y, groups, config):
        self.X = X
        self.y = y
        self.groups = groups
        self.config = config

    def run(self):
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score

        models_to_run = self.config.get("models", ["Logistic Regression"])
        metrics_to_run = self.config.get("metrics", ["accuracy", "balanced_accuracy", "roc_auc", "f1"])
        
        n_splits = self.config.get("n_splits", 5)
        random_state = self.config.get("cv_kwargs", {}).get("random_state", 42)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        results = {}
        
        for model_name in models_to_run:
            results[model_name] = {
                "metric_scores": {m: [] for m in metrics_to_run},
                "feature_importances": {col: [] for col in self.X.columns} if hasattr(self.X, 'columns') else {}
            }
            
            for train_idx, test_idx in skf.split(self.X, self.y):
                if hasattr(self.X, 'iloc'):
                    X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                else:
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    
                if hasattr(self.y, 'iloc'):
                    y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
                else:
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                from sklearn.impute import SimpleImputer
                
                imputer = SimpleImputer(strategy='mean')
                X_train_im = imputer.fit_transform(X_train)
                X_test_im = imputer.transform(X_test)
                
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train_im)
                X_test_sc = scaler.transform(X_test_im)
                
                if model_name == "Logistic Regression":
                    clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state, max_iter=1000)
                elif model_name == "Random Forest":
                    clf = RandomForestClassifier(random_state=random_state)
                else:
                    clf = LogisticRegression(random_state=random_state, max_iter=1000)
                    
                clf.fit(X_train_sc, y_train)
                preds = clf.predict(X_test_sc)
                probs = clf.predict_proba(X_test_sc)[:, 1] if hasattr(clf, "predict_proba") else preds
                
                if "accuracy" in metrics_to_run:
                    results[model_name]["metric_scores"]["accuracy"].append(accuracy_score(y_test, preds))
                if "balanced_accuracy" in metrics_to_run:
                    results[model_name]["metric_scores"]["balanced_accuracy"].append(balanced_accuracy_score(y_test, preds))
                if "roc_auc" in metrics_to_run:
                    try:
                        results[model_name]["metric_scores"]["roc_auc"].append(roc_auc_score(y_test, probs))
                    except ValueError:
                        results[model_name]["metric_scores"]["roc_auc"].append(0.5)
                if "f1" in metrics_to_run:
                    results[model_name]["metric_scores"]["f1"].append(f1_score(y_test, preds, average='weighted'))
                    
                if hasattr(clf, "coef_"):
                    imp = clf.coef_[0]
                elif hasattr(clf, "feature_importances_"):
                    imp = clf.feature_importances_
                else:
                    imp = np.zeros(X_train_sc.shape[1])
                    
                if hasattr(self.X, 'columns'):
                    for i, col in enumerate(self.X.columns):
                        results[model_name]["feature_importances"][col].append(imp[i])
                        
            agg_metrics = {}
            for m in metrics_to_run:
                vals = results[model_name]["metric_scores"][m]
                agg_metrics[m] = {"mean": np.mean(vals), "std": np.std(vals)}
            results[model_name]["metric_scores"] = agg_metrics
            
            agg_imps = {}
            if hasattr(self.X, 'columns'):
                for col in self.X.columns:
                    vals = results[model_name]["feature_importances"][col]
                    agg_imps[col] = {"mean": np.mean(vals), "std": np.std(vals), "weighted_mean": np.mean(vals)}
            results[model_name]["feature_importances"] = agg_imps
            
        return results

def load(kind, path):
    import pandas as pd
    return pd.read_csv(path)

def select_features(df, target_columns, covariates=None, spatial_units='all', feature_names='all', row_filter=None, sep='_', reverse=False, verbose=False):
    import numpy as np
    df_filtered = df.copy()
    
    if row_filter:
        for rf in row_filter:
            col, val, op = rf['column'], rf['values'], rf['operator']
            if op == '==':
                df_filtered = df_filtered[df_filtered[col].astype(str) == str(val)]
            elif op == '!=':
                df_filtered = df_filtered[df_filtered[col].astype(str) != str(val)]
    
    if isinstance(target_columns, list):
        y = df_filtered[target_columns[0]] if len(target_columns) == 1 else df_filtered[target_columns]
    else:
        y = df_filtered[target_columns]

    exclude_cols = []
    if target_columns:
        exclude_cols.extend(target_columns if isinstance(target_columns, list) else [target_columns])
    if covariates:
        exclude_cols.extend(covariates)
    exclude_cols.extend(["condition", "TDAH", "TSA", "Epilepsy", "participant_id", "subject_id"])
    
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    keep_cols = []
    for c in feature_cols:
        if sep in c:
            parts = c.split(sep, 1)
            if reverse:
                f_name, s_name = parts[0], parts[1]
            else:
                s_name, f_name = parts[0], parts[1]
            if f_name.startswith("feature-"):
                f_name = f_name.replace("feature-", "")
                
            keep_s = (spatial_units == 'all') or (spatial_units is None) or (s_name in spatial_units)
            keep_f = (feature_names == 'all') or (feature_names is None) or (f_name in feature_names)
            
            if keep_s and keep_f:
                keep_cols.append(c)
        else:
            # Do not auto-include non-sensor columns (like n_epilepsy_meds) when feature_names is 'all'
            keep_f = (isinstance(feature_names, list) and c in feature_names)
            if keep_f:
                keep_cols.append(c)

    X = df_filtered[keep_cols]
    return X, y

def balance_dataset(df, target, strategy="undersample", covariates=None, n_bins=5, binning="quantile", random_state=42, grid_balance=False, require_full_grid=False, prefer_clean=False, prefer_clean_rows=False):
    if strategy == "undersample":
        min_size = df[target].value_counts().min()
        return df.groupby(target).sample(n=min_size, random_state=random_state)
    return df

def clean_features(X, mode="any", sep="_", reverse=False, verbose=True, min_abs_value=1e-5, min_abs_fraction=0.1):
    n_before = X.shape[1]
    mask = (X.abs() > min_abs_value).mean(axis=0) >= min_abs_fraction
    mask = mask & (X.isna().mean(axis=0) < 0.5)
    X_cleaned = X.loc[:, mask]
    dropped_columns = X.columns[~mask].tolist()
    report = {
        "mode": mode,
        "n_before": n_before,
        "n_after": X_cleaned.shape[1],
        "dropped_columns": dropped_columns
    }
    return X_cleaned, report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_analysis(X, y, groups, analysis_cfg):
    """
    Run one or more MLPipeline runs according to analysis_cfg.

    Supports three multivariate variants (all / per-sensor / per-feature) plus
    univariate mode inside each slice handled by MLPipeline.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix, shape (n_samples, n_sensors * n_features) if DataFrame,
        or generic 2D array.
    y : pd.Series or np.ndarray
        Target vector or matrix.
    groups : pd.Series or np.ndarray
        Group labels for cross-validation (optional).
    analysis_cfg : dict
        Must include keys:
          - task, analysis_type, models, metrics, cv_kwargs, n_features, direction,
            search_type, n_iter, scoring, n_jobs, save_intermediate,
            results_dir, results_file, mode
        And for slicing:
          - spatial_units : list of sensor names or "all"
          - feature_names : list of feature names or "all"
          - analysis_unit: one of "all", "sensor", "feature"
          - sep           : string separator in column names
          - reverse       : bool, if True swap sensor/feature in names
    """
    # 1) Prepare X as DataFrame for easy column-based slicing
    X_df = X
    y_arr = y.values if hasattr(y, "values") else y
    groups_arr = groups.values if hasattr(groups, "values") else groups

    # 2) Extract slicing parameters
    spatial_units = analysis_cfg.get("spatial_units", "all")
    feature_names = analysis_cfg.get("feature_names", "all")
    sep = analysis_cfg.get("sep", "_")
    reverse = analysis_cfg.get("reverse", False)
    unit = analysis_cfg.get("analysis_unit", "all")  # "all", "sensor", or "feature"

    # 3) Build base config for MLPipeline (exclude X, y, groups)
    base_cfg = {
        "task":            analysis_cfg.get("task"),
        "analysis_type":   analysis_cfg.get("analysis_type"),
        "models":          analysis_cfg.get("models"),
        "metrics":         analysis_cfg.get("metrics"),
        "cv_strategy":     analysis_cfg.get("cv_kwargs", {}).get("cv_strategy"),
        "n_splits":        analysis_cfg.get("cv_kwargs", {}).get("n_splits"),
        "cv_kwargs":       analysis_cfg.get("cv_kwargs"),
        "n_features":      analysis_cfg.get("n_features"),
        "direction":       analysis_cfg.get("direction"),
        "search_type":     analysis_cfg.get("search_type"),
        "n_iter":          analysis_cfg.get("n_iter"),
        "scoring":         analysis_cfg.get("scoring"),
        "n_jobs":          analysis_cfg.get("n_jobs"),
        "save_intermediate": analysis_cfg.get("save_intermediate", True),
        "results_dir":     analysis_cfg.get("results_dir"),
        "results_file":    analysis_cfg.get("results_file"),
        "mode":            analysis_cfg.get("mode", "multivariate"),
    }
    # Drop any None values so defaults inside MLPipeline apply
    base_cfg = {k: v for k, v in base_cfg.items() if v is not None}

    logger.info(
        "Launching '%s' (%s, unit=%s) on data %s",
        base_cfg["task"],
        base_cfg["mode"],
        unit,
        X_df.shape,
    )

    # Helper to run one slice of X through MLPipeline
    def _run_slice(X_sub, label):
        logger.info("  • pipeline slice=%r, shape=%s", label, X_sub.shape)
        X_arr = X_sub
        # copy base config and update results_file to include slice label
        cfg_slice = deepcopy(base_cfg)
        # amend results filename if specified
        if "results_file" in cfg_slice:
            base_name, ext = os.path.splitext(cfg_slice["results_file"])
            cfg_slice["results_file"] = f"{base_name}_{label}{ext}"
        pipeline = MLPipeline(
            X=X_arr,
            y=y_arr,
            groups=groups_arr,
            config=cfg_slice
        )
        return pipeline.run()

    # 4) Build map of label → DataFrame slice
    slice_map = {}

    if unit == "all":
        slice_map["all"] = X_df

    elif unit == "sensor":
        # one run per sensor name in spatial_units
        for sensor in spatial_units:
            if not reverse:
                cols = [c for c in X_df.columns if c.startswith(f"{sensor}{sep}")]
            else:
                cols = [c for c in X_df.columns if c.endswith(f"{sep}{sensor}")]
            if not cols:
                logger.warning("No columns found for sensor=%r", sensor)
            else:
                slice_map[sensor] = X_df[cols]

    elif unit == "feature":
        # one run per feature name in feature_names
        for feat in feature_names:
            if not reverse:
                cols = [c for c in X_df.columns if c.endswith(f"{sep}{feat}")]
            else:
                cols = [c for c in X_df.columns if f"{feat}{sep}" in c]
            if not cols:
                logger.warning("No columns found for feature=%r", feat)
            else:
                slice_map[feat] = X_df[cols]

    else:
        raise ValueError(
            "Invalid analysis_unit %r; must be one of 'all', 'sensor', 'feature'",
            unit
        )

    if not slice_map:
        raise ValueError(f"No valid slices generated for unit={unit!r}")

    # 5) Run each slice and collect results
    results = {
        label: _run_slice(X_sub, label)
        for label, X_sub in slice_map.items()
    }

    # 6) If only the "all" slice was requested, unwrap the dict
    if list(slice_map.keys()) == ["all"]:
        return results["all"]
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ML analyses as defined in a YAML config")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML file with defaults + analyses"
    )
    args = parser.parse_args()

    # 0) Load global config and data
    cfg = yaml.safe_load(open(args.config, "r"))
    df = load("tabular", cfg["data_path"])
    all_results = {}
    defaults = cfg.get("defaults", {})

    # 1) Loop over each analysis block
    for analysis in cfg["analyses"]:
        analysis_cfg = deepcopy(defaults)
        analysis_cfg.update(analysis)
        feature_names = analysis_cfg.get("feature_names", "all")
        if isinstance(feature_names, list):
            # In the new data, the features don't have a "feature-" prefix
            # so we just use them exactly as they are.
            pass
        # 2) Feature selection & target extraction
        X, y = select_features(
            df,
            target_columns=analysis_cfg["target_columns"],
            covariates=analysis_cfg.get("covariates"),
            spatial_units=analysis_cfg.get("spatial_units"),
            feature_names=feature_names,
            row_filter=analysis_cfg.get("row_filter"),
            sep=analysis_cfg.get("sep", ".spaces-"),
            reverse=analysis_cfg.get("reverse", False),
            verbose=True,
        )

        logger.info(
            "Analysis %r selected %d features × %d samples, target=%r",
            analysis["id"], X.shape[1], X.shape[0], getattr(y, "name", None)
        )
        logger.info("  First features: %s", X.columns.tolist()[:5])

        # 2b) Optional class balancing
        bal_cfg = analysis_cfg.get("balance")
        if bal_cfg:
            tcols = analysis_cfg.get("target_columns")
            target_col = (
                (tcols[0] if isinstance(tcols, (list, tuple)) else tcols)
                if tcols is not None else (getattr(y, "name", None) or "target")
            )

            df_bal = X.copy()
            df_bal[target_col] = y

            covs = bal_cfg.get("covariates", analysis_cfg.get("covariates"))
            if covs:
                try:
                    df_bal = df_bal.join(df.loc[X.index, covs])
                except Exception as e:
                    logger.warning("Could not join covariates for balancing: %s", e)

            logger.info("Class distribution before balance:\n%s", df_bal[target_col].value_counts())
            prefer_flag = bal_cfg.get("prefer_clean_rows", bal_cfg.get("prefer_clean", True))
            bd_kwargs = dict(
                df=df_bal,
                target=target_col,
                strategy=bal_cfg.get("strategy", "undersample"),
                covariates=covs,
                n_bins=bal_cfg.get("qbins", bal_cfg.get("n_bins", 5)),
                binning=bal_cfg.get("binning", "quantile"),
                random_state=bal_cfg.get("seed", analysis_cfg.get("cv_kwargs", {}).get("random_state", 42)),
                grid_balance=analysis_cfg.get("grid_balance"),
                require_full_grid=analysis_cfg.get("require_full_grid"),
                prefer_clean=prefer_flag,
                prefer_clean_rows=prefer_flag,
            )
            sig = inspect.signature(balance_dataset)
            bd_kwargs = {k: v for k, v in bd_kwargs.items() if k in sig.parameters}
            balanced = balance_dataset(**bd_kwargs)
            logger.info("Class distribution after balance:\n%s", balanced[target_col].value_counts())

            y = balanced[target_col]
            X = balanced[X.columns]
            
            # 1.5) Optional: clean features based on config
            clean_cfg = analysis_cfg.get("clean_cfg") or analysis_cfg.get("clean_features")
            if clean_cfg:
                if isinstance(clean_cfg, dict):
                    mode = clean_cfg.get("mode", "any")
                    sep_clean = clean_cfg.get("sep", "_")
                    reverse = clean_cfg.get("reverse", False)
                    verbose_clean = clean_cfg.get("verbose", True)
                    min_abs_value = clean_cfg.get("min_abs_value", 1e-5)
                    min_abs_fraction = clean_cfg.get("min_abs_fraction", 0.1)
                else:
                    mode, sep_clean, reverse, verbose_clean = "any", "_", False, True
                    min_abs_value, min_abs_fraction = None, 0.0
                # Build kwargs and filter by signature for compatibility
                cf_kwargs = dict(
                    mode=mode,
                    sep=sep_clean,
                    reverse=reverse,
                    verbose=verbose_clean,
                    min_abs_value=min_abs_value,
                    min_abs_fraction=min_abs_fraction,
                )
                try:
                    sig_cf = inspect.signature(clean_features)
                    cf_kwargs = {k: v for k, v in cf_kwargs.items() if k in sig_cf.parameters}
                except Exception:
                    pass
                X, report = clean_features(X, **cf_kwargs)
                logger.info(
                    "Cleaned features (mode=%s): dropped %d columns; %d -> %d",
                    report.get("mode"),
                    len(report.get("dropped_columns", [])),
                    report.get("n_before"),
                    report.get("n_after"),
                )
        # 2c) Optional: PCA transform after cleaning/balancing
        pca_cfg = analysis_cfg.get("pca")
        use_pca = False
        if isinstance(pca_cfg, dict):
            use_pca = pca_cfg.get("enabled", True)
        elif isinstance(pca_cfg, bool):
            use_pca = pca_cfg
            pca_cfg = {}
        elif pca_cfg is not None:
            # any truthy non-dict non-bool: enable with defaults
            use_pca = True
            pca_cfg = {}

        if use_pca:
            # Defaults
            n_components = pca_cfg.get("n_components", 0.95)  # float (variance) or int
            scale = pca_cfg.get("scale", True)
            whiten = pca_cfg.get("whiten", False)
            feature_prefix = pca_cfg.get("feature_prefix", "PC")
            add_suffix = pca_cfg.get("suffix_results_file", True)

            logger.info(
                "Applying PCA (n_components=%r, scale=%s, whiten=%s) to X shape %s",
                n_components, scale, whiten, X.shape
            )

            X_mat = X.values
            if scale:
                scaler = StandardScaler(with_mean=True, with_std=True)
                X_mat = scaler.fit_transform(X_mat)
            pca = PCA(n_components=n_components, whiten=whiten)
            X_pca = pca.fit_transform(X_mat)
            # Rebuild DataFrame with PC names
            ncomp_real = X_pca.shape[1]
            pc_cols = [f"{feature_prefix}{i+1}" for i in range(ncomp_real)]
            X = pd.DataFrame(X_pca, index=X.index, columns=pc_cols)

            # Ensure slicing won't attempt per-sensor/feature on PCA features
            analysis_cfg["analysis_unit"] = "all"
            # feature_names list no longer applies to PCA features
            if "feature_names" in analysis_cfg:
                analysis_cfg["feature_names"] = []
            # Make results filename reflect PCA
            if add_suffix and analysis_cfg.get("results_file"):
                base_name, ext = os.path.splitext(analysis_cfg["results_file"])
                analysis_cfg["results_file"] = f"{base_name}_pca{ext}"

            # Expose PCA explained variance in logs
            try:
                evr_cum = np.cumsum(pca.explained_variance_ratio_)
                logger.info(
                    "PCA kept %d comps; cumulative EVR (first 5): %s",
                    ncomp_real,
                    np.array2string(evr_cum[:5], precision=3)
                )
            except Exception:
                pass

        # 2d) Sync analysis_cfg.feature_names and spatial_units to remaining columns
        orig_feats = analysis_cfg.get("feature_names")
        orig_sensors = analysis_cfg.get("spatial_units")
        sep_name = analysis_cfg.get("sep", "_")
        rev = analysis_cfg.get("reverse", False)
        
        present_feats = set()
        present_sensors = set()
        for col in X.columns:
            if sep_name in col:
                head = col.split(sep_name)[0] if rev else col.split(sep_name)[-1]
                tail = col.split(sep_name)[-1] if rev else col.split(sep_name)[0]
                if head.startswith("feature-"):
                    head = head[len("feature-") :]
                present_feats.add(head)
                present_sensors.add(tail)
                
        if orig_feats == "all" or orig_feats is None:
            analysis_cfg["feature_names"] = list(present_feats)
            logger.info("Auto-extracted %d feature_names from columns", len(present_feats))
        elif isinstance(orig_feats, (list, tuple)):
            filtered = [f for f in orig_feats if f in present_feats]
            if len(filtered) != len(orig_feats):
                analysis_cfg["feature_names"] = filtered
                logger.info(
                    "Filtered feature_names based on X columns: %d -> %d",
                    len(orig_feats), len(filtered)
                )

        if orig_sensors == "all" or orig_sensors is None:
            analysis_cfg["spatial_units"] = list(present_sensors)
            logger.info("Auto-extracted %d spatial_units from columns", len(present_sensors))
        elif isinstance(orig_sensors, (list, tuple)):
            filtered_sens = [s for s in orig_sensors if s in present_sensors]
            if len(filtered_sens) != len(orig_sensors):
                analysis_cfg["spatial_units"] = filtered_sens
                logger.info(
                    "Filtered spatial_units based on X columns: %d -> %d",
                    len(orig_sensors), len(filtered_sens)
                )
        # 3) Persist the exact X/y used for this analysis to CSV (after cleaning/balancing and optional PCA)
        try:
            tcols = analysis_cfg.get("target_columns")
            target_name = (
                (tcols[0] if isinstance(tcols, (list, tuple)) else tcols)
                if tcols is not None else (getattr(y, "name", None) or "target")
            )
            df_to_save = X.copy()
            df_to_save[target_name] = y
            out_root = cfg.get("results_dir", ".")
            os.makedirs(os.path.join(out_root, "prepared"), exist_ok=True)
            base_file = analysis_cfg.get("results_file") or f"{cfg['global_experiment_id']}_{analysis['id']}"
            csv_path = os.path.join(out_root, "prepared", f"{base_file}_Xy.csv")
            df_to_save.to_csv(csv_path, index=False)
            logger.info("Saved prepared X/y to %s", csv_path)
        except Exception as e:
            logger.warning("Could not save prepared X/y CSV: %s", e)

        # 4) Run the configured analysis
        groups = None

        results = run_analysis(X, y, groups, analysis_cfg)
        all_results[analysis["id"]] = results

    # 4) Save aggregated results
    out_dir = cfg.get("results_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg['global_experiment_id']}.pkl")
    logger.info("Saving all results to %r", out_path)
    pd.to_pickle(all_results, out_path)


if __name__ == "__main__":
    main()
