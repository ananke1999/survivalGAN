#!/usr/bin/env python3
"""
Survival Metrics Evaluator
===========================
Accepts a real CSV and a synthetic CSV, computes key metrics using
synthcity's Metrics API, and saves results to a CSV file.

Metrics computed:
  COVARIATE QUALITY
    - Jensen-Shannon distance (marginal)
    - Wasserstein distance (marginal)
    - Inverse KL divergence (marginal)
    - Feature correlation (joint)
    - Max Mean Discrepancy (joint)
    - Precision, Recall, Density, Coverage (PRDC)

  SURVIVAL-SPECIFIC
    - KM distance: optimism, abs_optimism, sightedness

  DOWNSTREAM PERFORMANCE
    - C-Index and Brier score for CoxPH, MLP, XGBoost
      (gt = train real/test real, syn_id = train synth/test train-real,
       syn_ood = train synth/test test-real)

  DETECTION
    - XGBoost, MLP, GMM detection scores

Environment: Python 3.10 | synthcity installed
Usage:
    python evaluate_survival_metrics.py real.csv synthetic.csv
    python evaluate_survival_metrics.py real.csv synthetic.csv --target status --time time
    python evaluate_survival_metrics.py real.csv synthetic.csv --output metrics_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =========================================================================== #
#  CONFIGURATION                                                               #
# =========================================================================== #

# Which metric groups to compute. Comment out any you don't need.
# Full list from synthcity tutorial notebook.
DEFAULT_METRICS = {
    "stats": [
        "jensenshannon_dist",       # JSD per feature (marginal)
        "wasserstein_dist",         # Wasserstein per feature (marginal)
        "inv_kl_divergence",        # Inverse KL divergence (marginal)
        "ks_test",                  # Kolmogorov-Smirnov test (marginal)
        "feature_corr",             # Feature correlation (joint)
        "max_mean_discrepancy",     # MMD (joint)
        "prdc",                     # Precision, Recall, Density, Coverage
        "survival_km_distance",     # KM optimism, abs_optimism, sightedness
    ],
    "performance": [
        "linear_model",             # CoxPH: C-Index + Brier
        "mlp",                      # Neural net: C-Index + Brier
        "xgb",                      # XGBoost: C-Index + Brier
    ],
    "detection": [
        "detection_xgb",            # Can XGB tell real from synthetic?
        "detection_mlp",            # Can MLP tell real from synthetic?
        "detection_gmm",            # Can GMM tell real from synthetic?
    ],
}


# =========================================================================== #
#  CORE EVALUATION FUNCTION                                                    #
# =========================================================================== #

def load_and_validate(path: str, target_col: str, time_col: str) -> pd.DataFrame:
    """Load CSV and validate required columns exist."""
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    assert target_col in df.columns, (
        f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}"
    )
    assert time_col in df.columns, (
        f"Time column '{time_col}' not found. Columns: {df.columns.tolist()}"
    )

    # Basic sanity checks
    assert df[target_col].isin([0, 1]).all(), (
        f"Target column '{target_col}' must be binary (0/1). "
        f"Found values: {df[target_col].unique()}"
    )
    assert (df[time_col] >= 0).all(), (
        f"Time column '{time_col}' must be non-negative."
    )

    log.info(f"  Loaded {path}: shape={df.shape}, "
             f"event_rate={df[target_col].mean():.3f}, "
             f"median_time={df[time_col].median():.2f}")
    return df


def evaluate_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    target_col: str = "status",
    time_col: str = "time",
    metrics: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Evaluate synthetic data quality using synthcity Metrics API.

    Returns a DataFrame with columns: [metric_group, metric_name, metric_value]
    """
    from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
    from synthcity.metrics import Metrics

    if metrics is None:
        metrics = DEFAULT_METRICS

    # ---- Align columns ----
    # Synthetic data must have exactly the same columns as real data.
    shared_cols = [c for c in real_df.columns if c in synthetic_df.columns]
    missing_in_syn = set(real_df.columns) - set(synthetic_df.columns)
    extra_in_syn = set(synthetic_df.columns) - set(real_df.columns)

    if missing_in_syn:
        log.warning(f"Columns in real but NOT in synthetic (dropped): {missing_in_syn}")
    if extra_in_syn:
        log.warning(f"Columns in synthetic but NOT in real (dropped): {extra_in_syn}")

    real_aligned = real_df[shared_cols].copy()
    syn_aligned = synthetic_df[shared_cols].copy()

    # ---- Remove rows with non-positive time (synthcity requirement) ----
    real_aligned = real_aligned[real_aligned[time_col] > 0].reset_index(drop=True)
    syn_aligned = syn_aligned[syn_aligned[time_col] > 0].reset_index(drop=True)

    log.info(f"  After alignment: real={real_aligned.shape}, synth={syn_aligned.shape}")

    # ---- Wrap in SurvivalAnalysisDataLoader ----
    log.info("Building SurvivalAnalysisDataLoaders...")
    real_loader = SurvivalAnalysisDataLoader(
        real_aligned,
        target_column=target_col,
        time_to_event_column=time_col,
    )
    syn_loader = SurvivalAnalysisDataLoader(
        syn_aligned,
        target_column=target_col,
        time_to_event_column=time_col,
    )

    # ---- Evaluate ----
    log.info(f"Running metrics evaluation (this may take several minutes)...")
    log.info(f"  Metric groups: {list(metrics.keys())}")
    for group, names in metrics.items():
        log.info(f"    {group}: {names}")

    try:
        results = Metrics.evaluate(
            X_gt=real_loader,
            X_syn=syn_loader,
            task_type="survival_analysis",
            metrics=metrics,
        )
    except TypeError:
        # Some synthcity versions use positional args
        log.warning("Trying alternative Metrics.evaluate() signature...")
        results = Metrics.evaluate(
            real_loader,
            syn_loader,
            task_type="survival_analysis",
            metrics=metrics,
        )

    # ---- Format results into a clean DataFrame ----
    rows = []
    if isinstance(results, dict):
        for key, value in results.items():
            # key format: "group.metric_name.sub_metric"
            parts = key.split(".")
            group = parts[0] if len(parts) > 0 else ""
            name = ".".join(parts[1:]) if len(parts) > 1 else key

            if isinstance(value, (int, float, np.floating, np.integer)):
                rows.append({"group": group, "metric": name, "value": float(value)})
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    rows.append({
                        "group": group,
                        "metric": f"{name}.{sub_key}",
                        "value": float(sub_val) if isinstance(sub_val, (int, float, np.floating, np.integer)) else str(sub_val),
                    })
    elif isinstance(results, pd.DataFrame):
        # Some synthcity versions return a DataFrame directly
        for col in results.columns:
            for idx in results.index:
                val = results.loc[idx, col]
                rows.append({
                    "group": str(idx).split(".")[0] if "." in str(idx) else str(idx),
                    "metric": str(idx),
                    "value": float(val) if isinstance(val, (int, float, np.floating, np.integer)) else str(val),
                })

    result_df = pd.DataFrame(rows)
    return result_df


# =========================================================================== #
#  PRETTY PRINT                                                                #
# =========================================================================== #

def print_results(result_df: pd.DataFrame):
    """Print results grouped by metric category."""
    if result_df.empty:
        log.warning("No results to display.")
        return

    print("\n" + "=" * 70)
    print("  SURVIVAL SYNTHETIC DATA — EVALUATION RESULTS")
    print("=" * 70)

    for group in result_df["group"].unique():
        group_df = result_df[result_df["group"] == group]
        print(f"\n--- {group.upper()} ---")
        for _, row in group_df.iterrows():
            val = row["value"]
            if isinstance(val, float):
                print(f"  {row['metric']:50s}  {val:.6f}")
            else:
                print(f"  {row['metric']:50s}  {val}")

    print("\n" + "=" * 70)


# =========================================================================== #
#  CLI ENTRY POINT                                                             #
# =========================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic survival data against real data."
    )
    parser.add_argument("real_csv", help="Path to real data CSV")
    parser.add_argument("synthetic_csv", help="Path to synthetic data CSV")
    parser.add_argument("--target", default="status",
                        help="Event indicator column name (default: status)")
    parser.add_argument("--time", default="time",
                        help="Time-to-event column name (default: time)")
    parser.add_argument("--output", default="metrics_results.csv",
                        help="Output CSV for metric results (default: metrics_results.csv)")
    parser.add_argument("--skip-performance", action="store_true",
                        help="Skip downstream performance metrics (faster)")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip detection metrics (faster)")
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=== Survival Metrics Evaluator ===")

    # ---- Load data ----
    log.info("Loading real data...")
    real_df = load_and_validate(args.real_csv, args.target, args.time)

    log.info("Loading synthetic data...")
    syn_df = load_and_validate(args.synthetic_csv, args.target, args.time)

    # ---- Build metric selection ----
    metrics = dict(DEFAULT_METRICS)  # copy
    if args.skip_performance:
        metrics.pop("performance", None)
        log.info("Skipping performance metrics (--skip-performance)")
    if args.skip_detection:
        metrics.pop("detection", None)
        log.info("Skipping detection metrics (--skip-detection)")

    # ---- Evaluate ----
    result_df = evaluate_metrics(
        real_df, syn_df,
        target_col=args.target,
        time_col=args.time,
        metrics=metrics,
    )

    # ---- Display ----
    print_results(result_df)

    # ---- Save ----
    result_df.to_csv(args.output, index=False)
    log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
