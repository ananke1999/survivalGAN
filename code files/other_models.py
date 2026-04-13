#!/usr/bin/env python3
"""
Baseline Synthetic Survival Data Generator
============================================
Accepts a real CSV, trains multiple baseline generative models via synthcity,
and outputs a synthetic CSV for each method.

Baselines (from SurvivalGAN paper):
  - adsgan       : ADS-GAN (Anonymization through Data Synthesis)
  - ctgan        : Conditional Tabular GAN
  - tvae         : Tabular VAE
  - nflow        : Normalizing Flows for tabular data
  - survae       : SurvivalVAE (survival-specific VAE)
  - survival_gan : SurvivalGAN via synthcity (for comparison with your local version)

Each baseline is trained using synthcity's plugin system with
SurvivalAnalysisDataLoader, matching the paper's evaluation protocol.

Environment: Python 3.10 | synthcity installed (pip install -e .)
Usage:
    python generate_baselines.py real_data.csv
    python generate_baselines.py real_data.csv --target status --time time --count 2000
    python generate_baselines.py real_data.csv --methods ctgan tvae nflow
    python generate_baselines.py real_data.csv --output-dir baselines_output/
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import random
import argparse
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =========================================================================== #
#  CONFIGURATION                                                               #
# =========================================================================== #

@dataclass
class Config:
    # ---- Data ----
    input_csv: str = "rotterdam_2232_survival.csv"
    target_column: str = "status"
    time_column: str = "time"

    # ---- Generation ----
    synthetic_count: int = 2000
    output_dir: str = "baselines_output"

    # ---- Methods to run ----
    # Comment out any you don't want. Order = execution order.
    methods: List[str] = field(default_factory=lambda: [
        "adsgan",
        "ctgan",
        "tvae",
        "nflow",
        "survae",
        # "survival_gan",   # uncomment to also run synthcity's SurvivalGAN
    ])

    # ---- Per-method overrides (optional) ----
    # Keys = method name, values = dict of kwargs passed to Plugins().get(method, **kwargs)
    # The paper's hyperparameters are embedded as defaults in synthcity;
    # override here if you want to experiment.
    method_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "adsgan": {
            "n_iter": 1500,
            "generator_n_layers_hidden": 3,
            "generator_n_units_hidden": 250,
            "discriminator_n_layers_hidden": 2,
            "discriminator_n_units_hidden": 250,
            "batch_size": 500,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        },
        "ctgan": {
            "n_iter": 1500,
            "generator_n_layers_hidden": 3,
            "generator_n_units_hidden": 250,
            "discriminator_n_layers_hidden": 2,
            "discriminator_n_units_hidden": 250,
            "batch_size": 500,
        },
        "tvae": {
            "n_iter": 1500,
            "decoder_n_layers_hidden": 3,
            "decoder_n_units_hidden": 250,
            "encoder_n_layers_hidden": 3,
            "encoder_n_units_hidden": 250,
            "batch_size": 500,
        },
        "nflow": {
            "n_iter": 1500,
            "n_layers_hidden": 3,
            "n_units_hidden": 250,
            "batch_size": 500,
        },
        "survae": {},           # uses synthcity defaults
        "survival_gan": {},     # uses synthcity defaults
    })

    # ---- System ----
    seed: int = 42
    device: str = "auto"        # "auto", "cuda", "cpu"
    n_seeds: int = 1            # set to 5 to match the paper's 5-seed protocol


# =========================================================================== #
#  HELPERS                                                                     #
# =========================================================================== #

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device(preference: str) -> str:
    if preference == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = (preference == "cuda") and torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        log.info(f"GPU: {name} ({mem:.1f} GB)")
        return "cuda"
    log.info("Using CPU")
    return "cpu"


def load_and_validate(path: str, target_col: str, time_col: str) -> pd.DataFrame:
    """Load CSV and validate required columns."""
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
    assert df[target_col].isin([0, 1]).all(), (
        f"Target column '{target_col}' must be binary (0/1)."
    )

    # Drop non-positive times (required by synthcity survival loaders)
    n_before = len(df)
    df = df[df[time_col] > 0].reset_index(drop=True)
    if len(df) < n_before:
        log.warning(f"Dropped {n_before - len(df)} rows with time <= 0")

    log.info(f"Loaded {path}: shape={df.shape}, "
             f"event_rate={df[target_col].mean():.3f}, "
             f"median_time={df[time_col].median():.2f}")
    return df


# =========================================================================== #
#  METHOD CLASSIFICATION                                                       #
# =========================================================================== #

# Survival-specific plugins use SurvivalAnalysisDataLoader natively.
# Generic plugins also accept it — synthcity handles the routing internally.
SURVIVAL_SPECIFIC_METHODS = {"survival_gan", "survae"}
GENERIC_METHODS = {"adsgan", "ctgan", "tvae", "nflow"}
ALL_SUPPORTED = SURVIVAL_SPECIFIC_METHODS | GENERIC_METHODS


# =========================================================================== #
#  CORE: TRAIN AND GENERATE FOR ONE METHOD                                     #
# =========================================================================== #

def train_and_generate(
    method: str,
    loader,       # SurvivalAnalysisDataLoader
    count: int,
    device: str,
    kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Train a single synthcity plugin and generate synthetic data.
    Returns a DataFrame of synthetic samples.
    """
    from synthcity.plugins import Plugins

    log.info(f"  Instantiating '{method}' plugin...")

    # Build kwargs, injecting device
    plugin_kwargs = dict(kwargs)
    plugin_kwargs["device"] = device

    try:
        syn_model = Plugins().get(method, **plugin_kwargs)
    except Exception as e:
        log.error(f"  Failed to instantiate '{method}': {e}")
        raise

    log.info(f"  Training '{method}' on {len(loader)} samples...")
    t0 = time.time()
    syn_model.fit(loader)
    elapsed = time.time() - t0
    log.info(f"  Training '{method}' complete in {elapsed:.1f}s")

    log.info(f"  Generating {count} synthetic samples...")
    synthetic_data = syn_model.generate(count=count)

    # synthcity returns a DataLoader — extract DataFrame
    if hasattr(synthetic_data, "dataframe"):
        synthetic_df = synthetic_data.dataframe()
    elif isinstance(synthetic_data, pd.DataFrame):
        synthetic_df = synthetic_data
    else:
        synthetic_df = pd.DataFrame(synthetic_data)

    log.info(f"  Generated shape: {synthetic_df.shape}")
    return synthetic_df


# =========================================================================== #
#  MAIN PIPELINE                                                               #
# =========================================================================== #

def run_baselines(cfg: Config):
    """Run all baseline methods and save outputs."""
    from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader

    seed_everything(cfg.seed)
    device = setup_device(cfg.device)

    # ---- Load data ----
    df = load_and_validate(cfg.input_csv, cfg.target_column, cfg.time_column)

    # ---- Create output directory ----
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- Build DataLoader ----
    loader = SurvivalAnalysisDataLoader(
        df,
        target_column=cfg.target_column,
        time_to_event_column=cfg.time_column,
    )

    # ---- Validate methods ----
    for m in cfg.methods:
        if m not in ALL_SUPPORTED:
            log.warning(f"Unknown method '{m}' — skipping. "
                        f"Supported: {sorted(ALL_SUPPORTED)}")

    # ---- Run each method ----
    results_summary = []

    for method in cfg.methods:
        if method not in ALL_SUPPORTED:
            continue

        log.info(f"\n{'='*60}")
        log.info(f"  METHOD: {method.upper()}")
        log.info(f"{'='*60}")

        kwargs = cfg.method_kwargs.get(method, {})

        for seed_idx in range(cfg.n_seeds):
            current_seed = cfg.seed + seed_idx
            seed_everything(current_seed)

            seed_suffix = f"_seed{current_seed}" if cfg.n_seeds > 1 else ""
            output_path = os.path.join(
                cfg.output_dir,
                f"synthetic_{method}{seed_suffix}.csv"
            )

            try:
                t0 = time.time()
                synthetic_df = train_and_generate(
                    method=method,
                    loader=loader,
                    count=cfg.synthetic_count,
                    device=device,
                    kwargs=kwargs,
                )
                elapsed = time.time() - t0

                # ---- Save ----
                synthetic_df.to_csv(output_path, index=False)
                log.info(f"  Saved: {output_path}")

                # ---- Quick comparison ----
                log.info(f"  --- Real vs {method} Summary ---")
                for col in df.columns:
                    r_mean = df[col].mean()
                    s_mean = (synthetic_df[col].mean()
                              if col in synthetic_df.columns else float("nan"))
                    log.info(f"    {col:20s}  real={r_mean:8.3f}  synth={s_mean:8.3f}")

                results_summary.append({
                    "method": method,
                    "seed": current_seed,
                    "status": "success",
                    "time_seconds": round(elapsed, 1),
                    "output_file": output_path,
                    "synth_event_rate": round(
                        synthetic_df[cfg.target_column].mean(), 4
                    ) if cfg.target_column in synthetic_df.columns else None,
                })

            except Exception as e:
                log.error(f"  FAILED: {method} (seed={current_seed}): {e}")
                traceback.print_exc()
                results_summary.append({
                    "method": method,
                    "seed": current_seed,
                    "status": f"FAILED: {str(e)[:100]}",
                    "time_seconds": None,
                    "output_file": None,
                    "synth_event_rate": None,
                })

            # Free GPU memory between methods
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Save run summary ----
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(cfg.output_dir, "run_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    log.info(f"\n{'='*60}")
    log.info(f"Run summary saved to {summary_path}")
    log.info(f"{'='*60}")
    print("\n" + summary_df.to_string(index=False))

    return summary_df


# =========================================================================== #
#  CLI ENTRY POINT                                                             #
# =========================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic survival data using baseline methods."
    )
    parser.add_argument("input_csv", help="Path to real data CSV")
    parser.add_argument("--target", default="status",
                        help="Event indicator column (default: status)")
    parser.add_argument("--time", default="time",
                        help="Time-to-event column (default: time)")
    parser.add_argument("--count", type=int, default=2000,
                        help="Number of synthetic samples per method (default: 2000)")
    parser.add_argument("--output-dir", default="baselines_output",
                        help="Output directory (default: baselines_output/)")
    parser.add_argument("--methods", nargs="+",
                        default=["adsgan", "ctgan", "tvae", "nflow", "survae"],
                        help="Methods to run (default: adsgan ctgan tvae nflow survae)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds per method (default: 1, paper uses 5)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.input_csv = args.input_csv
    cfg.target_column = args.target
    cfg.time_column = args.time
    cfg.synthetic_count = args.count
    cfg.output_dir = args.output_dir
    cfg.methods = args.methods
    cfg.n_seeds = args.seeds
    cfg.device = args.device

    log.info("=== Baseline Synthetic Survival Data Generator ===")
    log.info(f"  Input:   {cfg.input_csv}")
    log.info(f"  Methods: {cfg.methods}")
    log.info(f"  Count:   {cfg.synthetic_count}")
    log.info(f"  Seeds:   {cfg.n_seeds}")
    log.info(f"  Output:  {cfg.output_dir}/")

    run_baselines(cfg)


if __name__ == "__main__":
    main()
