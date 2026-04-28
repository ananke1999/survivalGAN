#!/usr/bin/env python3
"""
train_survgan_aws.py
====================
AWS/EC2 driver for the standalone SurvivalGAN (survGAN.py) on MSK-IMPACT.

This script is a THIN wrapper around survGAN.py's SurvivalPipeline. It does
not implement its own training loop — pipeline.fit() already runs WGAN-GP
training, DeepHit, and the XGBoost time regressor internally.

What this wrapper adds vs just running survGAN.py:
  - CLI args (input, output, n_iter, synthetic count, seed, checkpoint dir)
  - File logging to train.log AND stdout
  - GPU memory reporting + cudnn benchmark
  - Pickled checkpoint of the fitted pipeline after training
  - Graceful ctrl-C / failure handling with full traceback to the log file
  - Post-generation time clip + precision match + physical constraints

Usage on EC2:
    python3 train_survGan_aws.py --input "data sets/survival_gan_train.csv" --output "synthetic.csv" --n_iter 3000 --checkpoint-dir ckpt
"""

import argparse
import datetime
import io
import json
import logging
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch

from survGAN import (
    Config,
    SurvivalPipeline,
    seed_everything,
    setup_device,
    retrofit_tte_bounds,
)


def setup_logging(log_path: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SurvivalGAN training on AWS/EC2")

    # I/O
    p.add_argument("--input", required=True, help="Path to clean CSV (from preprocessing.py)")
    p.add_argument("--output", default="synthetic.csv", help="Where to write synthetic CSV")
    p.add_argument("--checkpoint-dir", default="ckpt",
                   help="Directory for pickled pipeline + log file")
    p.add_argument("--metadata-json", default=None,
                   help="Column metadata JSON written by preprocessing.py. "
                        "Default: same dir as --input, "
                        "'survival_gan_column_metadata.json'. Pass empty string "
                        "to disable and fall back to the built-in default map.")

    # Training
    p.add_argument("--n-iter", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--synthetic-count", type=int, default=None)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # Target columns
    p.add_argument("--time-column", default="time")
    p.add_argument("--status-column", default="status")

    # Skip stages
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--regenerate", action="store_true",
                   help="Load pipeline + retrofit TTE bounds (no retraining).")

    # Post-generation time clip
    p.add_argument("--clip-output-time", action="store_true", default=True)
    p.add_argument("--no-clip-output-time", dest="clip_output_time",
                   action="store_false")

    # Post-generation precision matching
    p.add_argument("--match-real-precision", action="store_true", default=True)
    p.add_argument("--no-match-real-precision", dest="match_real_precision",
                   action="store_false")
    p.add_argument("--precision-unique-threshold", type=int, default=1000)

    # Post-generation physical constraints
    p.add_argument("--enforce-constraints", action="store_true", default=True,
                   help="After generation, clip to column bounds and cast "
                        "integer columns to int (default ON). Uses "
                        "--metadata-json if available, else a name-based "
                        "default map.")
    p.add_argument("--no-enforce-constraints", dest="enforce_constraints",
                   action="store_false")

    return p.parse_args()


# =========================================================================== #
#  Precision matching (unchanged from prior version)                           #
# =========================================================================== #

def match_real_precision(real_df: pd.DataFrame,
                         syn_df: pd.DataFrame,
                         unique_threshold: int = 1000,
                         skip_cols: Optional[set] = None) -> Tuple[pd.DataFrame, dict]:
    """Snap synthetic numeric columns to match the precision of real."""
    skip_cols = skip_cols or set()
    out = syn_df.copy()
    report: dict = {}

    for col in syn_df.columns:
        if col in skip_cols or col not in real_df.columns:
            continue
        if not np.issubdtype(real_df[col].dtype, np.number):
            continue
        if not np.issubdtype(syn_df[col].dtype, np.number):
            continue

        real_vals = real_df[col].dropna()
        if len(real_vals) == 0:
            continue

        syn_arr = syn_df[col].to_numpy(dtype=float).copy()
        nan_mask = np.isnan(syn_arr)
        n_uniq_real = int(real_vals.nunique())

        if n_uniq_real <= unique_threshold:
            real_sorted = np.sort(real_vals.unique().astype(float))
            idx = np.searchsorted(real_sorted, syn_arr, side="left")
            idx_clip = np.clip(idx, 1, len(real_sorted) - 1)
            left = real_sorted[idx_clip - 1]
            right = real_sorted[idx_clip]
            snapped = np.where(
                np.abs(syn_arr - left) <= np.abs(syn_arr - right),
                left, right,
            )
            snapped[nan_mask] = np.nan
            out[col] = snapped
            method = f"snap→{n_uniq_real}_vals"
        else:
            sample = real_vals.astype(str).head(2000)
            max_dec = 0
            for v in sample:
                if "." in v:
                    dec_part = v.split(".")[1].rstrip("0")
                    if len(dec_part) > max_dec:
                        max_dec = len(dec_part)
            if max_dec > 0:
                out[col] = np.round(syn_arr, max_dec)
            method = f"round→{max_dec}dp"

        if np.issubdtype(real_df[col].dtype, np.integer) and not nan_mask.any():
            try:
                out[col] = out[col].astype(real_df[col].dtype)
                method += ",int"
            except (ValueError, TypeError):
                pass

        n_uniq_syn_before = int(pd.Series(syn_df[col]).nunique())
        n_uniq_syn_after = int(pd.Series(out[col]).nunique())
        report[col] = {
            "real_nuniq": n_uniq_real,
            "syn_nuniq_before": n_uniq_syn_before,
            "syn_nuniq_after": n_uniq_syn_after,
            "method": method,
        }

    return out, report


# =========================================================================== #
#  NEW                                  #
# =========================================================================== #

DEFAULT_CONSTRAINT_MAP: Dict[str, Dict[str, Any]] = {
    "Age at Diagnosis":                   {"dtype": "int",   "bounds": (0, 120)},
    "Mutation Count":                     {"dtype": "int",   "bounds": (0, None)},
    "Number of Other Cancer Types":       {"dtype": "int",   "bounds": (0, None)},
    "Sample coverage":                    {"dtype": "int",   "bounds": (0, None)},
    "Fraction Genome Altered":            {"dtype": "float", "bounds": (0.0, 1.0)},
    "FACETS Estimated Purity":            {"dtype": "float", "bounds": (0.0, 1.0)},
    "Pathologist Estimated Tumor Purity": {"dtype": "float", "bounds": (0.0, 100.0)},
    "Ploidy (FACETS)":                    {"dtype": "float", "bounds": (0.0, None)},
    "MSI Score":                          {"dtype": "float", "bounds": (0.0, None)},
    "TMB Score":                          {"dtype": "float", "bounds": (0.0, None)},
    "time":                               {"dtype": "float", "bounds": (1e-3, None)},
    "status":                             {"dtype": "int",   "bounds": (0, 1)},
}


def _load_constraint_map(metadata_json_path: Optional[Path],
                         real_df: pd.DataFrame,
                         log: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Build the effective column -> constraint mapping.

    Priority order for each column:
      1. metadata JSON entry (if present)
      2. DEFAULT_CONSTRAINT_MAP entry
      3. inferred from real data (integer if all-integer-valued, bounds from min/max)
    """
    constraints: Dict[str, Dict[str, Any]] = {}
    json_entries = {}

    if metadata_json_path is not None and metadata_json_path.exists():
        try:
            with open(metadata_json_path, "r") as f:
                meta = json.load(f)
            json_entries = meta.get("columns", {})
            log.info(f"Loaded column metadata from {metadata_json_path} "
                     f"({len(json_entries)} columns)")
        except Exception as e:
            log.warning(f"Failed to parse {metadata_json_path}: {e}. "
                        f"Falling back to default constraint map.")

    for col in real_df.columns:
        if col in json_entries:
            e = json_entries[col]
            constraints[col] = {
                "dtype":  e.get("dtype", "float"),
                "bounds": tuple(e.get("bounds", (None, None))),
                "source": "json",
            }
            continue

        if col in DEFAULT_CONSTRAINT_MAP:
            e = DEFAULT_CONSTRAINT_MAP[col]
            constraints[col] = {
                "dtype":  e["dtype"],
                "bounds": e["bounds"],
                "source": "default_map",
            }
            continue

        # Inference fallback — applies to label-encoded categoricals.
        # If real is integer-valued, treat as int and clip to observed range.
        if np.issubdtype(real_df[col].dtype, np.number):
            real_vals = pd.to_numeric(real_df[col], errors="coerce").dropna()
            if len(real_vals) == 0:
                continue
            is_int = (real_vals == real_vals.round()).all()
            constraints[col] = {
                "dtype":  "int" if is_int else "float",
                "bounds": (float(real_vals.min()), float(real_vals.max())),
                "source": "inferred",
            }

    return constraints


def enforce_physical_constraints(real_df: pd.DataFrame,
                                 syn_df: pd.DataFrame,
                                 constraints: Dict[str, Dict[str, Any]],
                                 log: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Apply dtype + bounds constraints to each synthetic column.

    For every column present in both `syn_df` and `constraints`:
      - Clip to `bounds` (skip endpoints set to None).
      - If `dtype == "int"`, round values and cast to int64.

    Returns the modified dataframe and a per-column report of how many
    values were clipped/rounded.
    """
    out = syn_df.copy()
    report: Dict[str, Dict] = {}

    for col, spec in constraints.items():
        if col not in out.columns:
            continue
        if not np.issubdtype(out[col].dtype, np.number):
            continue

        arr = out[col].to_numpy(dtype=float).copy()
        n = len(arr)
        lo, hi = spec["bounds"]
        dtype = spec["dtype"]

        n_below = int((arr < lo).sum()) if lo is not None else 0
        n_above = int((arr > hi).sum()) if hi is not None else 0

        # Clip
        if lo is not None:
            arr = np.where(arr < lo, lo, arr)
        if hi is not None:
            arr = np.where(arr > hi, hi, arr)

        # Integer cast (round first — post-clip rounding is safe)
        n_nonint = 0
        if dtype == "int":
            rounded = np.round(arr)
            n_nonint = int((arr != rounded).sum())
            arr = rounded
            try:
                out[col] = arr.astype(np.int64)
            except (ValueError, TypeError):
                out[col] = arr
        else:
            out[col] = arr

        report[col] = {
            "dtype":    dtype,
            "bounds":   (lo, hi),
            "n_below":  n_below,
            "n_above":  n_above,
            "n_rounded": n_nonint,
            "pct_modified": 100.0 * (n_below + n_above + n_nonint) / max(n, 1),
            "source":   spec.get("source", "?"),
        }

    return out, report


# =========================================================================== #
#  Save trained GAN as .pt + human-readable summary                            #
# =========================================================================== #

def save_model_pt_and_summary(pipeline,
                              cfg: Config,
                              ckpt_dir: Path,
                              log: logging.Logger,
                              training_elapsed_sec: Optional[float] = None) -> None:
    """Save the trained GAN's PyTorch weights to model.pt and a human-readable
    summary to model_summary.txt, alongside pipeline.pkl in `ckpt_dir`.

    `model.pt` contains a single dict (loadable with `torch.load`) with:
      - generator_state_dict      : weights of the generator MLP
      - discriminator_state_dict  : weights of the discriminator MLP
      - gan_state_dict            : full GAN module state_dict (incl. both)
      - architecture              : dims/layers needed to rebuild the GAN
      - config                    : serializable snapshot of training config
      - pipeline_meta             : column names, target/time cols, censoring rate
      - tte_bounds                : training-time T range for post-processing
      - training                  : n_iter, batch_size, seed, device, elapsed

    The full pipeline (encoders, TTE regressor, XGBoost censoring predictor)
    still lives in `pipeline.pkl`. `model.pt` is the *neural network* checkpoint
    the instructor asked for — the generator + discriminator weights and
    everything needed to rebuild the architecture.
    """
    model_pt_path = ckpt_dir / "model.pt"
    summary_path  = ckpt_dir / "model_summary.txt"

    # ----- Locate generator/discriminator -----
    try:
        gan = pipeline.gan.model           # the inner GAN nn.Module
        G   = gan.generator                # MLP nn.Module
        D   = gan.discriminator            # MLP nn.Module
    except AttributeError:
        log.error("Pipeline has no .gan.model — cannot save model.pt. "
                  "Was fit() called successfully?")
        log.error(traceback.format_exc())
        return

    # ----- Build .pt payload -----
    safe_cfg = {
        k: v for k, v in vars(cfg).items()
        if isinstance(v, (int, float, str, bool, tuple, list, type(None)))
    }

    payload: Dict[str, Any] = {
        "generator_state_dict":     G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "gan_state_dict":           gan.state_dict(),
        "architecture": {
            "n_features":                   int(gan.n_features),
            "n_units_latent":               int(gan.n_units_latent),
            "n_units_conditional":          int(gan.n_units_conditional),
            "generator_n_layers_hidden":    int(cfg.generator_n_layers_hidden),
            "generator_n_units_hidden":     int(cfg.generator_n_units_hidden),
            "generator_nonlin":             cfg.generator_nonlin,
            "generator_residual":           bool(cfg.generator_residual),
            "generator_batch_norm":         bool(cfg.generator_batch_norm),
            "generator_dropout":            float(cfg.generator_dropout),
            "discriminator_n_layers_hidden":int(cfg.discriminator_n_layers_hidden),
            "discriminator_n_units_hidden": int(cfg.discriminator_n_units_hidden),
            "discriminator_nonlin":         cfg.discriminator_nonlin,
            "discriminator_batch_norm":     bool(cfg.discriminator_batch_norm),
            "discriminator_dropout":        float(cfg.discriminator_dropout),
        },
        "config": safe_cfg,
        "pipeline_meta": {
            "columns":         list(getattr(pipeline, "columns", [])),
            "target_column":   getattr(pipeline, "target_column", None),
            "time_column":     getattr(pipeline, "time_column", None),
            "censoring_ratio": float(getattr(pipeline, "censoring_ratio", 0.0)),
        },
        "training": {
            "n_iter":      int(cfg.n_iter),
            "batch_size":  int(cfg.batch_size),
            "seed":        int(cfg.seed),
            "device":      str(cfg.device),
            "elapsed_sec": (float(training_elapsed_sec)
                            if training_elapsed_sec is not None else None),
            "saved_at":    datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }

    # TTE training-time bounds — needed for any inference-side time clamping.
    tte = getattr(pipeline, "tte_model", None)
    if tte is not None:
        payload["tte_bounds"] = {
            "T_min":        getattr(tte, "_T_min", None),
            "T_max":        getattr(tte, "_T_max", None),
            "Tlog_min":     getattr(tte, "_Tlog_min", None),
            "Tlog_max":     getattr(tte, "_Tlog_max", None),
            "residual_std": getattr(tte, "_residual_std", None),
        }

    # ----- Save model.pt -----
    try:
        torch.save(payload, model_pt_path)
        log.info(f"Saved model weights → {model_pt_path}  "
                 f"({model_pt_path.stat().st_size / 1e6:.2f} MB)")
    except Exception:
        log.error("Saving model.pt failed; continuing.")
        log.error(traceback.format_exc())
        return

    # ----- Build model_summary.txt -----
    def _count_params(module: torch.nn.Module) -> Tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, train

    g_total, g_train = _count_params(G)
    d_total, d_train = _count_params(D)

    buf = io.StringIO()
    print("=" * 70, file=buf)
    print("SurvivalGAN — model summary", file=buf)
    print(f"Saved at: {payload['training']['saved_at']}", file=buf)
    print("=" * 70, file=buf)

    print("\n[ Configuration ]", file=buf)
    for k, v in safe_cfg.items():
        print(f"  {k:35s} = {v}", file=buf)

    print("\n[ Architecture ]", file=buf)
    for k, v in payload["architecture"].items():
        print(f"  {k:35s} = {v}", file=buf)

    print("\n[ Parameter counts ]", file=buf)
    print(f"  Generator     : total={g_total:>10,}  trainable={g_train:>10,}", file=buf)
    print(f"  Discriminator : total={d_total:>10,}  trainable={d_train:>10,}", file=buf)
    print(f"  COMBINED      : total={g_total + d_total:>10,}  "
          f"trainable={g_train + d_train:>10,}", file=buf)

    print("\n[ Training ]", file=buf)
    for k, v in payload["training"].items():
        if k == "elapsed_sec" and v is not None:
            print(f"  {k:35s} = {v:.1f} sec  ({v/60:.2f} min)", file=buf)
        else:
            print(f"  {k:35s} = {v}", file=buf)

    print("\n[ Pipeline metadata ]", file=buf)
    pm = payload["pipeline_meta"]
    print(f"  target_column   = {pm['target_column']}", file=buf)
    print(f"  time_column     = {pm['time_column']}", file=buf)
    print(f"  censoring_ratio = {pm['censoring_ratio']:.4f}", file=buf)
    print(f"  n_columns       = {len(pm['columns'])}", file=buf)
    print(f"  columns         = {pm['columns']}", file=buf)

    if "tte_bounds" in payload:
        print("\n[ TTE bounds (used for post-processing of synthetic times) ]", file=buf)
        for k, v in payload["tte_bounds"].items():
            print(f"  {k:15s} = {v}", file=buf)

    print("\n[ Generator (torch.nn) ]", file=buf)
    print(G, file=buf)

    print("\n[ Discriminator (torch.nn) ]", file=buf)
    print(D, file=buf)

    # ----- Save summary -----
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        log.info(f"Saved model summary → {summary_path}  "
                 f"({summary_path.stat().st_size / 1e3:.1f} KB)")
    except Exception:
        log.error("Saving model_summary.txt failed; continuing.")
        log.error(traceback.format_exc())


# =========================================================================== #
#  main                                                                        #
# =========================================================================== #

def main() -> int:
    args = parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train.log"
    setup_logging(log_path)
    log = logging.getLogger("aws_driver")

    log.info("=" * 70)
    log.info("SurvivalGAN AWS training run")
    log.info(f"Args: {vars(args)}")
    log.info(f"Working directory: {os.getcwd()}")
    log.info(f"Python: {sys.version.split()[0]}")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        log.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info("=" * 70)

    seed_everything(args.seed)
    device = setup_device(args.device)

    # ---- Load data ----
    log.info(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    log.info(f"Shape: {df.shape}")
    log.info(f"Columns: {df.columns.tolist()}")
    if args.status_column not in df.columns or args.time_column not in df.columns:
        log.error(f"Input CSV must contain '{args.time_column}' and '{args.status_column}' columns")
        return 2
    log.info(f"Event rate: {df[args.status_column].mean():.3f}")
    log.info(f"Time range: [{df[args.time_column].min():.2f}, {df[args.time_column].max():.2f}]")

    # ---- Resolve metadata JSON path ----                                  (NEW)
    if args.metadata_json is None:
        guess = Path(args.input).parent / "survival_gan_column_metadata.json"
        metadata_json_path = guess if guess.exists() else None
    elif args.metadata_json == "":
        metadata_json_path = None
    else:
        metadata_json_path = Path(args.metadata_json)

    # ---- Build config ----
    cfg = Config()
    cfg.input_csv = args.input
    cfg.output_csv = args.output
    cfg.target_column = args.status_column
    cfg.time_column = args.time_column
    cfg.n_iter = args.n_iter
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    cfg.device = device
    cfg.synthetic_count = args.synthetic_count or len(df)
    log.info(f"Effective config: n_iter={cfg.n_iter}, batch_size={cfg.batch_size}, "
             f"device={cfg.device}, synthetic_count={cfg.synthetic_count}")

    # ---- Train or load ----
    pipeline_path = ckpt_dir / "pipeline.pkl"
    training_elapsed: Optional[float] = None  # set if we actually train

    if args.skip_train or args.regenerate:
        if not pipeline_path.exists():
            log.error(f"--skip-train/--regenerate set but no checkpoint at {pipeline_path}")
            return 3
        log.info(f"Loading pipeline from {pipeline_path} ...")
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        if args.regenerate and getattr(pipeline, "tte_model", None) is not None:
            log.info("Retrofitting TTE model with training T bounds (no retraining)...")
            retrofit_tte_bounds(
                pipeline.tte_model,
                df[args.time_column],
                add_residual_noise=False,
            )
    else:
        pipeline = SurvivalPipeline(cfg, device=device)
        log.info("Starting pipeline.fit() — this runs TTE → GAN → censoring predictor ...")
        t0 = time.time()
        try:
            pipeline.fit(df)
        except KeyboardInterrupt:
            log.warning("Interrupted by user during fit()")
            return 130
        except Exception:
            log.error("fit() crashed; full traceback:")
            log.error(traceback.format_exc())
            return 1
        elapsed = time.time() - t0
        training_elapsed = elapsed
        log.info(f"Training finished in {elapsed/60:.1f} min ({elapsed:.1f}s)")

        if torch.cuda.is_available():
            log.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        log.info(f"Saving pipeline to {pipeline_path} ...")
        try:
            with open(pipeline_path, "wb") as f:
                pickle.dump(pipeline, f)
            log.info(f"Checkpoint size: {pipeline_path.stat().st_size / 1e6:.1f} MB")
        except Exception:
            log.error("Checkpoint save failed — continuing to generation anyway.")
            log.error(traceback.format_exc())

    # ---- Save model.pt + model_summary.txt ----
    # Done here (after the train/load if-else) so the .pt and summary are
    # always written next to pipeline.pkl, whether we trained fresh, used
    # --skip-train, or used --regenerate.
    save_model_pt_and_summary(pipeline, cfg, ckpt_dir, log,
                              training_elapsed_sec=training_elapsed)

    if args.skip_generate:
        log.info("--skip-generate set; stopping after training.")
        return 0

    # ---- Generate ----
    log.info(f"Generating {cfg.synthetic_count} synthetic rows ...")
    t0 = time.time()
    try:
        synthetic_df = pipeline.generate(cfg.synthetic_count)
    except Exception:
        log.error("generate() crashed; full traceback:")
        log.error(traceback.format_exc())
        return 1
    log.info(f"Generation finished in {time.time() - t0:.1f}s")
    log.info(f"Synthetic shape: {synthetic_df.shape}")

    # ---- Post-generation time-column sanity check + optional clip ----
    tcol = args.time_column
    if tcol in synthetic_df.columns and tcol in df.columns:
        r_lo, r_hi = float(df[tcol].min()), float(df[tcol].max())
        s = synthetic_df[tcol]
        s_lo, s_hi = float(s.min()), float(s.max())
        q = s.quantile([0.01, 0.5, 0.99]).to_list()
        n_below = int((s < r_lo).sum())
        n_above = int((s > r_hi).sum())
        log.info(
            f"Synthetic {tcol}: real=[{r_lo:.3f}, {r_hi:.3f}]  "
            f"syn=[{s_lo:.3f}, {s_hi:.3f}]  "
            f"syn q[1/50/99]={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}  "
            f"out-of-range: {n_below} below / {n_above} above"
        )
        if n_below + n_above > 0.1 * len(s):
            log.warning(
                "More than 10% of synthetic times are outside the real range. "
                "This usually means the loaded checkpoint was trained before "
                "the TTE clamp fix. Re-run with --regenerate to retrofit, or "
                "leave --clip-output-time on to sanitize the CSV."
            )

        if args.clip_output_time:
            clipped = s.clip(lower=max(r_lo, 1e-3), upper=r_hi)
            if (clipped != s).any():
                log.info(
                    f"Clipping {(clipped != s).sum()} synthetic time values into "
                    f"[{r_lo:.3f}, {r_hi:.3f}] before saving."
                )
                synthetic_df[tcol] = clipped

    # ---- Post-generation precision matching ----
    if args.match_real_precision:
        log.info("Matching synthetic precision to real...")
        skip = {args.time_column, args.status_column}
        synthetic_df, prec_report = match_real_precision(
            df, synthetic_df,
            unique_threshold=args.precision_unique_threshold,
            skip_cols=skip,
        )
        log.info("=== Precision match report ===")
        log.info(f"  {'column':<40} {'real_nuniq':>10} {'syn_before':>10} "
                 f"{'syn_after':>10}  method")
        for col, info in prec_report.items():
            log.info(
                f"  {col:<40} {info['real_nuniq']:>10} "
                f"{info['syn_nuniq_before']:>10} {info['syn_nuniq_after']:>10}  "
                f"{info['method']}"
            )

    # ---- Post-generation physical constraint enforcement ----            (NEW)
    # Do this AFTER precision matching: snap to real values first, then clip
    # any remaining out-of-bound values and cast integer columns.
    if args.enforce_constraints:
        log.info("Enforcing physical constraints (bounds + integer dtype)...")
        constraints = _load_constraint_map(metadata_json_path, df, log)
        synthetic_df, constr_report = enforce_physical_constraints(
            df, synthetic_df, constraints, log,
        )
        log.info("=== Physical constraint report ===")
        log.info(f"  {'column':<40} {'dtype':>6} {'bounds':<22} "
                 f"{'n_below':>8} {'n_above':>8} {'n_rounded':>10} "
                 f"{'pct_mod':>7}  source")
        for col, info in constr_report.items():
            lo, hi = info["bounds"]
            bstr = f"[{'None' if lo is None else f'{lo:g}'}, " \
                   f"{'None' if hi is None else f'{hi:g}'}]"
            log.info(
                f"  {col:<40} {info['dtype']:>6} {bstr:<22} "
                f"{info['n_below']:>8} {info['n_above']:>8} "
                f"{info['n_rounded']:>10} {info['pct_modified']:>6.2f}%  "
                f"{info['source']}"
            )
        total_mod = sum(
            v["n_below"] + v["n_above"] + v["n_rounded"] for v in constr_report.values()
        )
        if total_mod > 0:
            log.warning(
                f"{total_mod} synthetic values were clipped or rounded to "
                f"satisfy physical constraints. Check the report above — "
                f"a high percentage for any column suggests the GAN is "
                f"struggling to model that feature's support."
            )

    # ---- Save ----
    synthetic_df.to_csv(args.output, index=False)
    log.info(f"Wrote synthetic data to {args.output}")

    # ---- Quick real vs synthetic comparison ----
    log.info("=== Real vs Synthetic means (numeric cols) ===")
    for col in df.columns:
        if col not in synthetic_df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        r = df[col].mean()
        s = synthetic_df[col].mean()
        rel_diff = abs(r - s) / (abs(r) + 1e-8) * 100
        log.info(f"  {col:30s}  real={r:10.3f}  synth={s:10.3f}  diff={rel_diff:5.1f}%")

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())