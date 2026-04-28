"""
CTGAN Baseline Training Script
================================
Baseline comparison model for SurvivalGAN project.
Paper: Norcliffe et al. (2023) - SurvivalGAN

Dataset: MSK-IMPACT 50k
Split: 80% train / 20% test (matches preprocessing.py)

Usage:
    python train_ctgan.py

Output:
    models/ctgan_model.pkl         <- saved model (for frontend)
    data_sets/synthetic_ctgan.csv  <- synthetic patients (for evaluation)
"""

import pandas as pd
import numpy as np
import pickle
import os
from ctgan import CTGAN

# ── 0. LOAD DATA ──────────────────────────────────────────────────────────────
df_train = pd.read_csv("data_sets/survival_gan_train.csv")
df_test  = pd.read_csv("data_sets/survival_gan_test.csv")

print(f"Train shape: {df_train.shape}")
print(f"Test shape:  {df_test.shape}")
print(f"Train event rate: {df_train['status'].mean():.1%}")
print(f"\nColumns: {df_train.columns.tolist()}")

# ── 1. DEFINE DISCRETE COLUMNS ────────────────────────────────────────────────
# Categorical columns — label encoded in preprocessing.py
# Integer columns     — treated as discrete by CTGAN
# status              — event indicator 0 or 1

discrete_columns = [
    # Categorical (label encoded)
    "Cancer Type",
    "Genetic Ancestry",
    "Disease Status",
    "FACETS QC",
    "MSI Type",
    "Sex",
    "Whole Genome Doubling Status (FACETS)",
    # Integer valued
    "Age at Diagnosis",
    "Mutation Count",
    "Sample coverage",
    "Number of Other Cancer Types",
    # Target
    "status",
]

# Keep only columns that actually exist in the dataframe
discrete_columns = [c for c in discrete_columns if c in df_train.columns]
print(f"\nDiscrete columns ({len(discrete_columns)}): {discrete_columns}")

# Continuous columns — everything else except time
continuous_columns = [
    c for c in df_train.columns
    if c not in discrete_columns and c != "time"
]
print(f"Continuous columns ({len(continuous_columns)}): {continuous_columns}")

# ── 2. TRAIN CTGAN ────────────────────────────────────────────────────────────
print("\nTraining CTGAN...")
print("This will take approximately 6 hours on AWS...")

ctgan = CTGAN(
    epochs=500,
    batch_size=500,
    verbose=True
)

ctgan.fit(df_train, discrete_columns=discrete_columns)
print("\nCTGAN training complete!")

# ── 3. SAVE THE TRAINED MODEL ─────────────────────────────────────────────────
# Save model to models/ folder — separate from data and code.
# Standard ML project structure:
#   data_sets/ → CSV data files
#   models/    → saved model files  ← model goes here
#   code/      → Python scripts
#
# pkl is used because CTGAN is a Python object containing
# PyTorch inside — not a pure PyTorch model.
# .pt is for pure PyTorch models only.
#
# The frontend backend loads this file to generate
# new synthetic patients on demand without retraining.

os.makedirs("models", exist_ok=True)
model_path = "models/ctgan_model.pkl"

print(f"\nSaving trained CTGAN model to models/ folder...")
with open(model_path, "wb") as f:
    pickle.dump(ctgan, f)

file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"Model saved → {model_path}")
print(f"File size:     {file_size_mb:.1f} MB")

# ── 4. GENERATE SYNTHETIC PATIENTS ───────────────────────────────────────────
n_synthetic = len(df_test)
print(f"\nGenerating {n_synthetic} synthetic patients...")

synthetic_ctgan = ctgan.sample(n_synthetic)
print(f"Synthetic shape: {synthetic_ctgan.shape}")

# ── 5. QUICK SANITY CHECK ─────────────────────────────────────────────────────
print("\n── Sanity Check ──")
print(f"Real    event rate:     {df_test['status'].mean():.1%}")
print(f"Synthetic event rate:   {synthetic_ctgan['status'].mean():.1%}")

print(f"\nReal    time — min: {df_test['time'].min():.1f}  "
      f"max: {df_test['time'].max():.1f}")
print(f"Synthetic time — min: {synthetic_ctgan['time'].min():.1f}  "
      f"max: {synthetic_ctgan['time'].max():.1f}")

print(f"\nReal    Age mean:       {df_test['Age at Diagnosis'].mean():.1f}")
print(f"Synthetic Age mean:     {synthetic_ctgan['Age at Diagnosis'].mean():.1f}")

print(f"\nReal    TMB mean:       {df_test['TMB Score'].mean():.2f}")
print(f"Synthetic TMB mean:     {synthetic_ctgan['TMB Score'].mean():.2f}")

print(f"\nReal    Mutation mean:  {df_test['Mutation Count'].mean():.1f}")
print(f"Synthetic Mutation mean:{synthetic_ctgan['Mutation Count'].mean():.1f}")

# ── 6. SAVE SYNTHETIC DATA ────────────────────────────────────────────────────
output_path = "data_sets/synthetic_ctgan.csv"
synthetic_ctgan.to_csv(output_path, index=False)
print(f"\nSaved synthetic data → {output_path}")

# ── 7. VERIFY SAVED MODEL WORKS ───────────────────────────────────────────────
# Load the saved model back and generate a small test
# to confirm pkl file is valid and not corrupted
print("\nVerifying saved model loads correctly...")
with open(model_path, "rb") as f:
    ctgan_loaded = pickle.load(f)

test_sample = ctgan_loaded.sample(10)
assert len(test_sample) == 10, "Model verification failed!"
print(f"Verification passed ✓")
print(f"Model loads correctly and generates patients")

# ── 8. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("TRAINING COMPLETE — SUMMARY")
print("="*55)
print(f"  Model saved:          models/ctgan_model.pkl")
print(f"  Model size:           {file_size_mb:.1f} MB")
print(f"  Synthetic data saved: data_sets/synthetic_ctgan.csv")
print(f"  Synthetic patients:   {len(synthetic_ctgan)}")
print(f"  Real event rate:      {df_test['status'].mean():.1%}")
print(f"  Synthetic event rate: {synthetic_ctgan['status'].mean():.1%}")
print("="*55)

print("""
HOW FRONTEND LOADS THIS MODEL:

  import pickle
  from ctgan import CTGAN

  # Load saved model — no retraining needed
  with open('models/ctgan_model.pkl', 'rb') as f:
      ctgan = pickle.load(f)

  # Generate patients instantly
  synthetic = ctgan.sample(100)
""")