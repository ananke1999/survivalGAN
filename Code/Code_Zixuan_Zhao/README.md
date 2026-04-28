# SurvivalGAN on MSK-IMPACT 50K

A reproduction and extension of **SurvivalGAN** (Norcliffe et al., AISTATS 2023) applied to the MSK-IMPACT 50K oncology cohort. The pipeline trains a WGAN-GP–based tabular generator with a separate time-to-event (TTE) head (DeepHit + XGBoost) to produce synthetic survival data, and benchmarks it against baseline tabular generators (CTGAN, TVAE, AdsGAN, NFlows) on covariate fidelity, survival-specific metrics, and downstream predictive performance.

## Research Questions

1. Can SurvivalGAN generate faithful synthetic survival data on a large-scale, high-dimensional real-world oncology dataset (MSK-IMPACT 50K), and does it outperform baseline tabular generators on covariate quality (JSD, Wasserstein), survival-specific metrics (Optimism, KM Divergence, Short-sightedness), and downstream predictive performance?
2. Does SurvivalGAN's performance vary across cancer types within the MSK dataset, and can conditioning on cancer type improve synthetic data quality for underrepresented subtypes?

---

## Files in this directory

### Source code

| File | Role |
| --- | --- |
| `preprocessing.py` | Cleans the raw MSK-IMPACT 50K CSV. Drops high-missingness and high-cardinality columns (HLA alleles, Primary Site, Oncotree Code), buckets Cancer Type to top-15 + "Other", de-duplicates to patient level (preferring Primary samples), clips outliers and physically-impossible values, restores integer dtype after imputation, label-encodes categoricals, and writes an 80/20 train/test split + a column-metadata JSON used downstream for constraint enforcement. |
| `survGAN.py` | Standalone SurvivalGAN implementation. Contains the `Config` dataclass, the WGAN-GP `GAN` + `MLP` backbone, the `TabularGAN` wrapper (BayesianGMM-based continuous encoding + one-hot for discrete), a local two-stage TTE model (`LocalSurvivalFunctionTTE`: DeepHit → XGBRegressor on log-time), and the orchestrating `SurvivalPipeline` class. Can be run directly as a script via its own `main()`. |
| `train_survGAN_aws.py` | Thin CLI driver around `SurvivalPipeline` for EC2 / GPU runs. Adds file logging, GPU-memory reporting, pickled checkpointing, post-generation time clipping, real-precision matching (snap synthetic numerics to real unique values), and physical-constraint enforcement (clip to bounds, cast integers) using the metadata JSON written by `preprocessing.py`. |
| `evaluate_synthetic_data.py` | Publication-quality evaluator. Wraps synthcity's `Metrics` runner (JSD, Wasserstein, PRDC, α-precision, detection AUC, MMD, downstream C-Index via XGB) and adds custom survival-specific metrics from the SurvivalGAN paper (Optimism, KM Divergence, Short-sightedness). Supports a single `--synthetic` CSV or a `--synthetic-dir` for cross-method comparison and produces plots (KM curves, t-SNE, correlation panels, headline dashboard). |

### Trained artifacts

| File | Description |
| --- | --- |
| `model.pt` | PyTorch state dict of the trained Generator + Discriminator (~685k params combined). |
| `pipeline.pkl` | Pickled fitted `SurvivalPipeline` (GAN + DeepHit + XGB time regressor + XGB censoring predictor + encoders). Load with `pickle.load` and call `.generate(n)` to produce more synthetic rows without retraining. |
| `model_summary.txt` | Human-readable summary of the trained model: config, architecture, parameter counts, training time, TTE log-time bounds, residual std, and full `torch.nn` module strings. |
| `train.log` | Full training log (file handler from `train_survGAN_aws.py`). |

---

## Pipeline overview

```
   raw MSK-IMPACT CSV
            │
            ▼
  ┌──────────────────────┐
  │   preprocessing.py   │   80/20 split, imputation, encoding
  └──────────────────────┘
            │
            ├──▶ survival_gan_train.csv     (34,938 rows × 19 cols)
            ├──▶ survival_gan_test.csv      (held-out)
            └──▶ survival_gan_column_metadata.json
                      │
                      ▼
            ┌──────────────────────────┐
            │  train_survGAN_aws.py    │   imports SurvivalPipeline from survGAN.py
            │   ├── DeepHit            │     S(t | X)
            │   ├── XGBRegressor       │     log T from [X, S(t|X), E]
            │   ├── TabularGAN (WGAN-GP) │   covariates + E + placeholder T
            │   └── XGBClassifier      │     P(censoring | X)
            └──────────────────────────┘
                      │
                      ├──▶ pipeline.pkl, model.pt, model_summary.txt, train.log
                      └──▶ synthetic.csv  (34,938 rows by default)
                                   │
                                   ▼
                      ┌──────────────────────────────┐
                      │  evaluate_synthetic_data.py  │
                      └──────────────────────────────┘
                                   │
                                   ▼
                      eval_results/  (metrics CSV, plots, report)
```

---

## Quick start

### 1. Preprocess

```bash
python preprocessing.py
# writes:  data sets/survival_gan_train.csv
#          data sets/survival_gan_test.csv
#          data sets/survival_gan_column_metadata.json
```

### 2. Train + generate

```bash
python train_survGAN_aws.py \
    --input "data sets/survival_gan_train.csv" \
    --output synthetic.csv \
    --checkpoint-dir ckpt \
    --n-iter 3000 \
    --batch-size 256 \
    --seed 99
```

Useful flags:
- `--skip-train` — load `ckpt/pipeline.pkl` and only generate.
- `--regenerate` — load checkpoint and retrofit the TTE clamp bounds (no retraining) before generating. Useful if the checkpoint was trained before the TTE clamp fix.
- `--no-enforce-constraints` / `--no-match-real-precision` — disable post-processing.

### 3. Evaluate

Single synthetic CSV against real train/test:

```bash
python evaluate_synthetic_data.py \
    --real-train "data sets/survival_gan_train.csv" \
    --real-test  "data sets/survival_gan_test.csv" \
    --synthetic  synthetic.csv \
    --outdir eval_results
```

Cross-method comparison — drop one CSV per method into a directory:

```bash
python evaluate_synthetic_data.py \
    --real-train "data sets/survival_gan_train.csv" \
    --real-test  "data sets/survival_gan_test.csv" \
    --synthetic-dir all_methods/ \
    --outdir eval_results
```

`all_methods/` should contain e.g. `survgan.csv`, `ctgan.csv`, `tvae.csv`, `adsgan.csv`, `nflows.csv`. Add `--no-heavy` if you want to skip the O(n²) metrics (MMD, Wasserstein, PRDC, α-precision) for a fast pass.

---

## Trained-checkpoint summary

Values read from `model_summary.txt` (run saved 2026-04-28).

| | |
| --- | --- |
| Training set | 34,938 patients × 19 columns (after preprocessing) |
| Censoring ratio | 0.6138 (event rate ≈ 38.6%) |
| Time range | [0.033, 90.838] months |
| `n_iter` × `batch_size` | 3,000 × 256 |
| Generator | 3 residual hidden layers × 250 units, tanh, dropout 0.1, lr 1e-3 |
| Discriminator | 2 hidden layers × 250 units, leaky_relu, dropout 0.1, n_critic = 5 |
| Latent / conditional dim | 250 / 51 |
| Gradient penalty / identifiability | λ = 10.0 / 0.1 |
| TTE strategy | `survival_function` (DeepHit + XGBRegressor on log-T) |
| Censoring strategy | `covariate_dependent` (XGBClassifier predicts E from X) |
| Survival conditional | enabled (BinEncoder over [T, E, X]) |
| Parameter count | Generator 571,800 + Discriminator 113,501 = **685,301** |
| Training time | 24,133 s ≈ **6.7 h** on a single CUDA GPU |
| Seed | 99 |

The pickled pipeline is self-contained: encoders, the DeepHit model, the XGBoost time/censoring models, and the GAN weights all travel together.

---

## Environment

Tested with:

- **Python 3.10** (3.12+ has known incompatibilities in this stack)
- PyTorch with CUDA (training was performed on CUDA; CPU works but is slow)
- `synthcity` (provides `SurvivalAnalysisDataLoader`, the DeepHit template, and the metric runner)
- `lifelines`, `scikit-learn`, `xgboost`, `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`

Recommended setup:

```bash
conda create -n survivalgan python=3.10 -y
conda activate survivalgan
pip install synthcity lifelines xgboost scikit-learn scipy pandas matplotlib tqdm
# pin opacus to 1.4.0 to avoid the torch.nn.RMSNorm AttributeError
# under synthcity's torch<2.3 constraint
pip install "opacus==1.4.0"
```

If you intend to modify the SurvivalGAN plugin itself rather than use this standalone implementation, do an editable install of synthcity from a local clone and edit `plugin_survival_gan.py` / the TTE model directory there.

---

## Notes & known issues

- `preprocessing.py` writes `survival_gan_column_metadata.json` next to the CSVs. `train_survGAN_aws.py` auto-discovers it; you can also point at it explicitly with `--metadata-json` or disable it with `--metadata-json ""`.
- A subset of the original raw column names is dropped during preprocessing (`HLA-A/B/C` allele genotypes, `Primary Site`, `Oncotree Code`, `Cancer Type Detailed`, `Gene Panel`, IDs). The 19 surviving columns are listed in the `[Pipeline metadata]` block of `model_summary.txt`.
- The TTE model clamps log-T predictions into `[Tlog_min, Tlog_max]` (with a small margin) to prevent OOD covariates from producing astronomically large or sub-millisecond synthetic times. If you load a checkpoint trained before this fix, pass `--regenerate` to retrofit the bounds.
- `privbayes` is excluded from the baseline list when running cross-method comparisons on this dataset due to memory pressure on 34k × 19.

---

## References

- Norcliffe, A. et al. *SurvivalGAN: Generating Time-to-Event Data for Survival Analysis*. AISTATS 2023.
- Qian, Z., Cebere, B.-C., van der Schaar, M. *Synthcity: facilitating innovative use cases of synthetic data in different data modalities*. 2023.
- Cerami, E. et al. *The cBio Cancer Genomics Portal* (MSK-IMPACT cohort source).
