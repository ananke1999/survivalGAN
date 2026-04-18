"""
MSK-IMPACT 50k → Survival GAN Cleaning Pipeline
================================================

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# The random seed
random_seed = 42

# Set random seed in numpy
np.random.seed(random_seed)

# ── 0. LOADING DATA ───────────────────────────────────────────────────────────
RAW_PATH = "data sets/msk_impact 50K.csv"

df = pd.read_csv(RAW_PATH, low_memory=False)
df.columns = df.columns.str.strip()
print(f"Raw shape: {df.shape}")


# ── 1. REQUIRE SURVIVAL LABELS ────────────────────────────────────────────────
# Both OS fields must be present — rows without them are unusable for survival.
df = df.dropna(subset=["Overall Survival (Months)", "Overall Survival Status"])

# Parsing event indicator: "1:DECEASED" → 1 | "0:LIVING" → 0
df["status"] = df["Overall Survival Status"].str.startswith("1").astype(int)
df["time"]   = df["Overall Survival (Months)"].astype(float)

# Drop raw OS columns — replaced by time / status
df = df.drop(columns=["Overall Survival (Months)", "Overall Survival Status"])

# Dropping biologically impossible values (must have time > 0)
df = df[(df["time"] > 0) & (df["time"] < 600)].reset_index(drop=True)
print(f"After survival filter: {df.shape}")


# ── 2. DE-DUPLICATE TO PATIENT LEVEL ─────────────────────────────────────────
# Some patients have multiple samples. Keep the PRIMARY sample when available,
# otherwise the first sample encountered.
def pick_sample(group):
    primary = group[group["Sample Type"] == "Primary"]
    return primary.iloc[0] if not primary.empty else group.iloc[0]

df = df.groupby("Patient ID", group_keys=False).apply(pick_sample)
df = df.reset_index(drop=True)
print(f"After patient-level de-duplication: {df.shape}")


# ── 3. SPLITTING THE DATA ─────────────────────────────────────────────────────
# Split BEFORE any imputation or encoding to prevent leakage.
# 60% train | 20% val | 20% test — stratified on status.

df_train, df_temp = train_test_split(
    df, train_size=0.6, random_state=random_seed, stratify=df["status"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=random_seed, stratify=df_temp["status"]
)

df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

print(pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=["# rows", "# columns"]))
print(pd.DataFrame([[df_val.shape[0],   df_val.shape[1]]],   columns=["# rows", "# columns"]))
print(pd.DataFrame([[df_test.shape[0],  df_test.shape[1]]],  columns=["# rows", "# columns"]))

# Verify event rate is preserved across all three splits
for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
    print(f"  {name} event rate: {split['status'].mean():.1%}")


# ── 4. HANDLING IDENTIFIERS ───────────────────────────────────────────────────
# Droping ID columns from all splits — they must never be model inputs.
# Also droping Sample Type (used for dedup only) and other non-feature columns.
drop_id_cols = [
    "Patient ID", "Sample ID", "Study ID",
    "Sample Type",                          # used for dedup, not a feature
    "Gene Panel",                           # assay label, not a patient feature
    "FACETS Suite Version",                 # version string
    "Somatic Status",                       # near-constant (~100% Matched)
    "Purity Estimate from Mutations",       # binary flag, low info
    "HLA Genotype Available",               # derived flag
    "Number of Samples Per Patient",        # dedup artifact
    "Oncotree Code",                        # redundant with Cancer Type
    "Cancer Type Detailed",                 # too high cardinality
]

df_train = df_train.drop(columns=drop_id_cols, errors="ignore")
df_val   = df_val.drop(columns=drop_id_cols,   errors="ignore")
df_test  = df_test.drop(columns=drop_id_cols,  errors="ignore")


# ── 5. HANDLING MISSING VALUES ────────────────────────────────────────────────

def nan_checker(df):
    """
    The NaN checker

    Parameters
    ----------
    df : the dataframe

    Returns
    ----------
    The dataframe of variables with NaN, their proportion of NaN and data type
    """
    # NOTE: dtype stored as string so .isin(['float64']) comparisons work.
    # Storing the raw dtype object causes isin() to fail silently because
    # dtype('float64') object != 'float64' string inside a pandas Series.
    df_nan = pd.DataFrame(
        [[var, df[var].isna().sum() / df.shape[0], str(df[var].dtype)]
         for var in df.columns if df[var].isna().sum() > 0],
        columns=["var", "proportion", "dtype"]
    )
    df_nan = df_nan.sort_values(by="proportion", ascending=False).reset_index(drop=True)
    return df_nan

# Run nan_checker on train only (do NOT concat — that causes leakage)
df_nan = nan_checker(df_train)
print("\nMissing values in train:")
print(df_nan)
print(pd.DataFrame(df_nan["dtype"].unique(), columns=["dtype"]))


# ── 5a. DROP COLUMNS WITH > 40% MISSING (based on train) ─────────────────────
threshold = 0.40
cols_to_drop = df_nan[df_nan["proportion"] > threshold]["var"].tolist()
print(f"\nColumns dropped (>40% missing): {cols_to_drop}")

df_train = df_train.drop(columns=cols_to_drop, errors="ignore")
df_val   = df_val.drop(columns=cols_to_drop,   errors="ignore")
df_test  = df_test.drop(columns=cols_to_drop,  errors="ignore")


# ── 5b. TYPE COERCIONS ────────────────────────────────────────────────────────
# Pathologist Estimated Tumor Purity is stored as string ("50", "20" …)
# → convert to float before imputation.
# Whole Genome Doubling Status is mixed bool/object → normalise to string.

def coerce_types(split):
    split = split.copy()
    if "Pathologist Estimated Tumor Purity" in split.columns:
        split["Pathologist Estimated Tumor Purity"] = pd.to_numeric(
            split["Pathologist Estimated Tumor Purity"], errors="coerce"
        )
    if "Whole Genome Doubling Status (FACETS)" in split.columns:
        split["Whole Genome Doubling Status (FACETS)"] = (
            split["Whole Genome Doubling Status (FACETS)"]
            .astype(str)
            .replace("nan", np.nan)
        )
    return split

df_train = coerce_types(df_train)
df_val   = coerce_types(df_val)
df_test  = coerce_types(df_test)


# ── 5c. RE-CHECKING MISSING AFTER DROPPING HIGH-MISSINGNESS COLS ────────────────
# Run nan_checker again on train to get the updated missing value list.
df_nan = nan_checker(df_train)
print("\nMissing values after dropping >40% columns:")
print(df_nan)


# ── 5d. IMPUTE NUMERIC COLUMNS — FIT ON TRAIN ONLY ───────────────────────────
# Use median (more robust than mean for skewed clinical data).
# NOTE: dtype check must include 'str' columns that were coerced above.

df_miss_num = df_nan[df_nan["dtype"].isin(["float64", "int64", "Float64", "Int64", "float32"])].reset_index(drop=True)
print("\nNumeric columns to impute:")
print(df_miss_num)

if len(df_miss_num) > 0:
    num_vars = df_miss_num["var"].tolist()
    si_num = SimpleImputer(missing_values=np.nan, strategy="median")
    df_train[num_vars] = si_num.fit_transform(df_train[num_vars])
    df_val[num_vars]   = si_num.transform(df_val[num_vars])
    df_test[num_vars]  = si_num.transform(df_test[num_vars])


# ── 5e. IMPUTE CATEGORICAL COLUMNS — FIT ON TRAIN ONLY ───────────────────────
# Fill missing categoricals with "Unknown" sentinel.
# This includes str-typed columns and object-typed columns.

df_miss_cat = df_nan[df_nan["dtype"].isin(["object", "str", "string"])].reset_index(drop=True)
print("\nCategorical columns to impute:")
print(df_miss_cat)

if len(df_miss_cat) > 0:
    cat_vars = df_miss_cat["var"].tolist()
    si_cat = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Unknown")
    df_train[cat_vars] = si_cat.fit_transform(df_train[cat_vars])
    df_val[cat_vars]   = si_cat.transform(df_val[cat_vars])
    df_test[cat_vars]  = si_cat.transform(df_test[cat_vars])


# ── 5f. CLIP OUTLIERS — USING TRAIN QUANTILES ONLY ───────────────────────────
# TMB Score and Mutation Count have extreme right tails and occasional
# negative QC artefacts. Fit bounds on train only.

clip_cols = ["TMB Score", "Mutation Count", "MSI Score"]
for col in clip_cols:
    if col not in df_train.columns:
        continue
    lo  = 0.0
    hi  = df_train[col].quantile(0.99)
    for split in [df_train, df_val, df_test]:
        split[col] = split[col].clip(lower=lo, upper=hi)
    print(f"  Clipped '{col}' to [0, {hi:.2f}] using train p99")


# ── 6. CATEGORICAL VARIABLE CHECKER ──────────────────────────────────────────

def cat_var_checker(df, target_cols):
    """
    The categorical variable checker

    Parameters
    ----------
    df         : the dataframe
    target_cols: list of target column names to exclude

    Returns
    ----------
    The dataframe of categorical variables and their number of unique values
    """
    # Including both 'object' and 'str'/'string' dtypes
    df_cat = pd.DataFrame(
        [[var, df[var].nunique(dropna=False)]
         for var in df.columns
         if var not in target_cols and df[var].dtype in ["object", "str", "string"]],
        columns=["var", "nunique"]
    )
    df_cat = df_cat.sort_values(by="nunique", ascending=False).reset_index(drop=True)
    return df_cat

target = ["time", "status"]
df_cat = cat_var_checker(df_train, target_cols=target)
print("\nCategorical variables:")
print(df_cat)


# ── 7. ENCODING ───────────────────────────────────────────────────────────────
# Using Label Encoding (not One-Hot) for survival GANs.
# One-Hot creates too many columns for high-cardinality features like
# Primary Site (200+ categories) and Cancer Type (60+ categories),
# which makes the GAN input space extremely sparse and hard to train.
#
# Label Encoding keeps dimensionality manageable.
# Fit encoder on train only; apply same mapping to val and test.
# Unseen categories in val/test are mapped to a safe fallback.

cat_cols = df_cat["var"].tolist()

encoders = {}
for col in cat_cols:
    for split in [df_train, df_val, df_test]:
        split[col] = split[col].astype(str)

    le = LabelEncoder()
    le.fit(df_train[col])
    encoders[col] = le

    train_classes = set(le.classes_)
    fallback = le.classes_[0]   # saving default for unseen categories

    for split in [df_train, df_val, df_test]:
        split[col] = split[col].apply(
            lambda x: x if x in train_classes else fallback
        )
        split[col] = le.transform(split[col])

    print(f"  Label encoded '{col}' → {len(le.classes_)} classes")


# ── 8. FINAL QC ───────────────────────────────────────────────────────────────
print("\n── Final QC ──")
for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
    assert split.isnull().sum().sum() == 0,       f"{name}: NaNs remaining!"
    assert (split["time"] > 0).all(),              f"{name}: non-positive times!"
    assert split["status"].isin([0, 1]).all(),     f"{name}: bad status codes!"
    print(f"  {name}: {split.shape} | event rate: {split['status'].mean():.1%} ✓")


# ── 9. SAVE ───────────────────────────────────────────────────────────────────
df_train.to_csv("data sets/survival_gan_train.csv", index=False)
df_val.to_csv("data sets/survival_gan_val.csv",     index=False)
df_test.to_csv("data sets/survival_gan_test.csv",   index=False)

print("\nSaved:")
print("  data sets/survival_gan_train.csv")
print("  data sets/survival_gan_val.csv")
print("  data sets/survival_gan_test.csv")
print(f"\nFinal columns ({len(df_train.columns)}): {df_train.columns.tolist()}")