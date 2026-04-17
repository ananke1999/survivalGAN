"""
MSK-IMPACT 50k → Survival GAN Cleaning Pipeline
================================================

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# The random seed
random_seed = 42



# Set random seed in numpy
np.random.seed(random_seed)

# ── 0. LOADING DATA ──────────────────────────────────────────────────────────────────
RAW_PATH  = "data sets/msk_impact 50K.csv"
OUT_PATH  = "msk_impact_50k_survival_gan_clean.csv"
 
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Raw shape: {df.shape}")   # (54 331, 45)


# ── 1. REQUIRE SURVIVAL LABELS  ──────────────────────────────────────────────
# Both OS fields must be present — rows without them are unusable for survival.
df = df.dropna(subset=["Overall Survival (Months)", "Overall Survival Status"])
 
# Parse event indicator: "1:DECEASED" → 1 | "0:LIVING" → 0
df["status"] = df["Overall Survival Status"].str.startswith("1").astype(int)
df["time"]      = df["Overall Survival (Months)"].astype(float)
 
# Drop biologically impossible values
df = df.dropna(subset=["time"]).copy()
df = df[(df["time"] > 0) & (df["time"] < 600)].copy()

df = df.drop(columns=["Overall Survival (Months)", "Overall Survival Status"])
print(f"After survival filter: {df.shape}")


# ── 2. DE-DUPLICATE TO PATIENT LEVEL ─────────────────────────────────────────
# Some patients have multiple samples.  Keep the PRIMARY sample when available,
# otherwise the first sample encountered (already sorted by Sample ID).
def pick_sample(group):
    primary = group[group["Sample Type"] == "Primary"]
    return primary.iloc[0] if not primary.empty else group.iloc[0]
 
df = df.groupby("Patient ID", group_keys=False).apply(pick_sample)
df = df.reset_index(drop=True)
print(f"After patient-level de-duplication: {df.shape}")


# ── 3. Splitting the data ────────────────────────────
# The code below shows how to divide the  data into training (60%) ,validation (20%) and testing (20%)

from sklearn.model_selection import train_test_split

# Divide the training data into training (60%) , temp (40%)
df_train, df_temp = train_test_split(df, train_size=0.6, random_state=random_seed ,stratify=df["status"])

# Second split: split temp into validation and test (20% each)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=random_seed ,stratify=df_temp["status"])

# Reset the index
df_train, df_val , df_test = df_train.reset_index(drop=True), df_val.reset_index(drop=True) , df_test.reset_index(drop=True)

# Print the dimension of df_train
print(pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns']))

# Print the dimension of df_val
print(pd.DataFrame([[df_val.shape[0], df_val.shape[1]]], columns=['# rows', '# columns']))

# Print the dimension of df_test
print(pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns']))


# ── 4. Handling Idefiers ────────────────────────────        
# Removing Identifiers

drop_cols = ["Patient ID", "Sample ID", "Study ID"]

df_train = df_train.drop(columns=drop_cols, errors="ignore")
df_val = df_val.drop(columns=drop_cols, errors="ignore")
df_test = df_test.drop(columns=drop_cols, errors="ignore")

# ── 5. Handling Missing Values ────────────────────────────  
# Combine df_train, df_val and df_test
df = pd.concat([df_train, df_val, df_test], sort=False)


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

    # Get the dataframe of variables with NaN, their proportion of NaN and data type
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])

    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)

    return df_nan

 #Call nan_checker on df
# See the implementation in pmlm_utilities.ipynb
df_nan = nan_checker(df)

# Print df_nan
print(df_nan)

# Print the unique data type of variables with NaN
print(pd.DataFrame(df_nan['dtype'].unique(), columns=['dtype']))

threshold = 0.40
cols_to_drop = df_nan[df_nan['proportion'] > threshold]['var'].tolist()
print(f"Columns dropped (>40% missing): {cols_to_drop}")

# Drop from all sets (Train, Val, Test)
df_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
df_val.drop(columns=cols_to_drop, inplace=True, errors='ignore')
df_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Step B: Separate columns by type for imputation
# Combine remaining columns to check types
df_combined = pd.concat([df_train, df_val, df_test], sort=False)