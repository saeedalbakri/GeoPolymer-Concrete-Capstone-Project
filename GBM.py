#libraries that we gnna need for this ML
import re
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#Hides some noisy user warnings so the console is cleaner.
warnings.filterwarnings("ignore", category=UserWarning)

#Input workbook and where we want to save the filled result.
INPUT_XLSX = "Polymer_Data.xlsx"
OUTPUT_XLSX = "Output_Data_.xlsx"
RANDOM_STATE = 0 #Keeps train/test split repeatable; 
TEST_SIZE = 0.25 #uses 25% of the observed data as a quick test.

SHEETS = ["Polymer 1", "Polymer 2", "Polymer 3"]#The sheet names to process (“Polymer 1/2/3”)

#ow many decimals to round to for each target. Keep predicted vaules lookin nice
FIXED_DECIMALS = {
    "initial setting time": 0,
    "final setting time": 0,
    "3 days": 0,
    "7 days": 0,
    "28 days": 0,
    "at 170 degrees f after 24 hours": 0,
    "flexural strength": 1,
}

#If you set ROUND_KEEP_FLOAT to False, integer like targets would be stored as integer type instead of float.
ROUND_KEEP_FLOAT = True

#few regex patterns that match possible ways the columns are named (e.g., “raw material a”, “flyash c class”, “initial setting time”, “3 days”, etc.).
COMPOSITION_PATTERNS = {
    "comp_A": [
        r"\braw\s*material\s*a\b",
        r"\braw\s*material\s*a\s*\(slake\)\b",
        r"\bflyash\s*f\s*class\b",
        r"\bflyash\s*c\s*class\b",
    ],
    "comp_B": [r"\braw\s*material\s*b\b"],
    "comp_C": [r"\braw\s*material\s*c\b"],
}
TARGET_PATTERNS = [
    r"\binitial\s*setting\s*time\b",
    r"\bfinal\s*setting\s*time\b",
    r"\b3\s*days\b",
    r"\b7\s*days\b",
    r"\b28\s*days\b",
    r"\bat\s*170\s*degrees?\s*f\s*after\s*24\s*hours\b",
    r"\bflexural\s*strength\b",
]

#hyperparameters so this is the settings we choose for GBM.
GBM_KW = dict(
    n_estimators=300, # small trees to build, one after another. More trees, more learning capacity
    learning_rate=0.05, #How big each step is when adding a new tree. Smaller values make learning more cautious and usually need more trees. Common trade-off: lower learning_rate, raise n_estimators.
    max_depth=3, #How deep each little tree can grow.
    subsample=0.8, # here this Use only 80% of the rows (randomly) for each new tree. This adds randomness (a bit like bagging), typically improving robustness and reducing overfitting. 1.0 would use all rows.
    random_state=RANDOM_STATE, #Seed for the model’s randomness so you get the same results every run (reproducible splits and sampling).
)

def normalize_col(c): #Lowercases, trims, and collapses symbols/spaces so “Raw Material A (Slake)” and “raw material a” both look the same.
    c = str(c).strip().lower()
    c = re.sub(r"[•·_|\-]+", " ", c)
    c = re.sub(r"\s+", " ", c)
    return c

def find_first_match(col_names_norm, patterns):#this scans normalized names and returns the first real column that matches a pattern.
    for pat in patterns:
        rx = re.compile(pat, flags=re.I)
        for real, norm in col_names_norm.items():
            if rx.search(norm):
                return real
    return None

def detect_columns(df): #Uses the functions above to find the 3 composition columns (A, B, C) and a list of target columns (setting times, strengths).
    norm_map = {c: normalize_col(c) for c in df.columns}
    comp_A = find_first_match(norm_map, COMPOSITION_PATTERNS["comp_A"])
    comp_B = find_first_match(norm_map, COMPOSITION_PATTERNS["comp_B"])
    comp_C = find_first_match(norm_map, COMPOSITION_PATTERNS["comp_C"])
    features = [c for c in [comp_A, comp_B, comp_C] if c is not None]

    targets = []
    for pat in TARGET_PATTERNS:
        rx = re.compile(pat, flags=re.I)
        for real, norm in norm_map.items():
            if rx.search(norm) and real not in targets:
                targets.append(real)

    return features, targets

def enforce_numeric(df, cols):#Forces the listed columns to numeric.
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def get_decimals_for(target_name):#Decides how many decimals to keep for a given target
    name = str(target_name).strip().lower()
    for key, d in FIXED_DECIMALS.items():
        if key in name:
            return d
    return None

def apply_rounding_policy(series, target_name):#Rounds the predictions/values according to that rule and returns either float or integer-like values based on ROUND_KEEP_FLOAT.
    dec = get_decimals_for(target_name)
    if dec is None:
        return series
    series = series.round(dec)
    if dec == 0 and not ROUND_KEEP_FLOAT:
        return series.astype("Int64")   # nullable integer keeps NaN
    return series.astype(float)

def train_and_impute_column(df, features, target):
    # features & target already numeric at sheet level
    feat_ok = df[features].notna().all(axis=1)#rows where A, B, and C are all present
    obs = feat_ok & df[target].notna()#rows where target is present (observed)
    miss = feat_ok & df[target].isna()#rows where target is missing (to be imputed)

    metrics = {
        "target": target,
        "n_observed": int(obs.sum()),
        "n_missing": int(miss.sum()),
        "r2_test": np.nan,
        "mae_test": np.nan,
        "rmse_test": np.nan,
    }

    filled = df[target].copy()

    if obs.sum() < 10:
        return apply_rounding_policy(filled, target), metrics

    X = df.loc[obs, features].values
    y = df.loc[obs, target].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = GradientBoostingRegressor(**GBM_KW).fit(Xtr, ytr)

   
    yhat = model.predict(Xte)
    metrics["r2_test"] = float(r2_score(yte, yhat))
    metrics["mae_test"] = float(mean_absolute_error(yte, yhat))
    metrics["rmse_test"] = float(np.sqrt(mean_squared_error(yte, yhat)))

    
    if miss.any():
        filled.loc[miss] = model.predict(df.loc[miss, features].values)

    
    filled = apply_rounding_policy(filled, target)
    return filled, metrics

def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Could not find {INPUT_XLSX} in the working folder.")

    print(f"Reading: {INPUT_XLSX}")
    xls = pd.ExcelFile(INPUT_XLSX)

    out_frames = {}
    all_metrics = []

    for sheet in SHEETS:
        if sheet not in xls.sheet_names:
            print(f"[WARN] Sheet '{sheet}' not found. Skipping.")
            continue

        print(f"\n--- Processing sheet: {sheet} ---")
        df = pd.read_excel(xls, sheet_name=sheet, header=1, dtype=object)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        print("Columns detected:", list(df.columns))

        features, targets = detect_columns(df)

        if len(features) != 3:
            raise ValueError(
                f"Sheet '{sheet}': expected 3 composition columns; detected {features}. "
                f"Check column names or COMPOSITION_PATTERNS."
            )
        if len(targets) == 0:
            print(f"[WARN] Sheet '{sheet}': no target columns detected. Skipping.")
            out_frames[sheet] = df
            continue

        print(f"Features (composition): {features}")
        print(f"Targets to impute:     {targets}")

        df_filled = df.copy()
        
        enforce_numeric(df_filled, features + targets)

        for tgt in targets:
            print(f"  -> Modeling target: {tgt}")
            filled_col, m = train_and_impute_column(df_filled, features, tgt)
            df_filled[tgt] = filled_col
            m["sheet"] = sheet
            all_metrics.append(m)

        out_frames[sheet] = df_filled

    
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for sheet, frame in out_frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)

        if len(all_metrics):
            met = pd.DataFrame(all_metrics)[
                ["sheet", "target", "n_observed", "n_missing", "r2_test", "mae_test", "rmse_test"]
            ].sort_values(["sheet", "target"])
            met.to_excel(writer, sheet_name="Imputation_Metrics", index=False)

    print(f"\nSaved completed workbook to: {OUTPUT_XLSX}")
    if len(all_metrics):
        print("Quick test metrics (observed 75/25 splits) were written to 'Imputation_Metrics'.")

if __name__ == "__main__":
    main()

