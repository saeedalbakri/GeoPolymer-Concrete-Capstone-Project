#  2-fold cross-validated hyperparameter 

import re, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

# --------- Config ---------
INPUT_XLSX   = "Polymer_Data.xlsx"
OUTPUT_XLSX  = "Output_Data_.xlsx"
RANDOM_STATE = 0               # repeatable split & CV shuffling
TEST_SIZE    = 0.25            # 25% test → 75% train
SHEETS       = ["Polymer 1", "Polymer 2", "Polymer 3"]

# How many decimals to keep by target substring
FIXED_DECIMALS = {
    "initial setting time": 0,
    "final setting time": 0,
    "3 days": 0,
    "7 days": 0,
    "28 days": 0,
    "at 170 degrees f after 24 hours": 0,
    "flexural strength": 1,
}
ROUND_KEEP_FLOAT = True

# Column detection patterns
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

# Hyperparameter search space (compact but effective)
PARAM_GRID = {
    "n_estimators":   [150, 300, 500],
    "learning_rate":  [0.02, 0.05, 0.1],
    "max_depth":      [2, 3, 4],
    "subsample":      [0.6, 0.8, 1.0],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 3, 5],
    "max_features":   [None, "sqrt"],  # None = all; "sqrt" helps regularize
}

# --------- Helpers ---------
def normalize_col(c):
    c = str(c).strip().lower()
    c = re.sub(r"[•·_|\-]+", " ", c)
    c = re.sub(r"\s+", " ", c)
    return c

def find_first_match(col_names_norm, patterns):
    for pat in patterns:
        rx = re.compile(pat, flags=re.I)
        for real, norm in col_names_norm.items():
            if rx.search(norm):
                return real
    return None

def detect_columns(df):
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

def enforce_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def get_decimals_for(target_name):
    name = str(target_name).strip().lower()
    for key, d in FIXED_DECIMALS.items():
        if key in name:
            return d
    return None

def apply_rounding_policy(series, target_name):
    dec = get_decimals_for(target_name)
    if dec is None:
        return series
    series = series.round(dec)
    if dec == 0 and not ROUND_KEEP_FLOAT:
        return series.astype("Int64")
    return series.astype(float)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# --------- Modeling / Imputation ---------
def train_and_impute_column(df, features, target):
    feat_ok = df[features].notna().all(axis=1)
    obs  = feat_ok & df[target].notna()
    miss = feat_ok & df[target].isna()

    metrics = {
        "target": target,
        "n_observed": int(obs.sum()),
        "n_missing":  int(miss.sum()),
        "cv_folds": 2,              # fixed at 2 folds
        "cv_mean_rmse": np.nan,
        "cv_best_params": "",
        "r2_test": np.nan,
        "mae_test": np.nan,
        "rmse_test": np.nan,
    }

    filled = df[target].copy()
    n_obs = int(obs.sum())
    if n_obs < 10:
        # too little signal; pass-through with rounding for consistency
        return apply_rounding_policy(filled, target), metrics

    X = df.loc[obs, features].values
    y = df.loc[obs, target].values

    # --- 75/25 holdout ---
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --- 2-fold CV on TRAIN ONLY ---
    kf = KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    base = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        scoring="neg_root_mean_squared_error",  # minimize RMSE
        cv=kf,
        n_jobs=-1,
        refit=True,          # refit best on the TRAIN split
        verbose=0
    )
    gs.fit(Xtr, ytr)

    metrics["cv_mean_rmse"] = float(-gs.best_score_)
    metrics["cv_best_params"] = str(gs.best_params_)

    # --- Test-set evaluation with the refit best model ---
    best_model_train = gs.best_estimator_
    yhat_test = best_model_train.predict(Xte)
    metrics["r2_test"]  = float(r2_score(yte, yhat_test))
    metrics["mae_test"] = float(mean_absolute_error(yte, yhat_test))
    metrics["rmse_test"] = rmse(yte, yhat_test)

    # --- Refit on ALL observed rows with best params for final imputation ---
    final_model = GradientBoostingRegressor(random_state=RANDOM_STATE, **gs.best_params_)
    final_model.fit(X, y)

    if miss.any():
        filled.loc[miss] = final_model.predict(df.loc[miss, features].values)

    # --- Policy rounding ---
    filled = apply_rounding_policy(filled, target)
    return filled, metrics

# --------- Main ---------
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
            met = (
                pd.DataFrame(all_metrics)[
                    [
                        "sheet", "target",
                        "n_observed", "n_missing",
                        "cv_folds", "cv_mean_rmse", "cv_best_params",
                        "r2_test", "mae_test", "rmse_test",
                    ]
                ]
                .sort_values(["sheet", "target"])
            )
            met.to_excel(writer, sheet_name="Imputation_Metrics", index=False)

    print(f"\nSaved completed workbook to: {OUTPUT_XLSX}")
    if len(all_metrics):
        print("Wrote best params and 75/25 test metrics to 'Imputation_Metrics' (cv_folds=2).")

if __name__ == "__main__":
    main()

