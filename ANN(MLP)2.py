# === ANN_MLP_random_safe_fixed.py — 75/25 split + RandomizedSearchCV (3-fold) + robust Excel save ===
import os, re, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import loguniform, uniform

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Config ----------
INPUT_XLSX   = "Polymer_Data.xlsx"
OUTPUT_XLSX  = "ANN_Output_Data_.xlsx"
RANDOM_STATE = 0
TEST_SIZE    = 0.25
CV_FOLDS     = 3
N_ITER       = 120
SCORING      = "neg_root_mean_squared_error"
SHEETS       = ["Polymer 1", "Polymer 2", "Polymer 3"]

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

# ---------- Column detection ----------
COMPOSITION_PATTERNS = {
    "A": [
        r"\braw\s*material\s*a\b",
        r"\braw\s*material\s*a\s*\(slake\)\b",
        r"\bflyash\s*f\s*class\b",
        r"\bflyash\s*c\s*class\b",
    ],
    "B": [r"\braw\s*material\s*b\b"],
    "C": [r"\braw\s*material\s*c\b"],
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

# ---------- Helpers ----------
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
    comp_A = find_first_match(norm_map, COMPOSITION_PATTERNS["A"])
    comp_B = find_first_match(norm_map, COMPOSITION_PATTERNS["B"])
    comp_C = find_first_match(norm_map, COMPOSITION_PATTERNS["C"])
    features = [c for c in (comp_A, comp_B, comp_C) if c is not None]

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

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------- MLP builder + tuner ----------
def _scaler_choices():
    # Always valid for a Pipeline step: either 'passthrough' or a transformer
    return ['passthrough',
            MinMaxScaler(),
            StandardScaler(with_mean=True, with_std=True)]

def _layer_choices():
    return [(32,), (64,), (64, 32), (128, 64)]

def build_mlp_pipeline():
    """
    Always include an 'xscaler' step so hyperparam search over 'regressor__xscaler'
    is valid. Use 'passthrough' to represent 'no scaling'.
    """
    pipe = Pipeline(steps=[
        ("xscaler", 'passthrough'),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=False,
            shuffle=True
        ))
    ])
    model = TransformedTargetRegressor(
        regressor=pipe,
        transformer=StandardScaler(with_mean=True, with_std=True)
    )
    return model

def randomized_tune_mlp(X, y):
    inner_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    base = build_mlp_pipeline()

    # Parameter names follow: TTR -> Pipeline('xscaler','mlp') -> mlp params
    param_dist = {
        "regressor__xscaler": _scaler_choices(),
        "regressor__mlp__hidden_layer_sizes": _layer_choices(),
        "regressor__mlp__activation": ["relu", "tanh"],
        "regressor__mlp__alpha": loguniform(1e-6, 1e-2),
        "regressor__mlp__learning_rate_init": loguniform(1e-4, 1e-1),
        "regressor__mlp__beta_1": uniform(0.85, 0.14),   # 0.85–0.99
        "regressor__mlp__batch_size": ["auto", 8, 16, 32],
    }

    rnd = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=inner_cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0
    )
    rnd.fit(X, y)
    best_cv = -rnd.best_score_
    return rnd.best_estimator_, rnd.best_params_, best_cv

# ---------- Modeling / Imputation ----------
def train_and_impute_column(df, features, target):
    feat_ok = df[features].notna().all(axis=1)
    obs  = feat_ok & df[target].notna()
    miss = feat_ok & df[target].isna()

    metrics = {
        "target": target,
        "n_observed": int(obs.sum()),
        "n_missing":  int(miss.sum()),
        "cv_folds": CV_FOLDS,
        "cv_best_score": np.nan,
        "best_params": "",
        "r2_test": np.nan,
        "mae_test": np.nan,
        "rmse_test": np.nan,
        "tuned": True,
    }

    filled = df[target].copy()
    if int(obs.sum()) < 10:
        metrics["tuned"] = False
        return apply_rounding_policy(filled, target), metrics

    X = df.loc[obs, features].values
    y = df.loc[obs, target].values

    # 75/25
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Tune on train only
    best_model, best_params, cv_score = randomized_tune_mlp(Xtr, ytr)
    metrics["cv_best_score"] = float(cv_score)
    metrics["best_params"]   = str(best_params)

    # Test once
    yhat = best_model.predict(Xte)
    metrics["r2_test"]  = float(r2_score(yte, yhat))
    metrics["mae_test"] = float(mean_absolute_error(yte, yhat))
    metrics["rmse_test"] = rmse(yte, yhat)

    # Refit on ALL observed rows and impute
    best_model.fit(X, y)
    if miss.any():
        filled.loc[miss] = best_model.predict(df.loc[miss, features].values)

    filled = apply_rounding_policy(filled, target)
    return filled, metrics

# ---------- Safe Excel writer ----------
def open_excel_writer_safely(target_path: str):
    p = Path(target_path)
    try:
        writer = pd.ExcelWriter(str(p), engine="openpyxl")
        alt_path = None
    except PermissionError:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        alt = p.with_name(f"{p.stem}_{ts}{p.suffix}")
        print(f"[WARN] '{p.name}' is locked (Excel/OneDrive?). Writing to '{alt.name}' instead.")
        writer = pd.ExcelWriter(str(alt), engine="openpyxl")
        alt_path = str(alt)
    return writer, alt_path

# ---------- Main ----------
def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Could not find {INPUT_XLSX}")

    print(f"Reading: {INPUT_XLSX}")
    xls = pd.ExcelFile(INPUT_XLSX)

    out_frames, all_metrics = {}, []
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
            raise ValueError(f"Sheet '{sheet}': expected 3 composition columns; detected {features}.")
        if len(targets) == 0:
            print(f"[WARN] Sheet '{sheet}': no target columns detected. Skipping.")
            out_frames[sheet] = df
            continue

        print(f"Features: {features}")
        print(f"Targets : {targets}")

        df_filled = df.copy()
        enforce_numeric(df_filled, features + targets)

        for tgt in targets:
            print(f"  -> Modeling target: {tgt}")
            filled_col, m = train_and_impute_column(df_filled, features, tgt)
            df_filled[tgt] = filled_col
            m["sheet"] = sheet
            all_metrics.append(m)

        out_frames[sheet] = df_filled

    writer, alt_path = open_excel_writer_safely(OUTPUT_XLSX)
    try:
        for s, frame in out_frames.items():
            frame.to_excel(writer, sheet_name=s, index=False)
        if len(all_metrics):
            met = (
                pd.DataFrame(all_metrics)[
                    ["sheet","target","n_observed","n_missing",
                     "cv_folds","cv_best_score","best_params",
                     "r2_test","mae_test","rmse_test","tuned"]
                ]
                .sort_values(["sheet","target"])
            )
            met.to_excel(writer, sheet_name="Imputation_Metrics", index=False)
    finally:
        writer.close()

    out_path = alt_path if alt_path is not None else OUTPUT_XLSX
    print(f"\nSaved completed workbook to: {out_path}")

if __name__ == "__main__":
    main()
