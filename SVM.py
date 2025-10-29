import os, re, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

# Here we point to our data, output the results
# Read all 3 data sheets
# lock/keep radomness same so you get the same train/test split every run. if None, splits/initializations vary each run (not reproducible).
#Test size of 25%
INPUT_XLSX  = "Polymer_Data.xlsx"
OUTPUT_XLSX = "SVM_Output.xlsx"
SHEETS      = ["Polymer 1", "Polymer 2", "Polymer 3"]

RANDOM_STATE = 0
TEST_SIZE    = 0.25

# Rounding: this Keeps integers as floats. many decimals to keep when writing predictions
FIXED_DECIMALS = {
    "initial setting time": 0,
    "final setting time": 0,
    "3 days": 0,
    "7 days": 0,
    "28 days": 0,
    "at 170 degrees f after 24 hours": 0,
    "flexural strength": 1,
}
# Keep 0-decimal columns as float.
ROUND_KEEP_FLOAT = True   # if set False to store 0-decimal columns as pandas nullable Int64

#For detecting our targets, even if headers vary slightly
TARGET_PATTERNS = [
    r"\binitial\s*setting\s*time\b",
    r"\bfinal\s*setting\s*time\b",
    r"\bcompressive\s*strength\s*in\s*psi\s*3\s*days\b|\b3\s*days\b",
    r"\bcompressive\s*strength\s*in\s*psi\s*7\s*days\b|\b7\s*days\b",
    r"\bcompressive\s*strength\s*in\s*psi\s*28\s*days\b|\b28\s*days\b",
    r"\bat\s*170\s*degrees?\s*f\s*after\s*24\s*hours\b",
    r"\bflexural\s*strength\b",
]

# same, For detecting our targets
COMPOSITION_PATTERNS = {
    "A": [r"\braw\s*material\s*a\b", r"\braw\s*material\s*a\s*\(slake\)\b",
          r"\bflyash\s*f\s*class\b", r"\bflyash\s*c\s*class\b"],
    "B": [r"\braw\s*material\s*b\b"],
    "C": [r"\braw\s*material\s*c\b"],
}

# For Tuning. enable auto tuner in sklearn lib. 
#set 5 fold cross validatioon (CV)
ENABLE_TUNING = True
CV_FOLDS      = 5
#  score by neg-MAE during CV (more robust to outliers than MSE).
CV_SCORING    = "neg_mean_absolute_error"
# Small, safe grid—expand if you want deeper searches:
PARAM_GRID = {
    "regressor__svr__C":       [3, 10, 30],
    "regressor__svr__gamma":   ["scale", 0.1, 0.03],
    "regressor__svr__epsilon": [0.05, 0.1, 0.2],
}

# _norm is for normalize header text and find the first column that matches a regex pattern. Regex is pattern for searching , matchin, transforming text
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[•·_\-|\s]+", " ", s)
    return s
# also for pattern for searching , matchin, transforming text
def _find_first(colnames, patterns):
    cmap = {c: _norm(c) for c in colnames}
    for pat in patterns:
        rx = re.compile(pat, flags=re.I)
        for real, n in cmap.items():
            if rx.search(n):
                return real
    return None
# this auto-detect A/B/C and all target columns; errors out if A/B/C aren’t all found.
def detect_features(df: pd.DataFrame):
    A = _find_first(df.columns, COMPOSITION_PATTERNS["A"])
    B = _find_first(df.columns, COMPOSITION_PATTERNS["B"])
    C = _find_first(df.columns, COMPOSITION_PATTERNS["C"])
    feats = [c for c in (A, B, C) if c is not None]
    if len(feats) != 3:
        raise ValueError(f"Could not detect 3 composition columns. Detected: {feats}")
    return feats

def detect_targets(df: pd.DataFrame):
    targets = []
    for pat in TARGET_PATTERNS:
        hit = _find_first(df.columns, [pat])
        if hit and hit not in targets:
            targets.append(hit)
    if not targets:
        raise ValueError("No target columns detected—check headers.")
    return targets

def enforce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def decimals_for(name: str):
    name = _norm(name)
    for key, d in FIXED_DECIMALS.items():
        if key in name:
            return d
    return None

def apply_rounding(series: pd.Series, name: str):
    dec = decimals_for(name)
    if dec is None:
        return series
    series = series.round(dec)
    if dec == 0 and not ROUND_KEEP_FLOAT:
        return series.astype("Int64")  # nullable int preserves NaN
    return series.astype(float)

def build_svr():
  
    inner = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])
    model = TransformedTargetRegressor(
        regressor=inner,
        transformer=StandardScaler()
    )
    return model

#Return a fitted SVR model. If tuning is enabled, run GridSearchCV
#over the TransformedTargetRegressor (parameters refer to regressor__svr__*).
def tune_or_default_svr(X, y):
   
    base = build_svr()
    if not ENABLE_TUNING:
        base.fit(X, y)
        return base, {"tuned": False, "best_params_": None, "best_score_": None}

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        scoring=CV_SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    gs.fit(X, y)
    best = gs.best_estimator_
    info = {"tuned": True, "best_params_": gs.best_params_, "best_score_": gs.best_score_}
    return best, info

#Train SVR on observed rows, evaluate on a 25% holdout, and impute missing rows.
#Returns: (filled_series, metrics_dict)
def train_and_impute_column(df, features, target):
   

    feat_ok = df[features].notna().all(axis=1)
    obs  = feat_ok & df[target].notna()
    miss = feat_ok & df[target].isna()

    metrics = {
        "target": target,
        "n_observed": int(obs.sum()),
        "n_missing":  int(miss.sum()),
        "r2_test": np.nan, "mae_test": np.nan, "rmse_test": np.nan,
        "cv_best_score": np.nan, "tuned": False, "best_params": None
    }

    filled = df[target].copy()

    if obs.sum() < 10:
        
        return apply_rounding(filled, target), metrics

    X = df.loc[obs, features].to_numpy()
    y = df.loc[obs, target].to_numpy()

    # Tune (optional) and fit
    model, info = tune_or_default_svr(X, y)
    metrics["tuned"]       = info["tuned"]
    metrics["best_params"] = info["best_params_"]
    if info["best_score_"] is not None:
        # if neg-MAE, flip sign for readability
        metrics["cv_best_score"] = float(-info["best_score_"]) if CV_SCORING.startswith("neg_") else float(info["best_score_"])

    # Hold-out evaluation (fresh split)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    metrics["r2_test"]  = float(r2_score(yte, yhat))
    metrics["mae_test"] = float(mean_absolute_error(yte, yhat))
    metrics["rmse_test"]= float(np.sqrt(mean_squared_error(yte, yhat)))

    # Refit on ALL observed, then impute missing
    model.fit(X, y)
    if miss.any():
        Xmiss = df.loc[miss, features].to_numpy()
        preds = model.predict(Xmiss)
        filled.loc[miss] = preds

    # Final rounding/formatting
    filled = apply_rounding(filled, target)
    return filled, metrics

#Open the Excel file and loop through the three sheets.
# For each sheet:
#Read with header=1 (row 2 are headers; data start at row 3).
#Drop empty rows/cols.
#Auto-detect features (A/B/C) and targets.
#Make a working copy and coerce those columns to numeric.
#For each target: call train_and_impute_column, replace the column with the filled version, and store metrics.
#Save:
#Each completed sheet (with imputed values).
#An Imputation_Metrics sheet: sheet, target, n_observed, n_missing, r2_test, mae_test, rmse_test, cv_best_score, tuned, best_params.
def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Put {INPUT_XLSX} in the working folder.")

    xls = pd.ExcelFile(INPUT_XLSX)
    out_frames = {}
    all_metrics = []

    for sheet in SHEETS:
        if sheet not in xls.sheet_names:
            print(f"[WARN] Sheet '{sheet}' not found; skipping.")
            continue

        print(f"\n--- Processing sheet: {sheet} ---")
        df = pd.read_excel(xls, sheet_name=sheet, header=1, dtype=object)
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Detect columns
        features = detect_features(df)
        targets  = detect_targets(df)
        print("Features:", features)
        print("Targets :", targets)

        # Coerce once per sheet for speed
        df_filled = df.copy()
        enforce_numeric(df_filled, features + targets)

        # Per-target SVR -> fill & metrics
        for tgt in targets:
            print(f"  -> Modeling: {tgt}")
            filled_col, m = train_and_impute_column(df_filled, features, tgt)
            df_filled[tgt] = filled_col
            m["sheet"] = sheet
            all_metrics.append(m)

        out_frames[sheet] = df_filled

    # Write outputs
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        for s, frame in out_frames.items():
            frame.to_excel(w, sheet_name=s, index=False)

        if all_metrics:
            cols = ["sheet","target","n_observed","n_missing",
                    "r2_test","mae_test","rmse_test","cv_best_score","tuned","best_params"]
            pd.DataFrame(all_metrics)[cols].sort_values(["sheet","target"]).to_excel(
                w, sheet_name="Imputation_Metrics", index=False
            )

    print(f"\nSaved completed workbook to: {OUTPUT_XLSX}")
    print("Tip: set ENABLE_TUNING=False for a faster run with defaults.")

if __name__ == "__main__":
    main()
