#necessary libraries
import os, re, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

#Where to read data from, where to save results, and which sheets to process.
INPUT_XLSX  = "Polymer_Data.xlsx"
OUTPUT_XLSX = "MLP_Output.xlsx"
SHEETS      = ["Polymer 1", "Polymer 2", "Polymer 3"]

RANDOM_STATE = 0 #Fixed randomness so runs are repeatable.
TEST_SIZE    = 0.25 #25% of observed rows are held out for a quick test.
MIN_OBS_ROWS = 10     # skip modeling if fewer observed rows than this

# Rounding how many decimals to keep for each targe
FIXED_DECIMALS = {
    "initial setting time": 0,
    "final setting time": 0,
    "3 days": 0,
    "7 days": 0,
    "28 days": 0,
    "at 170 degrees f after 24 hours": 0,
    "flexural strength": 1,
}
ROUND_KEEP_FLOAT = True  # False -> store 0-decimal columns as nullable Int64

# Target detection patterns Flexible regex patterns to find targets like “Initial/Final Setting Time”, 
#“3/7/28 days strength”, “170°F after 24 hours”, and “Flexural Strength”, even if the headers vary slightly.
TARGET_PATTERNS = [
    r"\binitial\s*setting\s*time\b",
    r"\bfinal\s*setting\s*time\b",
    r"\bcompressive\s*strength\s*in\s*psi\s*3\s*days\b|\b3\s*days\b",
    r"\bcompressive\s*strength\s*in\s*psi\s*7\s*days\b|\b7\s*days\b",
    r"\bcompressive\s*strength\s*in\s*psi\s*28\s*days\b|\b28\s*days\b",
    r"\bat\s*170\s*degrees?\s*f\s*after\s*24\s*hours\b",
    r"\bflexural\s*strength\b",
]

# Composition find the three composition columns (Raw material A/B/C, etc.), allowing minor header variations.
COMPOSITION_PATTERNS = {
    "A": [r"\braw\s*material\s*a\b", r"\braw\s*material\s*a\s*\(slake\)\b",
          r"\bflyash\s*f\s*class\b", r"\bflyash\s*c\s*class\b"],
    "B": [r"\braw\s*material\s*b\b"],
    "C": [r"\braw\s*material\s*c\b"],
}

# CV search space 3options for feature scaling.
SCALERS = {
    "No Scaling": FunctionTransformer(lambda z: z, feature_names_out="one-to-one", validate=False),
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
}
MLP_DEFAULTS = dict(
    solver="adam",
    max_iter=800,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.15,
    alpha=1e-4,
    random_state=RANDOM_STATE,
)
MLP_CONFIGS = [
    ("relu_64x32_lr0.01",  dict(hidden_layer_sizes=(64,32), activation="relu", learning_rate_init=0.01)),
    ("relu_128x64_lr0.01", dict(hidden_layer_sizes=(128,64),activation="relu", learning_rate_init=0.01)),
    ("relu_64x32_lr0.005", dict(hidden_layer_sizes=(64,32), activation="relu", learning_rate_init=0.005)),
    ("tanh_64x32_lr0.01",  dict(hidden_layer_sizes=(64,32), activation="tanh", learning_rate_init=0.01)),
]
KFOLDS = 4 #Use 4-fold CV; scale the target y during training (often stabilizes neural nets).
SCALE_Y = True  # scale targets via TransformedTargetRegressor(StandardScaler)

#Lowercases, trims, and compresses separators/spaces so messy headers still match consistently.
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[•·_\-|\s]+", " ", s)
    return s
#Searches normalized column names for the first regex match from a list of patterns.
def _find_first(colnames, patterns):
    cmap = {c: _norm(c) for c in colnames}
    for pat in patterns:
        rx = re.compile(pat, flags=re.I)
        for real, n in cmap.items():
            if rx.search(n):
                return real
    return None
#Uses _find_first with COMPOSITION_PATTERNS to locate A, B, C. Errors out if it can’t find all three.
def detect_features(df: pd.DataFrame):
    A = _find_first(df.columns, COMPOSITION_PATTERNS["A"])
    B = _find_first(df.columns, COMPOSITION_PATTERNS["B"])
    C = _find_first(df.columns, COMPOSITION_PATTERNS["C"])
    feats = [c for c in (A, B, C) if c is not None]
    if len(feats) != 3:
        raise ValueError(f"Could not detect 3 composition columns. Detected: {feats}")
    return feats
#Loops through TARGET_PATTERNS, records the first match for each, and returns the list. Errors if none are found.
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
#Picks the feature scaler.
#Creates MLPRegressor with defaults + the specific config.
#Wraps scaler+MLP in a Pipeline.
#Wraps that pipeline in TransformedTargetRegressor to scale y as well (if SCALE_Y).
def build_model(x_scaler_name: str, mlp_name: str, mlp_params: dict):
    x_scaler = SCALERS[x_scaler_name]
    mlp = MLPRegressor(**{**MLP_DEFAULTS, **mlp_params})
    inner = Pipeline([("xscaler", x_scaler), ("mlp", mlp)])
    y_transformer = StandardScaler() if SCALE_Y else FunctionTransformer(lambda z: z, validate=False)
    return TransformedTargetRegressor(regressor=inner, transformer=y_transformer)

def cv_choose_best(X: np.ndarray, y: np.ndarray):

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)
    best = None
    for s_name in SCALERS.keys():
        for m_name, m_params in MLP_CONFIGS:
            r2s, maes, rmses = [], [], []
            for tr, va in kf.split(X):
                X_tr, X_va = X[tr], X[va]
                y_tr, y_va = y[tr], y[va]
                model = build_model(s_name, m_name, m_params)
                model.fit(X_tr, y_tr)
                p = model.predict(X_va)
                r2s.append(r2_score(y_va, p))
                maes.append(mean_absolute_error(y_va, p))
                rmses.append(np.sqrt(mean_squared_error(y_va, p)))
            metrics = {"R2": float(np.mean(r2s)),
                       "MAE": float(np.mean(maes)),
                       "RMSE": float(np.mean(rmses))}
            # choose by (max R2, then min MAE)
            key = (metrics["R2"], -metrics["MAE"])
            if (best is None) or (key > best["key"]):
                best = {"scaler": s_name, "mlp": m_name, "metrics": metrics, "key": key}
    return best["scaler"], best["mlp"], best["metrics"]


def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Put {INPUT_XLSX} in this folder.")

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

        features = detect_features(df)
        targets  = detect_targets(df)
        print("Features:", features)
        print("Targets :", targets)

        df_filled = df.copy()
        enforce_numeric(df_filled, features + targets)

        for tgt in targets:
            print(f"  -> Target: {tgt}")
            feat_ok = df_filled[features].notna().all(axis=1)
            obs  = feat_ok & df_filled[tgt].notna()
            miss = feat_ok & df_filled[tgt].isna()

            metrics_row = {
                "sheet": sheet, "target": tgt,
                "n_observed": int(obs.sum()), "n_missing": int(miss.sum()),
                "r2_test": np.nan, "mae_test": np.nan, "rmse_test": np.nan,
                "best_scaler": None, "best_mlp": None,
                "cv_R2": np.nan, "cv_MAE": np.nan, "cv_RMSE": np.nan,
            }

            filled = df_filled[tgt].copy()

            if obs.sum() < MIN_OBS_ROWS:
                print("    Not enough observed rows. Skipping model.")
                df_filled[tgt] = apply_rounding(filled, tgt)
                all_metrics.append(metrics_row)
                continue

            X = df_filled.loc[obs, features].to_numpy()
            y = df_filled.loc[obs, tgt].to_numpy()

            # ---- CV to pick scaler + MLP config
            best_scaler, best_mlp, cvm = cv_choose_best(X, y)
            metrics_row.update({"best_scaler": best_scaler, "best_mlp": best_mlp,
                                "cv_R2": cvm["R2"], "cv_MAE": cvm["MAE"], "cv_RMSE": cvm["RMSE"]})

            # ---- Hold-out evaluation (fresh split)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            # recover the params for best_mlp
            mlp_params = dict([cfg for cfg in MLP_CONFIGS if cfg[0] == best_mlp][0][1])
            model = build_model(best_scaler, best_mlp, mlp_params)
            model.fit(Xtr, ytr)
            yhat = model.predict(Xte)
            metrics_row["r2_test"]  = float(r2_score(yte, yhat))
            metrics_row["mae_test"] = float(mean_absolute_error(yte, yhat))
            metrics_row["rmse_test"]= float(np.sqrt(mean_squared_error(yte, yhat)))

            # ---- Refit on ALL observed, then impute missing
            model = build_model(best_scaler, best_mlp, mlp_params)
            model.fit(X, y)
            if miss.any():
                Xmiss = df_filled.loc[miss, features].to_numpy()
                preds = model.predict(Xmiss)
                filled.loc[miss] = preds

            df_filled[tgt] = apply_rounding(filled, tgt)
            all_metrics.append(metrics_row)

        out_frames[sheet] = df_filled

   
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        for s, frame in out_frames.items():
            frame.to_excel(w, sheet_name=s, index=False)

        if all_metrics:
            cols = ["sheet","target","n_observed","n_missing",
                    "r2_test","mae_test","rmse_test",
                    "cv_R2","cv_MAE","cv_RMSE","best_scaler","best_mlp"]
            pd.DataFrame(all_metrics)[cols].sort_values(["sheet","target"]).to_excel(
                w, sheet_name="Imputation_Metrics", index=False
            )

    print(f"\nSaved completed workbook to: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
