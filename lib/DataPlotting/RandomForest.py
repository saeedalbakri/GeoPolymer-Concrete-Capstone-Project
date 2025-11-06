import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

EXCEL_PATH = "PolymerData2.xlsx"
OUTPUT_PATH = Path("Geo_Polymer_Data_with_Predictions_Optimized.xlsx")
SHEETS = ["Polymer 1", "Polymer 2", "Polymer 3"]

def find_col(df, *keywords):
    for col in df.columns:
        name = str(col).lower()
        if all(k.lower() in name for k in keywords):
            return col
    return None

def rmse_compat(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# Build inverse-frequency sample weights from quantile bins
def quantile_sample_weights(y, n_bins=5, min_weight=0.7, max_weight=1.8):
    y = np.asarray(y, dtype=float)
    uniq = np.unique(y[~np.isnan(y)])
    if uniq.size < 3:
        return np.ones_like(y, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y, qs)
    bins = np.unique(bins)
    if bins.size < 3:
        return np.ones_like(y, dtype=float)
    idx = np.clip(np.digitize(y, bins, right=True) - 1, 0, bins.size - 2)
    counts = np.bincount(idx, minlength=bins.size - 1).astype(float)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
    w = min_weight + (max_weight - min_weight) * inv[idx]
    w[np.isnan(y)] = 1.0
    return w

def cv_mae_for_params(X, y, params, sample_weight, random_state=42, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        sw_tr = sample_weight[tr_idx] if sample_weight is not None else None

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            min_impurity_decrease=params["min_impurity_decrease"],
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        pred = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, pred))
    return float(np.mean(maes))

def sample_params(rng):
    max_features_pool_floats = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0]
    max_features_pool_strs   = ["sqrt", "log2"]
    if rng.rand() < 0.75:
        max_features = float(max_features_pool_floats[int(rng.rand()*len(max_features_pool_floats))])
    else:
        max_features = max_features_pool_strs[int(rng.rand()*len(max_features_pool_strs))]

    params = {
        "n_estimators": int(rng.randint(800, 1601)),     # more trees → less collapse
        "max_depth": (None if rng.rand() < 0.5 else int(rng.randint(12, 41))),  # allow deep trees
        "min_samples_leaf": int(rng.randint(1, 4)),      # small leaves = more granularity
        "min_samples_split": int(rng.randint(2, 10)),
        "max_features": max_features,                    # valid dtype
        "min_impurity_decrease": float(rng.choice([0.0, 1e-8, 1e-7, 1e-6])),
    }
    return params

def tune_rf_params(X, y, sample_weight=None, random_state=42, n_iter=48):
    rng = np.random.RandomState(random_state)
    best_params, best_mae = None, float("inf")
    for _ in range(n_iter):
        params = sample_params(rng)
        mae = cv_mae_for_params(X, y, params, sample_weight, random_state=random_state, n_splits=5)
        if mae < best_mae:
            best_mae, best_params = mae, params
    return best_params, best_mae

# Start fresh
if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

for sheet in SHEETS:
    print(f"\n=== Processing {sheet} ===")
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    # Features (inputs)
    X_cols = [
        find_col(df, "raw", "a"),
        find_col(df, "raw", "b"),
        find_col(df, "raw", "c"),
    ]
    # Targets (outputs)
    y_cols = [
        find_col(df, "initial", "setting"),
        find_col(df, "final", "setting"),
        find_col(df, "cs", "3"),
        find_col(df, "cs", "7"),
        find_col(df, "cs", "28"),
        find_col(df, "at170"),
        find_col(df, "flexural"),
    ]
    X_cols = [c for c in X_cols if c]
    y_cols = [c for c in y_cols if c]

    if not X_cols or not y_cols:
        print(f"Could not detect expected columns in {sheet}.")
        mode = "a" if OUTPUT_PATH.exists() else "w"
        with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl", mode=mode) as writer:
            df.to_excel(writer, sheet_name=f"{sheet} (Predicted)", index=False)
        continue

    print("Features:", X_cols)
    print("Targets:", y_cols)

    # Ensure numeric
    df[X_cols + y_cols] = df[X_cols + y_cols].apply(pd.to_numeric, errors="coerce")

    for target in y_cols:
        print(f"\n— Target: {target}")
        sub = df[X_cols + [target]].dropna()
        if len(sub) < 5:
            print(f"  Not enough data to train {target} — skipping.")
            continue

        X = sub[X_cols].values.astype(float)
        y = sub[target].values.astype(float)

        # Quantile-balanced weights to avoid mean-collapse
        sw = quantile_sample_weights(y, n_bins=5, min_weight=0.7, max_weight=1.8)

        # Small holdout for reporting; final model refits on ALL data
        X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
            X, y, sw, test_size=0.2, random_state=42
        )

        best_params, cv_mae = tune_rf_params(X_tr, y_tr, sample_weight=sw_tr, random_state=42, n_iter=48)
        print("  Best (CV MAE):", round(cv_mae, 4), "Params:", best_params)

        final_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            min_samples_split=best_params["min_samples_split"],
            max_features=best_params["max_features"],
            min_impurity_decrease=best_params["min_impurity_decrease"],
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
        )
        final_model.fit(X, y, sample_weight=sw)

        # Report metrics on the holdout
        y_hat = final_model.predict(X_te)
        mae = mean_absolute_error(y_te, y_hat)
        rmse = rmse_compat(y_te, y_hat)
        r2 = r2_score(y_te, y_hat)
        print(f"  Holdout: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

        # Fill missing rows for this target (and round to 2 decimals)
        miss_mask = df[target].isna() & df[X_cols].notna().all(axis=1)
        if miss_mask.any():
            filled = final_model.predict(df.loc[miss_mask, X_cols].values.astype(float)).astype(float)
            df.loc[miss_mask, target] = np.round(filled, 2)
            print(f"  Filled {miss_mask.sum()} rows for {target}.")
        else:
            print(f"  No missing rows to fill for {target}.")

    # Write the current sheet immediately
    mode = "a" if OUTPUT_PATH.exists() else "w"
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl", mode=mode) as writer:
        df.to_excel(writer, sheet_name=f"{sheet} (Predicted)", index=False)

print(f"\nOptimized predictions saved to {OUTPUT_PATH.resolve()}")