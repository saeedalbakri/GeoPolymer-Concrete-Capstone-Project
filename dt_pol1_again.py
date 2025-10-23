import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------
# Config
# -------------------------
excel_path = "PolymerData.xlsx"
PREFERRED_SHEET_NAME = "polymer 1"  # case-insensitive

id_col = "Composition Number"
X_raw = ["Raw material A (Slake)", "Raw Material B", "Raw Material C"]

y_cols = [
    "Initial Setting Time",
    "Final Setting Time",
    "CS 3 Days",
    "CS 7 Days",
    "CS 28 Days",
    "at 170 degrees F after 24 hours",
    "Flexural Strength",
]

MIN_KNOWN    = 5
TEST_SIZE    = 0.25
RANDOM_STATE = 42
N_SPLITS     = 5        # <-- K for outer ensembling
N_JOBS       = -1

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

# -------------------------
# Metrics (your version)
# -------------------------
def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  # manually take sqrt
    return r2, rmse

# -------------------------
# Helpers
# -------------------------
def load_polymer1_sheet(path, preferred_name="polymer 1"):
    xls = pd.ExcelFile(path)
    sheetnames_lc = {s.lower(): s for s in xls.sheet_names}
    use_sheet = sheetnames_lc.get(preferred_name, xls.sheet_names[0])
    df_ = pd.read_excel(xls, sheet_name=use_sheet)
    return df_, use_sheet

def add_mixture_features(df, a_col, b_col, c_col):
    """Add proportions and simple interactions to make the tree sensitive to ratio changes."""
    for c in [a_col, b_col, c_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If A is missing you could compute it as 100 - B - C; here A exists in your sheet.
    total = (df[a_col] + df[b_col] + df[c_col]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["A_prop"] = df[a_col] / total
        df["B_prop"] = df[b_col] / total
        df["C_prop"] = df[c_col] / total

    # simple curvature and pairwise terms
    df["AB"] = df["A_prop"] * df["B_prop"]
    df["AC"] = df["A_prop"] * df["C_prop"]
    df["BC"] = df["B_prop"] * df["C_prop"]
    df["A2"] = df["A_prop"] ** 2
    df["B2"] = df["B_prop"] ** 2
    df["C2"] = df["C_prop"] ** 2

    # Use both raw % and engineered features
    X_cols = X_raw + ["A_prop", "B_prop", "C_prop", "AB", "AC", "BC", "A2", "B2", "C2"]
    return df, X_cols

# -------------------------
# Load & clean
# -------------------------
df, used_sheet = load_polymer1_sheet(excel_path, PREFERRED_SHEET_NAME)
df = df.replace(["", " ", "NA", "N/A", "na", "--", "unknown", "Unknown"], np.nan)

keep = [c for c in [id_col] + X_raw + y_cols if c in df.columns]
df = df[keep].copy()

# numeric
for c in X_raw + y_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# engineered features
df, X_cols = add_mixture_features(df, X_raw[0], X_raw[1], X_raw[2])

summary_rows = []

# -------------------------
# Train / Predict: Decision Tree + K-fold ensemble
# -------------------------
for target_col in y_cols:
    if target_col not in df.columns:
        continue

    known_mask   = df[target_col].notna()
    unknown_mask = ~known_mask

    n_known, n_unknown = int(known_mask.sum()), int(unknown_mask.sum())
    pred_col = f"{target_col} (pred)"
    df[pred_col] = np.nan

    if n_known < MIN_KNOWN:
        summary_rows.append({
            "Target": target_col,
            "Known": n_known,
            "Missing to fill": n_unknown,
            "CV best R2": np.nan,
            "OOF R2": np.nan,
            "OOF RMSE": np.nan,
            "Params": {}
        })
        continue

    # complete-X rows only
    X_all = df.loc[:, X_cols]
    feat_ok = X_all.notna().all(axis=1)
    train_mask = known_mask & feat_ok
    pred_mask  = unknown_mask & feat_ok

    X_known = X_all.loc[train_mask].astype(float).values
    y_known = df.loc[train_mask, target_col].astype(float).values

    # ---------------------
    # Inner CV: hyperparams for a single tree
    # ---------------------
    param_grid = {
        "max_depth":       [3, 4, 5, 6, 7, 8, None],
        "min_samples_leaf":[1, 2, 3, 5, 8],
        "min_samples_split":[2, 4, 6, 8, 12, 16],
        "ccp_alpha":       [0.0, 0.0005, 0.001, 0.002],
    }
    base_tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
    inner_cv  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=base_tree,
        param_grid=param_grid,
        scoring="r2",
        cv=inner_cv,
        n_jobs=N_JOBS,
        refit=True
    )
    gs.fit(X_known, y_known)
    best_params = gs.best_params_

    # quick sanity hold-out (optional)
    if len(X_known) >= 10:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_known, y_known, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        m_tmp = DecisionTreeRegressor(random_state=RANDOM_STATE, **best_params)
        m_tmp.fit(X_tr, y_tr)
        te_pred = m_tmp.predict(X_te)
        test_r2, test_rmse = metrics(y_te, te_pred)
    else:
        test_r2, test_rmse = np.nan, np.nan

    # ---------------------
    # Outer KFold ensemble: fit K trees on different folds
    # ---------------------
    oof_pred = np.full_like(y_known, np.nan, dtype=float)

    if pred_mask.any():
        X_to_fill = X_all.loc[pred_mask].astype(float).values
        fill_accum = np.zeros(X_to_fill.shape[0], dtype=float)
        fill_counts = 0
    else:
        X_to_fill = None

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, va_idx in kf.split(X_known, y_known):
        model = DecisionTreeRegressor(random_state=RANDOM_STATE, **best_params)
        model.fit(X_known[tr_idx], y_known[tr_idx])

        # Out-of-fold preds for generalization estimate
        oof_pred[va_idx] = model.predict(X_known[va_idx])

        # Average predictions for missing rows across folds
        if X_to_fill is not None:
            fill_accum += model.predict(X_to_fill)
            fill_counts += 1

    # OOF metrics
    ok = ~np.isnan(oof_pred)
    oof_r2, oof_rmse = (np.nan, np.nan)
    if ok.any():
        oof_r2, oof_rmse = metrics(y_known[ok], oof_pred[ok])

    # Final predictions for missing rows = mean across K trees
    if X_to_fill is not None and fill_counts > 0:
        final = fill_accum / fill_counts
        df.loc[pred_mask, pred_col] = final

    summary_rows.append({
        "Target": target_col,
        "Known": n_known,
        "Missing to fill": n_unknown,
        "CV best R2": round(gs.best_score_, 3),
        "OOF R2": round(float(oof_r2), 3) if not np.isnan(oof_r2) else np.nan,
        "OOF RMSE": round(float(oof_rmse), 3) if not np.isnan(oof_rmse) else np.nan,
        "Test R2": round(float(test_r2), 3) if not np.isnan(test_r2) else np.nan,
        "Test RMSE": round(float(test_rmse), 3) if not np.isnan(test_rmse) else np.nan,
        "Params": best_params
    })

# -------------------------
# Export
# -------------------------
ordered_cols = (
    [c for c in [id_col] if c in df.columns] +
    [c for c in X_raw if c in df.columns] +
    ["A_prop", "B_prop", "C_prop"] +
    sum([[y, f"{y} (pred)"] for y in y_cols if y in df.columns], [])
)
export_cols = [c for c in ordered_cols if c in df.columns]
out_df = df[export_cols].copy()

num_cols = out_df.select_dtypes(include="number").columns
out_df[num_cols] = out_df[num_cols].round(3)

out_path = Path(excel_path).with_name("DT_Polymer1_predictions.xlsx")
out_df.to_excel(out_path, index=False)

print(f"\n✓ Used sheet: '{used_sheet}'")
print(f"✓ Wrote predictions to: {out_path}")

if summary_rows:
    print("\nModel summary (per target):")
    print(pd.DataFrame(summary_rows).to_string(index=False))




