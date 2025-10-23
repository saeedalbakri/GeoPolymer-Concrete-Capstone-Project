import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

excel_path = "PolymerData.xlsx"
PREFERRED_SHEET_NAME = "polymer 1"

id_col = "Composition Number"
X_cols = ["Raw material A (Slake)", "Raw Material B", "Raw Material C"]
y_cols = [
    "Initial Setting Time",
    "Final Setting Time",
    "CS 3 Days",
    "CS 7 Days",
    "CS 28 Days",
    "at 170 degrees F after 24 hours",
    "Flexural Strength",
]

MIN_KNOWN     = 5
TEST_SIZE     = 0.25
RANDOM_STATE  = 42

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return r2, rmse


def load_polymer1_sheet(path, preferred_name="polymer 1"):
    """Open 'Polymer 1' by name if present otherwise first sheet."""
    xls = pd.ExcelFile(path)
    sheetnames_lc = {s.lower(): s for s in xls.sheet_names}
    if preferred_name in sheetnames_lc:
        use_sheet = sheetnames_lc[preferred_name]
    else:
        use_sheet = xls.sheet_names[0]
    df_ = pd.read_excel(xls, sheet_name=use_sheet)
    return df_, use_sheet

df, used_sheet = load_polymer1_sheet(excel_path, PREFERRED_SHEET_NAME)

df = df.replace(
    ["", " ", "NA", "N/A", "na", "--", "unknown", "Unknown"], np.nan
)

#creating prediction column
keep = [c for c in [id_col] + X_cols + y_cols if c in df.columns]
df = df[keep].copy()

#make floats
for c in X_cols + y_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

summary_rows = []

#go over empty columns to predict
for target_col in y_cols:
    if target_col not in df.columns:
        continue

    known_mask   = df[target_col].notna()
    unknown_mask = ~known_mask

    n_known, n_unknown = int(known_mask.sum()), int(unknown_mask.sum())
    if n_known < MIN_KNOWN:
        # Not enough training data for this target — leave predictions as NaN
        df[f"{target_col}_pred"] = np.nan
        summary_rows.append({
            "Target": target_col,
            "Known": n_known,
            "Missing to fill": n_unknown,
            "CV best R2": np.nan,
            "Test R2": np.nan,
            "Test RMSE": np.nan,
            "Params": {}
        })
        continue

    #training on columns with known data
    X_known_full = df.loc[known_mask, X_cols]
    X_complete_mask = X_known_full.notna().all(axis=1)
    X_known = X_known_full.loc[X_complete_mask].astype(float)
    y_known = df.loc[known_mask, target_col].loc[X_complete_mask].astype(float)

    if len(X_known) < MIN_KNOWN:
        df[f"{target_col}_pred"] = np.nan
        summary_rows.append({
            "Target": target_col,
            "Known": int(len(X_known)),
            "Missing to fill": n_unknown,
            "CV best R2": np.nan,
            "Test R2": np.nan,
            "Test RMSE": np.nan,
            "Params": {}
        })
        continue

    #5-fold CV
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_samples_leaf": [2, 3, 4, 5, 6, 8, 10],
        "min_samples_split": [4, 6, 8, 10, 12, 16],
        "ccp_alpha": [0.0, 0.0005, 0.001, 0.002],
    }
    base_tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=base_tree,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1
    )
    gs.fit(X_known, y_known)
    best_params = gs.best_params_

    #held out test fon known data
    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = DecisionTreeRegressor(random_state=RANDOM_STATE, **best_params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_r2, test_rmse = metrics(y_test, y_pred_test)

    #refit on all known data and predict empty data
    model.fit(X_known, y_known)

    pred_col = f"{target_col}_pred"
    df[pred_col] = np.nan

    X_unknown_full = df.loc[unknown_mask, X_cols]
    X_unknown_complete_mask = X_unknown_full.notna().all(axis=1)
    if X_unknown_complete_mask.any():
        X_unknown = X_unknown_full.loc[X_unknown_complete_mask].astype(float)
        preds = model.predict(X_unknown)
        df.loc[X_unknown_complete_mask.index[X_unknown_complete_mask], pred_col] = preds

    summary_rows.append({
        "Target": target_col,
        "Known": n_known,
        "Missing to fill": n_unknown,
        "CV best R2": round(gs.best_score_, 3),
        "Test R2": round(test_r2, 3),
        "Test RMSE": round(test_rmse, 3),
        "Params": best_params
    })

# Nice ordering: ID, Xs, then each target + its _pred
ordered_cols = (
    [c for c in [id_col] if c in df.columns] +
    [c for c in X_cols if c in df.columns] +
    sum([[y, f"{y}_pred"] for y in y_cols if y in df.columns], [])
)
export_df = df[ordered_cols].copy()

# Round numerics and rename prediction headers to "(pred)"
num_cols = export_df.select_dtypes(include="number").columns
export_df[num_cols] = export_df[num_cols].round(3)
rename_map = {f"{y}_pred": f"{y} (pred)" for y in y_cols if f"{y}_pred" in export_df.columns}
export_df = export_df.rename(columns=rename_map)

out_name = Path(excel_path).with_name("DT_Polymer1_predictions.xlsx")
export_df.to_excel(out_name, index=False)

print(f"\n✓ Used sheet: '{used_sheet}'")
print(f"✓ Wrote predictions to: {out_name}")

if summary_rows:
    print("\nModel summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
