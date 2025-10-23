import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

EXCEL_PATH = "PolymerData.xlsx"
OUT_PATH = "RandomForest_Filled_Polymers_All.xlsx"

#helpers
def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

WANTED = {
    # X features: allow Slake OR Flyash variants for X1
    "X1": [
        "rawmateriala(slake)", "rawmaterialaslake", "rawmateriala",
        "a(slake)", "a",
        "flyashfclass", "flyashcclass", "flyashclassf", "flyashclassc"
    ],
    "Raw material B": ["rawmaterialb", "b"],
    "Raw material C": ["rawmaterialc", "c"],

    # Targets
    "Initial Setting Time": ["initialsettingtime", "initialtime", "ist"],
    "Final Setting Time":   ["finalsettingtime", "f1nalsettingtime", "finaltime", "fst"],
    "CS 3 Days":            ["cs3days", "compressivestrength3days", "cs3", "psi3days"],
    "CS 7 Days":            ["cs7days", "compressivestrength7days", "cs7", "psi7days"],
    "CS 28 Days":           ["cs28days", "compressivestrength28days", "cs28", "psi28days"],
    "at170 degrees F after 24 hours": [
        "at170degreesfafter24hours", "at170fafter24hours", "170f24h", "170fafter24hours"
    ],
    "Flexural Strength": ["flexuralstrength", "fs"],
}

TARGET_KEYS = [
    "Initial Setting Time",
    "Final Setting Time",
    "CS 3 Days",
    "CS 7 Days",
    "CS 28 Days",
    "at170 degrees F after 24 hours",
    "Flexural Strength",
]

def map_columns(df: pd.DataFrame):
    """Return mapping from logical names to actual df columns, plus list of missing logicals."""
    avail = {norm(c): c for c in df.columns}
    mapping = {}
    missing = []
    for logical, variants in WANTED.items():
        hit = None
        for v in variants:
            if v in avail:
                hit = avail[v]
                break
        if hit is None:
            missing.append(logical)
        else:
            mapping[logical] = hit
    return mapping, missing

def fill_per_target(df: pd.DataFrame, X_cols: list[str], y_cols: list[str], rnd_state=42):
    work = df.copy()
    # ensure numeric
    work[X_cols + y_cols] = wo
    rk[X_cols + y_cols].apply(pd.to_numeric, errors='coerce')

    total_filled_cells = 0
    any_row_filled = np.zeros(len(work), dtype=bool)
    scores = []

    for y_col in y_cols:
        # rows with complete X and this target present (for training)
        mask_X_ok = work[X_cols].notna().all(axis=1)
        mask_y_present = work[y_col].notna()
        train_idx = work.index[mask_X_ok & mask_y_present]

        if len(train_idx) < 2:
            # not enough to learn anything robust
            scores.append({"Target": y_col, "R2": np.nan, "MSE": np.nan})
            continue

        X_train = work.loc[train_idx, X_cols]
        y_train = work.loc[train_idx, y_col]

        rf = RandomForestRegressor(n_estimators=400, random_state=rnd_state)

        # if we have room, do a quick holdout to report R^2/MSE; else fit-all
        if len(train_idx) >= 8:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train, y_train, test_size=0.25, random_state=rnd_state
            )
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_te)
            r2 = r2_score(y_te, y_pred)
            mse = mean_squared_error(y_te, y_pred)
            scores.append({"Target": y_col, "R2": float(r2), "MSE": float(mse)})
        else:
            rf.fit(X_train, y_train)
            scores.append({"Target": y_col, "R2": np.nan, "MSE": np.nan})

        # rows to fill: complete X and this target missing
        fill_idx = work.index[mask_X_ok & work[y_col].isna()]
        if len(fill_idx) > 0:
            preds = rf.predict(work.loc[fill_idx, X_cols])
            preds = np.round(preds, 2)  # <-- round to 2 decimals ONLY for filled values
            work.loc[fill_idx, y_col] = preds
            total_filled_cells += len(fill_idx)
            any_row_filled[fill_idx] = True

    filled_rows_any = int(any_row_filled.sum())
    return work, pd.DataFrame(scores), total_filled_cells, filled_rows_any

#main: process all sheets
xls = pd.ExcelFile(EXCEL_PATH)
summary_rows = []

with pd.ExcelWriter(OUT_PATH, engine="openpyxl", mode="w") as writer:
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=0)
        mapping, missing = map_columns(df_sheet)

        # Require X1, B, C, plus at least one target to proceed
        required = ["X1", "Raw material B", "Raw material C"]
        missing_required = [r for r in required if r not in mapping]
        if missing_required:
            df_sheet.to_excel(writer, sheet_name=f"{sheet} (unmodified)", index=False)
            summary_rows.append({
                "Sheet": sheet, "Status": "Missing required columns",
                "Filled Cells": 0, "Filled Rows": 0,
                "Target": "", "R2": np.nan, "MSE": np.nan,
                "Note": f"Missing: {', '.join(missing_required)}"
            })
            continue

        X_cols = [mapping["X1"], mapping["Raw material B"], mapping["Raw material C"]]

        # Only include targets that were successfully mapped on this sheet
        y_cols = [mapping[k] for k in TARGET_KEYS if k in mapping]
        if not y_cols:
            df_sheet.to_excel(writer, sheet_name=f"{sheet} (unmodified)", index=False)
            summary_rows.append({
                "Sheet": sheet, "Status": "No mapped targets",
                "Filled Cells": 0, "Filled Rows": 0,
                "Target": "", "R2": np.nan, "MSE": np.nan,
                "Note": "No target columns recognized on this sheet"
            })
            continue

        filled_df, scores_df, filled_cells, filled_rows = fill_per_target(df_sheet, X_cols, y_cols)

        # write filled sheet
        filled_df.to_excel(writer, sheet_name=f"{sheet} (filled)", index=False)

        # add per-target metrics
        for _, r in scores_df.iterrows():
            summary_rows.append({
                "Sheet": sheet, "Status": "OK",
                "Filled Cells": filled_cells, "Filled Rows": filled_rows,
                "Target": r["Target"], "R2": r["R2"], "MSE": r["MSE"],
                "Note": ""
            })

    # Summary tab
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

print(f"Processed {len(xls.sheet_names)} sheets. Output -> {OUT_PATH}")