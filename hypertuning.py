import re
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


data = "PolymerData.xlsx"
predictions = "dt_pred.xlsx"

random_state = 0
test_split   = 0.25
sheets       = ["Polymer 1", "Polymer 2", "Polymer 3"]

# rounding for float predictions
decimals = {
    "initial setting time": 0,
    "final setting time": 0,
    "3 days": 0,
    "7 days": 0,
    "28 days": 0,
    "at 170 degrees f after 24 hours": 0,
    "flexural strength": 1,
}


def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


input_var = {
    # Features (X): allow Slake OR Flyash variants for X1
    "X1": [
        "rawmateriala(slake)","rawmaterialaslake","rawmateriala",
        "a(slake)","a",
        "flyashfclass","flyashcclass","flyashclassf","flyashclassc"
    ],
    "Raw material B": ["rawmaterialb","b"],
    "Raw material C": ["rawmaterialc","c"],

    # Targets
    "Initial Setting Time": ["initialsettingtime","initialtime","ist"],
    "Final Setting Time":   ["finalsettingtime","finaltime","fst"],
    "CS 3 Days":            ["cs3days","compressivestrength3days","cs3","psi3days"],
    "CS 7 Days":            ["cs7days","compressivestrength7days","cs7","psi7days"],
    "CS 28 Days":           ["cs28days","compressivestrength28days","cs28","psi28days"],
    "at 170 degrees F after 24 hours": [
        "at170degreesfafter24hours","at170fafter24hours","170f24h","170fafter24hours"
    ],
    "Flexural Strength":    ["flexuralstrength","fs"],
}

target_var = [
    "Initial Setting Time",
    "Final Setting Time",
    "CS 3 Days",
    "CS 7 Days",
    "CS 28 Days",
    "at 170 degrees F after 24 hours",
    "Flexural Strength",
]

def map_columns(df: pd.DataFrame):
    """
    Map logical names -> actual df columns using normalized-name lookup.
    Returns (mapping, missing_list).
    """
    avail = {norm(c): c for c in df.columns}
    mapping, missing = {}, []
    for logical, variants in input_var.items():
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

#DT hyperparameters
DT_defaults = dict(random_state=random_state)
#DT_grid = {
#    "max_depth":         [3, 4, 5, 6, 8, None],
#    "min_samples_split": [2, 4, 6, 8, 12, 16],
#    "min_samples_leaf":  [1, 2, 3, 5, 8],
#    "ccp_alpha":         [0.0, 1e-4, 5e-4, 1e-3, 2e-3],
#}

DT_grid = {
    "max_depth":         [8, 12, 16, 20, None],  #deeper trees
    "min_samples_split": [2, 3, 4, 6], #smaller splits
    "min_samples_leaf":  [1, 2], #smaller leave
    "ccp_alpha":         [0.0],  #no cost-complexity pruning
}
min_rows_for_grid = 20

ADD_TRAIN_NOISE = True
NOISE_FRAC = 0.005

#helpers
def to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def decimals_for(name):
    name = str(name).lower().strip()
    for k, d in decimals.items():
        if k in name:
            return d
    return None

def nice_round(s, target_name):
    d = decimals_for(target_name)
    if d is None:
        return s
    return s.round(d).astype(float)

#train and fill cells
def fill_target(df, X_cols, y_col):
    okX  = df[X_cols].notna().all(axis=1)
    have = okX & df[y_col].notna()
    need = okX & df[y_col].isna()

    out = df[y_col].copy()
    stats = {
        "target": y_col,
        "n_observed": int(have.sum()),
        "n_missing":  int(need.sum()),
        "r2_test": np.nan,
        "mae_test": np.nan,
        "rmse_test": np.nan,
        "used_grid": False,
    }

    if have.sum() < 10:
        return nice_round(out, y_col), stats

    X = df.loc[have, X_cols].values
    y = df.loc[have, y_col].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )

    #copy training data and optionally add noise
    Xtr_fit = Xtr.copy()
    if ADD_TRAIN_NOISE and Xtr_fit.shape[0] >= 2:
        rng = np.random.default_rng(seed=random_state)
        std = Xtr_fit.std(axis=0, ddof=0)
        std[std == 0.0] = 1.0
        noise = rng.normal(
            loc=0.0,
            scale=NOISE_FRAC * std,
            size=Xtr_fit.shape
        )
        Xtr_fit = Xtr_fit + noise

    # use Xtr_fit (possibly noisy) when fitting
    if len(ytr) >= min_rows_for_grid:
        k = max(2, min(5, len(ytr)))
        cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
        base = DecisionTreeRegressor(**DT_defaults)
        gs = GridSearchCV(
            base,
            DT_grid,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(Xtr_fit, ytr)
        model = gs.best_estimator_
        stats["used_grid"] = True
    else:
        model = DecisionTreeRegressor(**DT_defaults).fit(Xtr_fit, ytr)

    yhat = model.predict(Xte)
    stats["r2_test"]  = float(r2_score(yte, yhat))
    stats["mae_test"] = float(mean_absolute_error(yte, yhat))
    stats["rmse_test"]= float(np.sqrt(mean_squared_error(yte, yhat)))

    if need.any():
        X_need = df.loc[need, X_cols].values
        preds = model.predict(X_need)

        #tiny jitter so each prediction is slightly different
        n = preds.shape[0]
        if n > 1:
            rng = np.random.default_rng(seed=random_state)
            base_scale = np.std(preds) if np.std(preds) > 0 else 1.0
            jitter = rng.normal(
                loc=0.0,
                scale=NOISE_FRAC * base_scale,
                size=n
            )
            preds = preds + jitter

        out.loc[need] = preds

    return nice_round(out, y_col), stats


#main
def main():
    if not os.path.exists(data):
        raise FileNotFoundError(f"{data} not found")

    xls = pd.ExcelFile(data)
    results = {}
    metrics_rows = []

    for sheet in sheets:
        if sheet not in xls.sheet_names:
            print(f"[skip] {sheet} not in workbook")
            continue

        print(f"\n Processing: {sheet} ")

        # try header=1 first, if mapping fails, fall back to header=0
        df = pd.read_excel(xls, sheet_name=sheet, header=1, dtype=object)
        df = df.dropna(how="all").dropna(axis=1, how="all")

        mapping, missing = map_columns(df)
        if any(req not in mapping for req in ["X1", "Raw material B", "Raw material C"]):
            # retry with header=0 automatically
            df = pd.read_excel(xls, sheet_name=sheet, header=0, dtype=object)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            mapping, missing = map_columns(df)

        required = ["X1", "Raw material B", "Raw material C"]
        missing_required = [r for r in required if r not in mapping]
        y_cols = [mapping[k] for k in target_var if k in mapping]

        if missing_required or not y_cols:
            print(f"[warn] {sheet}: missing {missing_required if missing_required else 'no targets recognized'}")
            results[sheet] = df
            continue

        X_cols = [mapping["X1"], mapping["Raw material B"], mapping["Raw material C"]]

        work = df.copy()
        to_numeric(work, X_cols + y_cols)

        for y in y_cols:
            print("  ->", y)
            filled, info = fill_target(work, X_cols, y)
            work[y] = filled
            info["sheet"] = sheet
            metrics_rows.append(info)

        results[sheet] = work

    with pd.ExcelWriter(predictions, engine="openpyxl") as w:
        for sheet, frame in results.items():
            frame.to_excel(w, sheet_name=sheet, index=False)

        if metrics_rows:
            md = (
                pd.DataFrame(metrics_rows)[
                    ["sheet","target","n_observed","n_missing","r2_test","mae_test","rmse_test","used_grid"]
                ].sort_values(["sheet","target"])
            )
            md.to_excel(w, sheet_name="Imputation_Metrics", index=False)


if __name__ == "__main__":
    main()
