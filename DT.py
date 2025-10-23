import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

excel_path = "PolymerData.xlsx"
sheet_name = 2  # change based on polymer

id_col = "Composition Number"
# Because A + B + C = 100
X_cols = ["Flyash C class", "Raw Material B", "Raw Material C"]

y_cols = [
    "Initial Setting Time",
    "Final Setting Time",
    "CS 3 Days",
    "CS 7 Days",
    "CS 28 Days",
    "at 170 degrees F after 24 hours",
    "Flexural Strength",
]

MIN_KNOWN = 5
TEST_SIZE = 0.25
RANDOM_STATE = 42
PLOT_PARITY = True

# Pretty printing
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)

df = pd.read_excel(excel_path, sheet_name=sheet_name)
df = df.replace(["", " ", "NA", "N/A", "na", "--", "unknown", "Unknown"], np.nan)

# keep only columns we care about - using known data to predict blank cells
keep = [c for c in [id_col] + X_cols + y_cols if c in df.columns]
df = df[keep].copy()

summary_rows = []


def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, rmse


# pre target loop
for target_col in y_cols:
    if target_col not in df.columns:
        continue

    print(f"\n {target_col} ")
    known_mask = df[target_col].notna()
    unknown_mask = ~known_mask

    n_known, n_unknown = int(known_mask.sum()), int(unknown_mask.sum())
    if n_known < MIN_KNOWN:
        print(f"Not enough known rows (have {n_known}). Skipping.")
        continue

    # Prepare known data
    X_known = df.loc[known_mask, X_cols].astype(float)
    y_known = df.loc[known_mask, target_col].astype(float)

    # GridSearchCV (5-fold) to tune the tree
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_samples_leaf": [2, 3, 4, 5, 6, 8, 10],
        "min_samples_split": [4, 6, 8, 10, 12, 16],
        "ccp_alpha": [0.0, 0.0005, 0.001, 0.002],  # pruning(removing sections that are not useful)
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
    print(f"Best parameters: {best_params}")
    print(f"Cross-val best R²: {gs.best_score_:.3f}")

    # traing and testing splits of data
    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = DecisionTreeRegressor(random_state=RANDOM_STATE, **best_params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_r2, test_rmse = metrics(y_test, y_pred_test)
    print(f"Held-out test: R²={test_r2:.3f} | RMSE={test_rmse:.3f}")

    # parity plot (Predicted vs True) for this target
    if PLOT_PARITY and len(X_test) > 0:
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred_test, s=35, alpha=0.8)
        plt.xlabel(f"True {target_col}")
        plt.ylabel(f"Predicted {target_col}")
        plt.title(f"Decision Tree — {target_col}: Prediction vs Truth")
        lo = min(y_test.min(), y_pred_test.min())
        hi = max(y_test.max(), y_pred_test.max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")  # 1:1 line
        plt.tight_layout()
        plt.show()

    # refits known data, predicts unknown
    model.fit(X_known, y_known)

    pred_col = f"{target_col}_predicted"
    if pred_col not in df.columns:
        df[pred_col] = np.nan

    if n_unknown > 0:
        X_unknown = df.loc[unknown_mask, X_cols].astype(float)
        preds = model.predict(X_unknown)
        df.loc[unknown_mask, pred_col] = preds
        print(f"Filled {len(preds)} missing values.")

    summary_rows.append({
        "Target": target_col,
        "Known": n_known,
        "Missing to fill": n_unknown,
        "CV best R2": round(gs.best_score_, 3),
        "Test R2": round(test_r2, 3),
        "Test RMSE": round(test_rmse, 3),
        "Params": best_params
    })

# table
final_cols = (
        [c for c in [id_col] if c in df.columns] +
        X_cols +
        sum([[y, f"{y}_predicted"] for y in y_cols if y in df.columns], [])
)
preview = df[final_cols].copy()

# Round numeric columns and rename predicted headers
num_cols = preview.select_dtypes(include="number").columns
preview[num_cols] = preview[num_cols].round(3)
for y in y_cols:
    pc = f"{y}_predicted"
    if pc in preview.columns:
        preview = preview.rename(columns={pc: f"{y} (pred)"})

print("\nFinal Data Prediction")
print(preview.to_string(index=False))

df.to_excel("DT_Polymer3_predictions.xlsx", index=False)

if summary_rows:
    print("\nSummary (CV + held-out test):")
    print(pd.DataFrame(summary_rows).to_string(index=False))