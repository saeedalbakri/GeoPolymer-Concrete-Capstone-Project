#!/usr/bin/env python3
import argparse, re
import numpy as np, pandas as pd

SHEETS = ["Polymer 1", "Polymer 2", "Polymer 3"]

# Forgiving targets after normalizing headers (lowercase, strip non-alphanum)
NEEDLES = {
    "A": ["rawmateriala", "slake"],
    "B": ["rawmaterialb"],
    "C": ["rawmaterialc"],
    "FST": ["finalsettingtime", "fst"],
    "IST": ["initialsettingtime", "ist"],
    "CS_3d": ["compressivestrength3days", "3days", "3d"],
    "CS_7d": ["compressivestrength7days", "7days", "7d"],
    "CS_28d": ["compressivestrength28days", "28days", "28d"],
    "CS_24h170": ["at170degreesfafter24hours", "170f24h", "170f"],
}
LABELS = {
    "FST": "Final Setting Time (FST)",
    "IST": "Initial Setting Time (IST)",
    "CS_3d": "Compressive Strength — 3 days",
    "CS_7d": "Compressive Strength — 7 days",
    "CS_28d": "Compressive Strength — 28 days",
    "CS_24h170": "Compressive Strength — 24h @ 170°F",
}
ORDER = ["FST", "IST", "CS_3d", "CS_7d", "CS_28d", "CS_24h170"]

def norm(s): return re.sub(r"[^0-9a-z]+", "", str(s).lower())

def first_match(cols, needles):
    cn = {c: norm(c) for c in cols}
    for n in needles:
        for c, v in cn.items():
            if n in v: return c
    return None

def find_cols(df):
    mats, props = {}, {}
    for k in ["A","B","C"]:
        c = first_match(df.columns, NEEDLES[k])
        if not c: raise ValueError(f"Missing Raw material {k}. Columns: {list(df.columns)}")
        mats[k] = c
    for k in ORDER:
        c = first_match(df.columns, NEEDLES[k])
        if c: props[k] = c
        elif k != "CS_24h170":
            raise ValueError(f"Missing property {k}. Columns: {list(df.columns)}")
    return mats, props

def corr(x,y):
    d = pd.concat([x,y], axis=1).dropna()
    return np.nan if len(d)<2 else d.corr(numeric_only=True).iloc[0,1]

def s_label(v): 
    a = abs(v); 
    return "strong" if a>=0.70 else ("moderate" if a>=0.30 else "weak")

def d_label(v):
    if np.isnan(v) or v==0: return "neutral"
    return "increases" if v>0 else "reduces"

def fmt(v): return "r=NA" if np.isnan(v) else f"r={v:+.2f}"

def summarize(prop_key, R):
    pick = max(R, key=lambda k: (-1 if np.isnan(R[k]) else abs(R[k])))
    most = abs(R[pick]) if not np.isnan(R[pick]) else float("nan")
    head = f"- **{LABELS[prop_key]}** → **Most influential: {pick}** (|r|={most:.2f})."
    body = "  " + " | ".join(f"{m}: {fmt(R[m])} ({s_label(R[m])}, {d_label(R[m])})" for m in ["A","B","C"])
    return head + "\n" + body + "\n"

def run_sheet(sheet, df):
    mats, props = find_cols(df)
    print(f"# {sheet}")
    for pkey, pcol in props.items():
        R = {m: corr(df[mcol], df[pcol]) for m, mcol in mats.items()}
        print(summarize(pkey, R))
    print()  # spacer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="Polymer_Data.xlsx")
    args = ap.parse_args()

    xls = pd.ExcelFile(args.excel)
    sheets = [s for s in SHEETS if s in xls.sheet_names] or xls.sheet_names
    print(f"Detected sheets: {sheets}\n")

    for s in sheets:
        # Your file uses a 2-row header; real labels are on row 2 => header=1
        df = pd.read_excel(args.excel, sheet_name=s, header=1)
        df.columns = [str(c).strip() for c in df.columns]
        run_sheet(s, df)

if __name__ == "__main__":
    main()
