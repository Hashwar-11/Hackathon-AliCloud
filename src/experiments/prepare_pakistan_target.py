"""
Convert Maheen's Pakistani dataset into the SGCC target format Stage 5 needs.

Handles BOTH plausible layouts of pakistan_sgcc_format.xlsx defensively
(the file could not be inspected when this was written — the terminal was
down — so this script diagnoses the schema itself and fails loudly with a
printout of what it found if neither layout matches):

  WIDE (SGCC-like):  House_ID/CONS_NO | FLAG/Label | <one column per day>
  LONG (smart meter): House_ID | Date_Time | Usage_kW | ... | Label

Output: data/raw/pakistan/pakistan_target.csv with columns
    CONS_NO, FLAG, <daily kWh columns sorted chronologically>

Run:
    python -m src.experiments.prepare_pakistan_target \
        --input pakistan_sgcc_format.xlsx [pakistan_theft_cases.xlsx ...] \
        [--out data/raw/pakistan/pakistan_target.csv]

Multiple --input files are merged: sheets from every workbook are converted
and concatenated (the normal houses and the theft scenarios live in separate
workbooks), then deduplicated on CONS_NO.
"""
import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_OUT = "data/raw/pakistan/pakistan_target.csv"

# columns we recognise (matched case-insensitively)
ID_ALIASES = {"cons_no", "house_id", "customer_id", "id"}
LABEL_ALIASES = {"flag", "label", "theft", "is_theft"}
TIME_ALIASES = {"date_time", "datetime", "date", "timestamp", "time"}
USAGE_ALIASES = {"usage_kw", "usage_kwh", "usage", "consumption", "total_kw"}


def _norm(col: str) -> str:
    return str(col).strip().lower()


def load_sheets(path: str):
    """Return a list of (sheet_name, DataFrame). For xlsx/xls this reads EVERY
    sheet (theft rows are often on a separate sheet from the normal rows, and
    pd.read_excel's default reads only the first one -> silently drops them).
    For csv there is a single anonymous sheet."""
    p = str(path).lower()
    if p.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(path)
        sheets = [(name, xl.parse(name)) for name in xl.sheet_names]
        print(f"[prepare-pk] Workbook sheets: {xl.sheet_names}")
        return sheets
    return [("csv", pd.read_csv(path))]


def find_col(df: pd.DataFrame, aliases: set):
    for c in df.columns:
        if _norm(c) in aliases:
            return c
    return None


def diagnose(df: pd.DataFrame):
    """Print what the file actually contains, so a schema mismatch is debuggable."""
    print("[prepare-pk] Columns found:", list(df.columns))
    print(f"[prepare-pk] Rows: {len(df)}")
    print("[prepare-pk] Head:")
    print(df.head(3).to_string())


def convert_long(df: pd.DataFrame, id_col: str, time_col: str, usage_col: str,
                 label_col: str) -> pd.DataFrame:
    """Long smart-meter layout -> one row per house with daily kWh columns."""
    df = df[[id_col, time_col, usage_col, label_col]].copy()
    df.columns = ["CONS_NO", "Date_Time", "Usage", "FLAG"]
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    df["Usage"] = pd.to_numeric(df["Usage"], errors="coerce")
    df = df.dropna(subset=["Date_Time", "Usage"])

    # reading interval -> converts kW readings into kWh per interval
    intervals = df.groupby("CONS_NO")["Date_Time"].apply(
        lambda s: s.sort_values().diff().median()
    )
    median_interval = intervals.median()
    if pd.isna(median_interval):
        raise RuntimeError("Could not determine the reading interval from Date_Time.")
    hours = median_interval.total_seconds() / 3600.0

    if hours >= 20:  # already (roughly) one reading per day
        print("[prepare-pk] Readings look daily — using values as daily kWh directly.")
        daily = df.groupby(["CONS_NO", df["Date_Time"].dt.date])["Usage"].mean()
    else:
        print(f"[prepare-pk] Readings every {hours:.2f} h — energy per interval "
              f"= Usage_kW x {hours:.2f} h, then summed per day.")
        df["kWh"] = df["Usage"] * hours
        daily = df.groupby(["CONS_NO", df["Date_Time"].dt.date])["kWh"].sum()

    wide = daily.unstack()
    wide.columns = [str(c) for c in wide.columns]
    # chronological order (columns are date strings, sort by parsed date)
    wide = wide[sorted(wide.columns, key=lambda c: pd.to_datetime(c))]

    labels = df.groupby("CONS_NO")["FLAG"].agg(lambda s: int(s.mode().iloc[0]))
    out = wide.reset_index()
    out = out.merge(labels.reset_index(), on="CONS_NO")
    cols = ["CONS_NO", "FLAG"] + [c for c in out.columns if c not in ("CONS_NO", "FLAG")]
    return out[cols]


def convert_wide(df: pd.DataFrame, id_col: str, label_col: str) -> pd.DataFrame:
    """Already one row per consumer — just normalise names and order the days."""
    df = df.rename(columns={id_col: "CONS_NO", label_col: "FLAG"})
    day_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    try:  # date-like columns -> sort chronologically
        ordered = sorted(day_cols, key=lambda c: pd.to_datetime(c))
        print("[prepare-pk] Day columns parsed as dates and sorted chronologically.")
    except Exception:  # e.g. day_0..day_N or plain numeric names -> keep file order
        ordered = day_cols
        print("[prepare-pk] Day columns are not date-named — keeping file order.")
    out = df[["CONS_NO", "FLAG"] + ordered].copy()
    # Do NOT force int here — FLAG may hold -1 / NaN / strings; normalize_labels
    # coerces it to {0,1} loudly downstream.
    return out


def convert_sheet(name: str, df: pd.DataFrame):
    """Detect one sheet's layout and convert it to CONS_NO|FLAG|<days>. Returns
    None if the sheet has no usable id+label columns (e.g. a README sheet)."""
    print(f"\n[prepare-pk] === sheet '{name}' ({df.shape[0]} rows x {df.shape[1]} cols) ===")
    diagnose(df)

    id_col = find_col(df, ID_ALIASES)
    label_col = find_col(df, LABEL_ALIASES)
    time_col = find_col(df, TIME_ALIASES)
    usage_col = find_col(df, USAGE_ALIASES)

    if id_col is None or label_col is None:
        print(f"[prepare-pk] sheet '{name}': no id/label columns "
              f"(id={id_col!r}, label={label_col!r}) — skipping it.")
        return None

    if time_col is not None and usage_col is not None:
        print("[prepare-pk] LONG smart-meter layout detected.")
        return convert_long(df, id_col, time_col, usage_col, label_col)
    print("[prepare-pk] WIDE SGCC-like layout detected.")
    return convert_wide(df, id_col, label_col)


def normalize_labels(out: pd.DataFrame) -> pd.DataFrame:
    """Coerce FLAG to {0,1}. Some exports encode normal as -1 (or True/False,
    'normal'/'theft'). Loud about anything it remaps; refuses to guess silently."""
    flag = out["FLAG"]
    uniq = sorted(pd.unique(flag), key=str)
    print(f"[prepare-pk] Raw FLAG values: {uniq}")

    # string labels -> binary
    if flag.dtype == object:
        mapping = {}
        for v in pd.unique(flag):
            lv = str(v).strip().lower()
            mapping[v] = 1 if lv in ("1", "theft", "steal", "fraud", "yes", "true") else 0
        out["FLAG"] = flag.map(mapping).astype(int)
        print(f"[prepare-pk] Remapped string labels via {mapping}")
        return out

    out["FLAG"] = pd.to_numeric(flag, errors="coerce")
    # CONFIRMED convention for this Pakistani export (data owner, 2026-09): FLAG = -1 is
    # the THEFT sentinel, NOT a 'normal' marker. finetune_pakistan.py documents the same
    # convention ("FLAG = -1 means THEFT"). Map -1 -> 1 so the on-disk CSV obeys the
    # project-wide 0=normal / 1=theft convention.
    if (out["FLAG"] == -1).any():
        n_theft = int((out["FLAG"] == -1).sum())
        out.loc[out["FLAG"] == -1, "FLAG"] = 1
        print(f"[prepare-pk] Remapped {n_theft} FLAG=-1 (confirmed THEFT sentinel) -> 1 (theft).")
    if (out["FLAG"] < 0).any():  # any other unexpected negative sentinel
        n_other = int((out["FLAG"] < 0).sum())
        out.loc[out["FLAG"] < 0, "FLAG"] = 1
        print(f"[prepare-pk] WARNING: {n_other} unexpected negative FLAG values (not -1) "
              "-> treated as theft (1); inspect the source export.")
    out["FLAG"] = (out["FLAG"] >= 1).astype(int)  # any positive -> theft
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["pakistan_sgcc_format.xlsx"],
                        help="one or more workbooks/CSVs; all sheets are merged")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    parts = []
    for path in args.input:
        print(f"\n[prepare-pk] >>> reading file: {path}")
        for name, df in load_sheets(path):
            conv = convert_sheet(f"{path}::{name}", df)
            if conv is not None and len(conv):
                parts.append(conv)

    if not parts:
        raise SystemExit("[prepare-pk] No sheet contained usable id+label data. "
                         "See the diagnostic printouts above.")

    # concat across sheets, aligning day columns (outer join keeps all days)
    out = pd.concat(parts, ignore_index=True, sort=False)
    # CONS_NO + FLAG first, then day columns in stable order
    day_cols = [c for c in out.columns if c not in ("CONS_NO", "FLAG")]
    out = out[["CONS_NO", "FLAG"] + day_cols]

    # drop exact-duplicate consumers if the same house appears on several sheets
    before = len(out)
    out = out.drop_duplicates(subset=["CONS_NO"], keep="first")
    if len(out) < before:
        print(f"[prepare-pk] Dropped {before - len(out)} duplicate CONS_NO rows.")

    out = normalize_labels(out)

    n_days = out.shape[1] - 2
    counts = out["FLAG"].value_counts().to_dict()
    print(f"\n[prepare-pk] Converted: {len(out)} consumers x {n_days} daily readings")
    print(f"[prepare-pk] Class balance (0=normal, 1=theft): {counts}")
    if 1 not in counts:
        print("[prepare-pk] WARNING: NO theft (FLAG=1) rows in any source sheet -- this "
              "target file contains NORMALS ONLY. Stage 5 cannot be evaluated.")
    elif 0 not in counts:
        print("[prepare-pk] NOTE: target holds THEFT ONLY (confirmed cases, no normal rows). "
              "A discriminative fine-tune is IMPOSSIBLE on one class (ROC-AUC/precision "
              "undefined; BCE pos_weight=0). Stage 5 therefore runs a ZERO-SHOT transfer "
              "evaluation -- the FROZEN SGCC-trained model scores these confirmed theft "
              "cases and we report detection recall. Run: "
              "python -m src.experiments.stage5_zeroshot_pakistan --config config/config.yaml")
    else:
        print("[prepare-pk] Both classes present -> full fine-tune / transfer is possible.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[prepare-pk] Saved -> {args.out}")
    print("[prepare-pk] Next: python -m src.experiments.run_all_benchmark "
          f"--config config/config.yaml --target_csv {args.out}")


if __name__ == "__main__":
    main()
