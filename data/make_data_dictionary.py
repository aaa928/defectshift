"""Write the data dictionary: type, dtype, missing rate and number of distinct values of every column (not given for the target and the timestamps).

Usage:
    python data/make_data_dictionary.py                       # data/data.parquet -> data/data_dictionary.csv
    python data/make_data_dictionary.py --check               # fail if the existing dictionary differs
"""
import argparse
import os
import sys
from io import StringIO

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "y"
TIME_COLUMNS = ("x_time", "y_time")


def column_type(name):
    if name == TARGET:
        return "target (regression)"
    if name in TIME_COLUMNS:
        return "timestamp"
    if name.startswith("cat_"):
        return "categorical"
    if name.startswith("num_"):
        return "numerical"
    raise ValueError(f"unexpected column name: {name}")


def build(df):
    """One row per column, in the column order of the data file."""
    return pd.DataFrame({
        "column": df.columns,
        "type": [column_type(c) for c in df.columns],
        "dtype": [str(t) for t in df.dtypes],
        "missing_rate": df.isna().mean().round(4).to_numpy(),
        "n_unique": [pd.NA if c == TARGET or c in TIME_COLUMNS else n for c, n in df.nunique().items()],   # not given for the target and the timestamps
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(HERE, "data.parquet"))
    ap.add_argument("--out", default=os.path.join(HERE, "data_dictionary.csv"))
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    args = ap.parse_args()
    table = build(pd.read_parquet(args.data))
    if args.check:
        existing = pd.read_csv(args.out)
        fresh = pd.read_csv(StringIO(table.to_csv(index=False)))
        if not existing.equals(fresh):
            sys.exit(f"{args.out} differs from the dictionary computed from {args.data}")
        print(f"{args.out} matches {args.data}")
        return
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(table)} columns)")


if __name__ == "__main__":
    main()
