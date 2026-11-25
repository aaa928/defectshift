"""Write the partition table of the data splits.

For every challenge domain and every randomized lot-split control, the table assigns each record of data/data.parquet to exactly one
subset of the split: the source subsets `A_heldout` and `A_dev`, and the target subsets `B_tr`, `B_val`, `B_screen` and `B_te`
(see docs/usage.md). The table is the output of the released split code, written to a file so that the splits can be used without
running the code; it changes no split.

Columns of the output
    slot        candidate name, as accepted by defectshift.candidates.build_one
    row         row position of the record in data/data.parquet
    partition   A_heldout, A_dev, B_tr, B_val, B_screen or B_te

The per-repetition subsets (A_train and A_val, the matched samples, the internal early-stopping split) are not in this table; they are
stored for every fit in the `.npz` files of the benchmark runs.

Usage:
    python splits/make_partitions.py            # data/data.parquet -> splits/partitions.csv
    python splits/make_partitions.py --check    # fail if the existing table differs from a fresh computation
"""
import argparse
import os
import sys
import tempfile
from io import StringIO

# Split caches are not needed for a one-off computation; keep them out of the repository.
os.environ.setdefault("DEFECTSHIFT_CACHE_DIR", tempfile.mkdtemp(prefix="defectshift_partitions_"))

import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from defectshift import config  # noqa: E402
from defectshift.candidates import build_one, load_data  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402
from make_lot_overlap import slots  # noqa: E402

ORDER = ["A_heldout", "A_dev", "B_tr", "B_val", "B_screen", "B_te"]


def build(df, y, lot):
    parts = []
    for name in slots():
        slot, mA, mB = build_one(df, name)
        a = get_A(slot, df.index[mA], y, lot)
        b = B_view(slot, df.index[mB], y, lot, "benchmark")
        b["B_screen"] = B_view(slot, df.index[mB], y, lot, "screening")["B_screen"]
        subsets = {**a, **b}
        rows = pd.DataFrame({"slot": name, "row": [i for k in ORDER for i in subsets[k]],
                             "partition": [k for k in ORDER for _ in subsets[k]]})
        assert not rows.duplicated("row").any() and len(rows) == int(mA.sum() + mB.sum()), name
        parts.append(rows.sort_values("row").reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "partitions.csv"))
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    args = ap.parse_args()
    df, y = load_data()
    table = build(df, y, df[config.LOT_COL])
    if args.check:
        existing = pd.read_csv(args.out)
        fresh = pd.read_csv(StringIO(table.to_csv(index=False)))
        if not existing.equals(fresh):
            sys.exit(f"{args.out} differs from the table computed from {config.DATA_PATH}")
        print(f"{args.out} matches {config.DATA_PATH}")
    else:
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(table)} rows, {table['slot'].nunique()} splits)")


if __name__ == "__main__":
    main()
