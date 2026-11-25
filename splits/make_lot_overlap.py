"""Write the lot-overlap annotation of the target test records.

For every challenge domain and every randomized lot-split control, the table lists each record of the target test set B_te and
marks whether its production lot also occurs in the source development set A_dev of the same split. A record with the mark is
trained on indirectly in the sense that the model saw other records of the same lot; the lot-disjoint evaluation of the paper
leaves such records out. The marks are an evaluation mask: they do not change any split.

Columns of the output
    slot                        candidate name, as accepted by defectshift.candidates.build_one
    row                         row position of the record in data/data.parquet
    source_train_lot_overlap    True if the lot of the record occurs in A_dev of the same split

The companion table source_heldout_lot_overlap.csv lists each record of the source test set A_heldout and marks whether its
production lot also occurs in the target test set B_te of the same split (column target_test_lot_overlap). It is also an
evaluation mask and changes no split.

By construction, the randomized controls assign whole lots to the source or the target side, so their marks are all False.

Usage:
    python splits/make_lot_overlap.py            # data/data.parquet -> splits/target_test_lot_overlap.csv and splits/source_heldout_lot_overlap.csv
    python splits/make_lot_overlap.py --check    # fail if the existing table differs from a fresh computation
"""
import argparse
import os
import sys
import tempfile
from io import StringIO

# Split caches are not needed for a one-off computation; keep them out of the repository.
os.environ.setdefault("DEFECTSHIFT_CACHE_DIR", tempfile.mkdtemp(prefix="defectshift_lot_overlap_"))

import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from defectshift import config  # noqa: E402
from defectshift.candidates import build_one, load_data  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402

FINAL_DOMAINS = os.path.join(HERE, "final_domains.json")


def slots():
    """Challenge domains (splits/final_domains.json, temporal first) followed by the ten randomized controls."""
    import json
    final = json.load(open(FINAL_DOMAINS))
    return final["temporal"] + final["categorical"] + [f"placebo@{s}" for s in config.PLACEBO_B_TARGET]


def build(df, y, lot):
    """Return (target_test table, source_heldout table)."""
    parts, src_parts = [], []
    for name in slots():
        slot, mA, mB = build_one(df, name)
        a = get_A(slot, df.index[mA], y, lot)
        b_te = list(B_view(slot, df.index[mB], y, lot, "benchmark")["B_te"])
        a_dev_lots = set(lot.loc[list(a["A_dev"])])
        parts.append(pd.DataFrame({
            "slot": name,
            "row": b_te,
            "source_train_lot_overlap": [lot.loc[i] in a_dev_lots for i in b_te],
        }))
        a_ho = list(a["A_heldout"])
        b_te_lots = set(lot.loc[b_te])
        src_parts.append(pd.DataFrame({
            "slot": name,
            "row": a_ho,
            "target_test_lot_overlap": [lot.loc[i] in b_te_lots for i in a_ho],
        }))
    return pd.concat(parts, ignore_index=True), pd.concat(src_parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "target_test_lot_overlap.csv"))
    ap.add_argument("--out-source", default=os.path.join(HERE, "source_heldout_lot_overlap.csv"))
    ap.add_argument("--check", action="store_true", help="compare with --out and --out-source instead of writing them")
    args = ap.parse_args()
    df, y = load_data()
    pairs = zip((args.out, args.out_source), build(df, y, df[config.LOT_COL]), ("source_train_lot_overlap", "target_test_lot_overlap"))
    for path, table, col in pairs:
        if args.check:
            existing = pd.read_csv(path)
            fresh = pd.read_csv(StringIO(table.to_csv(index=False)))
            if not existing.equals(fresh):
                sys.exit(f"{path} differs from the table computed from {config.DATA_PATH}")
            print(f"{path} matches {config.DATA_PATH}")
        else:
            table.to_csv(path, index=False)
            print(f"wrote {path} ({len(table)} rows, {table['slot'].nunique()} splits, {int(table[col].sum())} marked)")


if __name__ == "__main__":
    main()
