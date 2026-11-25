"""Minimal example: evaluate your own regressor on the DefectShift splits and read it against the randomized controls.

For each selected split the script follows the protocol of the paper for one repetition:
  - the source-trained model f_S is fitted on the matched source sample A_match and the target-trained reference f_T, of the same
    model class, on the matched target sample B_match (both with n* records, drawn from A_train and B_tr);
  - both are evaluated on the source test set A_heldout and on the target test set B_te;
  - the ID-OOD gap compares f_S on B_te with f_S on A_heldout, and the target-reference gap compares f_S with f_T on B_te
    (RMSE and MAE, in percent; positive values mean a loss of the source-trained model);
  - the target rows are the lot-disjoint rows of splits/target_test_lot_overlap.csv (add --all-rows to use every B_te record).
A challenge domain is then placed among the randomized lot-split controls by the number k of controls whose target-reference gap is
smaller. k locates the domain among the observed controls; it is not a significance test.

Simplifications compared with the paper: one repetition, default hyperparameters (the paper selects them on the source side), and no
early stopping. To evaluate another model, replace make_model() with any scikit-learn style regressor.

Usage:
    python examples/evaluate_new_model.py                           # seven challenge domains and ten controls, repetition 0
    python examples/evaluate_new_model.py --slots cat_050 placebo@9001 --out gaps.csv
    python examples/evaluate_new_model.py --extra-controls          # k is counted among all 50 controls (40 more, regenerated from fixed seeds)
"""
import argparse
import json
import os
import sys
import tempfile

# Split caches are not needed for a one-off computation; keep them out of the repository.
os.environ.setdefault("DEFECTSHIFT_CACHE_DIR", tempfile.mkdtemp(prefix="defectshift_example_"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from defectshift import config, extra_controls  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_data  # noqa: E402
from defectshift.evaluation import gaps_from_predictions, k_among_controls  # noqa: E402
from defectshift.splits import B_view, get_A, matched, repeat_split  # noqa: E402

OVERLAP = os.path.join(ROOT, "splits", "target_test_lot_overlap.csv")


def make_model():
    """The model to evaluate. Replace it with your own estimator (fit(X, y) / predict(X), NaN allowed in X)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=0)


def evaluate_slot(df, y, lot, name, overlap, repeat=0, model_factory=make_model, all_rows=False):
    """ID-OOD gap and target-reference gap (RMSE and MAE, %) of one split in one repetition."""
    slot, mA, mB = build_one(df, name)
    a = get_A(slot, df.index[mA], y, lot)
    bv = B_view(slot, df.index[mB], y, lot, "benchmark")
    a_train, _ = repeat_split(a["A_dev"], y, lot, repeat)
    n_star, a_match, b_match = matched(a_train, bv["B_tr"], repeat)
    cat_cols, num_cols = feature_sets(df, name)
    cols = cat_cols + num_cols
    f_s = model_factory().fit(df.loc[a_match, cols], y.loc[a_match])
    f_t = model_factory().fit(df.loc[b_match, cols], y.loc[b_match])
    b_te = list(bv["B_te"])
    marked = overlap.loc[overlap["slot"] == name].set_index("row")["source_train_lot_overlap"]
    rows = b_te if all_rows or name.startswith("placebo@") else [i for i in b_te if not marked.loc[i]]     # the controls contain no shared lots
    held = list(a["A_heldout"])
    out = dict(slot=name, n_star=n_star, n_target_rows=len(rows))
    out.update(gaps_from_predictions(y.loc[held], f_s.predict(df.loc[held, cols]), y.loc[rows], f_s.predict(df.loc[rows, cols]), f_t.predict(df.loc[rows, cols])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slots", nargs="*", default=None, help="split names; default: challenge domains and controls")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--all-rows", action="store_true", help="evaluate all B_te records instead of the lot-disjoint rows")
    ap.add_argument("--out", default=None, help="write the table to this CSV file")
    ap.add_argument("--extra-controls", action="store_true",
                    help="also use the 40 additional controls (seeds 9011 to 9050; regenerated from fixed seeds, not canonical files)")
    args = ap.parse_args()
    if args.extra_controls or any(s.startswith("placebo@") and int(s.split("@")[1]) > 9010 for s in args.slots or []):
        extra_controls.register()
    final = json.load(open(os.path.join(ROOT, "splits", "final_domains.json")))
    domains = final["temporal"] + final["categorical"]
    controls = [f"placebo@{s}" for s in config.PLACEBO_B_TARGET]
    slots = args.slots or domains + controls
    df, y = load_data()
    lot, overlap = df[config.LOT_COL], pd.read_csv(OVERLAP)
    table = pd.DataFrame([evaluate_slot(df, y, lot, s, overlap, args.repeat, all_rows=args.all_rows) for s in slots])
    ctl = table[table.slot.str.startswith("placebo@")]
    for measure in ("rmse", "mae"):
        col = f"target_reference_gap_{measure}"
        table[f"k_{measure}"] = [k_among_controls(g, ctl[col]) if not s.startswith("placebo@") and len(ctl) else np.nan
                                 for s, g in zip(table.slot, table[col])]
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    if len(ctl):
        print(f"\nk: number of the {len(ctl)} controls with a smaller target-reference gap (a position among the controls, not a significance test)")
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
