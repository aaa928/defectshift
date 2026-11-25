"""Runner of the redrawn target test splits: the target test split of a challenge domain is redrawn with the screening partition
B_screen fixed (defectshift.splits_redraw) and the existing runner (06_benchmark.py) is run unchanged. At run time only the benchmark
stage of splits.B_view is replaced by the redrawn split; the shared training code and the configuration file are not changed.
The source-side tuning does not depend on the target split, so the tuning run of the canonical benchmark is reused. Eight models
(Ridge, kNN, XGBoost, CatBoost, MLP, TabM, TabPFN, TabICL), regimes ID_match and OOD_match only. The redrawn splits are a sensitivity
analysis, not canonical. A file redrawn_split.json is written to the run folder (the manifest records an ordinary lot split, so this
file is the marker).
Usage: python scripts/extra/run_redrawn_target_splits.py --screening-run-id <screening> --resplit 1 --dry-run
           # only checks split sizes and overlap (no training, no GPU)
       python scripts/extra/run_redrawn_target_splits.py --screening-run-id <screening> --hpo-run-id <hpo> --resplit 1 \
           --run-id redrawn_s1 --models Ridge,kNN,... [06_benchmark.py arguments...]
"""
import argparse
import json
import os
import runpy
import sys

from defectshift import config, splits  # noqa: E402
from defectshift.candidates import build_one, load_data  # noqa: E402
from defectshift.runio import load_final_domains  # noqa: E402
from defectshift.splits_redraw import RESPLIT_SEEDS, TE_FRAC, VAL_FRAC, B_view_resplit  # noqa: E402

M8 = ("Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL")
REGIMES = "ID_match,OOD_match"
FORBIDDEN = ("--controls", "--no-save")


def summarize(k, screening_run_id, only=None):
    """Size and number of lots of the canonical and the redrawn split of every domain. Checks that B_screen is the same and
    that nothing overlaps. Returns {domain: {...}}"""
    domains, _ = load_final_domains(screening_run_id)
    if only:
        domains = [d for d in domains if d in only]
    df, y = load_data()
    lot = df[config.LOT_COL]
    out = {}
    for name in domains:
        slot, mA, mB = build_one(df, name)
        B_idx = df.index[mB]
        canon = splits.B_view(slot, B_idx, y, lot, "screening")
        bv = B_view_resplit(slot, B_idx, y, lot, "benchmark", k)
        scr = B_view_resplit(slot, B_idx, y, lot, "screening", k)
        if not scr["B_screen"].equals(canon["B_screen"]):
            raise SystemExit(f"{name}: B_screen of the redrawn split differs from the canonical one")
        a = splits.get_A(slot, df.index[mA], y, lot)
        te_c = splits.B_view(slot, B_idx, y, lot, "benchmark")["B_te"]
        shared = lambda ix: float(lot.loc[ix].isin(lot.loc[a["A_dev"]].unique()).mean())
        out[name] = dict(n_B=len(B_idx), n_screen=len(canon["B_screen"]),
                         canonical=dict(B_tr=len(canon["B_tr"]), B_te=len(te_c), te_lots=int(lot.loc[te_c].nunique()), te_shared_rows=shared(te_c)),
                         resplit=dict(B_tr=len(bv["B_tr"]), B_val=len(bv["B_val"]), B_te=len(bv["B_te"]), te_lots=int(lot.loc[bv["B_te"]].nunique()),
                                      te_shared_rows=shared(bv["B_te"]), te_overlap_with_canonical_te=float(bv["B_te"].isin(te_c).mean()),
                                      tr_pos=float((y.loc[bv["B_tr"]] > 0).mean()), te_pos=float((y.loc[bv["B_te"]] > 0).mean())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("--resplit", type=int, required=True, choices=(1, 2, 3, 4, 5), help="redraw number (split seed RESPLIT_SEEDS[k-1])")
    ap.add_argument("--dry-run", action="store_true", help="only build the splits and check sizes and overlap (no training)")
    ap.add_argument("--screening-run-id", required=True)
    ap.add_argument("--hpo-run-id", default=None, help="source-side tuning run of the canonical benchmark (required unless --dry-run)")
    ap.add_argument("--only", default=None, help="comma-separated domain names (for smoke runs)")
    a, rest = ap.parse_known_args()
    bad = [x for x in rest if x.split("=")[0] in FORBIDDEN]
    if bad:
        raise SystemExit(f"arguments not allowed in a redrawn-split run: {bad}")
    flag = lambda name: rest[rest.index(name) + 1] if name in rest else None
    rid = flag("--run-id")
    if a.dry_run:
        info = summarize(a.resplit, a.screening_run_id, a.only.split(",") if a.only else None)
        print(json.dumps(dict(resplit=a.resplit, seed=RESPLIT_SEEDS[a.resplit - 1], te_frac_of_rest=TE_FRAC, val_frac_of_rest=VAL_FRAC, domains=info),
                         ensure_ascii=False, indent=1))
        return
    if not a.hpo_run_id:
        raise SystemExit("--hpo-run-id is required")
    if not rid or not rid.startswith(f"redrawn_s{a.resplit}"):
        raise SystemExit(f"--run-id must start with redrawn_s{a.resplit}: {rid}")
    models = (flag("--models") or "").split(",")
    if not models[0] or not set(models) <= set(M8):
        raise SystemExit(f"--models is required and must be within {M8}: {models}")
    if flag("--regimes") not in (None, REGIMES):
        raise SystemExit(f"--regimes must be {REGIMES}")
    inject = []
    if "--screening-run-id" not in rest:
        inject += ["--screening-run-id", a.screening_run_id]
    if "--hpo-run-id" not in rest:
        inject += ["--hpo-run-id", a.hpo_run_id]
    if "--regimes" not in rest:
        inject += ["--regimes", REGIMES]
    info = summarize(a.resplit, a.screening_run_id, a.only.split(",") if a.only else None)
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "runs", rid)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(dict(resplit=a.resplit, seed=RESPLIT_SEEDS[a.resplit - 1], te_frac_of_rest=TE_FRAC, val_frac_of_rest=VAL_FRAC, canonical_b_screen_fixed=True,
                   canonical=False, domains=info), open(os.path.join(run_dir, "redrawn_split.json"), "w"), ensure_ascii=False, indent=1)
    orig = splits.B_view

    def patched(slot, B_idx, y, groups, stage):
        if stage == "benchmark":
            return B_view_resplit(slot, B_idx, y, groups, stage, a.resplit)
        return orig(slot, B_idx, y, groups, stage)

    splits.B_view = patched
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "06_benchmark.py")
    sys.argv = [script] + inject + rest + (["--only", a.only] if a.only else [])
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
