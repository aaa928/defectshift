"""Runner of the time-ordered configuration: runs the existing runner (06_benchmark.py) unchanged and replaces, at run time, only
splits.B_view (benchmark stage), splits.repeat_split and splits.matched. The shared training code and the configuration file are not changed.

Modes
  time_ordered : temporal domains (x_time, y_time). The target is split in time order (splits_time_ordered); the source training candidates are
                 only the lots that precede the start of the test period, with one buffer lot left out.
                 n*_FO = min(|A_train past|, |B_tr past|). The source-side tuning of the canonical benchmark is reused.
  size_only    : temporal domains. The canonical randomized lot split is kept and only the number of training rows is reduced to n*_FO of the
                 same repetition (separates the size effect).
  controls     : randomized lot-split controls (placebo@seed). The same constraint as time_ordered is applied with the time of --axis
                 (source and reference use earlier lots only). The tuning run of the controls is reused.
Eight models (Ridge, kNN, XGBoost, CatBoost, MLP, TabM, TabPFN, TabICL), regimes ID_match and OOD_match only. The results are a sensitivity
analysis, not canonical. A file time_ordered.json (mode, axis, split figures, n* per repetition) is written to the run folder (the manifest
records an ordinary lot split, so this file is the marker).
Usage: python scripts/extra/run_time_ordered.py --screening-run-id <screening> --mode time_ordered --dry-run --only 'x_time<=2099-05-04 21:18:39'
       python scripts/extra/run_time_ordered.py --screening-run-id <screening> --mode controls --axis x_time --dry-run --controls 9001
       python scripts/extra/run_time_ordered.py --screening-run-id <screening> --hpo-run-id <hpo> --mode time_ordered --run-id time_ordered_time_ordered_x_time \
           --models Ridge,XGBoost --repeats 1 --only '...' [06_benchmark.py arguments...]
"""
import argparse
import json
import os
import runpy
import sys

import numpy as np
import pandas as pd

from defectshift import config, splits  # noqa: E402
from defectshift.candidates import build_one, load_data, parse_candidate  # noqa: E402
from defectshift.runio import control_names, load_final_domains  # noqa: E402
from defectshift.splits_time_ordered import forward_view  # noqa: E402

M8 = ("Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL")
REGIMES = "ID_match,OOD_match"
FORBIDDEN = ("--no-save", "--regimes")
AXES = ("x_time", "y_time")


def key(ix):
    ix = pd.Index(ix)
    return (len(ix), ix[0], ix[-1])


def same(a, b, what):
    """Checks that the prepared result found by its fingerprint (key) equals the index actually received (guards against a different content with the same length and end values)."""
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise SystemExit(f"{what}: the index of the preparation step differs from the index of the runner")


def prepare(mode, names, axis, repeats):
    """Per name: the time-ordered split, the filtered (A_train, A_val) of every repetition and n*_FO. Returns {name: dict}"""
    df, y = load_data()
    lot = df[config.LOT_COL]
    out = {}
    for name in names:
        kind, _ = parse_candidate(name)
        ax = kind if mode in ("size_only", "time_ordered") else axis
        if ax not in AXES:
            raise SystemExit(f"{name}: the time axis is not a temporal domain ({ax})")
        slot, mA, mB = build_one(df, name)
        a = splits.get_A(slot, df.index[mA], y, lot)
        fv = forward_view(a["A_dev"], df.index[mB], lot, df[ax])
        allowed = set(fv["A_allowed"])
        reps = {}
        for i in range(repeats):
            a_train, a_val = splits.repeat_split(a["A_dev"], y, lot, i)
            tr = a_train[a_train.isin(allowed)]
            va = a_val[a_val.isin(allowed)]
            reps[i] = dict(n_A_train=len(a_train), n_A_train_fo=len(tr), n_A_val=len(a_val), n_A_val_fo=len(va),
                           n_star_fo=min(len(tr), len(fv["B_tr"])))
        canon = splits.B_view(slot, df.index[mB], y, lot, "benchmark")
        out[name] = dict(slot=slot, axis=ax, fv=fv, allowed=allowed, reps=reps, B_idx=pd.Index(df.index[mB]), A_dev=pd.Index(a["A_dev"]), A_dev_key=key(a["A_dev"]),
                         canon_B_tr=pd.Index(canon["B_tr"]), canon_B_tr_key=key(canon["B_tr"]), n_B_tr_canonical=len(canon["B_tr"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("--mode", required=True, choices=("size_only", "time_ordered", "controls"))
    ap.add_argument("--axis", choices=AXES, default=None, help="time axis of the controls mode")
    ap.add_argument("--dry-run", action="store_true", help="only build the splits and print the figures (no training, no GPU)")
    ap.add_argument("--screening-run-id", required=True)
    ap.add_argument("--hpo-run-id", default=None, help="source-side tuning run to reuse (required unless --dry-run)")
    ap.add_argument("--only", default=None, help="comma-separated temporal domain names (modes size_only and time_ordered)")
    ap.add_argument("--controls", default=None, help="comma-separated control seeds, e.g. 9001,9002 (mode controls)")
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS_BENCHMARK)
    a, rest = ap.parse_known_args()
    bad = [x for x in rest if x.split("=")[0] in FORBIDDEN]
    if bad:
        raise SystemExit(f"arguments not allowed in this run: {bad}")
    if a.mode == "controls":
        if not a.controls or a.only or a.axis is None:
            raise SystemExit("mode controls needs --controls and --axis and does not use --only")
        names = control_names(a.controls, None)
    else:
        if a.controls or a.axis or not a.only:
            raise SystemExit("modes size_only and time_ordered use only --only (temporal domains), not --controls or --axis")
        domains, _ = load_final_domains(a.screening_run_id)
        names = a.only.split(",")
        if [n for n in names if n not in domains or parse_candidate(n)[0] not in AXES]:
            raise SystemExit(f"names that are not selected temporal domains: {names}")
    prep = prepare(a.mode, names, a.axis, a.repeats)
    report = {n: dict(axis=p["axis"], split=p["fv"]["info"], n_B_tr_canonical=p["n_B_tr_canonical"],
                      repeats=p["reps"]) for n, p in prep.items()}
    if a.dry_run:
        print(json.dumps(dict(mode=a.mode, axis=a.axis, domains=report), ensure_ascii=False, indent=1))
        return
    if not a.hpo_run_id:
        raise SystemExit("--hpo-run-id is required")
    flag = lambda name: rest[rest.index(name) + 1] if name in rest else None
    rid = flag("--run-id")
    if not rid or not rid.startswith(f"time_ordered_{a.mode}"):
        raise SystemExit(f"--run-id must start with time_ordered_{a.mode}: {rid}")
    models = (flag("--models") or "").split(",")
    if not models[0] or not set(models) <= set(M8):
        raise SystemExit(f"--models is required and must be within {M8}: {models}")
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "runs", rid)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(dict(mode=a.mode, axis=a.axis, canonical=False, domains=report),
              open(os.path.join(run_dir, "time_ordered.json"), "w"), ensure_ascii=False, indent=1, default=str)

    by_slot = {p["slot"]: p for p in prep.values()}
    by_A = {p["A_dev_key"]: p for p in prep.values()}
    by_canonB = {p["canon_B_tr_key"]: p for p in prep.values()}
    by_foB = {key(p["fv"]["B_tr"]): p for p in prep.values()}
    orig_view, orig_rep, orig_match = splits.B_view, splits.repeat_split, splits.matched

    def view(slot, B_idx, y, groups, stage):
        if stage == "benchmark" and a.mode in ("time_ordered", "controls"):
            fv = by_slot[slot]["fv"]
            same(by_slot[slot]["B_idx"], B_idx, "B_view B_idx")
            return {k: fv[k] for k in ("B_tr", "B_val", "B_te")}
        return orig_view(slot, B_idx, y, groups, stage)

    def rep(A_dev, y, groups, i):
        a_train, a_val = orig_rep(A_dev, y, groups, i)
        if a.mode in ("time_ordered", "controls"):
            p = by_A[key(A_dev)]
            same(p["A_dev"], A_dev, "repeat_split A_dev")
            ok = p["allowed"]
            a_train, a_val = a_train[a_train.isin(ok)], a_val[a_val.isin(ok)]
        return a_train, a_val

    def match(A_train, B_tr, i):
        if a.mode == "size_only":
            p = by_canonB[key(B_tr)]
            same(p["canon_B_tr"], B_tr, "matched canonical B_tr")
            n = p["reps"][i]["n_star_fo"]
            assert n <= min(len(A_train), len(B_tr))
            A_m, B_m = splits.subsample(A_train, n, config.RANDOM_STATE + i), splits.subsample(B_tr, n, config.RANDOM_STATE + i)
            return n, A_m, B_m
        p = by_foB[key(B_tr)]
        same(p["fv"]["B_tr"], B_tr, "matched time-ordered B_tr")
        n, A_m, B_m = orig_match(A_train, B_tr, i)
        assert n == p["reps"][i]["n_star_fo"], "n*_FO differs from the preparation step"
        return n, A_m, B_m

    splits.B_view, splits.repeat_split, splits.matched = view, rep, match
    inject = ["--screening-run-id", a.screening_run_id, "--hpo-run-id", a.hpo_run_id, "--regimes", REGIMES, "--repeats", str(a.repeats)]
    inject += ["--controls", a.controls] if a.mode == "controls" else ["--only", a.only]
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "06_benchmark.py")
    sys.argv = [script] + inject + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
