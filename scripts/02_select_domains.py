"""Domain selection: applies the selection rule (defectshift/selection.py) to rows.csv of a screening run and fixes the final domains.

Order (if a step stops, no file is written):
  1) stop if selection/final_domains.json already exists (the final domains are made only once)
  2) run completeness (selection.validate_run): main-run settings, candidate set = the 78 listed + 10 random-split controls, exactly
     24 run keys per candidate (duplicate successful rows of the same key are allowed as retraining history; the last row is the
     record of reference, and their number is recorded)
  3) N_CAT is the fixed value selection.N_CAT (it cannot be changed from the command line) and must equal the value recorded in
     the screening manifest
  4) write final_domains.json (rule, N_CAT, rows/manifest hashes, selection-code hashes), then summary.csv (verdict per candidate)
B_te is not read (rows.csv has B_screen metrics only).

Usage: python scripts/02_select_domains.py --run-id <screening run>
"""
import argparse
import json
import os
import time

import pandas as pd

from defectshift import config, manifest, selection  # noqa: E402
from defectshift.candidates import load_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    run_dir = os.path.join(config.RESULTS_DIR, "screening", "runs", args.run_id)
    rows_path, man_path = os.path.join(run_dir, "rows.csv"), os.path.join(run_dir, "manifest.json")
    out_dir = os.path.join(run_dir, "selection")
    fpath, spath = os.path.join(out_dir, "final_domains.json"), os.path.join(out_dir, "summary.csv")
    if os.path.exists(fpath):
        raise SystemExit(f"{fpath} already exists; the final domains are not made again")

    man = json.load(open(man_path))
    rows = pd.read_csv(rows_path)
    base = load_candidates()
    placebo = [n for n in load_candidates(include_placebo=True) if n not in base]
    run_stats = selection.validate_run(rows, man["args"], base, placebo)   # also checks tune=True and the final budget
    for key, what in (("selection_py_sha256", "selection-rule code (selection.py)"), ("metrics_py_sha256", "selection-score code (metrics.py)")):
        if man["args"].get(key) != selection.code_hashes()[key]:
            raise SystemExit(f"the hash of the {what} at screening time differs from the current code")
    n_cat = selection.N_CAT
    if man["args"].get("selection_n_cat") != n_cat:
        raise SystemExit(f"N_CAT at screening time ({man['args'].get('selection_n_cat')}) differs from the current value ({n_cat})")

    summary = selection.candidate_summary(rows, list(config.SCREENING_MODELS), config.N_REPEATS_SCREENING)
    table, final = selection.select(summary, base, n_cat)
    final.update(run_id=args.run_id, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                 rows_sha256=manifest.file_sha256(rows_path), manifest_sha256=manifest.file_sha256(man_path),
                 **selection.code_hashes(), **run_stats,
                 rule="selection.py module documentation: run completeness check, complete cases, consistently positive models >= 3/4 "
                      "(repetition mean > 0 and > 0 in at least 2/3 of the repetitions), floor(0.25|B|) >= 500, top N_CAT by consensus "
                      "score (median over the 4 models of the repetition-mean target-reference gap (RMSE)) in descending order, ties by name. Temporal: "
                      "the 8th of the 15 cutoffs per axis (median split), pre-specified")
    os.makedirs(out_dir, exist_ok=True)
    selection.write_final_domains(fpath, final)
    table.sort_values(["kind", "cat_rank", "candidate"]).to_csv(spath, index=False)
    print(json.dumps(final, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
