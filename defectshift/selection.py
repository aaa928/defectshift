"""Domain selection rule of screening: result rows (rows.csv) -> per-candidate summary -> final domains.

Rule
- Only complete cases are eligible: candidates for which every run of the 4 models x 2 regimes x repetitions succeeded.
  Candidates with any failure are excluded, with the reason in the table.
- target-reference gap is computed from the ID_match and OOD_match rows of the same (candidate, model, repetition) (metrics.target_reference_gap, on B_screen).
- Model score = repetition mean of target-reference gap (RMSE). A model is "consistently positive" if its repetition mean is > 0 and the
  value is > 0 in at least 2/3 of the repetitions.
- A process-condition candidate passes if (1) at least 3 of the 4 models are consistently positive and (2) it is
  eligible, floor(0.25|B|) >= 500 (b_te_eligible, computed from |B| only).
- Consensus score = median of the four repetition-mean target-reference gap (RMSE). Passing process-condition candidates are ranked by
  descending consensus score (ties by candidate name) and the top n_cat are selected. If fewer than n_cat pass, all passing
  candidates are selected (the final domains become 2 + the number passing; n_cat_passed is recorded).
- Temporal domains are not screened: for each axis the 8th of the 15 cutoffs (the median split) is included as is (its
  scores appear in the table only).
- Random-split controls are not eligible and appear in the table only.
- Before selecting, the run is checked for completeness (validate_run): full-run settings (4 models x 3 repetitions, saved
  models) and 24 = 4 x 2 x 3 successful rows for each of the 78 candidates and the ten random-split controls. If anything is
  missing, nothing is selected (failed items are rerun by resuming the screening).
The selection reads only the result rows: the scores come from B_screen, eligibility from |B|, and B_te is never read.
"""
import json
import os

import numpy as np
import pandas as pd

from defectshift import config, metrics

# Number of selected process-condition domains (5; all passing candidates if fewer than 5 pass). It cannot be changed
# from the command line. The screening script imports this module and records the value and the file hash in the screening
# manifest.
N_CAT = 5


def temporal_primary(names):
    """For each axis, the 8th of the 15 cutoffs in time order (share of A = 0.50)."""
    out = []
    for ax in ("x_time", "y_time"):
        cand = sorted([n for n in names if n.startswith(ax + "<=")], key=lambda n: pd.Timestamp(n.split("<=")[1]))
        assert len(cand) == 15, f"{ax}: {len(cand)} candidates"
        out.append(cand[7])
    return out


def target_reference_gap_table(rows):
    """target-reference gap per (slot, candidate, kind, model, repeat), only where both regimes succeeded."""
    ok = rows[rows.status == "ok"].drop_duplicates(["slot", "model", "repeat", "regime"], keep="last")
    out = []
    for key, g in ok.groupby(["slot", "candidate", "kind", "model", "repeat"]):
        g = g.set_index("regime")
        if not {"ID_match", "OOD_match"} <= set(g.index):
            continue
        m = lambda r: dict(r2=g.loc[r, "r2_screen"], rmse=g.loc[r, "rmse_screen"], mae=g.loc[r, "mae_screen"])
        out.append(dict(zip(["slot", "candidate", "kind", "model", "repeat"], key), **metrics.target_reference_gap(m("ID_match"), m("OOD_match"))))
    return pd.DataFrame(out)


def candidate_summary(rows, models, n_repeats):
    """One row per candidate: complete case, repetition-mean target-reference gap per model, number of consistently positive models,
    eligibility, consensus score."""
    dt = target_reference_gap_table(rows)
    meta = rows.drop_duplicates("slot").set_index("slot")[["candidate", "kind", "n_A", "n_B", "n_star", "b_te_eligible"]]
    ok = rows[rows.status == "ok"].drop_duplicates(["slot", "model", "repeat", "regime"], keep="last")
    n_ok = ok.groupby("slot").size()
    need = len(models) * 2 * n_repeats
    out = []
    for slot, mrow in meta.iterrows():
        r = dict(slot=slot, **mrow.to_dict(), n_ok_runs=int(n_ok.get(slot, 0)), n_needed=need)
        r["complete"] = r["n_ok_runs"] == need
        g = dt[dt.slot == slot]
        means, n_cons = [], 0
        for m in models:
            gm = g[g.model == m]
            mean_rmse = gm.gap_rmse_pct.mean() if len(gm) else np.nan
            npos = int((gm.gap_rmse_pct > 0).sum())
            r[f"{m}_gap_rmse_pct"] = mean_rmse
            r[f"{m}_r2_difference"] = gm.r2_difference.mean() if len(gm) else np.nan
            r[f"{m}_n_pos"] = npos
            consistent = bool(len(gm) == n_repeats and mean_rmse > 0 and npos * 3 >= 2 * n_repeats)
            r[f"{m}_consistent"] = consistent
            n_cons += consistent
            means.append(mean_rmse)
        r["n_models_consistent"] = n_cons
        r["consensus_gap_rmse_pct"] = float(np.median(means)) if r["complete"] else np.nan
        r["consensus_r2_difference"] = float(np.median([r[f"{m}_r2_difference"] for m in models])) if r["complete"] else np.nan
        out.append(r)
    return pd.DataFrame(out)


REGIMES = ("ID_match", "OOD_match")


def expected_keys():
    """Set of run keys (model, repetition, regime) that each candidate must have: 4 x 3 x 2 = 24."""
    return {(m, i, r) for m in config.SCREENING_MODELS for i in range(config.N_REPEATS_SCREENING) for r in REGIMES}


def validate_run(rows, run_args, base_names, placebo_names=(), require_tuned=True):
    """Check that the run used for selection is complete. Raises ValueError with the list of problems; on success,
    returns statistics for the record.

    - full-run settings: 4 models x 3 repetitions, saved models, full_run=True
    - tuned screening (require_tuned=True, default): the tunable models were searched on the source side (tune=True) with
      the final budget (config.HPO_N_TRIALS); require_tuned=False only to reread an untuned screening run for comparison
    - candidate set: the candidates of the run arguments = the 78 listed candidates + the random-split controls (exactly,
      without duplicates, omissions or additions), and the result rows contain no other candidate
    - keys: for each candidate, the set of (model, repetition, regime) of the successful rows equals expected_keys()
    - several successful rows for the same key record a retraining after a damaged artifact; this is allowed, the last
      successful row is authoritative (it matches the target-reference gap computation and the saved model), and their number is returned."""
    problems = []
    if list(run_args.get("models", [])) != list(config.SCREENING_MODELS):
        problems.append(f"models {run_args.get('models')} != {list(config.SCREENING_MODELS)}")
    if run_args.get("repeats") != config.N_REPEATS_SCREENING:
        problems.append(f"repetition {run_args.get('repeats')} != {config.N_REPEATS_SCREENING}")
    if not run_args.get("save_models", False):
        problems.append("models were not saved (no selection audit possible)")
    if run_args.get("full_run") is not True:
        problems.append("not a full run (full_run)")
    if run_args.get("max_epochs") is not None:
        problems.append(f"run with a smoke-test epoch cap ({run_args.get('max_epochs')})")
    if require_tuned:
        if run_args.get("tune") is not True:
            problems.append("run with fixed settings, without the source-side search (--tune)")
        if run_args.get("trials") != config.HPO_N_TRIALS:
            problems.append(f"trials {run_args.get('trials')} != final budget {config.HPO_N_TRIALS}")
    expected = set(base_names) | set(placebo_names)
    args_c = list(run_args.get("candidates", []))
    if len(args_c) != len(set(args_c)):
        problems.append("duplicates among the candidates of the run arguments")
    if set(args_c) != expected:
        problems.append(f"candidates of the run arguments differ from the expected set (missing {sorted(expected - set(args_c))[:3]}, "
                        f"extra {sorted(set(args_c) - expected)[:3]})")
    unknown = sorted(set(rows.candidate) - expected)
    if unknown:
        problems.append(f"unknown candidates in the result rows {unknown[:3]}")
    ok = rows[rows.status == "ok"]
    exp = expected_keys()
    for c in sorted(expected):
        keys = set(zip(ok.loc[ok.candidate == c, "model"], ok.loc[ok.candidate == c, "repeat"].astype(int),
                       ok.loc[ok.candidate == c, "regime"]))
        if keys != exp:
            problems.append(f"{c}: successful keys {len(keys & exp)}/{len(exp)}, missing {len(exp - keys)}, invalid {len(keys - exp)}")
    if problems:
        raise ValueError("incomplete screening run, no selection:\n  - " + "\n  - ".join(problems[:20]))
    return dict(n_duplicate_ok_rows=int(ok.duplicated(["candidate", "model", "repeat", "regime"]).sum()))


def code_hashes():
    """Hashes of the selection code (this file and the selection script) and of metrics.py, which computes the selection
    score (target-reference gap)."""
    from defectshift.manifest import file_sha256
    return dict(selection_py_sha256=file_sha256(os.path.abspath(__file__)),
                select_domains_py_sha256=file_sha256(os.path.join(config.REPO_ROOT, "scripts", "02_select_domains.py")),
                metrics_py_sha256=file_sha256(os.path.join(config.REPO_ROOT, "defectshift", "metrics.py")))


def select(summary, all_names, n_cat):
    """Final domains = the 2 pre-specified temporal domains + the top n_cat process-condition candidates. Returns summary with
    the decision columns (passes, rank, selected, reason)."""
    assert n_cat is not None and n_cat > 0, "the number of process-condition domains (n_cat) is not set (selection.N_CAT)"
    s = summary.copy()
    temporal = set(temporal_primary(all_names))
    have = s[s.candidate.isin(temporal)]
    assert len(have) == 2 and have.complete.all(), "results of both pre-specified temporal domains are required"
    s["is_temporal_primary"] = s.candidate.isin(temporal)
    is_cat = s.kind == "cat"
    s["passes"] = is_cat & s.complete & (s.n_models_consistent >= 3) & s.b_te_eligible.astype(bool)

    def reason(r):
        if r.kind == "placebo":
            return "random-split control (not eligible)"
        if r.kind != "cat":
            return "pre-specified temporal (median split)" if r.is_temporal_primary else "temporal trajectory analysis only"
        if not r.complete:
            return f"incomplete ({r.n_ok_runs}/{r.n_needed} successful)"
        if r.n_models_consistent < 3:
            return f"consistently positive models {r.n_models_consistent}/4"
        if not r.b_te_eligible:
            return "B_te eligibility not met"
        return "passed"
    s["reason"] = s.apply(reason, axis=1)
    passed = s[s.passes].sort_values(["consensus_gap_rmse_pct", "candidate"], ascending=[False, True])
    s["cat_rank"] = np.nan
    s.loc[passed.index, "cat_rank"] = np.arange(1, len(passed) + 1)
    chosen_cat = list(passed.candidate.head(n_cat))
    s["selected"] = s.candidate.isin(chosen_cat) | s.is_temporal_primary
    return s, dict(temporal=temporal_primary(all_names), categorical=chosen_cat, n_cat=n_cat,
                   n_cat_passed=int(len(passed)))


def write_final_domains(path, final):
    """The final domain file is written once; if it exists, an exception (no rewriting after a change of rule)."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; the final domains are not overwritten")
    with open(path, "w") as f:
        json.dump(final, f, ensure_ascii=False, indent=1)
