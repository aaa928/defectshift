"""Merges several benchmark runs, by (domain, model), into one run for aggregation (models that did not change keep their first results
and are merged with the models that were rerun). No retraining or re-prediction.

Rules
- Source runs must use the same data and candidate list as now (hashes), the same categorical representation (benchmark) and
  foundation-model input (raw_native), and the same split constants (SPLIT_KEYS). Domains with the same name are not merged if these
  differ. The prediction files are actually read to check that their evaluation rows equal A_heldout and B_te recomputed from the
  current split and that the predictions are finite.
- Everything is written to temporary folders (<path>.partial) and renamed at the end (if the run stops halfway, the final path is empty;
  a rerun removes the leftover .partial folders and starts over).
- Scope = the final domains of --screening-run-id (or --domains) x model roster (--models) x config.N_REPEATS_BENCHMARK repetitions x 4
  regimes (OOD_match_tuned for tuned models only). Exactly one run must cover each (domain, model) of the scope (gaps and overlaps refused).
- The covering run's manifest must contain that domain, model and regime with the same number of repetitions, every expected key must
  be a successful row, and the prediction files must exist.
- Lineage: a source must have the current protocol version (config.PROTOCOL_VERSION), use only search records with the final budget
  (hpo_trials_required = config.HPO_N_TRIALS), have no smoke epoch cap, and, with --screening-run-id, come from that selection (hash of
  final_domains.json). If it covers tuned models, the tuning run it used must also be a main run (full_run) with the final budget.
- Between sources: all sources must have the same shared training code (per-file hashes of runio.COMPAT_FILES). The screening run of
  --screening-run-id must have the current version too.
- Merged run: results/benchmark/runs/<run_id>/{rows.csv (with a source_run_id column), manifest.json (sources recorded)};
  model artifacts in cache/benchmark/runs/<run_id>/models/ as symbolic links to the original files (not copied).
- 06c_aggregate recomputes and checks the evaluation sets (whether the rows of the prediction files are A_heldout and B_te of that domain).
Usage: python scripts/06b_merge_runs.py --run-id <merged run> --screening-run-id <screening> \\
         --take <run A>=TabPFN,TabICL,TabFM,Kumo,TabDPT,EXAONE --take <run B>=LinearRegression,Ridge,...
       (RUN=MODELS@DOMAIN1;DOMAIN2 narrows the domains)
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from defectshift import config, manifest  # noqa: E402
from defectshift.models import extra_models  # noqa: E402
from defectshift.candidates import build_one, load_data, slot_dir  # noqa: E402
from defectshift.runio import COMPAT_FILES, check_same_inputs, check_selection_current, load_final_domains  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402

KEY = ["slot", "repeat", "model", "regime"]
REGIMES = ("ID_full", "ID_match", "OOD_match", "OOD_match_tuned")
# Constants that fix the splits and seeds. protocol.constants of the source manifests and the current config must all match (constants that
# only change model settings, such as HPO_N_TRIALS, may differ in a rerun and are not included)
SPLIT_KEYS = ("TARGET_COL", "LOT_COL", "MIN_DOMAIN", "A_HELDOUT_FRAC", "A_HELDOUT_SEED", "HPO_SEED", "HPO_VAL_FRAC",
              "A_VAL_FRAC_OF_DEV", "RANDOM_STATE", "B_SPLIT_SEED", "B_FRACS", "ES_VAL_FRAC", "ES_SEED_BASE", "N_REPEATS_BENCHMARK",
              "SEED_OFFSET")


def eval_rows(domains):
    """Domain -> (A_heldout, B_te) recomputed by the split code (to check the evaluation rows of the prediction files)."""
    df, y = load_data()
    lot = df[config.LOT_COL]
    out = {}
    for d in domains:
        slot, mA, mB = build_one(df, d)
        out[d] = (np.asarray(get_A(slot, df.index[mA], y, lot)["A_heldout"]),
                  np.asarray(B_view(slot, df.index[mB], y, lot, "benchmark")["B_te"]))
    return out


def pred_file_ok(path, held, te):
    """Read a prediction file and check its evaluation rows, lengths and finite values. A reason string if something is wrong, None otherwise."""
    try:
        z = np.load(path)
        if not (np.array_equal(z["id_idx"], held) and np.array_equal(z["ood_idx"], te)):
            return "evaluation rows differ from the current split"
        if len(z["id_pred"]) != len(held) or len(z["ood_pred"]) != len(te):
            return "prediction length differs from the number of evaluation rows"
        if not (np.isfinite(z["id_pred"]).all() and np.isfinite(z["ood_pred"]).all()):
            return "NaN or Inf in predictions"
    except Exception as e:
        return f"read failure {type(e).__name__}: {e}"
    return None


def parse_take(s):
    """'RUN=M1,M2@D1;D2' -> (run, [models], [domains] or None)."""
    run, rest = s.split("=", 1)
    models, _, doms = rest.partition("@")
    return run, models.split(","), (doms.split(";") if doms else None)


def plan_cover(takes, domains, models):
    """(domain, model) -> covering run. SystemExit on gaps or overlaps."""
    cover = {}
    for run, ms, ds in takes:
        for d in (ds or domains):
            if d not in domains:
                raise SystemExit(f"{run}: not a domain of the scope {d}")
            for m in ms:
                if m not in models:
                    raise SystemExit(f"{run}: not a model of the scope {m}")
                if (d, m) in cover:
                    raise SystemExit(f"({d}, {m}) covered by both {cover[(d, m)]} and {run}")
                cover[(d, m)] = run
    missing = [(d, m) for d in domains for m in models if (d, m) not in cover]
    if missing:
        raise SystemExit(f"{len(missing)} (domain, model) pairs without a covering run (e.g. {missing[:3]})")
    return cover


def lineage_problems(a, consts, models, target_sha):
    """Lineage conditions of a regular source: protocol version, smoke marker, final budget, selection hash; for tuned models, the tuning
    run must also be a main run with the final budget."""
    bad = []
    if consts.get("PROTOCOL_VERSION") != config.PROTOCOL_VERSION:
        bad.append(f"protocol version {consts.get('PROTOCOL_VERSION')} != {config.PROTOCOL_VERSION}")
    if a.get("max_epochs") is not None:
        bad.append(f"run with a smoke-test epoch cap ({a.get('max_epochs')})")
    extra = a.get("stage") == "benchmark_extra"     # NODE/TabNet run: checked with the actual number of trials per model (NODE grid 25, TabNet 30)
    want = {m: extra_models.n_trials_final(m) for m in models if m in extra_models.EXTRA_TUNED}
    if extra:
        got = {m: (a.get("hpo_trials_per_model") or {}).get(m) for m in want}
        if got != want:
            bad.append(f"run accepting search records without the final number of trials (hpo_trials_per_model = {got}, required {want})")
    elif a.get("hpo_trials_required") != config.HPO_N_TRIALS:
        bad.append(f"run accepting search records without the final budget (hpo_trials_required = {a.get('hpo_trials_required')})")
    if target_sha is not None and a.get("screening_final_domains_sha256") != target_sha:
        bad.append("run from a different selection than the target (final_domains.json hash)")
    if set(models) & set(extra_models.tuned_all()):      # tuned models of the 20 + the additional models (NODE, TabNet)
        p = os.path.join(config.RESULTS_DIR, "benchmark", "hpo", str(a.get("hpo_run_id")), "manifest.json")
        if not os.path.exists(p):
            bad.append(f"no tuning-run manifest ({a.get('hpo_run_id')})")
        else:
            ha = json.load(open(p))["args"]
            if extra:
                tpm = {m: (ha.get("trials_per_model") or {}).get(m) for m in want}
                if (ha.get("stage") != "benchmark_tuning_extra" or ha.get("full_run") is not True or tpm != want
                        or ha.get("max_epochs") is not None):
                    bad.append(f"NODE/TabNet tuning run {a.get('hpo_run_id')} is not a main run with the final number of trials (trials_per_model = {tpm})")
            elif ha.get("full_run") is not True or ha.get("trials") != config.HPO_N_TRIALS or ha.get("max_epochs") is not None:
                bad.append(f"tuning run {a.get('hpo_run_id')} is not a main run with the final budget")
    return bad


def cross_source_problems(mans):
    """Condition between sources: every source has per-file code hashes and the same shared training code (runio.COMPAT_FILES) (refused if the
    training code differs even with the same version string)."""
    bad = []
    code = {r: {f: (m.get("code_file_sha256") or {}).get(f) for f in COMPAT_FILES} for r, m in mans.items()}
    no_hash = sorted(r for r, h in code.items() if any(v is None for v in h.values()))
    if no_hash:
        bad.append(f"sources without per-file code hashes {no_hash}")
    elif len({json.dumps(h, sort_keys=True) for h in code.values()}) > 1:
        bad.append(f"the shared training code (runio.COMPAT_FILES) differs between sources {sorted(code)}")
    return bad


def expected(slot, model, repeats):
    return {(slot, i, model, r) for i in range(repeats) for r in REGIMES if r != "OOD_match_tuned" or model in extra_models.tuned_all()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--screening-run-id", default=None, help="screening run to read the final domains from")
    ap.add_argument("--domains", default=None, help="comma-separated domains (instead of --screening-run-id)")
    ap.add_argument("--models", required=True, help="comma-separated model roster (all models of the merged run)")
    ap.add_argument("--take", action="append", required=True, help="RUN=MODELS[@D1;D2] (repeatable)")
    args = ap.parse_args()
    if (args.screening_run_id is None) == (args.domains is None):
        raise SystemExit("exactly one of --screening-run-id and --domains")
    target_sha = None
    if args.screening_run_id:     # includes the selection-time hash and input hash checks. A merged run made with --domains has no selection hash and does not pass the manuscript table checks
        domains, target_sha = load_final_domains(args.screening_run_id)
        check_selection_current(args.screening_run_id)          # only a screening of the current version
    else:
        domains = args.domains.split(",")
    models = args.models.split(",")
    if len(models) != len(set(models)) or len(domains) != len(set(domains)):
        raise SystemExit("a model or domain appears twice")
    takes = [parse_take(t) for t in args.take]
    cover = plan_cover(takes, domains, models)

    out_dir = os.path.join(config.RESULTS_DIR, "benchmark", "runs", args.run_id)
    out_models = os.path.join(config.CACHE_DIR, "benchmark", "runs", args.run_id, "models")
    if os.path.exists(out_dir) or os.path.exists(out_models):
        raise SystemExit(f"run already exists: {out_dir}")
    tmp_dir, tmp_models = out_dir + ".partial", out_models + ".partial"
    for t in (tmp_dir, tmp_models):          # remove traces of an earlier interrupted run and start over
        if os.path.exists(t):
            shutil.rmtree(t)
    cur = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items() if k in SPLIT_KEYS}
    cur = json.loads(json.dumps(cur))
    evals = eval_rows(domains)
    parts, sources, links, excluded_by_run = [], {}, [], {}
    mans = {}
    for run in sorted({r for r, _, _ in takes}):          # 1) check every source first (nothing is written)
        run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "runs", run)
        man = json.load(open(os.path.join(run_dir, "manifest.json")))
        mans[run] = man
        a = man["args"]
        if a["repeats"] != config.N_REPEATS_BENCHMARK or not set(REGIMES) <= set(a["regimes"]):
            raise SystemExit(f"{run}: repetitions {a['repeats']} and regimes {a['regimes']} differ from the scope")
        check_same_inputs(man, f"source run {run}")
        # a NODE/TabNet run has no categorical-representation or foundation-input markers (each model has its own official input, recorded in inputs) and may cover only the additional models
        if a.get("stage") == "benchmark_extra":
            if not {m for (_, m), r in cover.items() if r == run} <= set(extra_models.EXTRA_TUNED):
                raise SystemExit(f"{run}: a NODE/TabNet run covers models other than NODE and TabNet")
        elif a.get("cat_mode") != "benchmark" or a.get("fm_input") != "raw_native":
            raise SystemExit(f"{run}: categorical representation and foundation input are not benchmark and raw_native ({a.get('cat_mode')}, {a.get('fm_input')})")
        consts = (man.get("protocol") or {}).get("constants") or {}
        diff = [k for k in SPLIT_KEYS if consts.get(k) != cur.get(k)]
        if diff:
            raise SystemExit(f"{run}: split constants differ from the current configuration {diff}")
        rows = pd.read_csv(os.path.join(run_dir, "rows.csv"))
        ok = rows[rows.status == "ok"].drop_duplicates(KEY, keep="last")
        slots = dict(rows.drop_duplicates("candidate")[["candidate", "slot"]].values)
        mine = [(d, m) for (d, m), r in cover.items() if r == run]
        man_sha = manifest.file_sha256(os.path.join(run_dir, "manifest.json"))
        mine_models = sorted({m for _, m in mine})
        why = lineage_problems(a, consts, mine_models, target_sha)
        if why:
            raise SystemExit(f"{run}: source cannot be merged — " + "; ".join(why))
        src_models = os.path.join(config.CACHE_DIR, "benchmark", "runs", run, "models")
        for d, m in mine:
            if d not in a["domains"] or m not in a["models"]:
                raise SystemExit(f"({d}, {m}) not in the manifest of {run}")
            got = set(map(tuple, ok[(ok.candidate == d) & (ok.model == m)][KEY].itertuples(index=False)))
            need = expected(slots[d], m, config.N_REPEATS_BENCHMARK)
            if got != need:
                raise SystemExit(f"{run} ({d}, {m}): successful keys {len(got & need)}/{len(need)}, unexpected {len(got - need)}")
            for _, i, _, r in sorted(need):
                stem = os.path.join(slot_dir(slots[d]), f"{m}__{r}__r{i}")
                files = glob.glob(os.path.join(src_models, stem + ".*"))
                if not any(f.endswith(".pred.npz") for f in files):
                    raise SystemExit(f"{run}: no prediction file {stem}")
                why = pred_file_ok(os.path.join(src_models, stem + ".pred.npz"), *evals[d])
                if why:
                    raise SystemExit(f"{run} {stem}: {why}")
                links += [(os.path.realpath(f), os.path.join(tmp_models, os.path.relpath(f, src_models))) for f in files]
        sel = ok[[(c, m) in set(mine) for c, m in zip(ok.candidate, ok.model)]].copy()
        sel.insert(0, "source_run_id", run)
        parts.append(sel)
        excluded_by_run[run] = a.get("foundation_excluded")
        sources[run] = dict(pairs=sorted([list(p) for p in mine]), manifest_sha256=man_sha, hpo_run_id=a.get("hpo_run_id"),
                            screening_run_id=a.get("screening_run_id"))
    why = cross_source_problems(mans)                       # between sources: the shared training code
    if why:
        raise SystemExit("source combination cannot be merged — " + "; ".join(why))
    for src, dst in links:                                  # 2) link only after every check passed
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)
    merged = pd.concat(parts, ignore_index=True)
    assert not merged.duplicated(KEY).any()
    # models excluded by the qualification: the values recorded by the source runs must agree, and roster (--models) + excluded = the 20 candidates (+ NODE and TabNet; used by the manuscript table checks)
    excl = {json.dumps(sorted(v)) for v in excluded_by_run.values() if v is not None}
    if len(excl) > 1:
        raise SystemExit(f"the source runs differ in their excluded foundation models: {excl}")
    excluded = json.loads(excl.pop()) if excl else []
    os.makedirs(tmp_dir, exist_ok=True)
    manifest.write_or_check(tmp_dir, manifest.build(dict(
        stage="benchmark_merged", screening_run_id=args.screening_run_id, screening_final_domains_sha256=target_sha, domains=domains, models=models,
        roster=models, foundation_excluded=excluded,
        repeats=config.N_REPEATS_BENCHMARK,
        regimes=list(REGIMES), cat_mode="benchmark", fm_input="raw_native", sources=sources)))
    merged.to_csv(os.path.join(tmp_dir, "rows.csv"), index=False)
    os.makedirs(os.path.dirname(out_models), exist_ok=True)
    os.replace(tmp_models, out_models)       # only rename at the end: if the run does not get here, the final path stays empty
    os.replace(tmp_dir, out_dir)
    print(f"[merge] {args.run_id}: {len(merged)} rows, sources { {k: len(v['pairs']) for k, v in sources.items()} } (domain-model pairs)")


if __name__ == "__main__":
    main()
