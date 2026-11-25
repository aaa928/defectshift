"""Checks of merging benchmark runs (scripts/06b_merge_runs.py). Run: python -m pytest tests/test_merge_runs.py -q
Fake runs (rows, manifests, prediction files) built on the evaluation rows of 2 real domains (the temporal median cutoff and cat_050)
check the assignment by (domain, model), refusal of gaps, overlaps, missing keys, input hashes, split constants and prediction files, the
artifact links, and rerunning after an interruption.
Lineage: an earlier protocol version, search records without the final budget, a smoke epoch cap, a tuning run that is not a main run and
another selection hash are refused.
"""
import importlib.util
import json
import os
import sys

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from defectshift import config, hpo, manifest  # noqa: E402
from defectshift.candidates import build_one, load_data, slot_dir  # noqa: E402
from defectshift.runio import COMPAT_FILES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_MEDIAN = "x_time<=2099-05-04 21:18:39"
DOM_NAMES = [X_MEDIAN, "cat_050"]


def _mod():
    spec = importlib.util.spec_from_file_location("06b_merge_runs", os.path.join(ROOT, "scripts", "06b_merge_runs.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def evals():
    m = _mod()
    df, _ = load_data()
    return {d: (build_one(df, d)[0], *m.eval_rows([d])[d]) for d in DOM_NAMES}


def _fake_hpo_run(hpo_run, **over):
    """Manifest of the tuning run used by a source (a main run with the final budget by default)."""
    d = os.path.join(config.RESULTS_DIR, "benchmark", "hpo", hpo_run)
    os.makedirs(d, exist_ok=True)
    json.dump(dict(args=dict(dict(full_run=True, trials=config.HPO_N_TRIALS, max_epochs=None), **over)),
              open(os.path.join(d, "manifest.json"), "w"))


def _fake_screening(run_id, current=True):
    """Fake selection with DOM_NAMES as the final domains (the format read by runio.load_final_domains). Returns the hash of the final
    file. current=False gives a screening without a version record (the format of a first screening)."""
    d = os.path.join(config.RESULTS_DIR, "screening", "runs", run_id)
    os.makedirs(os.path.join(d, "selection"), exist_ok=True)
    pd.DataFrame(dict(candidate=DOM_NAMES)).to_csv(os.path.join(d, "rows.csv"), index=False)
    json.dump(dict(data_sha256=manifest.file_sha256(config.DATA_PATH), candidates_sha256=manifest.file_sha256(config.CANDIDATES_CSV),
                   protocol=dict(constants=json.loads(json.dumps(manifest.protocol_constants()))) if current else {},
                   args=dict(stage="screening")), open(os.path.join(d, "manifest.json"), "w"))
    json.dump(dict(temporal=[X_MEDIAN], categorical=["cat_050"], rows_sha256=manifest.file_sha256(os.path.join(d, "rows.csv")),
                   manifest_sha256=manifest.file_sha256(os.path.join(d, "manifest.json"))),
              open(os.path.join(d, "selection", "final_domains.json"), "w"))
    return manifest.file_sha256(os.path.join(d, "selection", "final_domains.json"))


def _fake_run(evals, run, models, drop=None, bad_pred=None, args_over=None, hpo_over=None, **man_over):
    """Successful rows and prediction files for every key of DOM_NAMES x models in run. drop = one key to leave out, bad_pred = one key
    whose evaluation rows are written wrong. Default lineage: the current protocol version (protocol_constants), search records with the
    final budget only, no epoch cap, a main tuning run. args_over = overrides of the manifest args, hpo_over = overrides of the tuning-run
    manifest args."""
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "runs", run)
    mdir = os.path.join(config.CACHE_DIR, "benchmark", "runs", run, "models")
    os.makedirs(run_dir, exist_ok=True)
    rows = []
    for d, (slot, held, te) in evals.items():
        for m in models:
            for i in range(config.N_REPEATS_BENCHMARK):
                for r in ("ID_full", "ID_match", "OOD_match", "OOD_match_tuned"):
                    if r == "OOD_match_tuned" and m not in hpo.TUNED:
                        continue
                    if drop == (d, i, m, r):
                        continue
                    rows.append(dict(slot=slot, candidate=d, repeat=i, model=m, regime=r, status="ok", r2_ood=0.1))
                    stem = os.path.join(mdir, slot_dir(slot), f"{m}__{r}__r{i}")
                    os.makedirs(os.path.dirname(stem), exist_ok=True)
                    te_ = te[::-1] if bad_pred == (d, i, m, r) else te
                    np.savez(stem + ".pred.npz", id_idx=held, id_pred=np.zeros(len(held)), ood_idx=te_, ood_pred=np.zeros(len(te)))
                    np.savez(stem + ".npz", fit_idx=np.arange(2))
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "rows.csv"), index=False)
    root = os.path.realpath(config.REPO_ROOT)
    man = dict(data_sha256=manifest.file_sha256(config.DATA_PATH), candidates_sha256=manifest.file_sha256(config.CANDIDATES_CSV),
               protocol=dict(constants=manifest.protocol_constants()), packages=manifest._packages(),
               code_file_sha256={f: manifest.file_sha256(os.path.join(root, f)) for f in COMPAT_FILES},
               args=dict(domains=DOM_NAMES, models=models, repeats=config.N_REPEATS_BENCHMARK, cat_mode="benchmark", fm_input="raw_native",
                         regimes=["ID_full", "ID_match", "OOD_match", "OOD_match_tuned"], hpo_run_id=f"hpo_{run}",
                         hpo_trials_required=config.HPO_N_TRIALS, max_epochs=None, screening_final_domains_sha256=None))
    man["args"].update(args_over or {})
    man.update(man_over)
    json.dump(man, open(os.path.join(run_dir, "manifest.json"), "w"))
    _fake_hpo_run(f"hpo_{run}", **(hpo_over or {}))


def _argv(run_id, *takes, screening=None):
    out = ["06b_merge_runs.py", "--run-id", run_id, "--models", "Ridge,TabPFN"]
    out += ["--screening-run-id", screening] if screening else ["--domains", ",".join(DOM_NAMES)]
    for t in takes:
        out += ["--take", t]
    return out


def test_plan_cover_rejects_gap_and_overlap():
    m = _mod()
    with pytest.raises(SystemExit, match="covered by both"):
        m.plan_cover([("r1", ["Ridge"], None), ("r2", ["Ridge"], ["dA"])], ["dA", "dB"], ["Ridge"])
    with pytest.raises(SystemExit, match="without a covering run"):
        m.plan_cover([("r1", ["Ridge"], ["dA"])], ["dA", "dB"], ["Ridge"])
    assert m.plan_cover([("r1", ["Ridge"], ["dA"]), ("r2", ["Ridge"], ["dB"])], ["dA", "dB"], ["Ridge"]) == \
        {("dA", "Ridge"): "r1", ("dB", "Ridge"): "r2"}


def test_merge_end_to_end(evals, monkeypatch):
    m = _mod()
    _fake_run(evals, "old_run", ["Ridge", "TabPFN"])
    _fake_run(evals, "new_run", ["Ridge"])
    monkeypatch.setattr(sys, "argv", _argv("merged_t", "old_run=TabPFN", "new_run=Ridge"))
    m.main()
    out = pd.read_csv(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "merged_t", "rows.csv"))
    n = config.N_REPEATS_BENCHMARK
    assert len(out) == 2 * (4 * n + 3 * n)
    assert set(out[out.model == "Ridge"].source_run_id) == {"new_run"} and set(out[out.model == "TabPFN"].source_run_id) == {"old_run"}
    slot = evals["cat_050"][0]
    link = os.path.join(config.CACHE_DIR, "benchmark", "runs", "merged_t", "models", slot_dir(slot), "Ridge__ID_full__r0.pred.npz")
    assert os.path.islink(link) and "new_run" in os.path.realpath(link)
    man = json.load(open(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "merged_t", "manifest.json")))["args"]
    assert man["models"] == ["Ridge", "TabPFN"] and set(man["sources"]) == {"old_run", "new_run"}
    assert not os.path.exists(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "merged_t.partial"))


@pytest.mark.parametrize("case", ["missing_key", "bad_pred", "other_data", "other_split", "other_cat_mode"])
def test_merge_refusals(evals, monkeypatch, case):
    """A missing key, a prediction file with wrong evaluation rows, other data, other split constants and another categorical
    representation are refused, and nothing remains at the final path."""
    m = _mod()
    run = f"src_{case}"
    kw = {}
    if case == "missing_key":
        kw = dict(drop=("cat_050", 3, "Ridge", "OOD_match_tuned"))
    elif case == "bad_pred":
        kw = dict(bad_pred=("cat_050", 1, "Ridge", "ID_match"))
    elif case == "other_data":
        kw = dict(data_sha256="0" * 64)
    elif case == "other_split":
        c = manifest.protocol_constants()
        kw = dict(protocol=dict(constants=dict(c, B_SPLIT_SEED=c["B_SPLIT_SEED"] + 1)))
    _fake_run(evals, run, ["Ridge"], **kw)
    if case == "other_cat_mode":
        p = os.path.join(config.RESULTS_DIR, "benchmark", "runs", run, "manifest.json")
        man = json.load(open(p))
        man["args"]["cat_mode"] = "integer"
        json.dump(man, open(p, "w"))
    _fake_run(evals, f"fm_{case}", ["TabPFN"])
    monkeypatch.setattr(sys, "argv", _argv(f"merged_{case}", f"fm_{case}=TabPFN", f"{run}=Ridge"))
    with pytest.raises(SystemExit):
        m.main()
    assert not os.path.exists(os.path.join(config.RESULTS_DIR, "benchmark", "runs", f"merged_{case}"))
    assert not os.path.exists(os.path.join(config.CACHE_DIR, "benchmark", "runs", f"merged_{case}", "models"))


def test_merge_rerun_after_partial(evals, monkeypatch):
    """Even if a .partial folder remains from an interruption, rerunning with the same run name builds the run from scratch."""
    m = _mod()
    _fake_run(evals, "old_p", ["TabPFN"])
    _fake_run(evals, "new_p", ["Ridge"])
    stale = os.path.join(config.CACHE_DIR, "benchmark", "runs", "merged_p", "models.partial", "junk")
    os.makedirs(stale, exist_ok=True)
    monkeypatch.setattr(sys, "argv", _argv("merged_p", "old_p=TabPFN", "new_p=Ridge"))
    m.main()
    assert not os.path.exists(stale) and os.path.exists(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "merged_p", "rows.csv"))


@pytest.mark.parametrize("case", ["no_version", "other_version", "nonfinal_hpo", "smoke_epochs", "hpo_not_full", "hpo_smoke_epochs",
                                  "other_selection"])
def test_merge_lineage_refusals(evals, monkeypatch, case):
    """A source covering a tuned model (Ridge) is refused if its lineage is off: no version record (the format of a first run), another
    version, a run with --allow-nonfinal-hpo, a smoke epoch cap, a tuning run that is not a main run, a tuning run with an epoch cap, another
    selection hash. Nothing remains at the final path."""
    m = _mod()
    sha = _fake_screening("screening_lin")
    kw = dict(args_over=dict(screening_final_domains_sha256=sha))
    c = manifest.protocol_constants()
    if case == "no_version":
        kw["protocol"] = dict(constants={k: v for k, v in c.items() if k != "PROTOCOL_VERSION"})
    elif case == "other_version":
        kw["protocol"] = dict(constants=dict(c, PROTOCOL_VERSION="0.9"))
    elif case == "nonfinal_hpo":
        kw["args_over"]["hpo_trials_required"] = None
    elif case == "smoke_epochs":
        kw["args_over"]["max_epochs"] = 10
    elif case == "hpo_not_full":
        kw["hpo_over"] = dict(full_run=False)
    elif case == "hpo_smoke_epochs":
        kw["hpo_over"] = dict(max_epochs=10)
    elif case == "other_selection":
        kw["args_over"]["screening_final_domains_sha256"] = "0" * 64
    _fake_run(evals, f"lin_{case}", ["Ridge"], **kw)
    _fake_run(evals, f"linfm_{case}", ["TabPFN"], args_over=dict(screening_final_domains_sha256=sha))
    monkeypatch.setattr(sys, "argv", _argv(f"mlin_{case}", f"linfm_{case}=TabPFN", f"lin_{case}=Ridge", screening="screening_lin"))
    with pytest.raises(SystemExit, match="source cannot be merged"):
        m.main()
    assert not os.path.exists(os.path.join(config.RESULTS_DIR, "benchmark", "runs", f"mlin_{case}"))


def test_merge_with_selection_hash(evals, monkeypatch):
    """Sources with the same selection hash pass, and the target selection hash remains in the merged manifest (checked by the manuscript
    table checks)."""
    m = _mod()
    sha = _fake_screening("screening_ok")
    _fake_run(evals, "sel_fm", ["TabPFN"], args_over=dict(screening_final_domains_sha256=sha))
    _fake_run(evals, "sel_r", ["Ridge"], args_over=dict(screening_final_domains_sha256=sha))
    monkeypatch.setattr(sys, "argv", _argv("msel", "sel_fm=TabPFN", "sel_r=Ridge", screening="screening_ok"))
    m.main()
    man = json.load(open(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "msel", "manifest.json")))
    assert man["args"]["screening_final_domains_sha256"] == sha and man["protocol"]["constants"]["PROTOCOL_VERSION"] == config.PROTOCOL_VERSION


def test_merge_refuses_old_selection_and_mixed_code(evals, monkeypatch):
    """Refused if the target screening has no version record (the format of a first screening), and if the shared training-code hashes
    differ between regular sources."""
    m = _mod()
    sha_old = _fake_screening("screening_old", current=False)
    _fake_run(evals, "o_fm", ["TabPFN"], args_over=dict(screening_final_domains_sha256=sha_old))
    _fake_run(evals, "o_r", ["Ridge"], args_over=dict(screening_final_domains_sha256=sha_old))
    monkeypatch.setattr(sys, "argv", _argv("m_old", "o_fm=TabPFN", "o_r=Ridge", screening="screening_old"))
    with pytest.raises(SystemExit, match="protocol version"):
        m.main()
    sha = _fake_screening("screening_mix")
    _fake_run(evals, "x_fm", ["TabPFN"], args_over=dict(screening_final_domains_sha256=sha))
    _fake_run(evals, "x_r", ["Ridge"], args_over=dict(screening_final_domains_sha256=sha))
    p = os.path.join(config.RESULTS_DIR, "benchmark", "runs", "x_r", "manifest.json")
    man = json.load(open(p))
    man["code_file_sha256"]["defectshift/regimes.py"] = "0" * 64
    json.dump(man, open(p, "w"))
    monkeypatch.setattr(sys, "argv", _argv("m_mix", "x_fm=TabPFN", "x_r=Ridge", screening="screening_mix"))
    with pytest.raises(SystemExit, match="shared training code"):
        m.main()
    assert not os.path.exists(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "m_mix"))
