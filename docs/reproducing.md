# Reproducing the results

Rerunning the pipeline below should reproduce the numbers of the paper up to rounding and GPU nondeterminism; Verification states what was and was not rerun at full scale. The release contains no results of the models (`DATA_LICENSE.md`).

## Installation

Python 3.11 with CUDA 11.8 PyTorch; `environment/requirements-lock.txt` pins the packages of the run behind the paper and `environment/versions.json` records the GPU, driver, foundation-model commits and weight snapshots. `environment/environment.yml` installs the imported packages at the versions of the lock file but does not pin their dependencies; the exact versions of the run behind the paper are in `requirements-lock.txt`.

That run used one NVIDIA L4 GPU (24 GB), four CPUs and 47 GB of memory available to the jobs (run times: Supplementary Table S18). The batch sizes of NODE are fixed for a 24 GB GPU; on a smaller GPU the NODE search stops with an out-of-memory error.

Install as in the README; the `rtdl` line declares `numpy<2` and `torch<2` although the run behind the paper used newer versions.

`pip check` still reports unmet requirements: `rtdl` pins `numpy<2` and `torch<2` (it runs with the installed versions), and `otdd` requires `attrdict`, `opentsne` and `torchtext`, which the code paths used here do not import.

`DEFECTSHIFT_DATA_PATH`, `DEFECTSHIFT_CACHE_DIR` and `DEFECTSHIFT_RESULTS_DIR` override `data/data.parquet`, `cache/` and `results/`. The model files that a run writes under `cache/` are local outputs; do not load model files (`joblib.load` can execute code) from a source you do not trust.

## Execution order

Script prefixes order the steps but are not step numbers: `01` and `02` belong to step 1, `03a` and `03b` to step 2, `04` to step 3, `05` to step 4, `06` to `06d` to steps 5 and 6 (`06a` also to step 3), `07*` to step 7 and `10a` to `10d` to step 8. Commands show the required arguments; every script prints all options with `--help`.

| Step | Commands | Run ids | Runs after |
|---|---|---|---|
| 0. Weights | `python scripts/00_fetch_weights.py` | | network access |
| 1. Screening and selection | `python scripts/01_screen_domains.py --run-id screening_main --placebo --tune`<br>`python scripts/02_select_domains.py --run-id screening_main` | `screening_main` | `data.parquet`, `splits/candidates.csv` |
| 2. Selection audit | `python scripts/03a_audit_selection.py --screening-run-id screening_main --run-id selection_audit_main`<br>`python scripts/03b_selection_followup.py --screening-run-id screening_main --selection-audit-run-id selection_audit_main` | `selection_audit_main` | step 1 (needs its saved screening models) |
| 3. Tuning | `python scripts/04_tune_hyperparameters.py --screening-run-id screening_main --run-id hpo_main`<br>NODE and TabNet: `python scripts/06a_benchmark_node_tabnet.py hpo --screening-run-id screening_main --run-id hpo_x`<br>original controls: see Control runs below | `hpo_main`, `hpo_x`, `<controls tuning run>`, `<NODE and TabNet controls tuning run>` (different ids from `hpo_main` and `hpo_x`) | step 1 |
| 4. Foundation-model qualification | `python scripts/05_qualify_foundation.py --screening-run-id screening_main --run-id <qualification run>` (about 1.5 hours on one NVIDIA L4) | your choice; passed to step 5 as `--fmq-run-id` | step 1 |
| 5. Benchmark | `python scripts/06_benchmark.py --screening-run-id screening_main --hpo-run-id hpo_main --fmq-run-id <qualification run> --run-id benchmark_part_main`<br>NODE and TabNet: `python scripts/06a_benchmark_node_tabnet.py bench --screening-run-id screening_main --hpo-run-id hpo_x --run-id benchmark_part_extra`<br>original controls: see Control runs below | `benchmark_part_main`, `benchmark_part_extra`, `controls_*` | steps 3 and 4 |
| 6. Merge and aggregate | `python scripts/06b_merge_runs.py --run-id benchmark_canonical --screening-run-id screening_main --models <the 22 models> --take <run>=<models>` (repeatable; merges `benchmark_part_main` and `benchmark_part_extra`)<br>`python scripts/06c_aggregate.py --run-id benchmark_canonical`<br>`python scripts/06d_diagnostics.py --benchmark-run-id benchmark_canonical` | `benchmark_canonical` | step 5 |
| 7. Sensitivity | `python scripts/07a_sensitivity.py --screening-run-id screening_main --hpo-run-id hpo_main [--extra-hpo-run-id hpo_x]`<br>`python scripts/07c_sensitivity_report.py --sensitivity-run-id <run> --benchmark-run-id benchmark_canonical` | | steps 3 and 5 |
| 8. Shift statistics | `python scripts/10a_shift_statistics.py --run-id main` (without `--placebo`)<br>`python scripts/10b_shift_association.py --screening-run-id screening_main --shift-statistics-run-id main --run-id <run>`<br>`python scripts/10c_shift_diagnostics.py --run-id main`<br>`python scripts/10d_shift_diag_summary.py --run-id main` | `main` | step 1 (`10b` also uses `03a`) |
| 9. Analyses of the paper | `scripts/extra/` runners (below), then `paper/analysis/` | below | steps 5 and 6 |
| 10. Tables | `paper/tools/` | | steps 6 to 9 |

**Control runs.** The analyses and generators of steps 9 and 10 read the run ids of the table. The ten original controls are in five runs (`CTRL_RUNS` of `paper/analysis/common.py`), and each of the 22 models must be in exactly one of them: `controls_batch_01`: CatBoost, MLP, Ridge, TabICL, TabM, TabPFN, XGBoost and kNN; `controls_batch_02`: DecisionTree, EXAONE, GBR, Kumo, LightGBM, LinearRegression, RandomForest, RealMLP, SVR, TabDPT and TabFM; `controls_batch_03`: FT-Transformer; `controls_node`: NODE; `controls_tabnet`: TabNet. The controls are tuned before they are benchmarked (step 3): `python scripts/04_tune_hyperparameters.py --screening-run-id screening_main --controls $(seq -s, 9001 9010) --run-id <controls tuning run>`; for NODE and TabNet: `python scripts/06a_benchmark_node_tabnet.py hpo --screening-run-id screening_main --controls $(seq -s, 9001 9010) --run-id <NODE and TabNet controls tuning run>`. In step 5 the controls use the same scripts with `--controls $(seq -s, 9001 9010)` and, as `--hpo-run-id`, the controls tuning run of the model (it must hold the search records of the controls); the run ids are those above.

Steps 0 and 3 to 5 need the extra installation steps below (weights, TabFM, NODE, TabNet), and `10c` in step 8 needs the shift-diagnostics packages. If a gated weight cannot be obtained, leave that model out with the `--models` option of both `05_qualify_foundation.py` and `06_benchmark.py`; such a run is not a full run (`full_run: false` in its manifest) and its tables are not those of the paper.

**Verification.** The pipeline was run end to end in a clean environment at reduced training scale (the screening in full); the tables recomputed from the run records matched those of the paper cell by cell, and one domain and repetition was retrained. 26 of the 32 scripts of `paper/analysis/` and the generators of eight table files (seven of the main text and the full lot-disjoint tables of the supplement) were then run, on CPU, on the saved run records of the paper (the run folders renamed to the names used here): the 23 scripts with a counterpart output give the same values as the code behind the paper (`control_sizes.py` writes one more column, `tol`), and the eight table files have the numbers of the paper's tables. The other six scripts (`control_screening`, `selection_audit_collapsed`, `sensitivity_lot_disjoint`, `sensitivity_directions`, `uniform_preprocessing_signs`, `time_ordered_verdict`) read run records of the later stages in the format of this release, which those saved records do not have; they were compared with the code behind the paper (syntax-tree comparison), and a few of them have unit tests on synthetic input.

## Extra installation steps

**Foundation-model weights.** `python scripts/00_fetch_weights.py` downloads the weights of TabPFN-3.5, TabICL, TabFM, TabDPT and EXAONE-Tabular at the commits pinned in `environment/weights.json` and verifies their sha256
(`--verify-only` only verifies). TabPFN-3.5 is gated: log in with a Hugging Face token (`hf auth login`) and accept its license first. The benchmark stops if a pinned file is missing or differs and never downloads weights by itself.
Kumo-Tabular is pinned by its Hub tag `v1.0.1`; the run stops if the cached snapshot is not the one of the paper (`3c3e10bb`). If the tag has been moved, restore the snapshot of the paper:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

commit = "3c3e10bbdb590ace29e7026847f92db3c603096d"
snapshot_download(repo_id="nvidia/Kumo-Tabular", revision=commit)
ref = Path(HF_HUB_CACHE) / "models--nvidia--Kumo-Tabular" / "refs" / "v1.0.1"
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(commit, encoding="utf-8")
```

Afterwards run with `HF_HUB_OFFLINE=1` so that the tag is not resolved again. `scripts/checks/check_offline_load.py --run-id <id>` checks that all six models load and run with the network blocked (exit code 0: ok, 1: a model failed, 2: a library tried to reach the network).

**TabFM** is accepted only from a clean checkout of the pinned commit:

```bash
git clone https://github.com/google-research/tabfm vendor/tabfm && git -C vendor/tabfm checkout fbb6655
pip install --no-deps -e vendor/tabfm
```

**NODE and TabNet** run the authors' code at the commits of the paper in `vendor/` (set `DEFECTSHIFT_NODE_DIR` and `DEFECTSHIFT_TABNET_DIR` to use other folders):

```bash
git clone https://github.com/Qwicen/node vendor/node && git -C vendor/node checkout 3bae6a8
git clone --filter=blob:none --sparse https://github.com/google-research/google-research vendor/google-research
git -C vendor/google-research sparse-checkout set tabnet && git -C vendor/google-research checkout 0a1b0fe
```

**TabNet environment.** TabNet needs a second environment (Python 3.8, NVIDIA TensorFlow 1.15 from `pypi.nvidia.com`), which the pipeline calls through `DEFECTSHIFT_TABNET_PYTHON`:

```bash
conda create -n defectshift-tabnet python=3.8
conda run -n defectshift-tabnet pip install -r environment/requirements-tabnet-tf1-lock.txt --extra-index-url https://pypi.nvidia.com
export DEFECTSHIFT_TABNET_PYTHON="$(conda run -n defectshift-tabnet which python)"
```

**Shift diagnostics** (`scripts/10c_shift_diagnostics.py`) need seven more packages (geomloss, the authors' `otdd` and `disde` at the commits of the paper, and their imports):
`pip install --no-deps -r environment/requirements-shift-diagnostics.txt`; run the script with `CUDA_VISIBLE_DEVICES=""` (CPU only).

## Non-canonical analyses

Runners in `scripts/extra/`; their results are not pooled with the 22-model results of the canonical splits (details: the paper and `docs/usage.md`).

| Analysis | Runner | Run ids (`results/benchmark/runs/<id>`) | Aggregation (`results/analysis/<name>`) |
|---|---|---|---|
| 40 additional controls (eight models) | `run_extra_controls.py` | `controls_supplementary` (original ten: `controls_batch_01`, `_02`, `_03`, `controls_tabnet`, `controls_node`) | `extra_controls_verdict` |
| Redrawn target test splits | `run_redrawn_target_splits.py` | `redrawn_s<k>_a`, `redrawn_s<k>_b`, `k = 1 ... 5` | `redrawn_verdict`, `redrawn_directions` |
| Time-ordered configuration | `run_time_ordered.py` | `time_ordered_{size_only,time_ordered,controls}_<axis>`, `<axis>` = `x_time`, `y_time` | `time_ordered_verdict` |
| Training-row rescaling of the numerical features | `run_trainscale.sh <screening-run> <hpo-run> <extra-hpo-run>` | arm `trainscale` of `07a_sensitivity.py` | `07c_sensitivity_report.py` |

In the redrawn splits the test part is 28% and the training part 45% of `B` (canonical 25% and 48%), so `n*` is about 5–6% lower in six domains; the controls keep the canonical proportions.

Do not reuse a run id after changing code: a run is resumed by comparing only its evaluation rows, so old and new results would be mixed.

## Analyses and tables of the paper

The scripts of `paper/analysis/` compute the supplementary numbers of the paper from the predictions and run records that a benchmark run saves (CPU only, no training); each writes to `results/analysis/<run-id>/` and prints its options with `--help`.
`M8` in the code is the set of eight models of the additional analyses (Ridge, kNN, XGBoost, CatBoost, MLP, TabM, TabPFN, TabICL). The generators in `paper/tools/` write `paper/tables/` and read fixed analysis folder names, so use these as `--run-id`:

| Folder | Script | Note |
|---|---|---|
| `lot_disjoint`, `lot_disjoint_accuracy` | `lot_disjoint_gaps.py`, `lot_disjoint_accuracy.py` | |
| `robust` | `verdict_robustness.py` | run with `--base-disjoint`; by default, the models that finished all ten original controls are used (the 22 models of the paper), read from the five control runs of `common.CTRL_RUNS` |
| `control_rank`, `criteria` | `control_rank.py`, `evaluation_criteria.py` | after `robust` |
| `scale_normalized_bootstrap`, `scale_normalized_contrast`, `lot_block_bootstrap` | scripts of the same name | |
| `direction_checks` | `direction_checks.py` | after the two bootstrap runs; with `--models`, `--only` or another `--n-boot` the values differ from the paper and the reference checks are skipped |
| `extra_controls_verdict`, `redrawn_verdict`, `redrawn_directions`, `time_ordered_verdict` | scripts of the same name | after `scripts/extra/` |
| `lot_match` | `lot_matched_controls.py` | after `lot_composition` |
| `size_matched_verdict` | `size_matched_verdict.py` | after `extra_controls_verdict`, `control_sizes`, `lot_matched_controls` |
| `selection_audit_collapsed` | `selection_audit_collapsed.py` | after `03a_audit_selection.py` (`--selection-audit-run-id`, default `selection_audit_main`); gives the `Test*` column of Supplementary Table S4 and the rank statistics of Section S3, with and without the collapsed cells |
| `sensitivity_lot_disjoint`, `lot_composition`, `diagnostics_lot_disjoint`, `control_screening`, `lot_order_proximity`, `time_axes_comparison` | scripts of the same name | |
| other supplementary numbers | `eight_model_verdict.py`, `extra_controls_compose.py`, `mae_gap_range.py`, `mean_bias_decomposition.py`, `repetition_uncertainty.py`, `uniform_preprocessing_signs.py`; helpers called by other scripts: `sensitivity_directions.py`, `tail_concentration.py`, `test_set_checks.py` | each docstring states its inputs and output folder |

| Generator (`paper/tools/`) | Tables (LaTeX label, number in the paper) |
|---|---|
| `tables_dataset.py` | `tab:domains` (Table 2), `tab:shift` (Table S19) |
| `tables_results.py` | `tab:models` (Table 3), `tab:rq1` (Table 4), `tab:rq2` (Table 5) |
| `tables_sensitivity.py` | `tab:ctrl50` (Table 6), `tab:resplit` (Table 7), `tab:forward` (Table 8), `tab:resplit-rq1` (Table 12) |
| `tables_appendix.py` | `tab:lotinfo` (Table 10), `tab:compositions` (Table 11), `tab:xy` (Table 13), `tab:diag` (Table 15), `tab:forward-split` (Table S20), `tab:compute` (Table S18) |
| `tables_direction.py` | `tab:block` (Table 14), `tab:scaleterm` (Table 16) |
| `tables_supp.py` | `tab:fullld-*` (Tables S6 to S12; written to `paper/tables/tab_full_disjoint.tex`) |

Tables 1 and 9 and Supplementary Tables S1 to S5 and S13 to S17 (literature, properties of the records, candidates, tuning summary, model settings, versions and search spaces) are typeset by hand in the paper; the repository has no generator for them.

Family-level summaries are descriptive because the families differ in size. The domain classifier of the shift statistics uses five-fold cross-validation grouped by lot, and the R^2 decomposition is a reporting device, not a method.

The inputs of each generator are listed in its docstring; some come from earlier stages: `shift_diag/main` (`10c`), `shift_statistics/main` (`10a`) and `cache/dry_run/redrawn_split_1.json` (save the output printed by `run_redrawn_target_splits.py --screening-run-id <screening> --resplit 1 --dry-run` to this file). Run a generator only in an environment that has the result folders.

## Run records

`pred_path` in `report/per_fit.csv` is relative to the cache folder (`DEFECTSHIFT_CACHE_DIR`, default `cache/`): the predictions of a fit are `<cache>/benchmark/runs/<run>/models/<domain>/<model>__<regime>__r<repetition>.pred.npz` and its training rows the `.npz` file of the same name without `.pred` (read both through `defectshift.utils.read_per_fit`). Screening predictions contain rows of `B_screen` only; the rows of `B_te` are opened only by the selection audit and its follow-up analyses. The variants of `sensitivity_*` store predictions and run records only (their training rows are those of the benchmark fit with the same domain, model, regime and repetition). `benchmark_canonical` is the merged final run of `benchmark_part_main` (without NODE and TabNet) and `benchmark_part_extra` (NODE and TabNet); the columns `source_run_id` and `run_id` of its `rows.csv` carry these names.

Every run folder holds `rows.csv` and `manifest.json` (time, interpreter, platform, GPU and driver; hashes of the data file, of `splits/candidates.csv` and of the loaded code files, which a resumed run must match; git state, protocol constants, package versions and the arguments of the run). `final_domains.json` of a screening run holds the selected domains and the hashes of the rows, manifest and code that the selection used.

### Paper quantities in the result files

| Paper | Code and files | Note |
|---|---|---|
| `Err_S`, `Err_T` | columns `rmse_id`, `mae_id`, `r2_id` and `rmse_ood`, `mae_ood`, `r2_ood` of `per_fit.csv` | `_id`: `A_heldout`; `_ood`: `B_te` |
| `100 g_TR` (target-reference gap) | `gap_rmse_pct`, `gap_mae_pct` of `target_reference_gap.csv` | `metrics.target_reference_gap`: `ID_match` against `OOD_match` on the same evaluation set; all `B_te` records |
| `100 g_ID-OOD` (ID-OOD gap) | `gap_rmse_pct`, `gap_mae_pct` of `id_ood_gap.csv` | `ID_match` and `ID_full`; `B_te` against `A_heldout` |
| outcome-scale ratio `s` | columns `s` (and `s_pct`) of `disjoint_domain.csv` | `Err_R` of the best constant prediction on the target rows over the same on `A_heldout`: ratio of standard deviations for RMSE, of mean absolute deviations around the median for MAE; constant within a domain |
| `d` | column `d` of `disjoint_cells.csv` | `Err_T(f_T) / Err_S(f_S)`, with `f_S` = `ID_match` |
| `q` (error ratio without scale) | columns `q`, `q_med` | `d / s` |
| scale-normalized ID-OOD contrast `g_SN` | `adj` columns of `scale_adj_*.csv` | `(1 + g_ID-OOD) / s - 1` |
| identity | checked in every fit by `lot_disjoint_gaps.py` | `1 + g_ID-OOD = (1 + g_TR) * s * q` (all terms as ratios, not percent) |

The tables `target_reference_gap.csv` and `id_ood_gap.csv` written by `scripts/06c_aggregate.py` use all `B_te` records. The lot-disjoint values of the paper are
computed from the stored predictions and the lot-overlap table. `gap_rmse_pct` and `gap_mae_pct` are the gaps in
percent (positive: the source-trained model is worse); `r2_difference` is the difference of R^2 (not a gap).

## NODE and TabNet implementation notes

Both models run their official implementations. The training procedure follows them except for the common stopping
rule and the NODE batch size, and differs from the official examples in the NODE input standardization.

- Stopping rule: the fixed iteration budgets of the official examples are replaced by the common epoch-based rule (at
  most 150 epochs, stop after 17 epochs without improvement), as Gorishniy et al. (2021) also did for both models.
- NODE: quantile transform, QHAdam optimizer and validation on the average of the last five checkpoints, as in the
  official code. With the epoch-based rule, validation and checkpointing move from every 100 updates to every epoch;
  the averaged checkpoint with the lowest validation loss is kept and restored, and its averaging window is shorter
  than in the official example.
- NODE batch size: 512, as in the official regression example. The four grid configurations that exceed the available
  GPU memory at this size (two or four layers, 2048 trees in total, depth eight) use 256. This follows the original
  paper's principle of setting the batch size by GPU memory, but the largest feasible size was not searched; the batch
  size of every trial is recorded.
- NODE input standardization: before its quantile transform, NODE standardizes the inputs with the training-row mean
  and standard deviation, the normalization option of its official data loader, which the official examples leave off.
  The official code adds noise whose scale depends on each column's standard deviation before fitting the quantile
  transform; standardizing first keeps this noise from depending on the rescaling applied for anonymization.
- TabNet: official TensorFlow 1 implementation with Adam, exponential learning-rate decay and gradient clipping, and a
  fixed batch size equal to its virtual batch size of 512, following its authors' choice of equal batch and virtual
  batch sizes for small training sets.

## Model licenses

Model weights are not redistributed; each model is installed from its official source under its own license (what this means for results: `DATA_LICENSE.md`). Versions, checkpoints and licenses of all 22 models are in the Supplementary Material of the paper.

| Model | Code license | Weights license |
|---|---|---|
| NODE (Qwicen/node) | MIT | trained from scratch |
| TabNet (google-research/google-research, `tabnet/`) | Apache-2.0 | trained from scratch |
| TabPFN | Apache-2.0 | TabPFN-3.5 License v1.0 (non-commercial, non-production; outputs: §2(d), §4(a)) |
| TabICL | BSD-3-Clause | BSD-3-Clause |
| TabFM | Apache-2.0 | TabFM Non-Commercial License v1.0 (outputs: §3(a)) |
| TabDPT | Apache-2.0 | Apache-2.0 |
| Kumo Tabular | Apache-2.0 | OpenMDW-1.1 |
| EXAONE-Tabular | BSD-3-Clause variant (LG AI Research) | EXAONE AI Model License 1.2-NC (outputs: §3.1) |
