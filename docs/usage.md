# Using the benchmark

## Data

`data.parquet` is in the Zenodo record [10.5281/zenodo.24313062](https://doi.org/10.5281/zenodo.24313062); place it in `data/`.
It holds all 14,276 production records of a display-manufacturing line over about seven months. No record was added, removed or resampled, and no value was clipped or imputed.

| Column | Count | Description |
|---|---|---|
| `num_001` to `num_173` | 173 | Numerical features |
| `cat_001` to `cat_050` | 50 | Categorical features, integer-coded (`cat_049` identifies the production lot and is used only for grouping) |
| `x_time` | 1 | Time at which the features were measured |
| `y_time` | 1 | Time at which the defect rate was measured |
| `y` | 1 | Defect rate in [0, 1] |

`x_time` and `y_time` only define the temporal domains; they are not model inputs. `data/data_dictionary.csv` lists the kind, data type, missing rate and
number of distinct values of every column; `data/make_data_dictionary.py --check` verifies it against `data.parquet`.

**Anonymization.**

- Numerical features are rescaled to [0, 1] per column over all records (up to floating-point rounding); the labels are not used. This is a release-level step, not preprocessing fitted on
  training rows: the extrema include target-side and evaluation records. Model-specific preprocessing is fitted on the training rows of each fit. Nine models (RealMLP, TabM,
  TabNet and the six foundation models) apply their own preprocessing to the released values and receive the released scale as it is. For the other 13 the rescaling has no effect; the only exception is the fixed floor (0.001) on the training-row standard deviation, which some training pools reached (`scripts/07a_sensitivity.py`, variant `trainscale`, re-scales with the extrema of each fit's training rows).
- Each timestamp column was shifted by its own constant to a non-identifying future year (2099), which preserves order and intervals within a column; differences between the two columns are not meaningful.
- Categorical values, including the lot identifier `cat_049`, were replaced by integer codes; the mapping to the original values is not released.
- The defect rate was min-max scaled to [0, 1]; the denominator of the rate is not released.

**Properties.**

- The defect rate is zero for 25.3% of the records and right-skewed otherwise.
- 152 of the 173 numerical columns have missing values (up to 99.3% per column, 45% of all numerical entries); the 152 columns fall into 29 groups with identical missing positions.
  The categorical columns have no missing values.
- Some numerical columns contain rare extreme readings, kept as recorded.

## Evaluating a model

1. **Use the canonical splits.** Fit the source model on `A_match` (`n*` records drawn from `A_dev`) and the reference on `B_match` (`n*` records drawn from `B_tr`; step 2), and evaluate both on `A_heldout` and `B_te`. Tune hyperparameters on the source side with `defectshift.splits.hpo_split`, the tuning split of `A_dev`. `B_te` is used only for the final evaluation (rules 1 and 6 below).
2. **Match the training budget.** The source-trained model `f_S` and the reference `f_T` use the same model class, the same hyperparameters (selected on the source side) and the same nominal number of records `n*`. The reference is not target-optimal: matching the number of records does not match the number of lots.
3. **Choose the target rows.** Records whose lot also occurs in the source development set can be left out with `splits/target_test_lot_overlap.csv` (lot-disjoint evaluation). State whether you report all `B_te` records or the lot-disjoint records.
4. **Compute both gaps.** With `Err` the RMSE or MAE, the ID-OOD gap is `Err_T(f_S) / Err_S(f_S) - 1` and the target-reference gap is `Err_T(f_S) / Err_T(f_T) - 1`; a positive value is a loss of the source-trained model. Report the outcome-scale ratio `s` of the two test sets (Notation below) next to the ID-OOD gap. The paper computes the gaps for each of the five repetitions (repetitions 0 to 4), averages them per model, and reports the median over models as the domain value; the example below evaluates one repetition.
5. **Read the gap against the controls.** Run the same procedure on the ten randomized lot-split controls and report, for each challenge domain, `k` (the number of controls with a smaller gap) with the number of controls. `k` is not a significance test; the verdict depends on the error measure and on the number of controls (see the paper). In the paper the RMSE range of the ten controls was narrower than that of all 50 controls, so read RMSE verdicts against all 50 (`python examples/evaluate_new_model.py --extra-controls`); the 40 additional controls (seeds 9011 to 9050) are regenerated from fixed seeds by `defectshift.extra_controls.register()` and are not distributed as files.

`examples/evaluate_new_model.py` runs this procedure for one repetition with a scikit-learn regressor (default hyperparameters, no early stopping) on the seven domains and the ten controls; replace `make_model()` with your model:

```bash
python examples/evaluate_new_model.py                                   # seven domains and ten controls, repetition 0 (prints both gaps and the outcome-scale ratio s)
python examples/evaluate_new_model.py --slots cat_050 placebo@9001 --out gaps.csv
python examples/evaluate_new_model.py --extra-controls                  # k is counted among all 50 controls
```

To score predictions of your own pipeline instead, `defectshift.evaluation.gaps_from_predictions` returns both gaps and the outcome-scale ratios from the predictions of `f_S` on `A_heldout` and `B_te` and of `f_T` on `B_te`, and `k_among_controls` counts the controls with a smaller gap.

`--slots` takes the challenge domains of `splits/final_domains.json` (for example `cat_050`; the temporal domains are named by their cutoff, `x_time<=2099-05-04 21:18:39` and `y_time<=2099-05-10 02:20:07.500000`) and the controls `placebo@9001` to `placebo@9010` (`placebo@9011` to `placebo@9050` after `extra_controls.register()`, which `--extra-controls` calls).

To build one split in code (fixed seeds, no tuning on `B_te`):

```python
from defectshift import config
from defectshift.candidates import build_one, load_data
from defectshift.splits import B_view, get_A, matched, repeat_split

df, y = load_data()
lot = df[config.LOT_COL]
slot, mA, mB = build_one(df, "cat_050")                          # or "placebo@9001" for a randomized lot-split control
a = get_A(slot, df.index[mA], y, lot)                            # a["A_dev"], a["A_heldout"]
b = B_view(slot, df.index[mB], y, lot, "benchmark")              # b["B_tr"], b["B_val"], b["B_te"]
a_train, a_val = repeat_split(a["A_dev"], y, lot, 0)             # repetition 0
n_star, a_match, b_match = matched(a_train, b["B_tr"], 0)        # matched samples of the source and the target model
```

## Splits and notation

Each domain divides the records into a source group A and a target group B: for a process-condition domain, A is
the most frequent category of the defining column and B all other categories; for a temporal domain, A is the part
before the median of the time axis and B the part after it. The defining column is not a model input.

| Subset | Share | Use | Seed |
|---|---|---|---|
| `A_heldout` | 20% of A | in-domain test set, fixed for the domain | 2027 |
| `A_dev` | 80% of A | development set; its tuning split (80/20) and, in each repetition, `A_train` (60% of A) and `A_val` (20% of A) | tuning split 999 |
| `B_tr` | 48% of B | target training pool (target-trained references, target-side tuning) | 7777 for the four-way split of B |
| `B_val` | 12% of B | validation set of target-side tuning only | |
| `B_screen` | 15% of B | domain screening only | |
| `B_te` | 25% of B | final test set, reserved for the final evaluation | |

Every split, including the internal early-stopping splits, is grouped by production lot, so that the
records of one lot never fall on both sides, and stratified by whether the defect rate is positive. The subsets are
created once per domain and do not change across repetitions. In repetition i, the matched
training-pool size n* = min(|A_train|, |B_tr|) is drawn from `A_train` and from `B_tr` (matched samples), and models with early stopping
hold out an internal 20% of their training rows, grouped by lot (seed 5000 + i).

Seeds (see the table for the fixed splits): in repetition `i`, the split of `A_dev` into
`A_train` and `A_val` and the matched draws use seed `42 + i`, and the internal early-stopping split uses seed `5000 + i`. The model
seed is `42 + o + i`, with offset `o = 0` for `ID_full`, 100 for `ID_match` and 200 for `OOD_match`. Hyperparameter searches fix the
global seed to 999, and the Optuna sampler uses seed 42. Every fit records its seed, settings and training-row indices.

The stratification by a positive defect rate only stabilizes the share of nonzero defect rates across subsets. Lots
that span both domains stay on both sides of A and B, which favors the source model. Models without early stopping
train on all n* records of the matched sample.

- Process-condition candidates: at least 300 records on each side (one column with a 92-record minority side is removed); the defining column is not a model input because it is constant within A and takes only unseen values in B.
- Random-split controls: seeds 9001 to 9010; the target sizes match the 5th to 95th percentiles of the eligible process-condition candidates.
- Screening eligibility: floor(0.25 |B|) >= 500, so that `B_te` has at least 500 records; ties in the consensus score are broken by name.
- Categorical vocabularies follow the fixed dataset schema; one-hot encodings are fitted on the training rows and map unseen levels to zero.
- When standardizing, a column whose training standard deviation is positive but below 1e-3 uses 1e-3 as the denominator.
- Gaussian-process regression is not included (cost on the largest domains); kernel regression is represented by support-vector regression.

| Paper | Code and files | Note |
|---|---|---|
| source domain A, target domain B | slot (= candidate name) built by `candidates.build_one` | candidate names: `cat_050`, `x_time<=2099-05-04 21:18:39`, `placebo@9001` |
| randomized lot-split control | `placebo@<seed>` | seeds 9001 to 9010; `config.PLACEBO_B_TARGET` |
| `A_held`, `A_dev` | `A_heldout`, `A_dev` | `splits.get_A` |
| `A_train`, `A_val` | `a_train`, `a_val` | `splits.repeat_split`, one pair per repetition |
| `B_tr`, `B_val`, `B_scr`, `B_te` | `B_tr`, `B_val`, `B_screen`, `B_te` | `splits.B_view`; the stage decides which subsets are returned |
| `n*`, `A_match`, `B_match` | `n_star`, `a_match`, `b_match` | `splits.matched` |
| `f_S,full` | regime `ID_full` | source model fitted on `A_train` (early stopping on `A_val`) |
| `f_S,match` | regime `ID_match` | source model fitted on `A_match` |
| `f_T` (target-trained reference) | regime `OOD_match` | fitted on `B_match`; `OOD_match_tuned` is the target-side tuning sensitivity |

## Rules

The code enforces which subsets a stage may read (`defectshift.splits.STAGE_ALLOWED`).

1. `B_te` is used only for the final evaluation, not for training, tuning, early stopping, model selection or domain selection (the one exception, unlabeled inputs for transductive test-time adaptation, is rule 6). Its labels are released, so the code cannot enforce this rule; compliance is the user's responsibility.
2. `B_screen` is used only to reproduce the screening that selected the challenge domains.
3. A challenge domain is read against the controls; the controls are not challenge domains.
4. Evaluating without the records whose lot also occurs in the source training pool is an evaluation mask; it changes no split.
5. The five training repetitions change the split of `A_dev`, the matched draws, the internal early-stopping split and the training seed, and keep `A_heldout` and the target partition fixed. A redrawn target split changes the target partition and does not replace the canonical split.
6. Label-free adaptation methods:
   - Target labels from any target-side partition (`B_tr`, `B_val`, `B_screen` or `B_te`) must not be used for training, tuning, early stopping, model selection or adaptation; `B_te` labels are used only to evaluate the final predictions.
   - Unlabeled input features from `B_tr` may be used. Unlabeled `B_te` input features may be used only when explicitly evaluating transductive test-time adaptation, and such results must be reported separately from methods that do not use test inputs.
   - The target-trained reference is a separate supervised comparator: it continues to use `B_tr` labels under the canonical protocol and is not a label-free adaptation method.

## Partition table

`splits/partitions.csv` assigns every record to one subset of each split, for the seven challenge domains and the ten controls (17 splits): `slot` (candidate name), `row` (row position in `data.parquet`), `partition` (`A_heldout`, `A_dev`, `B_tr`, `B_val`, `B_screen` or `B_te`).

```python
import pandas as pd

part = pd.read_csv("splits/partitions.csv")
a_dev = part.loc[(part["slot"] == "cat_050") & (part["partition"] == "A_dev"), "row"]
```

`splits/make_partitions.py --check` recomputes the table from `data.parquet` with the released split code and fails if the file differs. The per-repetition subsets (`A_train`, `A_val`, the matched samples, the early-stopping split) are not in the table; the `.npz` file of every fit of a benchmark run holds its training and early-stopping rows.

## Lot-overlap annotation

`splits/target_test_lot_overlap.csv` lists every record of `B_te` of the 17 splits with `slot`, `row` and `source_train_lot_overlap` (`True` if the lot also occurs in `A_dev` of the same split). The marks are all `False` for the controls. In the challenge domains 0% to 11.3% of the `B_te` records are marked (`cat_005` 197 of 1,736; `cat_019` 214 of 1,959; `cat_039` 84 of 2,080; `cat_045` 20 of 1,528; `y_time` 30 of 1,789; `cat_050` and `x_time` none).

```python
overlap = pd.read_csv("splits/target_test_lot_overlap.csv")
rows = overlap.loc[(overlap["slot"] == "cat_050") & ~overlap["source_train_lot_overlap"], "row"]   # lot-disjoint rows
```

For the ID-OOD gap, the paper restricts only the target side; `A_heldout` stays complete. `splits/source_heldout_lot_overlap.csv` is the matching annotation of the source test set (`target_test_lot_overlap` is `True` if the lot also occurs in `B_te`); only `cat_005`, `cat_019`, `cat_039`, `cat_045` and `y_time` have marked records (91 of the 10,619 `A_heldout` records of the seven challenge domains). `splits/make_lot_overlap.py --check` recomputes both tables; `tests/test_lot_overlap.py` does the same in the test suite.
