# DefectShift

[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data license: CC BY-NC 4.0](https://img.shields.io/badge/data-CC%20BY--NC%204.0-lightgrey.svg)](DATA_LICENSE.md)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg)](environment/environment.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24313062-blue.svg)](https://doi.org/10.5281/zenodo.24313062)

A tabular regression benchmark for display-manufacturing defect rates under process-condition and temporal shift. It contains 14,276 anonymized records of one production line (1,080 production lots),
seven canonical challenge domains (five process-condition, two temporal) with fixed splits, ten randomized lot-split controls, row-level lot-overlap annotations, and 22 evaluated models.
The data come from one facility, and the five process-condition domains are screened stress tests; see the paper for definitions, results and limitations.

- Data: [Zenodo 10.5281/zenodo.24313062](https://doi.org/10.5281/zenodo.24313062). Place `data.parquet` in `data/` (columns and anonymization: `docs/usage.md`).
- Splits: `splits/` (candidate list, final domains, partition table, lot-overlap annotations).

## Evaluate your own model

Needs `data.parquet`, Python 3.11 and a CPU (numpy, pandas, pyarrow and scikit-learn):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install numpy==2.4.6 pandas==2.2.3 pyarrow==25.0.1 scikit-learn==1.9.1
python examples/evaluate_new_model.py        # seven challenge domains and ten controls, repetition 0; add --extra-controls to read RMSE verdicts against all 50 controls
```

Replace `make_model()` in the example with your regressor. Rules and options: `docs/usage.md`.

## Reproduce the benchmark

Needs `data.parquet`, one 24 GB GPU and the run times in Supplementary Table S18 of the paper.

```bash
conda env create -f environment/environment.yml && conda activate defectshift && pip install --no-deps rtdl==0.0.13 && pip install --no-deps -e .
python scripts/00_fetch_weights.py           # pinned foundation-model weights
```

TabNet (a second environment) and the foundation models (pinned weights) need extra installation steps. `docs/reproducing.md` covers installation, pipeline order, the analyses of the paper and the model licenses.

```bash
pytest tests    # needs the installation above; tests that need data/data.parquet are skipped when it is absent, and the suite takes a while
```

`DATA_LICENSE.md` lists what the release contains and explains why it contains no model results. A rerun can be checked against the tables of the paper (printed tables are rounded, and GPU runs are not bitwise reproducible).

Report errors in the data or the splits, problems with the code and questions as GitHub issues; corrections are issued as new versions listed under Versions.

## Versions

The code repository and the Zenodo record use the same version numbers. v1.0 is the first release: the data (Zenodo record), the benchmark definition (`splits/`), the code of the pipeline and the code of the analyses of the paper.

## License and citation

Source code and documentation: MIT (`LICENSE`), except third-party components and pretrained weights, which keep their own licenses (`docs/reproducing.md`, "Model licenses").
Data and derived artifacts (the `*.csv` and `*.json` files in `splits/` and `data/`, and the Zenodo record): CC BY-NC 4.0 (`DATA_LICENSE.md`); the scripts in these folders are MIT.

Citation: the dataset record (Zenodo DOI above), as given in `DATA_LICENSE.md`.
