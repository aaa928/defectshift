# Data license

## DefectShift data and derived artifacts

The DefectShift dataset (`data.parquet`) and the artifacts derived from it that are distributed with this
project are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
The derived artifacts are:

- in this repository: the data files (`*.csv`, `*.json`) in `splits/` and `data/` (candidate list, final domains, partition table, row-level lot-overlap
  annotations, data dictionary); the scripts (`*.py`) in these folders are code and fall under the MIT license;
- on Zenodo: the tuning records (one CSV).

Under this license you may copy, redistribute and adapt the material for non-commercial purposes, provided that
you give appropriate credit, provide a link to the license and indicate if changes were made. Commercial use is
not permitted. This summary is not a substitute for the
[legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## Model results

No per-record predictions, per-fit metrics or machine-readable result tables of any model are part of the release (aggregate results are reported in the paper and its Supplementary Material). The licenses of TabPFN-3.5 (License v1.0), TabFM
(Non-Commercial License v1.0) and EXAONE-Tabular (EXAONE AI Model License 1.2-NC) are non-commercial and restrict the use of the outputs of these models, and
the release treats all 22 models alike and distributes the results of none of them. Anyone who runs the code is bound by the license of each model that is run.
See `docs/reproducing.md` (Model licenses).

When you use the data, please cite the dataset record (and the paper once it is published):

Choi, Y. DefectShift: A tabular benchmark of display production-line data. Zenodo, 2026. https://doi.org/10.5281/zenodo.24313062

## Code

The code in this repository is licensed separately under the MIT License (see `LICENSE`).
