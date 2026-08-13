# Personalized Federated Tensor Regression

This repository contains the core numerical implementation accompanying the
paper. It includes the proposed private personalized federated estimator, the
single-client and ADMM baselines, hyperparameter tuning, rank selection,
simulation data generation, and ADHD-200 preprocessing and analysis functions.

Plotting code, one-off command-line wrappers, cached runs, and historical
diagnostic scripts are intentionally excluded.

## Install

```bash
python -m pip install -e .
```

## Code map

- `pftr.dgp`: strict low-Tucker-rank common component, strict sparse deviations,
  and tensor-regression sampling.
- `pftr.single_client`: one-stage ADMM initialization and local two-stage model.
- `pftr.federated`: private Stage I and local FISTA personalization.
- `pftr.tuning_federated`, `pftr.tuning_single`: common-fold CV functions.
- `pftr.rank_selection`: ridge-type Tucker-rank selection.
- `pftr.data.adhd200`: raw T1/phenotype matching and db4 tensor construction.
- `pftr.experiments.simulation`: Figure 2 and Figure 3 numerical experiments.
- `pftr.experiments.real_data`: Section 7 fitting and Table 3 benchmarks.

## Reproducing the simulation

Read `configs/paper.yaml`, tune the ADMM, federated, and single-client parameter
grids with the functions in `pftr.tuning_*`, then pass the selected parameter
dictionaries to `run_sample_size_experiment` or `run_privacy_sensitivity`.

The final design generates `A0` and all `B_k` once. Every replication regenerates
only `X`, `E`, and `Y`, and all methods within a replication use the same data.
Figure 2 compares non-private federated, private federated, single-client
two-stage, and ADMM initialization. Figure 3 uses the same `K=5`, heterogeneity
ratio `0.5`, and fixed-truth design at `n_k=250`.

## Reproducing the ADHD-200 analysis

1. Obtain ADHD-200 T1 images and phenotypic tables from the source named in
   `configs/paper.yaml`.
2. Build tensors with `build_adhd200_dataset` and export only through
   `build_public_workbook`.
3. Load and select the paper cohort with `load_analysis_workbook` and
   `prepare_three_site_analysis`.
4. Convert it with `client_tensor_datasets`.
5. Tune on common five-fold splits, fit with `fit_real_data`, and compare the
   four Table 3 methods with `compare_real_data_benchmarks`.

The final paper cohort contains 556 subjects: KKI 80, Peking 221, and NYU 255.

## Data policy

Raw MRI files and participant-level identifiers are excluded. The repository
contains code to build the analysis data locally. Before publishing any derived
data file, verify the ADHD-200 redistribution terms and remove participant ids,
subject folders, and local paths.

The ADHD-200 source page describes unrestricted use for non-commercial research
but requires users to register for access. For that reason this release directs
users to the official source instead of redistributing participant-level data.

## Scope

This is research code. It is not a clinical tool, and estimated imaging
coefficients represent predictive associations rather than causal effects.
