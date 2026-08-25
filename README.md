# Personalized Federated Tensor Regression

This repository contains the core numerical implementation accompanying the paper. It includes the proposed private personalized federated estimator, the single-client and ADMM baselines, rank selection, simulation data generation, and ADHD-200 preprocessing and analysis functions.

Plotting code, one-off command-line wrappers, cached runs, and historical
diagnostic scripts are intentionally excluded.

## Code map

- `pftr.dgp`: strict low-Tucker-rank common component, strict sparse deviations,
  and tensor-regression sampling.
- `pftr.single_client`: one-stage ADMM initialization and local two-stage model.
- `pftr.federated`: private Stage I and local FISTA personalization.
- `pftr.rank_selection`: ridge-type Tucker-rank selection.
- `pftr.data.adhd200`: raw T1/phenotype matching and db4 tensor construction.
- `pftr.experiments.simulation`: Figure 2 and Figure 3 numerical experiments.
- `pftr.experiments.real_data`: Section 7 fitting and Table 3 benchmarks.

## Reproducing the simulation

The simulation module contains the implementations of the proposed personalized federated tensor regression method and the competing methods used in the paper, including the non-private federated estimator, single-client two-stage estimator, and ADMM-based baseline.

The main experimental routines are:

* `run_sample_size_experiment`: evaluates estimation performance under different client sample sizes.
* `run_privacy_sensitivity`: evaluates the effect of differential privacy on estimation performance.

Within each replication, all methods are evaluated on the same generated data to ensure a fair comparison. The simulation settings correspond to those used for Figure 2 and Figure 3 in the paper.


## Reproducing the ADHD-200 analysis

1. Obtain ADHD-200 T1 images and phenotypic tables from the official ADHD-200 source.
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
