# Release Manifest

Included:

- tensor regression DGP and tensor algebra;
- strictly low-Tucker-rank and strictly sparse parameter generation;
- one-stage ADMM initialization;
- single-client two-stage estimator;
- private and non-private personalized federated estimator;
- Gaussian privacy-noise calibration and gradient truncation;
- federated and single-client cross-validation functions;
- ridge-type Tucker-rank selection;
- ADHD-200 raw T1 and phenotype ingestion;
- db4 wavelet and `12 x 14 x 12` feature construction;
- de-identified processed-workbook export and analysis loading;
- Figure 2 and Figure 3 numerical experiment functions;
- Fed-2Stage, Fed-Avg, Fed-Common, Single-2Stage, and ADMM benchmark functions;
- final paper configuration and selected parameter values;
- smoke tests for the DGP, privacy branch, and de-identification boundary.

Excluded:

- plotting functions and generated figures;
- command-line wrappers and `main()` calls;
- historical experiments, diagnostics, caches, and logs;
- LaTeX/PDF manuscript sources;
- raw ADHD-200 MRI files;
- participant ids, subject-folder names, and local file paths;
- atlas mapping and MNI pilot code, which did not generate the fitted Figure 1 coefficients.

Before public release, choose a software license and confirm whether any derived
participant-level feature file may be redistributed under the source dataset's terms.
