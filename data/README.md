# Data

The raw ADHD-200 MRI files are not redistributed in this repository. Download
the structural T1 images and phenotypic tables from the ADHD-200 Sample
Initiative, then arrange the site directories and CSV files under one local
data root.

Expected source sites for the paper are `KKI`, `NYU`, `Peking_1`, `Peking_2`,
and `Peking_3`. The preprocessing functions in `pftr.data.adhd200` perform:

1. BIDS T1 image discovery and participant-id matching;
2. phenotypic column harmonization;
3. three-level Daubechies db4 wavelet decomposition;
4. resizing of approximation coefficients to `12 x 14 x 12`;
5. de-identified workbook export;
6. selection and pooling of the three analysis clients.

`build_adhd200_dataset` internally retains participant identifiers and relative
paths for matching and audit. `build_public_workbook` removes those columns.
Never commit the internal metadata or raw MRI tree.
