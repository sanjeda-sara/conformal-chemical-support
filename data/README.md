# Data

| File (data/raw/) | Source | Version / commit | Licence | Retrieved | Modification |
|---|---|---|---|---|---|
| BBBP.csv | MoleculeNet BBBP (Martins et al., JCIM 2012), via torch_geometric `MoleculeNet` (deepchem S3), copied from the MoML project folder | file as downloaded Sep 2026 | MoleculeNet distribution; cite Martins 2012 and Wu et al. 2018 | 2026-09-08 (MoML) | none (raw copy) |
| B3DB_classification.tsv | github.com/theochem/B3DB (Meng et al., Sci Data 8:289, 2021) | commit 75dab1cc607bb7a03f3de5c576cffd223e767844 (2025-09-10) | CC0-1.0 | 2026-10-08 | none |
| B3DB_classification_external.tsv | same | same | CC0-1.0 | 2026-10-08 | none |
| B3DB_regression.tsv | same (logBB subset, Tier 3, unused so far) | same | CC0-1.0 | 2026-10-08 | none |

All processing happens in code (`src/bbbcp/data/datasets.py`, `src/bbbcp/chemistry/standardize.py`). Rules and counts are in `docs/DATA_AUDIT.md`, `results/dataset_audit.csv` and `results/dataset_overlap_audit.csv`.

Label convention: y = 1 means BBB+ / permeable (the majority class); y = 0 means BBB− (the minority class).

`results/processed/*.pkl` are caches (standardized frame, ECFP4 bits, fixed k-means clusters). Delete them to rebuild.

Secondary MoleculeNet tasks (BACE, ClinTox, HIV) are Tier 2. They are not included, because the sandbox could not reach the deepchem S3 bucket. Fetch them with `python scripts/download_data.py --optional`.
