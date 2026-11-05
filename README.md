# Class-conditional conformal prediction masks chemical-support undercoverage in BBB screening

Code, data and outputs for the RECOMB 2027 submission by Sanjeda Sara Jennifer and Niloofar Yousefi
(University of Central Florida).

Class-conditional (Mondrian) conformal calibration restores coverage of the minority class (BBB−) on average, but
leaves BBB− compounds with low Tanimoto similarity to the training set under-covered. This repository reproduces
every number, table and figure of the paper and its supplement from the cached predictions, and retrains every
model from scratch if wanted.

## What is here

| Path | Contents |
|---|---|
| `src/bbbcp/` | Library: data curation, label-free splits, models (ECFP4 random forest, GIN, Chemprop), calibrators |
| `scripts/` | One script per analysis, table or figure (list below) |
| `configs/experiments/` | Experiment grids; `full.yaml` is the confirmatory grid (seeds 100-119) |
| `data/raw/` | BBBP (MoleculeNet) and B3DB (CC0) as downloaded |
| `results/raw/` | Cached predictions for VAL, CAL and TEST, one file per dataset, split regime, model and seed (100-119) |
| `results/processed/` | Curated datasets; Stage B per-seed calibration results (`full_v13_conformal.csv.gz`) |
| `tables/`, `figures/` | Analysis outputs used by the paper |
| `paper/latex/` | `main.tex`, `supplement.tex`; `generated/` holds every number as a macro, with provenance |
| `tests/` | Unit tests (split integrity, no leakage, conformal quantiles, coverage on synthetic data) |

Only the confirmatory seeds 100-119 are used in the paper. Pilot runs (seeds 0-9) used to design the study are not
included.

## Install

Python 3.12. Exact versions are pinned in `requirements-lock.txt` (RDKit 2026.03.6, scikit-learn 1.9.1,
PyTorch 2.14.1, torch-geometric 2.8.0, Chemprop 2.3.1).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e .
export PYTHONPATH=$PWD/src          # run every command from the repository root
```

## Check the paper's numbers (minutes, no training)

```bash
python scripts/make_paper_numbers.py        # every number in the paper, as LaTeX macros with provenance
python scripts/make_supplement_tables.py    # supplementary tables
python scripts/verify_paper_numbers.py      # independent recomputation from the caches; must end in PASS
python scripts/check_cited_keys.py          # every number cited in main.tex and supplement.tex is verified
```

`verify_paper_numbers.py` shares no code with the analysis pipeline: it re-implements the conformal thresholds,
support strata and statistics and recomputes each number from the cached predictions.
`paper/latex/generated/numbers_provenance.csv` records the source table and filter of every number.

## Reproduce tables and figures from the caches

| Paper item | Command |
|---|---|
| Table 1, Fig. 2 and the headline numbers | `python scripts/make_paper_numbers.py`, `python scripts/make_main_figures.py` |
| Fig. 1 (editable draw.io source and PDF) | `python scripts/make_fig1_overview.py` |
| Figs. 3 and 4 | `python scripts/make_main_figures.py` (after the analyses below) |
| Table S3 (predictive performance) | `python scripts/run_classifier_metrics.py` |
| Tables S4-S8, S11 | `python scripts/make_supplement_tables.py` |
| Table S9 (same-cell-size random strata) | `python scripts/run_random_strata_contrast.py` |
| Table S10, Fig. S1 (common grid, feasibility) | `python scripts/run_common_grid.py --workers 4` |
| Table S12 (existing conditional methods) | `python scripts/run_conditional_baselines.py` |
| Table S13, Fig. S2 (mechanism) | `python scripts/run_mechanism.py` |
| Fig. S3 (reliability by support) | `python scripts/run_support_calibration.py` |
| Fig. S4 (coverage against support) | `python scripts/run_support_curve.py` |
| Fig. S5 (calibration-set audit) | `python scripts/run_cal_audit.py` |
| Table S14, Fig. S6 (other tasks, prior protocol) | `python scripts/run_generality.py --runs bace/random bace/scaffold clintox/random clintox/scaffold tox21_sr_are/random tox21_sr_are/scaffold --models rf --tag generality` and `python scripts/run_generality.py --runs bbbp_raw/random5025 --models rf gcn --tag tursun_confirm` |
| Tables S16-S17, Fig. 3 (shared failures) | `python scripts/run_persistent_failures.py`; without the three curation-error records: `python scripts/run_persistent_failures.py --exclude-ids B3DB:2544 B3DB:3330 B3DB:3001 --stageb none --tag _excl3` |
| Table S15 (concordance against a covariate-stratified null) | `python scripts/run_concordance_conditional.py` |
| Table S18 (TPSA association adjusted for support) | `for m in rf gin chemprop; do python scripts/run_descriptor_adjustment.py --runs "results/raw/{bbbp,b3db}/{random,scaffold}/${m}_ce/seed1??.npz" --tag _confirmatory_$m; done` |
| Fig. S7 (structures), Fig. S9 (chemistry) | `python scripts/make_figS_structures.py`, `python scripts/make_fig_chemistry.py` |
| Table S19 (screening decisions) | `python scripts/run_screening_utility.py` |
| Table S20, Fig. 4 (calibration plus rejection filters) | `python scripts/run_calibrated_rejection.py` |
| Label, name and logBB checks | `python scripts/run_label_audit.py`, `python scripts/run_name_audit.py`, `python scripts/run_logbb_check.py` |
| Molecule and scaffold bootstrap | `python scripts/run_cluster_bootstrap.py` |
| Supplementary Figs. S1-S3, S5, S6, S8 | `python scripts/make_supp_figures.py` |
| Build the PDFs | `cd paper/latex && latexmk -pdf main.tex supplement.tex` |

Analysis scripts write CSV files to `tables/`; `make_supplement_tables.py`, `make_paper_numbers.py` and the figure
scripts turn them into the paper's tables, numbers and figures.

## Retrain from scratch (GPU recommended for GIN and Chemprop)

```bash
python scripts/run_grid.py configs/experiments/full.yaml            # BBBP, B3DB, transfer, cluster; seeds 100-119
python scripts/run_grid.py configs/experiments/generality.yaml --stage A
python scripts/run_grid.py configs/experiments/tursun_confirm.yaml --stage A
```

Splits depend only on the data and the seed, so they reproduce exactly. Model predictions can differ in the last
digits across platforms and RDKit versions; the conclusions do not.

## Data

B3DB (Meng et al., Scientific Data 2021) is distributed under CC0. BBBP (Martins et al., 2012) is distributed
through MoleculeNet. Curation (standardization, de-duplication, stereoisomer collapse, cross-dataset overlap audit)
is in `src/bbbcp/data/` and is reported in the supplement.

## License

MIT (see `LICENSE`).