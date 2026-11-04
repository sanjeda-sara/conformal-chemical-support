# recomb2027-bbb-conformal

Support-aware class-conditional conformal prediction for imbalanced molecular screening, with blood–brain barrier (BBB) permeability as the case study.

This is the working repository for the RECOMB 2027 submission. It grew out of the MoML 2026 workshop paper, which is preserved unchanged in `legacy_moml2026/`.

Start with:
- `docs/PROJECT_STATUS.md`: what is done, what was found, and the go/no-go assessment.
- `docs/DATA_AUDIT.md`: the 23-item audit of the MoML code and the BBBP/B3DB data audit.
- `docs/RESEARCH_GAP.md`: literature position (53-paper matrix in `docs/RELATED_WORK_MATRIX.csv`).
- `docs/THEORY_NOTES.md` and `docs/METHOD_SPECIFICATION.md`.
- `paper/recomb2027_outline.md`, `paper/abstract_candidate.md` and `paper/contribution_candidates.md`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda env create -f environment.yml
pip install -r requirements.txt && pip install -e .
```

The code runs on CPU, CUDA or Apple MPS. GIN uses `device_auto()`, and Chemprop uses `accelerator="auto"`.

## Commands

| What | Command | Time (2-core CPU) |
|---|---|---|
| Unit and integrity tests | `python -m pytest -q` | ~3 min |
| Smoke test | `bash scripts/run_smoke_test.sh` | ~1 min |
| Data audit tables | `python scripts/audit_datasets.py` | ~1 min |
| BBBP grid | `bash scripts/run_bbbp.sh` | ~45 min |
| B3DB + BBBP→B3DB transfer | `bash scripts/run_b3db.sh` | ~40 min |
| Executed pilot (core benchmark) | `bash scripts/run_core_benchmarks.sh` | ~2.5 h |
| Confirmatory grid (fresh seeds 100–119) | `sbatch --array=0-839%40 slurm/run_full_array.sbatch`, then `python scripts/run_grid.py configs/experiments/full.yaml --stage B` | HPC |
| Ablations on cached predictions | `bash scripts/run_ablations.sh` | ~40 min |
| Tables / figures from cached results | `make tables` / `make figures` | ~1–2 min |

**Figure quality.** Every figure is written as a 300-dpi PNG and a vector PDF with embedded fonts. Use the PDFs in the paper. To change this, set environment variables before `make figures`: `BBBCP_FIG_DPI=600 BBBCP_FIG_FORMATS=png,pdf,svg make figures`. No retraining is needed; the figures are rebuilt from the cached results in `results/`.

## Layout

```
src/bbbcp/
  chemistry/   standardize.py (salt strip, uncharge, InChIKey, scaffold); fingerprints.py (ECFP4, Tanimoto, descriptors)
  data/        datasets.py (BBBP, B3DB loaders + audit); splits.py (label-free random / balanced scaffold / cluster holdout)
  models/      gin.py (refactored MoML GIN, correct MC dropout); baselines.py (ECFP-RF, Chemprop v2)
  conformal/   scores.py (LAC, APS, RAPS); calibrators.py (marginal, Mondrian, Mondrian-PAC, class×stratum, SABM, weighted)
  metrics/     classification + prediction-set metrics, screening cost
  stats/       Wilson / Clopper-Pearson, paired bootstrap, Wilcoxon, Holm, scaffold-cluster bootstrap
  pipeline.py  Stage A: split -> fit -> cache predictions + provenance registry
  evaluate.py  Stage B: all scores x calibrators x alpha on cached predictions; calibration-support sweep
configs/experiments/   pilot_{rf,gin,chemprop}.yaml (executed), full.yaml (confirmatory), bbbp.yaml, b3db.yaml
scripts/               run_*.sh / run_*.py, analyze_pilot.py, run_biology.py, make_paper_figures.py
tests/                 split integrity, scaffold disjointness, leakage, quantiles, APS/RAPS, synthetic coverage
results/raw/           cached predictions per run (npz)       results/manifests/registry.jsonl  provenance per run
results/processed/     long-format conformal results           tables/  figures/  docs/  paper/
legacy_moml2026/       original MoML project (unchanged, for provenance)
```

## Status tags used in the docs

- **[IMPLEMENTED]**: the code exists and has been inspected.
- **[TESTED]**: a test passed.
- **[EXECUTED]**: the experiment was run.
- **[VERIFIED RESULT]**: the output was inspected and sanity-checked.
- **[PLANNED]**: not yet done.

No number in `docs/` or `paper/` comes from the legacy MoML pipeline unless it is explicitly labelled legacy.

## Licence

Code: MIT. B3DB data: CC0-1.0. BBBP: MoleculeNet distribution.

## v1.4 (2026-10-21, after an external PC-style review)

- **Read first: `docs/REVIEW_RESPONSE_2026-10-21.md`.** Every falsifiable claim in that review was checked against the data before acting on it.
- **The headline changed.** At the default operating point (3 support bins, full calibration, alpha = 0.1) SABM is bit-identical to a fixed class x support Mondrian in 5 of 7 settings: the coverage gain belongs to the textbook taxonomy, not to adaptive merging. SABM is now presented as a robustness mechanism for the regime where a partition outruns its calibration data, with the operating range measured (`scripts/run_adaptive_vs_fixed.py`).
- **Reproducibility fix.** Derived `*.pkl` caches are no longer load-bearing: an unreadable or version-incompatible cache rebuilds from `data/raw/`. Pandas pickles are not portable across major versions, which is why an outside reviewer saw 9 test failures. `rm -f results/processed/*.pkl && python -m pytest -q` now passes in 45 s.
- **Statistics corrected.** A pre-specified primary Holm family replaces the 48-method-per-cell design, which could not reject at any effect size with 10 seeds.
- **New analysis.** The TPSA/HBD reliability disparity survives adjustment for chemical support (stratified, adjusted logistic, and within-stratum permutation, p < 0.005 in all four exchangeable settings).
- Tests: 56 passing, with no cached artifacts required.

## v1.3 (2026-10-21)

- Every item of the Oct 21 request list was audited, and most are now implemented: **`docs/ITEM_AUDIT_2026-10-21.md`**.
- What to run, in which order, day by day to Nov 4: **`docs/RUNBOOK_2026-10-21.md`**. VS Code tasks 0–14 run each step.
- Stage B defaults to the `full` profile (every baseline, ablation, variant and metric). `--profile v12` reproduces v1.2 exactly; a unit test checks that the unchanged v1.2 methods give identical numbers.
- New configs: `tursun_repro.yaml`, `other_tasks.yaml`, `backbones.yaml`, `training_ablations.yaml`, `extra_regimes.yaml`.
- Tests: 59 passing, from an empty cache.

## v1.2 (2026-10-09)

- **n_req rule frozen** before the confirmatory seeds: ε = α, δ = 0.10 (n_req = 22 / 11 / 5). The original rule is kept as `sabm_abs05`. Tag the commit `freeze-nreq-2026-10-09` before running seeds 100–119.
- `pyproject.toml` now says `requires-python >= 3.9`.
- `docs/RESEARCH_DESIGN.md`: research questions, hypotheses, theorems, baselines, experiments, ablations, statistics, compute and acceptance strategy in one place (pre-registration).
- `scripts/run_simulation_s1.py`: synthetic simulation S1 (executed; `tables/s1_simulation_panel*.csv`, `figures/s1_simulation.*`).
- `paper/latex/`: RECOMB-format draft (`main.pdf`) and generated appendix (`appendix.pdf`). See `paper/latex/README.md`.
- No GPU needed: the 840-job confirmatory grid is about 17 CPU core-hours.

### Local setup on a Mac

Chemprop v2 has no Python 3.9 release, and the pinned versions in `requirements.txt` need Python ≥ 3.11. The core code (RF, conformal methods, tests) still runs on 3.9, but the full grid needs **Python 3.12**. `uv` installs it per user, with no admin rights and no global pip:

    curl -LsSf https://astral.sh/uv/install.sh | sh        # one-time; restart the terminal afterwards
    uv venv --python 3.12 .venv312
    source .venv312/bin/activate
    uv pip install -r requirements.txt
    export PYTHONPATH=$PWD/src
    python -m pytest -q                                     # expect 36 passed
