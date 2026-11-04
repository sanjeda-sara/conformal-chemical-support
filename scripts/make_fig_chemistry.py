"""Chemistry figure: where in BBB-relevant chemical space BBB- compounds are missed, and which
molecules are missed by every architecture.  Requires scripts/run_persistent_failures.py.

Panel A/B: every eligible unique BBB- molecule of one setting (default B3DB scaffold), placed by
TPSA and mean max-Tanimoto support to TRAIN, coloured by its miss rate averaged over RF, GIN and
Chemprop (A: class-conditional; B: class x support calibration).  Cross-architecture persistent
failures (strict definition) are outlined.
Panel C: exemplar structures selected by a fixed rule, not by appearance: all strict
cross-architecture failures of the setting that carry no label-conflict or curation flag, in
ascending TPSA, first six.  Each is paired with the never-missed BBB- molecule nearest in
(TPSA, support), shown underneath, as a matched control.

usage: python scripts/make_fig_chemistry.py [--setting b3db/scaffold]
Output: figures/fig_chemistry_<setting>.{png,pdf}, tables/fig_chemistry_exemplars_<setting>.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from bbbcp.utils.figio import save

REPO = Path(__file__).resolve().parents[1]
FLAGGED = {"B3DB:2544", "B3DB:3330", "B3DB:3001"}           # curation errors reported in the paper
BB = ["rf", "gin", "chemprop"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--setting", default="b3db/scaffold"); a = ap.parse_args()
    W = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False)
    g = W[(W.setting == a.setting) & W.eligible.astype(bool)].copy()
    g["miss_mond"] = g[[f"rate_mondrian__{b}" for b in BB]].mean(axis=1)
    g["miss_cxs"] = g[[f"rate_mondrian_x__{b}" for b in BB]].mean(axis=1)
    pers = g.cross_arch_mondrian.astype(bool)
    clean = pers & ~g.label_disagrees_other_dataset.astype(bool) & ~g.mol_id.isin(FLAGGED)
    ex = g[clean].sort_values("TPSA").head(6)
    never = g[g.category == "never_missed"]
    ctrl = []
    for _, r in ex.iterrows():
        d = ((never.TPSA - r.TPSA) / g.TPSA.std()) ** 2 + ((never.mean_sim_train - r.mean_sim_train) / g.mean_sim_train.std()) ** 2
        ctrl.append(never.loc[d.idxmin()])
    ctrl = pd.DataFrame(ctrl)
    rec = pd.concat([ex.assign(role="persistent failure"), ctrl.assign(role="matched never-missed control")])
    tag = a.setting.replace("/", "_")
    rec[["role", "mol_id", "name", "smiles", "TPSA", "HBD", "MW", "cLogP", "mean_sim_train", "miss_mond", "miss_cxs"]].to_csv(
        REPO / f"tables/fig_chemistry_exemplars_{tag}.csv", index=False)

    # Full-page layout for the supplement (printed at 6.5 in wide): A and B on top, structures below.
    fig = plt.figure(figsize=(7.0, 10.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.1], hspace=0.30)
    top = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.28)
    vmax = 1.0
    for k, (col, title) in enumerate((("miss_mond", "A  Class-conditional calibration"), ("miss_cxs", "B  Class $\\times$ support calibration"))):
        ax = fig.add_subplot(top[0, k])
        o = g.sort_values(col)
        sc = ax.scatter(o.TPSA, o.mean_sim_train, c=o[col], cmap="Oranges", vmin=0, vmax=vmax, s=10, lw=0)
        ax.scatter(g[pers].TPSA, g[pers].mean_sim_train, facecolors="none", edgecolors="#111", s=46, lw=1.0,
                   label="missed by all three, every split")
        ax.axvline(90, ls=":", c="#777", lw=0.8)
        ax.set_xlabel("TPSA (Å$^2$)", fontsize=9); ax.set_ylabel("support: max Tanimoto to TRAIN", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left"); ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper right")
    cax = fig.add_subplot(top[0, 2]); cb = fig.colorbar(sc, cax=cax)
    cb.set_label("miss rate (mean of 3 backbones)", fontsize=8.5); cb.ax.tick_params(labelsize=8)
    # structure panel: each rule-selected failure (left) beside its matched never-missed control (right)
    pairs = list(zip(ex.itertuples(), ctrl.itertuples()))[:4]
    sub = gs[1].subgridspec(len(pairs), 2, hspace=0.62, wspace=0.06)
    for i, (f, c) in enumerate(pairs):
        for j, (r, lab) in enumerate(((f, "persistent failure"), (c, "never-missed control"))):
            ax = fig.add_subplot(sub[i, j]); ax.axis("off")
            dr = Draw.rdMolDraw2D.MolDraw2DCairo(760, 330); dr.drawOptions().padding = 0.03; dr.drawOptions().minFontSize = 22
            dr.drawOptions().bondLineWidth = 2.2; Draw.rdMolDraw2D.PrepareAndDrawMolecule(dr, Chem.MolFromSmiles(r.smiles)); dr.FinishDrawing()
            import io; from PIL import Image; img = Image.open(io.BytesIO(dr.GetDrawingText()))
            ax.imshow(img)
            nm = str(r.name) if isinstance(r.name, str) and not str(r.name)[:1].isdigit() else r.mol_id
            nm = nm.replace(" hydrochloride", "").replace(" hydrochlor", "")
            ax.set_title(f"{nm[:34]}  ({lab})\nTPSA {r.TPSA:.0f} Å$^2$, support {r.mean_sim_train:.2f}, miss rate {r.miss_mond:.2f}",
                         fontsize=8.5)
    y_c = fig.axes[3].get_position().y1 + 0.045
    fig.text(0.125, y_c, "C  Persistent failures (rule-selected, left) and matched never-missed controls (right)", fontsize=10)
    save(fig, REPO / f"figures/fig_chemistry_{tag}"); plt.close(fig)
    print(f"{len(g)} eligible BBB- molecules, {int(pers.sum())} persistent, {len(ex)} clean exemplars -> figures/fig_chemistry_{tag}")


if __name__ == "__main__":
    main()
