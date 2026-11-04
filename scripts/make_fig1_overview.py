"""Figure 1: overview of the finding, as an editable diagrams.net (draw.io) project.

Every chart element is a native draw.io object (text, rectangles, lines, polylines, markers); only the two
molecule depictions are embedded images.  All plotted values are computed here from the confirmatory caches
(seeds 100-119) and the confirmatory summary table, written to tables/fig1_values.csv, turned into paper
keys by make_paper_numbers.py and recomputed independently by verify_paper_numbers.py.

Representative setting (chosen by a fixed rule, not by appearance):
  gap = mean high-support minus mean low-support BBB- coverage under class-conditional calibration (LAC,
  alpha 0.1, CE).  Among the six random-split cells (the regime in which exchangeability holds, so the figure
  can be read against Section 4), take the lower median of the gap (third of six, ties broken by setting and
  backbone name).  The lower median of all twelve cells is also recorded in tables/fig1_selection.csv.
Traced molecule: the BBB- test compound with the most splits in which it is excluded under class-conditional
  calibration and included under class x support (ties: more appearances, then identifier); the trace uses
  the lowest seed in which that happens.

usage: python scripts/make_fig1_overview.py            # writes the .drawio and, if draw.io is installed, the PDF
outputs: figures/fig1_chemical_support_overview.drawio (+ .pdf), tables/fig1_values.csv,
         tables/fig1_selection.csv, tables/fig1_scores.csv.gz
"""
import base64
import glob
import html
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.chemistry.fingerprints import ecfp4_bitvects
from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.pipeline import prepare

REPO = Path(__file__).resolve().parents[1]
T = REPO / "tables"; F = REPO / "figures"
ALPHA = 0.1
SEEDS = range(100, 120)
NAMES = {"rf": "random forest", "gin": "GIN", "chemprop": "Chemprop"}
DSN = {"bbbp": "BBBP", "b3db": "B3DB"}


# =============================================================================== 1. data
def select_cell():
    S = pd.read_csv(T / "confirmatory_summary.csv", low_memory=False)
    f = S[(S.score == "lac") & (S.alpha == ALPHA) & (S.loss == "ce") & (S.method == "mondrian")
          & S.setting.isin(["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"])]
    G = f[["setting", "model_name", "cov_min_bin0", "cov_min_bin2"]].copy()
    G["gap"] = G.cov_min_bin2 - G.cov_min_bin0
    G = G.sort_values(["gap", "setting", "model_name"]).reset_index(drop=True)
    G["rank_all12"] = np.arange(1, 13)
    R = G[G.setting.str.endswith("/random")].reset_index(drop=True)
    pick = R.iloc[2]                                          # lower median of six
    G["selected"] = (G.setting == pick.setting) & (G.model_name == pick.model_name)
    G["lower_median_all12"] = G.rank_all12 == 6
    G.to_csv(T / "fig1_selection.csv", index=False)
    return pick.setting, pick.model_name, G


def appearances(setting, model):
    rows, thr = [], []
    for s in SEEDS:
        z = np.load(REPO / f"results/raw/{setting}/{model}_ce/seed{s}.npz", allow_pickle=True)
        mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
        Sc = 1.0 - z["p_cal"][np.arange(len(yc)), yc]; St = 1.0 - z["p_test"]
        bc, _ = sim_bins(z["sim_val"], z["sim_cal"]); bt, _ = sim_bins(z["sim_val"], z["sim_test"])
        TM = C.mondrian(Sc, yc, St, ALPHA); TX = C.mondrian_x(Sc, yc, St, ALPHA, bin_cal=bc, bin_test=bt)
        qc = TM[0, mino]
        for b in (0, 1, 2):
            thr.append({"seed": s, "tert": b, "q_class": qc, "q_stratum": TX[bt == b, mino][0],
                        "n_cal_stratum": int(((yc == mino) & (bc == b)).sum())})
        for i in np.where(yt == mino)[0]:
            rows.append({"seed": s, "mol_id": z["test_ids"][i], "tert": int(bt[i]), "s_max": float(z["sim_test"][i]),
                         "score": float(St[i, mino]), "q_class": float(qc), "q_stratum": float(TX[i, mino])})
    D = pd.DataFrame(rows)
    D["excess"] = D.score - D.q_class
    D["in_class"] = D.score <= D.q_class; D["in_cxs"] = D.score <= D.q_stratum
    return D, pd.DataFrame(thr)


def pick_molecule(D):
    D = D.assign(rescued=~D.in_class & D.in_cxs)
    g = D.groupby("mol_id").agg(n=("seed", "size"), rescued=("rescued", "sum"), low_all=("tert", lambda t: bool((t == 0).all())))
    g = g[g.low_all].reset_index().sort_values(["rescued", "n", "mol_id"], ascending=[False, False, True])
    mol = g.iloc[0]
    seed = int(D[(D.mol_id == mol.mol_id) & D.rescued].seed.min())
    return mol, seed


def neighbour(setting, model, seed, mol_id):
    ds = setting.split("/")[0]
    df = prepare(ds)["df"]
    z = np.load(REPO / f"results/raw/{setting}/{model}_ce/seed{seed}.npz", allow_pickle=True)
    from rdkit import DataStructs
    q = df.set_index("mol_id").loc[mol_id]
    tr = df.iloc[z["idx_train"]]
    fq = ecfp4_bitvects([q.smiles])[0]; ft = ecfp4_bitvects(list(tr.smiles))
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fq, ft))
    j = int(np.argmax(sims))
    return q.smiles, tr.smiles.iloc[j], tr.mol_id.iloc[j], float(sims[j])


def provenance(setting, model, seed, mol_id, nn_id, thr, D):
    from bbbcp.chemistry.standardize import standardize_smiles
    from rdkit import Chem, DataStructs
    ds = setting.split("/")[0]
    df = prepare(ds)["df"].set_index("mol_id")
    raw = pd.read_csv(REPO / "data/raw/B3DB_classification.tsv", sep="\t")
    ik = [(lambda r: r.inchikey if r is not None and r.ok else None)(standardize_smiles(x) if isinstance(x, str) else None)
          for x in raw.SMILES]
    raw = raw.assign(_ik=ik)
    out = {}
    for tag, mid in (("mol", mol_id), ("nn", nn_id)):
        r = raw[raw._ik == df.loc[mid, "inchikey"]]
        assert len(r) == 1, (mid, len(r))
        r = r.iloc[0]
        out.update({f"{tag}.no": int(r["NO."]), f"{tag}.cid": int(r.CID), f"{tag}.rawname": r.compound_name,
                    f"{tag}.rawlabel": r["BBB+/BBB-"], f"{tag}.group": str(r.group),
                    f"{tag}.nref": len([x for x in str(r.reference).split("|") if x])})
    assert out["mol.rawlabel"] == "BBB-"
    bb = prepare("bbbp")["df"]
    out["mol.in_bbbp"] = int((bb.conn_key == df.loc[mol_id, "conn_key"]).sum())
    z = np.load(REPO / f"results/raw/{setting}/{model}_ce/seed{seed}.npz", allow_pickle=True)
    tr = df.loc[list(df.reset_index().iloc[z["idx_train"]].mol_id)]
    fq = ecfp4_bitvects([df.loc[mol_id, "smiles"]])[0]
    sims = np.sort(np.array(DataStructs.BulkTanimotoSimilarity(fq, ecfp4_bitvects(list(tr.smiles)))))[::-1]
    out["nn.sim2"] = float(sims[1])
    r = thr[(thr.seed == seed) & (thr.tert == 0)].iloc[0]; out["mol.ncal"] = int(r.n_cal_stratum)
    a = D[D.mol_id == mol_id]
    out["mol.smax.min"], out["mol.smax.max"] = float(a.s_max.min()), float(a.s_max.max())
    return out


def molecule_name(setting, mol_id):
    W = pd.read_csv(T / "persistent_failure_molecules_ce.csv", low_memory=False)
    n = W[(W.setting == setting) & (W.mol_id == mol_id)].name.dropna()
    nm = str(n.iloc[0]) if len(n) else mol_id
    return nm.replace(" hydrochloride", "")


# =============================================================================== 2. draw.io writer
class DrawIO:
    def __init__(self):
        self.cells = []; self.n = 2; self.parent = "1"; self.ox = 0.0; self.oy = 0.0

    def _id(self):
        self.n += 1; return f"c{self.n}"

    def group(self, name, x, y, w, h):
        """Open a named draw.io group (nestable); children use absolute canvas coordinates in this writer."""
        gid = self._id()
        self.cells.append(f'<mxCell id="{gid}" value="" style="group;" vertex="1" connectable="0" parent="{self.parent}">'
                          f'<mxGeometry x="{x - self.ox:.2f}" y="{y - self.oy:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
                          f'</mxCell><!-- {html.escape(name)} -->')
        self.stack = getattr(self, "stack", []) + [(self.parent, self.ox, self.oy)]
        self.parent, self.ox, self.oy = gid, x, y
        return gid

    def end_group(self):
        self.parent, self.ox, self.oy = self.stack.pop()

    def vertex(self, x, y, w, h, style, value=""):
        cid = self._id()
        self.cells.append(f'<mxCell id="{cid}" value="{html.escape(value, quote=True)}" style="{style}" vertex="1" '
                          f'parent="{self.parent}"><mxGeometry x="{x - self.ox:.2f}" y="{y - self.oy:.2f}" '
                          f'width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>')
        return cid

    def text(self, x, y, w, h, value, size=15, color="#1d2b3a", align="left", valign="middle", bold=False, italic=False,
             extra=""):
        fs = (1 if bold else 0) + (2 if italic else 0)
        st = (f"text;html=1;whiteSpace=wrap;overflow=visible;fontFamily=Arial;fontSize={size};fontColor={color};"
              f"align={align};verticalAlign={valign};spacing=0;fontStyle={fs};{extra}")
        return self.vertex(x, y, w, h, st, value)

    def rect(self, x, y, w, h, fill, stroke="none", rounded=0, sw=1, extra=""):
        st = (f"rounded={1 if rounded else 0};arcSize={rounded};whiteSpace=wrap;html=1;fillColor={fill};"
              f"strokeColor={stroke};strokeWidth={sw};{extra}")
        return self.vertex(x, y, w, h, st)

    def line(self, pts, color="#333333", width=1, dash=None, arrow=False, extra=""):
        cid = self._id()
        st = (f"endArrow={'block' if arrow else 'none'};endFill=1;endSize=7;html=1;rounded=0;curved=0;"
              f"strokeColor={color};strokeWidth={width};" + (f"dashed=1;dashPattern={dash};" if dash else "") + extra)
        p = [(x - self.ox, y - self.oy) for x, y in pts]
        mids = "".join(f'<mxPoint x="{x:.2f}" y="{y:.2f}"/>' for x, y in p[1:-1])
        self.cells.append(f'<mxCell id="{cid}" style="{st}" edge="1" parent="{self.parent}"><mxGeometry relative="1" as="geometry">'
                          f'<mxPoint x="{p[0][0]:.2f}" y="{p[0][1]:.2f}" as="sourcePoint"/>'
                          f'<mxPoint x="{p[-1][0]:.2f}" y="{p[-1][1]:.2f}" as="targetPoint"/>'
                          + (f'<Array as="points">{mids}</Array>' if mids else "") + '</mxGeometry></mxCell>')
        return cid

    def marker(self, cx, cy, shape, fill, stroke, size=10, sw=1.5):
        shp = {"circle": "ellipse;", "square": "", "diamond": "rhombus;", "triangle": "triangle;direction=north;",
               "ring": "ellipse;"}[shape]
        s = size * (1.25 if shape in ("diamond", "triangle") else 1)
        return self.vertex(cx - s / 2, cy - s / 2, s, s, f"{shp}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                           f"strokeWidth={sw};aspect=fixed;")

    def image(self, x, y, w, h, svg):
        b = base64.b64encode(svg.encode()).decode()
        return self.vertex(x, y, w, h, f"shape=image;html=1;imageAspect=1;aspect=fixed;image=data:image/svg+xml,{b};")

    def write(self, path, w, h):
        body = "".join(self.cells)
        xml = (f'<mxfile host="make_fig1_overview.py"><diagram name="Figure 1" id="fig1"><mxGraphModel dx="{w}" dy="{h}" grid="0" '
               f'gridSize="10" guides="1" tooltips="1" connect="0" arrows="1" fold="1" page="0" pageScale="1" '
               f'pageWidth="{w}" pageHeight="{h}" math="0" shadow="0" background="#FFFFFF"><root><mxCell id="0"/>'
               f'<mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>\n')
        path.write_text(xml)


ELEM = {"O": "#d32f2f", "N": "#1f4fbf", "Cl": "#2e7d32", "F": "#4f8a10", "S": "#a07000", "Br": "#8b3a1a"}


def draw_molecule(d, smi, x, y, w, h, fs=10.5, max_bond=14.0):
    """Native draw.io depiction: every bond is an editable line, every heteroatom an editable text label.
    2D coordinates from RDKit (CoordGen); Kekule form; ring double bonds drawn inside the ring."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(False)
    m = Chem.MolFromSmiles(smi); rdDepictor.Compute2DCoords(m); Chem.Kekulize(m, clearAromaticFlags=True)
    c = m.GetConformer(); P = np.array([[c.GetAtomPosition(i).x, -c.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
    bl = np.median([np.linalg.norm(P[b.GetBeginAtomIdx()] - P[b.GetEndAtomIdx()]) for b in m.GetBonds()])
    span = P.max(0) - P.min(0) + bl * 0.8
    sc = min(w / span[0], h / span[1], max_bond / bl)
    Q = (P - (P.max(0) + P.min(0)) / 2) * sc + np.array([x + w / 2, y + h / 2])
    L = bl * sc
    lab = {}
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        if sym != "C" or a.GetFormalCharge() != 0:
            nh = a.GetTotalNumHs()
            lab[a.GetIdx()] = sym + ("H" if nh == 1 else f"H{nh}" if nh > 1 else "")
    ri = m.GetRingInfo()
    for bnd in m.GetBonds():
        i, j = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx(); p, q = Q[i].copy(), Q[j].copy()
        u = (q - p) / np.linalg.norm(q - p)
        if i in lab: p = p + u * fs * 0.5
        if j in lab: q = q - u * fs * 0.5
        d.line([tuple(p), tuple(q)], "#222222", 1.1)
        if bnd.GetBondTypeAsDouble() == 2:
            nrm = np.array([-u[1], u[0]])
            rings = [r for r in ri.BondRings() if bnd.GetIdx() in r]
            if rings:
                ats = {m.GetBondWithIdx(k).GetBeginAtomIdx() for k in rings[0]} | {m.GetBondWithIdx(k).GetEndAtomIdx() for k in rings[0]}
                cen = Q[list(ats)].mean(0)
                if np.dot(cen - (p + q) / 2, nrm) < 0: nrm = -nrm
                off = nrm * L * 0.2; sh = u * L * 0.16
                d.line([tuple(p + off + sh), tuple(q + off - sh)], "#222222", 1.1)
            else:
                off = nrm * L * 0.18
                d.line([tuple(p + off), tuple(q + off)], "#222222", 1.1)
    for i, t in lab.items():
        wt = fs * 0.62 * len(t) + 2
        d.text(Q[i][0] - wt / 2, Q[i][1] - fs * 0.6, wt, fs * 1.2, t, fs, ELEM.get(m.GetAtomWithIdx(i).GetSymbol(), "#222222"),
               "center", bold=True)


# =============================================================================== 3. figure
INK = "#1d2b3a"; SUB = "#5b6573"
BLUE, ORANGE = "#1f6fd1", "#f07f13"                     # class-conditional, class x support
LOW, MID, HIGH = "#c2185b", "#8d8d8d", "#00897b"        # support tertiles
PANEL = "#fff8e8"; EXCL = "#fde3e3"


def ax_x(v, lo, hi, x0, x1):
    return x0 + (v - lo) / (hi - lo) * (x1 - x0)


def draw(V, ecdf, thr, mol):
    """Canvas 860 px wide = 6.5 in printed (0.544 pt/px): 15 px = 8.2 pt, 16 px = 8.7 pt."""
    d = DrawIO(); W = 860; PH = 296
    fmt = lambda v: f"{v:.3f}"
    FS, HS = 15, 16

    def heading(x0, y0, w, letter, title, sub):
        d.text(x0 + 10, y0 + 7, w - 20, 20, f"<b>{letter}&nbsp;&nbsp;{title}</b>", HS)
        d.text(x0 + 10, y0 + 28, w - 20, 18, sub, FS, SUB, italic=True)

    def xaxis(px0, px1, ya, lo, hi, ticks, title, tfmt="{:.2f}"):
        d.line([(px0, ya), (px1, ya)], "#333333", 1)
        for v in ticks:
            xv = ax_x(v, lo, hi, px0, px1)
            d.line([(xv, ya), (xv, ya + 4)], "#333333", 1)
            lab = tfmt.format(v) if isinstance(tfmt, str) else tfmt(v)
            d.text(xv - 24, ya + 5, 48, 18, lab, FS, INK, "center")
        d.text(px0 - 30, ya + 24, px1 - px0 + 60, 18, title, FS, INK, "center")

    def ci(xa, xb, yy, col, w=1.6, cap=4):
        d.line([(xa, yy), (xb, yy)], col, w)
        d.line([(xa, yy - cap), (xa, yy + cap)], col, w); d.line([(xb, yy - cap), (xb, yy + cap)], col, w)

    # ------------------------------------------------------------------------ panel A
    x0, y0, w = 0, 0, 245
    d.group("Panel A: class-conditional coverage by support", x0, y0, w, PH)
    d.rect(x0, y0, w, PH, PANEL, rounded=4)
    heading(x0, y0, w, "A", "Average hides a gap", V["sel.short"])
    lo_, hi_ = 0.80, 1.00; px0, px1 = x0 + 88, x0 + w - 60
    rows = [("All BBB−", "cls", INK, "diamond"), ("Low", "low", LOW, "triangle"),
            ("Mid", "mid", MID, "square"), ("High", "high", HIGH, "circle")]
    d.text(x0 + 8, y0 + 120, 24, 76, "support", FS, SUB, "center", extra="horizontal=0;")
    ys = [y0 + 72, y0 + 106, y0 + 140, y0 + 174]
    tx = ax_x(0.90, lo_, hi_, px0, px1)
    d.line([(tx, y0 + 56), (tx, y0 + 192)], "#555555", 1.2, dash="5 4")
    for (lab, k, col, shp), yy in zip(rows, ys):
        d.text(x0 + (10 if k == "cls" else 36), yy - 10, 80, 20, lab, FS, col, bold=(k != "mid"))
        a, b = V[f"A.{k}.lo"], V[f"A.{k}.hi"]
        ci(ax_x(a, lo_, hi_, px0, px1), ax_x(b, lo_, hi_, px0, px1), yy, col, 1.6, 4)
        d.marker(ax_x(V[f"A.{k}"], lo_, hi_, px0, px1), yy, shp, col if k != "mid" else "#ffffff", col, 10, 1.6)
        d.text(px1 + 10, yy - 10, 50, 20, fmt(V[f"A.{k}"]), FS, col, bold=(k in ("low", "high")))
    xaxis(px0, px1, y0 + 200, lo_, hi_, (0.80, 0.90, 1.00), "BBB− coverage")
    d.text(x0 + 10, y0 + 250, w - 20, 36, "dashed: 0.90 target<br>bars: 95% CI over 20 splits", FS, SUB)
    d.end_group()

    # ------------------------------------------------------------------------ panel B
    x0, y0, w = 257, 0, 346
    d.group("Panel B: threshold-relative score ECDFs", x0, y0, w, PH)
    d.rect(x0, y0, w, PH, PANEL, rounded=4)
    heading(x0, y0, w, "B", "One threshold, unequal scores", "same setting; 20 splits pooled (descriptive)")
    xlo, xhi, ylo, yhi = -1.0, 0.35, 0.0, 1.0
    px0, px1, py0, py1 = x0 + 64, x0 + w - 46, y0 + 120, y0 + 226
    X = lambda v: ax_x(v, xlo, xhi, px0, px1)
    Y = lambda v: py1 - (v - ylo) / (yhi - ylo) * (py1 - py0)
    d.text(x0 + 10, y0 + 47, w - 20, 36,
           "class × support thresholds relative to the class-conditional threshold, one tick per split:", FS, INK)
    for b, col, yy, lab in ((2, HIGH, y0 + 92, "high"), (0, LOW, y0 + 106, "low")):
        r = thr[thr.tert == b]
        for v in (r.q_stratum - r.q_class).values:
            d.line([(X(v), yy - 5), (X(v), yy + 5)], col, 1.5)
        d.text(x0 + 8, yy - 9, 38, 18, lab, FS, col, bold=True)
    d.rect(X(0), py0, X(xhi) - X(0), py1 - py0, EXCL)
    d.text(X(0) + 4, py1 - 40, X(xhi) - X(0) - 4, 36, "BBB−<br>excluded", FS, "#b0283a")
    for k, col, dash, wd in (("high", HIGH, None, 2.4), ("mid", MID, "1 3", 2.0), ("low", LOW, "7 3", 2.4)):
        xs, ys_ = ecdf[k]
        m = (xs >= xlo) & (xs <= xhi)
        d.line([(X(a), Y(b_)) for a, b_ in zip(xs[m], ys_[m])], col, wd, dash=dash)
    d.line([(X(0), y0 + 86), (X(0), py1)], "#111111", 1.8)
    # legend, upper left of the plot (empty there because every curve is below 0.4 at x < -0.6)
    for i, (lab, col, dash) in enumerate((("low", LOW, "7 3"), ("mid", MID, "1 3"), ("high", HIGH, None))):
        yy = py0 + 8 + i * 17
        d.line([(px0 + 6, yy), (px0 + 28, yy)], col, 2.4, dash=dash)
        d.text(px0 + 33, yy - 9, 60, 18, lab, FS, col if lab != "mid" else "#6f6f6f", bold=(lab != "mid"))
    # share above the class threshold (whole percent; pooled and split-mean values agree at this rounding)
    for k, col in (("high", HIGH), ("low", LOW)):
        f0 = V[f"B.F0.{k}"]
        d.line([(X(0), Y(f0)), (px1 + 2, Y(f0))], col, 1, dash="2 2")
        d.text(px1 + 4, Y(f0) - 9, 42, 18, f"<b>{V[f'B.exclint.{k}']}%</b>", FS, col)
    d.line([(px0, py0), (px0, py1)], "#333333", 1)
    for v in (0.0, 0.5, 1.0):
        d.line([(px0 - 4, Y(v)), (px0, Y(v))], "#333333", 1)
        d.text(px0 - 30, Y(v) - 9, 24, 18, f"{v:g}", FS, INK, "right")
    d.text(x0 + 2, py0 - 8, 34, py1 - py0 + 16, "cumulative<br>fraction", FS, INK, "center", extra="horizontal=0;")
    xaxis(px0, px1, py1, xlo, xhi, (-1.0, -0.5, 0.0),
          "score − class threshold of its split", lambda v: "0" if v == 0 else f"{v:g}".replace("-", "−"))
    d.end_group()

    # ------------------------------------------------------------------------ panel C
    x0, y0, w = 615, 0, 245
    d.group("Panel C: class-conditional vs class x support", x0, y0, w, PH)
    d.rect(x0, y0, w, PH, PANEL, rounded=4)
    heading(x0, y0, w, "C", "Coverage is redistributed", "same setting; mean of 20 splits")
    d.marker(x0 + 16, y0 + 60, "circle", BLUE, BLUE, 10); d.text(x0 + 26, y0 + 51, 200, 18, "class-conditional", FS, BLUE, bold=True)
    d.marker(x0 + 16, y0 + 79, "diamond", ORANGE, ORANGE, 10); d.text(x0 + 26, y0 + 70, 200, 18, "class × support", FS, ORANGE, bold=True)
    lo_, hi_ = 0.80, 1.00; px0, px1 = x0 + 84, x0 + w - 22
    tx = ax_x(0.90, lo_, hi_, px0, px1)
    d.line([(tx, y0 + 98), (tx, y0 + 200)], "#555555", 1.2, dash="5 4")
    d.text(x0 + 8, y0 + 110, 24, 76, "support", FS, SUB, "center", extra="horizontal=0;")
    for lab, k, col, yy in (("Low", "low", LOW, y0 + 124), ("High", "high", HIGH, y0 + 172)):
        d.text(x0 + 36, yy - 10, 60, 20, lab, FS, col, bold=True)
        for meth, mcol, off in (("mond", BLUE, -7), ("cxs", ORANGE, 7)):
            ci(ax_x(V[f"C.{k}.{meth}.lo"], lo_, hi_, px0, px1), ax_x(V[f"C.{k}.{meth}.hi"], lo_, hi_, px0, px1), yy + off, mcol, 1.4, 3)
        xa = ax_x(V[f"C.{k}.mond"], lo_, hi_, px0, px1); xb_ = ax_x(V[f"C.{k}.cxs"], lo_, hi_, px0, px1)
        d.line([(xa, yy), (xb_ + (-8 if xb_ > xa else 8), yy)], "#444444", 1.4, arrow=True)
        d.marker(xa, yy, "circle", BLUE, BLUE, 10); d.marker(xb_, yy, "diamond", ORANGE, ORANGE, 10)
        up, dn = yy - 29, yy + 10
        d.text(xa - 24, dn if k == "low" else up, 48, 18, fmt(V[f"C.{k}.mond"]), FS, BLUE, "center", extra=f"labelBackgroundColor={PANEL};")
        d.text(xb_ - 24, up if k == "low" else dn, 48, 18, fmt(V[f"C.{k}.cxs"]), FS, ORANGE, "center", extra=f"labelBackgroundColor={PANEL};")
    xaxis(px0, px1, y0 + 206, lo_, hi_, (0.80, 0.90, 1.00), "BBB− coverage")
    d.line([(x0 + 10, y0 + 252), (x0 + w - 10, y0 + 252)], "#e6cf9c", 1)
    d.text(x0 + 10, y0 + 255, w - 20, 18,
           f"mean set size <font color='{BLUE}'>{V['C.size.mond']:.3f}</font> → <font color='{ORANGE}'>{V['C.size.cxs']:.3f}</font>", FS, INK)
    d.text(x0 + 10, y0 + 273, w - 20, 18, "bars: 95% CI over 20 splits", FS, SUB)
    d.end_group()

    # ------------------------------------------------------------------------ study-design strip
    y0 = PH + 10; bh = 118
    d.group("Study design strip", 0, y0, W, bh)
    widths = [88, 104, 104, 338, 112, 90]; gap = (W - sum(widths)) / 5
    fills = [("#e3f2fd", "#90caf9"), ("#fff3c4", "#f2cd5c"), ("#e6f6ec", "#8fd3a8"), ("#fce4ec", "#f2a7c0"),
             ("#ffe8d1", "#f7b77a"), ("#ede7f6", "#bba6e3")]
    titles = ["Datasets", "Splits", "Predictors", "Chemical support: max Tanimoto to TRAIN", "Calibrate", "Evaluate"]
    bodies = ["BBBP<br>B3DB<br>curated", "TRAIN · VAL<br>CAL · TEST<br>×20", "RF · GIN<br>Chemprop<br>fit on TRAIN",
              "", "on CAL:<br>class-cond.<br>class × support", "on TEST:<br>coverage<br>set size"]
    xs_ = np.cumsum([0] + [wd + gap for wd in widths[:-1]])
    for i, (xx, wd) in enumerate(zip(xs_, widths)):
        f_, s_ = fills[i]
        d.rect(xx, y0, wd, bh, f_, s_, rounded=8, sw=1.2)
        d.text(xx + 3, y0 + 5, wd - 6, 18, f"<b>{titles[i]}</b>", FS, INK, "center")
        if bodies[i]:
            d.text(xx + 3, y0 + 26, wd - 6, bh - 28, bodies[i], FS, INK, "center", "top")
        if i < 5:
            d.line([(xx + wd + 1, y0 + bh / 2), (xx + wd + gap - 1, y0 + bh / 2)], "#555555", 1.4, arrow=True)
    xx = xs_[3]; wd = widths[3]
    d.group("Chemical support example (native bonds and atom labels)", xx, y0, wd, bh)
    draw_molecule(d, mol["smiles"], xx + 6, y0 + 30, 182, 60, fs=12.5, max_bond=16)
    draw_molecule(d, mol["nn_smiles"], xx + wd - 108, y0 + 30, 92, 60, fs=12.5, max_bond=16)
    d.line([(xx + 196, y0 + 62), (xx + wd - 122, y0 + 62)], LOW, 1.6, dash="3 2")
    d.text(xx + 192, y0 + 41, wd - 310, 18, f"<b>{mol['s_max']:.2f}</b>", FS, LOW, "center")
    d.text(xx + 4, y0 + 96, 186, 18, f"<i>{mol['name']}</i> (TEST)", FS, INK, "center")
    d.text(xx + wd - 126, y0 + 96, 122, 18, "nearest TRAIN", FS, INK, "center")
    d.end_group()
    return d, W, y0 + bh


# =============================================================================== 4. main
def find_drawio():
    for c in (os.environ.get("DRAWIO"), shutil.which("drawio"), shutil.which("draw.io"),
              "/Applications/draw.io.app/Contents/MacOS/draw.io"):
        if c and Path(c).exists():
            return c
    return None


def main():
    setting, model, G = select_cell()
    D, thr = appearances(setting, model)
    D.to_csv(T / "fig1_scores.csv.gz", index=False)
    S = pd.read_csv(T / "confirmatory_summary.csv", low_memory=False)
    c = S[(S.setting == setting) & (S.model_name == model) & (S.score == "lac") & (S.alpha == ALPHA) & (S.loss == "ce")]
    M = c[c.method == "mondrian"].iloc[0]; X = c[c.method == "mondrian_x"].iloc[0]
    ds, reg = setting.split("/")
    V = {"sel.label": f"{DSN[ds]}, {reg} split, {NAMES[model]}, 20 splits", "sel.short": f"{DSN[ds]}, {reg} split, {NAMES[model]}", "sel.setting": setting, "sel.model": model,
         "sel.gap": float(G[G.selected].gap.iloc[0]), "sel.gap12.low": float(G.gap.min()), "sel.gap12.high": float(G.gap.max()),
         "sel.all12.setting": G[G.lower_median_all12].setting.iloc[0], "sel.all12.model": G[G.lower_median_all12].model_name.iloc[0],
         "sel.all12.gap": float(G[G.lower_median_all12].gap.iloc[0])}
    a12 = G[G.lower_median_all12].iloc[0]
    V["sel.all12.label"] = f"{DSN[a12.setting.split('/')[0]]} {a12.setting.split('/')[1]} split, {NAMES[a12.model_name]}"
    for k, col in (("cls", "cov_minority"), ("low", "cov_min_bin0"), ("mid", "cov_min_bin1"), ("high", "cov_min_bin2")):
        V[f"A.{k}"] = float(M[col]); V[f"A.{k}.lo"] = float(M[col + "_lo"]); V[f"A.{k}.hi"] = float(M[col + "_hi"])
    V["A.gap"] = V["A.high"] - V["A.low"]
    for k, col in (("low", "cov_min_bin0"), ("high", "cov_min_bin2")):
        for meth, r in (("mond", M), ("cxs", X)):
            V[f"C.{k}.{meth}"] = float(r[col]); V[f"C.{k}.{meth}.lo"] = float(r[col + "_lo"]); V[f"C.{k}.{meth}.hi"] = float(r[col + "_hi"])
    V["C.size.mond"], V["C.size.cxs"] = float(M.set_size), float(X.set_size)
    V["C.full.mond"], V["C.full.cxs"] = float(M.full_rate), float(X.full_rate)
    ecdf = {}
    grid = np.round(np.arange(-1.0, 0.35001, 0.0125), 5)
    for b, k in ((0, "low"), (1, "mid"), (2, "high")):
        e = np.sort(D[D.tert == b].excess.values)
        ecdf[k] = (grid, np.searchsorted(e, grid, side="right") / len(e))
        V[f"B.F0.{k}"] = float((e <= 0).mean()); V[f"B.n.{k}"] = int(len(e))
        V[f"B.nmol.{k}"] = int(D[D.tert == b].mol_id.nunique())
        r = thr[thr.tert == b]; sh = (r.q_stratum - r.q_class).values
        V[f"B.shift.{k}.min"], V[f"B.shift.{k}.med"], V[f"B.shift.{k}.max"] = float(sh.min()), float(np.median(sh)), float(sh.max())
        V[f"B.shift.{k}.npos"] = int((sh > 0).sum()); V[f"B.shift.{k}.nneg"] = int((sh < 0).sum())
        V[f"B.ncal.{k}.min"] = int(r.n_cal_stratum.min()); V[f"B.ncal.{k}.max"] = int(r.n_cal_stratum.max())
    for k in ("low", "high"):
        pooled = round(100 * (1 - V[f"B.F0.{k}"])); split_mean = round(100 * (1 - V[f"A.{k}"]))
        assert pooled == split_mean, (k, pooled, split_mean)
        V[f"B.exclint.{k}"] = int(pooled)
    resc = ~D.in_class & D.in_cxs; lost = D.in_class & ~D.in_cxs
    V["B.rescued"] = int(resc.sum()); V["B.lost"] = int(lost.sum())
    for b, k in ((0, "low"), (1, "mid"), (2, "high")):
        V[f"B.rescued.{k}"] = int((resc & (D.tert == b)).sum()); V[f"B.lost.{k}"] = int((lost & (D.tert == b)).sum())
    mrow, seed = pick_molecule(D)
    tr = D[(D.mol_id == mrow.mol_id) & (D.seed == seed)].iloc[0]
    smi, nn_smi, nn_id, nn_sim = neighbour(setting, model, seed, mrow.mol_id)
    assert abs(nn_sim - tr.s_max) < 1e-9, (nn_sim, tr.s_max)
    mol = {"mol_id": mrow.mol_id, "name": molecule_name(setting, mrow.mol_id), "smiles": smi, "nn_smiles": nn_smi,
           "nn_id": nn_id, "s_max": float(tr.s_max), "excess": float(tr.excess), "seed": seed}
    V.update({"mol.id": mrow.mol_id, "mol.name": mol["name"], "mol.n": int(mrow.n), "mol.rescued": int(mrow.rescued),
              "mol.seed": seed, "mol.smax": float(tr.s_max), "mol.score": float(tr.score), "mol.qclass": float(tr.q_class),
              "mol.qstratum": float(tr.q_stratum), "mol.nn_id": nn_id, "mol.excess": float(tr.excess)})
    # provenance of the example pair: raw B3DB records matched by standardized InChIKey
    V.update(provenance(setting, model, seed, mrow.mol_id, nn_id, thr, D))
    for k in ("low", "mid", "high"):
        V[f"A.excl.{k}"] = 100 * (1 - V[f"A.{k}"])
    pd.DataFrame([{"key": k, "value": v} for k, v in V.items()]).to_csv(T / "fig1_values.csv", index=False)

    d, W, H = draw(V, ecdf, thr, mol)
    out = F / "fig1_chemical_support_overview.drawio"
    d.write(out, W, H)
    print(f"selected: {setting} {model} (gap {V['sel.gap']:.3f}); traced molecule {mol['name']} ({mrow.mol_id}), split {seed}")
    print(f"wrote {out.relative_to(REPO)}, tables/fig1_values.csv")
    exe = find_drawio()
    if exe:
        pdf = out.with_suffix(".pdf")
        cmd = [exe, "-x", "-f", "pdf", "--crop", "-o", str(pdf), str(out)]
        if exe.endswith("drawio") and not exe.startswith("/Applications"):
            cmd += ["--no-sandbox", "--disable-gpu"]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"wrote {pdf.relative_to(REPO)}")
    else:
        print("draw.io not found: open the .drawio file in diagrams.net and export as PDF (crop) to figures/")


if __name__ == "__main__":
    main()
