"""A2 (complete): BBBP/B3DB label-conflict audit, at dataset level and against the persistent
failures.  Run AFTER scripts/run_persistent_failures.py.

Part 1, dataset level (curated datasets as used in every experiment): molecules shared by
connectivity key (InChIKey first block), label agreement, and the direction of disagreements.
Part 2, failure level: among eligible BBB- test molecules that also occur in the other dataset,
is a label disagreement associated with being a persistent failure?  Fisher exact test per
setting, using the strict cross-architecture definition and the descriptive majority rule.
Disagreements are dataset conflicts, not biology; this table decides which failures may be
interpreted chemically.  Transfer molecules are absent from BBBP by construction and are skipped.

usage: python scripts/run_label_audit.py
Outputs: tables/label_audit_overlap.csv, tables/label_audit_failures.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from bbbcp.pipeline import prepare

REPO = Path(__file__).resolve().parents[1]


def main():
    a, b = prepare("bbbp")["df"], prepare("b3db")["df"]
    m = a[["conn_key", "y", "mol_id"]].merge(b[["conn_key", "y", "mol_id"]], on="conn_key", suffixes=("_bbbp", "_b3db"))
    ov = pd.DataFrame([{
        "n_bbbp": len(a), "n_b3db": len(b), "n_shared_connectivity": len(m),
        "frac_bbbp_in_b3db": len(m) / len(a), "label_agreement": float((m.y_bbbp == m.y_b3db).mean()),
        "n_disagree": int((m.y_bbbp != m.y_b3db).sum()),
        "n_bbbp_neg_b3db_pos": int(((m.y_bbbp == 0) & (m.y_b3db == 1)).sum()),
        "n_bbbp_pos_b3db_neg": int(((m.y_bbbp == 1) & (m.y_b3db == 0)).sum()),
        "frac_bbbp_neg_disagreeing": float(((m.y_bbbp == 0) & (m.y_b3db == 1)).sum() / max((m.y_bbbp == 0).sum(), 1))}])
    ov.to_csv(REPO / "tables/label_audit_overlap.csv", index=False)

    W = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False)
    W = W[W.eligible.astype(bool) & W.in_other_dataset.astype(bool)]
    rows = []
    for st, g in W.groupby("setting", sort=False):
        dis = g.label_disagrees_other_dataset.astype(bool)
        for defn, col in (("strict", "cross_arch_mondrian"), ("majority", "majority_all3_mondrian")):
            f = g[col].astype(bool)
            tab = [[int((f & dis).sum()), int((f & ~dis).sum())], [int((~f & dis).sum()), int((~f & ~dis).sum())]]
            rows.append({"setting": st, "definition": defn, "n_in_other": len(g), "n_disagree": int(dis.sum()),
                         "failures_disagree": tab[0][0], "failures_agree": tab[0][1],
                         "failure_rate_if_disagree": tab[0][0] / max(dis.sum(), 1),
                         "failure_rate_if_agree": tab[0][1] / max((~dis).sum(), 1),
                         "fisher_odds_ratio": fisher_exact(tab)[0], "fisher_p": fisher_exact(tab)[1]})
    F = pd.DataFrame(rows); F.to_csv(REPO / "tables/label_audit_failures.csv", index=False)
    pd.set_option("display.width", 200)
    print(ov.round(4).T.to_string()); print(); print(F.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
