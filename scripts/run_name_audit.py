"""B3DB-internal label conflicts by compound name (EXPLORATORY, added 1 Nov 2026).

Curation found BBB- records whose compound also appears, under the same name, as a BBB+ record with a
different structure drawing (phenylbutazone, cyclobenzaprine).  Such records escape InChIKey-based
de-duplication.  This audit asks how common that is and whether persistent failures are enriched for it.
A name conflict: the normalized compound name (lower case, parentheses and common salt words removed)
of an eligible BBB- molecule occurs among BBB+ records of the raw B3DB classification file.
Names are an imperfect key (synonyms, stereoisomers), so this is a screen, not a verdict on any label.

usage: python scripts/run_name_audit.py
Output: tables/b3db_name_conflicts.csv
"""
import re
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parents[1]
SALT = (r"\b(hydrochloride|dihydrochloride|hydrobromide|maleate|mesylate|citrate|sulfate|sulphate|tartrate|fumarate|"
        r"sodium|potassium|hcl|acetate|phosphate|bromide|chloride|succinate|besylate|tosylate|monohydrate|hydrate)\b")


def norm(n):
    if not isinstance(n, str):
        return ""                                  # missing names never match
    n = n.lower(); n = re.sub(r"\(.*?\)", "", n); n = re.sub(SALT, "", n)
    return re.sub(r"[^a-z0-9]", "", n)


def main():
    raw = pd.read_csv(REPO / "data/raw/B3DB_classification.tsv", sep="\t")
    pos = {norm(n) for n, l in zip(raw.compound_name, raw["BBB+/BBB-"]) if l == "BBB+"} - {""}
    W = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False)
    W = W[W.setting.isin(["b3db/random", "b3db/scaffold"]) & W.eligible.astype(bool)].copy()
    W["conflict"] = W["name"].map(lambda n: norm(n) in pos and norm(n) != "")
    rows = []
    for st, g in W.groupby("setting", sort=True):
        for defn, col in (("strict", "cross_arch_mondrian"), ("majority", "majority_all3_mondrian")):
            f = g[col].astype(bool); c = g.conflict
            tab = [[int((f & c).sum()), int((f & ~c).sum())], [int((~f & c).sum()), int((~f & ~c).sum())]]
            rows.append({"setting": st, "definition": defn, "n_eligible": len(g), "n_conflict": int(c.sum()),
                         "n_fail": int(f.sum()), "fail_conflict": tab[0][0],
                         "rate_fail": tab[0][0] / max(int(f.sum()), 1), "rate_rest": tab[1][0] / max(int((~f).sum()), 1),
                         "fisher_p": fisher_exact(tab)[1],
                         "conflicting_failures": "; ".join(str(x) for x in g[f & c]["name"].fillna(""))})
    R = pd.DataFrame(rows); R.to_csv(REPO / "tables/b3db_name_conflicts.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
