"""Final gate before submission: every number the paper prints must be a verified key.

Reads paper/latex/main.tex and supplement.tex, collects every \\V{key}, and checks each against
paper/latex/generated/numbers_verification.csv (written by scripts/verify_paper_numbers.py).
A cited key passes only if its status is OK.  Run after verify_paper_numbers.py ends in PASS.

usage: python scripts/check_cited_keys.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
L = REPO / "paper/latex"
V = pd.read_csv(L / "generated/numbers_verification.csv", dtype={"paper": str})
ok = set(V[V.status == "OK"].key)
known = set(V.key)
bad = 0
for tex in ("main.tex", "supplement.tex"):
    if not (L / tex).exists():
        print(f"{tex}: MISSING (save it into paper/latex/ first)"); bad += 1; continue
    keys = sorted(set(re.findall(r"\\V\{([^}]+)\}", (L / tex).read_text())))
    missing = [k for k in keys if k not in known]
    unver = [k for k in keys if k in known and k not in ok]
    print(f"{tex}: {len(keys)} cited keys, {len(keys) - len(missing) - len(unver)} verified OK")
    for k in missing:
        print(f"   NOT GENERATED  {k}")
    for k in unver:
        r = V[V.key == k].iloc[0]
        print(f"   NOT VERIFIED   {k}  ({r.level}, {r.status})")
    bad += len(missing) + len(unver)
print("PASS: every cited number is verified" if bad == 0 else f"FAIL: {bad} cited key(s) not verified")
sys.exit(1 if bad else 0)