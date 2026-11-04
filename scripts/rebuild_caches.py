"""Rebuild every derived cache in the CURRENT environment (reviewer item: do not rely on
shipped pickles).  Safe to run any time; it only deletes files it can regenerate.

usage: python scripts/rebuild_caches.py [--datasets bbbp b3db ...] [--keep-butina]
"""
import argparse

from bbbcp.pipeline import PROC, butina_labels, features, prepare

ALL = ["bbbp", "b3db", "bace", "clintox", "tox21_sr_are", "bbbp_raw"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=ALL)
    ap.add_argument("--keep-butina", action="store_true")
    a = ap.parse_args()
    for f in PROC.glob("*.pkl"):
        f.unlink(); print("removed", f.name)
    if not a.keep_butina:
        for f in PROC.glob("*_butina*.npy"):
            f.unlink(); print("removed", f.name)
    for name in a.datasets:
        try:
            d = prepare(name); features(name)
            print(f"rebuilt {name}: {len(d['df'])} molecules, minority label {d['audit']['minority_label']}")
        except FileNotFoundError as e:
            print(f"skipped {name}: {e} (run scripts/download_data.py --extra)")
