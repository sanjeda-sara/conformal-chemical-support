"""Predictive performance of the three backbones (context for the coverage results; EXPLORATORY).

For every split (seeds 100-119), from the cached TEST predictions: AUROC, average precision for the
minority class (BBB-), balanced accuracy and minority recall of the argmax decision.  Reported as the
mean and SD over splits per setting x backbone, for the four in-domain settings and BBBP->B3DB transfer.

usage: python scripts/run_classifier_metrics.py
Output: tables/classifier_metrics.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold", "bbbp__to__b3db/scaffold"]
MODELS = ["rf", "gin", "chemprop"]


def main():
    rows = []
    for st in SETTINGS:
        for md in MODELS:
            for seed in range(100, 120):
                z = np.load(REPO / "results/raw" / st / f"{md}_ce" / f"seed{seed}.npz", allow_pickle=True)
                y = z["y_test"].astype(int); mino = int(z["minority"]); p = z["p_test"].astype(float)
                ym = (y == mino).astype(int); pm = p[:, mino]; yhat = p.argmax(1)
                rows.append({"setting": st.replace("bbbp__to__b3db", "bbbp→b3db"), "model": md, "seed": seed,
                             "auroc": roc_auc_score(ym, pm), "auprc_minority": average_precision_score(ym, pm),
                             "minority_prevalence": ym.mean(), "balanced_accuracy": balanced_accuracy_score(y, yhat),
                             "minority_recall": recall_score(ym, (yhat == mino).astype(int))})
    D = pd.DataFrame(rows)
    out = D.groupby(["setting", "model"], sort=False).agg(
        **{f"{c}_{s}": (c, s) for c in ("auroc", "auprc_minority", "minority_prevalence", "balanced_accuracy", "minority_recall")
           for s in ("mean", "std")}, n_seeds=("seed", "nunique")).reset_index()
    out.to_csv(REPO / "tables/classifier_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["setting", "model", "auroc_mean", "auprc_minority_mean", "minority_prevalence_mean", "balanced_accuracy_mean",
               "minority_recall_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
