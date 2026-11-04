"""Stronger screening baseline (EXPLORATORY, added 1 Nov 2026 after internal review).

Question: does support-aware calibration add decision value beyond combining the established remedy
(class-conditional Mondrian calibration) with a standard screening filter at the same deferral budget?

Policies, per split, LAC, alpha 0.10, CE (singleton = accepted call with the set's label; else deferred):
  fixed_b3          class x support Mondrian (VAL tertiles)
  mondrian          class-conditional Mondrian alone
  mond_ad / mond_conf
      Mondrian, then additionally defer the accepted compounds with the lowest support (AD) or the lowest
      max probability (confidence) until total deferral equals that of fixed_b3 on the same split
      (matched on TEST using set sizes and features only; no TEST label).
  mond_ad_val / mond_conf_val
      fully held-out: the extra deferral fraction and the support / confidence cut-off are set on VAL
      (VAL sets use the CAL thresholds); no TEST quantity is used.
If Mondrian already defers more than fixed_b3 on a split, no extra deferral is added (recorded).

Output: tables/calibrated_rejection_summary.csv (per setting x model x policy, seed means)
        tables/calibrated_rejection_paired.csv  (fixed_b3 minus each Mondrian+filter policy)
usage: python scripts/run_calibrated_rejection.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.stats.stats import paired_compare

REPO = Path(__file__).resolve().parents[1]
SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold", "bbbp__to__b3db/scaffold"]
MODELS = ["rf", "gin", "chemprop"]
ALPHA = 0.10


def metrics(acc, lab, y, mino):
    m = y == mino
    return {"deferral_rate": 1 - acc.mean(),
            "accepted_error": ((acc & (lab != y)).sum() / acc.sum()) if acc.any() else np.nan,
            "minority_false_accept_rate": (acc & m & (lab != mino)).sum() / m.sum()}


def extra_defer(acc, score, k):
    """Defer the k accepted compounds with the lowest score (stable order)."""
    out = acc.copy()
    if k > 0:
        idx = np.where(acc)[0]
        out[idx[np.argsort(score[idx], kind="stable")[:k]]] = False
    return out


def run_file(z):
    mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    bc = sim_bins(ref, z["sim_cal"])[0]; bt = sim_bins(ref, z["sim_test"])[0]
    sc = 1.0 - z["p_cal"][np.arange(len(yc)), yc]; St = 1.0 - z["p_test"]
    Sx = C.predict_sets(St, C.mondrian_x(sc, yc, St, ALPHA, bin_cal=bc, bin_test=bt))
    Sm = C.predict_sets(St, C.mondrian(sc, yc, St, ALPHA))
    ax, am = Sx.sum(1) == 1, Sm.sum(1) == 1
    lx, lm = Sx.argmax(1), Sm.argmax(1)
    n = len(yt); k = int(round((1 - ax.mean()) * n)) - int((~am).sum())
    out = {"fixed_b3": metrics(ax, lx, yt, mino), "mondrian": metrics(am, lm, yt, mino)}
    conf_t = z["p_test"].max(1)
    out["mond_ad"] = metrics(extra_defer(am, z["sim_test"], k), lm, yt, mino)
    out["mond_conf"] = metrics(extra_defer(am, conf_t, k), lm, yt, mino)
    if len(z["sim_val"]):
        Sv = 1.0 - z["p_val"]; bv = sim_bins(ref, z["sim_val"])[0]
        avx = C.predict_sets(Sv, C.mondrian_x(sc, yc, Sv, ALPHA, bin_cal=bc, bin_test=bv)).sum(1) == 1
        avm = C.predict_sets(Sv, C.mondrian(sc, yc, Sv, ALPHA)).sum(1) == 1
        kv = int(round((1 - avx.mean()) * len(avx))) - int((~avm).sum())
        for nm, sv, stt in (("mond_ad_val", z["sim_val"], z["sim_test"]), ("mond_conf_val", z["p_val"].max(1), conf_t)):
            acc = am.copy()
            if kv > 0:
                acc_v = np.sort(sv[avm], kind="stable")
                tau = acc_v[min(kv, len(acc_v)) - 1]          # defer accepted TEST compounds with score <= tau
                acc = am & (stt > tau)
            out[nm] = metrics(acc, lm, yt, mino)
    return out, k


def main():
    rows = []
    for st in SETTINGS:
        for md in MODELS:
            for seed in range(100, 120):
                z = np.load(REPO / "results/raw" / st / f"{md}_ce" / f"seed{seed}.npz", allow_pickle=True)
                res, k = run_file(z)
                for pol, m in res.items():
                    rows.append({"setting": st.replace("bbbp__to__b3db", "bbbp→b3db"), "model": md, "seed": seed,
                                 "policy": pol, "extra_deferrals_needed": k, **m})
    D = pd.DataFrame(rows)
    S = D.groupby(["setting", "model", "policy"], sort=False).mean(numeric_only=True).reset_index().drop(columns="seed")
    S.to_csv(REPO / "tables/calibrated_rejection_summary.csv", index=False)
    P = []
    for (st, md), g in D.groupby(["setting", "model"], sort=False):
        base = g[g.policy == "fixed_b3"].set_index("seed").sort_index()
        for pol in ("mondrian", "mond_ad", "mond_conf", "mond_ad_val", "mond_conf_val"):
            o = g[g.policy == pol].set_index("seed").sort_index()
            if o.empty:
                continue
            for ep in ("minority_false_accept_rate", "accepted_error", "deferral_rate"):
                P.append({"setting": st, "model": md, "contrast": f"fixed_b3 - {pol}", "endpoint": ep,
                          **paired_compare(base[ep].values, o[ep].values),
                          "n_splits_unmatched": int((base.extra_deferrals_needed < 0).sum())})
    P = pd.DataFrame(P); P.to_csv(REPO / "tables/calibrated_rejection_paired.csv", index=False)
    pd.set_option("display.width", 220)
    q = P[P.endpoint == "minority_false_accept_rate"]
    q = q.assign(tr=q.setting.str.contains("→"))
    print(q.groupby(["contrast", "tr"]).agg(mn=("mean_diff", "min"), mx=("mean_diff", "max"), n_ci_below0=("ci_hi", lambda x: int((x < 0).sum())),
                                           n_ci_above0=("ci_lo", lambda x: int((x > 0).sum())), cells=("ci_hi", "size")).round(4).to_string())
    q = P[P.endpoint.isin(["accepted_error", "deferral_rate"])]
    q = q.assign(tr=q.setting.str.contains("→"))
    print(q.groupby(["contrast", "endpoint", "tr"]).mean_diff.agg(["min", "max"]).round(4).to_string())
    print("splits where Mondrian already defers more than fixed_b3:", int((D[D.policy == "fixed_b3"].extra_deferrals_needed < 0).sum()), "of", int((D.policy == "fixed_b3").sum()))


if __name__ == "__main__":
    main()
