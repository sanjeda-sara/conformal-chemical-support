"""Run a grid from a YAML config: Stage A (train + cache) then Stage B (conformal eval).

usage:
  python scripts/run_grid.py CONFIG [--stage A|B|AB] [--only-model rf] [--only-dataset bbbp]
                                    [--profile full|v12|core] [--workers 4]

Stage B profiles
  full  (default; writes <config>_v13_conformal.csv and <config>_v13_selective.csv)
        every method in conformal/{calibrators,sota,ablations}.py, scores LAC/APS/RAPS/margin,
        probability variants MC-dropout / logit-adjusted / temperature-scaled, PAS and
        macro-coverage-PAS, worst-slab metrics, plus the non-conformal baselines
        (evaluate_selective).
  core  the v1.2 methods with the v1.3 metrics (fast).
  v12   byte-for-byte the v1.2 Stage B (methods, scores and variants).
A config may set  stageB: {profile: full}  to choose the default.
"""
import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbbcp.evaluate import METHOD_SETS, SCORES, evaluate_file, evaluate_selective
from bbbcp.pipeline import REPO, RAW, features, prepare, run_one


def jobs(cfg):
    for block in cfg["blocks"]:
        for ds, reg, m, loss, seed in itertools.product(
                block["datasets"], block["regimes"], block["models"], block["losses"],
                range(block.get("seed_start", 0), block.get("seed_start", 0) + block["n_seeds"])):
            yield dict(dataset=ds, regime=reg, model_name=m, loss=loss, seed=seed,
                       external=block.get("external"))


def result_path(j):
    tag = j["dataset"] + (f"__to__{j['external']}" if j["external"] else "")
    return RAW / tag / j["regime"] / f"{j['model_name']}_{j['loss']}" / f"seed{j['seed']}.npz"


def build_meta(j, z):
    d = prepare(j["dataset"]); fd = features(j["dataset"])
    ext = prepare(j["external"]) if j["external"] else d
    fe = features(j["external"]) if j["external"] else fd
    it, iv, ic, ix = (z[k] for k in ("idx_train", "idx_val", "idx_cal", "idx_test"))
    return {"X_train": d["X"][it], "X_val": d["X"][iv], "X_cal": d["X"][ic], "X_test": ext["X"][ix],
            "M_train": fd["maccs"][it], "M_val": fd["maccs"][iv], "M_cal": fd["maccs"][ic],
            "M_test": fe["maccs"][ix], "desc_cal": fd["desc"][ic], "desc_test": fe["desc"][ix]}


PROFILES = {
    "full": dict(methods=METHOD_SETS["all"], scores=SCORES, prob_variants=("_mcdropout", "_la", "_ts"),
                 score_free=True, slab=True),
    "core": dict(methods=METHOD_SETS["core"], scores=SCORES[:3], prob_variants=("_mcdropout",),
                 score_free=False, slab=True),
    "v12": dict(methods=METHOD_SETS["v12"], scores=SCORES[:3], prob_variants=("_mcdropout",),
                score_free=False, slab=False),
}


def stage_b_one(j, profile):
    p = result_path(j)
    if not p.exists():
        return None, None
    z = np.load(p)
    meta = build_meta(j, z)
    df = evaluate_file(p, extra_meta=meta, **PROFILES[profile])
    sel = evaluate_selective(p, extra_meta=meta) if profile == "full" else None
    for k, v in j.items():
        df[k] = v
        if sel is not None:
            sel[k] = v
    return df, sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--stage", default="AB")
    ap.add_argument("--only-model", default=None); ap.add_argument("--only-dataset", default=None)
    ap.add_argument("--profile", default=None); ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    profile = a.profile or cfg.get("stageB", {}).get("profile", "full")
    J = [j for j in jobs(cfg) if a.only_model in (None, j["model_name"]) and a.only_dataset in (None, j["dataset"])]
    if "A" in a.stage:
        for i, j in enumerate(J):
            t0 = time.time(); p = run_one(**j)
            print(f"[A {i+1}/{len(J)}] {j} -> {p.name} ({time.time()-t0:.1f}s)", flush=True)
    if "B" in a.stage:
        frames, sels = [], []
        for j in J:                                     # warm caches once in the parent
            prepare(j["dataset"]); features(j["dataset"])
            if j["external"]:
                prepare(j["external"]); features(j["external"])
        t0 = time.time()
        if a.workers > 1:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(stage_b_one, j, profile): j for j in J}
                for i, f in enumerate(as_completed(futs)):
                    df, sel = f.result()
                    if df is not None:
                        frames.append(df)
                    if sel is not None:
                        sels.append(sel)
                    print(f"[B {i+1}/{len(J)}] {futs[f]} ({time.time()-t0:.0f}s)", flush=True)
        else:
            for i, j in enumerate(J):
                df, sel = stage_b_one(j, profile)
                if df is not None:
                    frames.append(df)
                if sel is not None:
                    sels.append(sel)
                print(f"[B {i+1}/{len(J)}] {j} ({time.time()-t0:.0f}s)", flush=True)
        # v12 keeps the v1.2 file name; full -> <config>_v13_*, core -> <config>_core_* (never overwrites v1.2 outputs)
        stem = Path(a.config).stem + {"full": "_v13", "core": "_core", "v12": ""}[profile]
        if a.only_model:
            stem += f"_{a.only_model}"
        if a.only_dataset:
            stem += f"_{a.only_dataset}"
        out = pd.concat(frames, ignore_index=True)
        dest = REPO / "results/processed" / f"{stem}_conformal.csv"
        out.to_csv(dest, index=False); print("wrote", dest, out.shape)
        if sels:
            s = pd.concat(sels, ignore_index=True)
            d2 = REPO / "results/processed" / f"{stem}_selective.csv"
            s.to_csv(d2, index=False); print("wrote", d2, s.shape)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
