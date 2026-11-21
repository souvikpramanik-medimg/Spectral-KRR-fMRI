"""Grid evaluation: vanilla KRR vs Spectral-KRR (fixed / adaptive alpha),
plus optional latent-ridge and diffusion-augmentation baselines.

Usage: python src/evaluate.py --config configs/base.yaml
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_abide, target_config, site_indices, train_test_split
from spectral import build_cache
from utils import (fit_ridge, fit_logreg, metrics_reg, metrics_cls,
                   select_alpha_nested)


def fit_predict(task, X_tr, y_tr, X_te, y_te, ridge_alphas):
    if task == "reg":
        return metrics_reg(y_te, fit_ridge(X_tr, y_tr, X_te, ridge_alphas))
    return metrics_cls(y_te, fit_logreg(X_tr, y_tr, X_te))


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    dc, sc_, mc, ec = cfg["data"], cfg["spectral"], cfg["model"], cfg["eval"]
    np.random.seed(42)

    data = load_abide(dc["path"])
    X, site = data["X"], data["site"]
    print(f"X: {X.shape}")

    Z_all = synth_Z = synth_ages = None
    if dc.get("z_path") and dc.get("synth_path"):
        Z_all = np.load(dc["z_path"]).astype(np.float32)
        syn = np.load(dc["synth_path"])
        synth_Z = syn["synth_Z"].astype(np.float32)
        synth_ages = syn["synth_ages"].astype(np.float32)
    use_latent = Z_all is not None
    if not use_latent:
        print("No latent files given: skipping latent_noaug and diffaug_cont.")

    fixed = sc_["alpha"]
    alphas = sc_["adaptive_alphas"]
    assert fixed in alphas, "spectral.alpha must be in spectral.adaptive_alphas"
    cache = build_cache(X, alphas, sc_["atlas_size"])

    rows = []
    t_start = time.time()
    for tname in ec["targets"]:
        tconf = target_config(data)[tname]
        task, y_all = tconf["task"], tconf["y"]
        hold = site_indices(tconf["valid"], site, dc["site"])
        y_hold = y_all[hold]
        print(f"\nTARGET {tname.upper()} ({task}), {dc['site']} n={len(y_hold)}")

        for n in ec["n_values"]:
            for seed in ec["seeds"]:
                tr, te = train_test_split(len(y_hold), n, seed)
                tr_g, te_g = hold[tr], hold[te]
                y_tr, y_te = y_hold[tr], y_hold[te]

                a_star = (select_alpha_nested(cache, tr_g, y_tr, alphas, fixed,
                                              mc["ridge_alphas"])
                          if task == "reg" else fixed)

                def record(method, m):
                    row = dict(target=tname, n=n, seed=seed, method=method, **m)
                    if task == "reg":
                        row["alpha_star"] = a_star
                    rows.append(row)

                # 1. vanilla KRR on raw edges
                sc = StandardScaler().fit(X[tr_g])
                record("krr_raw", fit_predict(task, sc.transform(X[tr_g]), y_tr,
                                              sc.transform(X[te_g]), y_te, mc["ridge_alphas"]))
                # 2. Spectral-KRR, fixed alpha; 3. adaptive alpha
                for name, a in (("spec_fixed", fixed), ("spec_adaptive", a_star)):
                    F = cache[a]
                    sc = StandardScaler().fit(F[tr_g])
                    record(name, fit_predict(task, sc.transform(F[tr_g]), y_tr,
                                             sc.transform(F[te_g]), y_te, mc["ridge_alphas"]))

                if use_latent:
                    Z_tr, Z_te = Z_all[tr_g], Z_all[te_g]
                    # 4. latent ridge, no augmentation
                    record("latent_noaug", fit_predict(task, Z_tr, y_tr, Z_te, y_te,
                                                       mc["ridge_alphas"]))
                    # 5. diffusion augmentation (synthetic latents matched on age)
                    aug_Z, aug_y = [Z_tr], [y_tr]
                    rr = np.random.RandomState(seed * 137)
                    for yv in y_tr:
                        cand = np.argsort(np.abs(synth_ages - yv))[:20]
                        pick = rr.choice(cand, 1, replace=False)
                        aug_Z.append(synth_Z[pick])
                        aug_y.append(synth_ages[pick])
                    record("diffaug_cont", fit_predict(task, np.concatenate(aug_Z), np.concatenate(aug_y),
                                                       Z_te, y_te, mc["ridge_alphas"]))

            sub = [r for r in rows if r["target"] == tname and r["n"] == n]
            metric = "r" if task == "reg" else "auc"
            k = np.mean([r[metric] for r in sub if r["method"] == "krr_raw"])
            s = np.mean([r[metric] for r in sub if r["method"] == "spec_fixed"])
            print(f"  n={n:3d}  krr {metric}={k:+.3f}  spec_fixed {metric}={s:+.3f}")

    print(f"\nGrid: {(time.time() - t_start) / 60:.1f} min, {len(rows)} rows")

    os.makedirs(ec["out_dir"], exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ec["out_dir"], "results_all.csv"), index=False)
    for tname in ec["targets"]:
        sub = df[df["target"] == tname]
        metric = "auc" if tname == "asd" else "r"
        piv = sub.groupby(["n", "method"])[metric].mean().unstack().round(3)
        if tname != "asd":
            piv["alpha*"] = sub[sub["method"] == "spec_adaptive"].groupby("n")["alpha_star"].mean().round(3)
        piv.to_csv(os.path.join(ec["out_dir"], f"summary_{tname}.csv"))
        print(f"\n=== {tname.upper()} ===\n{piv.to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
