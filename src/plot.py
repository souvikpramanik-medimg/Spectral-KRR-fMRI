"""Plot Spectral-KRR gain over vanilla KRR vs training size.

Reads results/summary_{age,fiq,asd}.csv (written by evaluate.py) and saves
results/figure_gain_vs_n.png.

Usage: python src/plot.py [--results results]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [("age", "Age (Pearson r)"), ("fiq", "FIQ (Pearson r)"), ("asd", "ASD (AUC)")]


def main(results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (t, title) in zip(axes, PANELS):
        df = pd.read_csv(os.path.join(results_dir, f"summary_{t}.csv")).sort_values("n")
        gain = df["spec_fixed"] - df["krr_raw"]
        ax.plot(df["n"], gain, marker="o", color="tab:blue")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Training size n")
        ax.set_xticks(df["n"])
    axes[0].set_ylabel("Gain (Spectral-KRR − vanilla KRR)")
    fig.tight_layout()
    out = os.path.join(results_dir, "figure_gain_vs_n.png")
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    main(ap.parse_args().results)
