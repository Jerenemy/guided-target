#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


VINA_COL = "vina_dock"


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if VINA_COL not in df.columns:
        raise ValueError(f"{path} is missing required column '{VINA_COL}'")
    return df


def summarize_scores(scores: pd.Series, name: str) -> None:
    x = scores.values.astype(float)
    print(f"\n=== {name} ({len(x)} molecules) ===")
    print(f"mean:   {x.mean():.3f}")
    print(f"std:    {x.std(ddof=1):.3f}")
    print(f"median: {np.median(x):.3f}")
    print(f"min:    {x.min():.3f}")
    print(f"max:    {x.max():.3f}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"q{int(q*100):2d}:   {np.quantile(x, q):.3f}")


def threshold_stats(scores: pd.Series, thresholds=None, name: str = "") -> None:
    if thresholds is None:
        thresholds = [-5.0, -6.0, -7.0, -8.0]
    x = scores.values.astype(float)
    print(f"\nFraction with {VINA_COL} <= threshold ({name}):")
    n = len(x)
    for t in thresholds:
        frac = (x <= t).mean()
        print(f"  <= {t:5.2f}: {frac*100:6.2f}%  ({int(frac*n)}/{n})")


def unpaired_test(baseline: pd.Series, guided: pd.Series) -> None:
    b = baseline.values.astype(float)
    g = guided.values.astype(float)
    print("\n=== Unpaired comparison (all molecules) ===")
    print(f"baseline mean {VINA_COL}: {b.mean():.3f}")
    print(f"guided   mean {VINA_COL}: {g.mean():.3f}")
    diff = g.mean() - b.mean()
    print(f"guided - baseline mean diff: {diff:.3f} (negative is better)")

    if not HAVE_SCIPY:
        print("SciPy not installed; skipping Mann–Whitney U test.")
        return

    # One-sided: guided more negative than baseline
    stat, p = mannwhitneyu(g, b, alternative="less")
    print(f"Mann–Whitney U (guided < baseline): U = {stat:.1f}, p = {p:.3e}")


def find_overlap(df_base: pd.DataFrame, df_guided: pd.DataFrame):
    keys = []
    for col in ["ligand_filename", "smiles", "index"]:
        if col in df_base.columns and col in df_guided.columns:
            keys.append(col)

    if not keys:
        return None, None, None

    # Try each key separately; pick the one with the largest overlap
    best_key = None
    best_merge = None
    best_n = 0
    for col in keys:
        merged = df_base[[col, VINA_COL]].merge(
            df_guided[[col, VINA_COL]],
            on=col,
            suffixes=("_base", "_guided"),
        )
        if len(merged) > best_n:
            best_n = len(merged)
            best_key = col
            best_merge = merged

    return best_merge, best_key, best_n


def paired_test(df_base: pd.DataFrame, df_guided: pd.DataFrame) -> None:
    merged, key, n = find_overlap(df_base, df_guided)
    if merged is None or n == 0:
        print("\nNo overlapping ligands found for paired analysis.")
        return

    print(f"\n=== Paired analysis on overlap ({n} shared by '{key}') ===")
    b = merged[f"{VINA_COL}_base"].values.astype(float)
    g = merged[f"{VINA_COL}_guided"].values.astype(float)
    delta = g - b  # negative = guided better

    print(f"mean delta (guided - baseline): {delta.mean():.3f}")
    print(f"median delta:                  {np.median(delta):.3f}")
    print(f"fraction improved (delta < 0): {(delta < 0).mean()*100:.2f}%")

    if not HAVE_SCIPY:
        print("SciPy not installed; skipping paired test.")
        return

    try:
        stat, p = wilcoxon(delta, alternative="less")
        print(f"Wilcoxon signed-rank (guided < baseline): W = {stat:.1f}, p = {p:.3e}")
    except ValueError as e:
        # e.g. all deltas identical
        print(f"Wilcoxon test failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Vina docking distributions between two CSV runs."
    )
    parser.add_argument("baseline_csv", help="CSV for unguided / baseline sampling")
    parser.add_argument("guided_csv", help="CSV for classifier-guided sampling")
    args = parser.parse_args()

    try:
        df_base = load_csv(args.baseline_csv)
        df_guided = load_csv(args.guided_csv)
    except Exception as e:
        print(f"Error loading CSVs: {e}", file=sys.stderr)
        sys.exit(1)

    summarize_scores(df_base[VINA_COL], "Baseline")
    summarize_scores(df_guided[VINA_COL], "Guided")

    threshold_stats(df_base[VINA_COL], name="Baseline")
    threshold_stats(df_guided[VINA_COL], name="Guided")

    unpaired_test(df_base[VINA_COL], df_guided[VINA_COL])
    paired_test(df_base, df_guided)


if __name__ == "__main__":
    main()
