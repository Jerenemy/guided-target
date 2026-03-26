#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


DEFAULT_METRICS = ["vina_score_only", "vina_minimize", "vina_dock"]
DEFAULT_THRESHOLDS = [-4.0, -5.0, -6.0, -7.0, -8.0]
DEFAULT_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50]
DEFAULT_TAIL_FRACS = [0.01, 0.05, 0.10]


def resolve_csv(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path

    candidates = [
        path / "metrics_extracted.csv",
        path / "eval_results" / "metrics_extracted.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise ValueError(
        f"Could not resolve a metrics CSV from '{path_str}'. "
        "Pass a metrics_extracted.csv file or a run directory containing one."
    )


def load_csv(path_str: str) -> pd.DataFrame:
    csv_path = resolve_csv(path_str)
    df = pd.read_csv(csv_path)
    df.attrs["source_path"] = str(csv_path)
    return df


def get_scores(df: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in df.columns:
        raise ValueError(f"{df.attrs.get('source_path', 'CSV')} is missing required column '{metric}'")
    return df[metric].dropna().astype(float).to_numpy()


def tail_mean(x: np.ndarray, frac: float) -> float:
    k = max(1, int(np.ceil(len(x) * frac)))
    return np.sort(x)[:k].mean()


def summarize_scores(x: np.ndarray, name: str, quantiles, tail_fracs) -> None:
    print(f"\n=== {name} ({len(x)} molecules) ===")
    print(f"mean:      {x.mean():.3f}")
    print(f"std:       {x.std(ddof=1) if len(x) > 1 else 0.0:.3f}")
    print(f"median:    {np.median(x):.3f}")
    print(f"min:       {x.min():.3f}")
    print(f"max:       {x.max():.3f}")

    print("lower quantiles:")
    for q in quantiles:
        print(f"  q{int(round(q * 100)):02d}:     {np.quantile(x, q):.3f}")

    print("best-tail means (lowest values):")
    for frac in tail_fracs:
        pct = int(round(frac * 100))
        print(f"  best {pct:2d}%:  {tail_mean(x, frac):.3f}")


def threshold_stats(x: np.ndarray, metric: str, thresholds, name: str = "") -> None:
    print(f"\nFraction with {metric} <= threshold ({name}):")
    n = len(x)
    for t in thresholds:
        frac = (x <= t).mean()
        print(f"  <= {t:6.2f}: {frac*100:6.2f}%  ({int(frac*n)}/{n})")


def compare_tails(metric: str, base: np.ndarray, guided: np.ndarray, quantiles, tail_fracs) -> None:
    print(f"\n=== Tail Comparison: {metric} (guided - baseline; negative is better) ===")
    for q in quantiles:
        diff = np.quantile(guided, q) - np.quantile(base, q)
        print(f"q{int(round(q * 100)):02d} delta:      {diff:.3f}")
    for frac in tail_fracs:
        diff = tail_mean(guided, frac) - tail_mean(base, frac)
        pct = int(round(frac * 100))
        print(f"best {pct:2d}% delta: {diff:.3f}")


def unpaired_test(metric: str, baseline: np.ndarray, guided: np.ndarray) -> None:
    print("\n=== Unpaired comparison (all molecules) ===")
    print(f"baseline mean {metric}: {baseline.mean():.3f}")
    print(f"guided   mean {metric}: {guided.mean():.3f}")
    diff = guided.mean() - baseline.mean()
    print(f"guided - baseline mean diff: {diff:.3f} (negative is better)")

    if not HAVE_SCIPY:
        print("SciPy not installed; skipping Mann-Whitney U test.")
        return

    stat, p = mannwhitneyu(guided, baseline, alternative="less")
    print(f"Mann-Whitney U (guided < baseline): U = {stat:.1f}, p = {p:.3e}")


def paired_test(df_base: pd.DataFrame, df_guided: pd.DataFrame, metric: str, pair_key: str) -> None:
    if pair_key not in df_base.columns or pair_key not in df_guided.columns:
        print(f"\nSkipping paired analysis: key '{pair_key}' not present in both CSVs.")
        return

    base = df_base[[pair_key, metric]].dropna()
    guided = df_guided[[pair_key, metric]].dropna()

    if base[pair_key].duplicated().any() or guided[pair_key].duplicated().any():
        print(
            f"\nSkipping paired analysis on '{pair_key}': key is not unique in one or both CSVs. "
            "This is common for 'smiles' and unsafe for 'ligand_filename=custom.pdb'."
        )
        return

    merged = base.merge(guided, on=pair_key, suffixes=("_base", "_guided"))
    if merged.empty:
        print(f"\nSkipping paired analysis: no shared rows for key '{pair_key}'.")
        return

    print(f"\n=== Paired analysis on overlap ({len(merged)} shared by '{pair_key}') ===")
    b = merged[f"{metric}_base"].to_numpy(dtype=float)
    g = merged[f"{metric}_guided"].to_numpy(dtype=float)
    delta = g - b

    print(f"mean delta (guided - baseline): {delta.mean():.3f}")
    print(f"median delta:                  {np.median(delta):.3f}")
    print(f"fraction improved (delta < 0): {(delta < 0).mean()*100:.2f}%")

    if not HAVE_SCIPY:
        print("SciPy not installed; skipping Wilcoxon signed-rank test.")
        return

    try:
        stat, p = wilcoxon(delta, alternative="less")
        print(f"Wilcoxon signed-rank (guided < baseline): W = {stat:.1f}, p = {p:.3e}")
    except ValueError as e:
        print(f"Wilcoxon test failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Vina-score tails between two TargetDiff runs."
    )
    parser.add_argument("baseline", help="Baseline CSV or run directory")
    parser.add_argument("guided", help="Guided CSV or run directory")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"Columns to compare (default: {' '.join(DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Thresholds for fraction-below-threshold summaries",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=DEFAULT_QUANTILES,
        help="Lower-tail quantiles to print, e.g. 0.01 0.05 0.1 0.25 0.5",
    )
    parser.add_argument(
        "--tail-fracs",
        nargs="+",
        type=float,
        default=DEFAULT_TAIL_FRACS,
        help="Fractions for best-tail means, e.g. 0.01 0.05 0.1",
    )
    parser.add_argument(
        "--pair-key",
        default=None,
        help="Optional unique key column for paired analysis. Disabled by default because pairing is often invalid.",
    )
    args = parser.parse_args()

    try:
        df_base = load_csv(args.baseline)
        df_guided = load_csv(args.guided)
    except Exception as e:
        print(f"Error loading CSVs: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Baseline: {df_base.attrs['source_path']}")
    print(f"Guided:   {df_guided.attrs['source_path']}")

    for metric in args.metrics:
        print(f"\n{'#' * 18} {metric} {'#' * 18}")
        try:
            base = get_scores(df_base, metric)
            guided = get_scores(df_guided, metric)
        except Exception as e:
            print(f"Skipping metric '{metric}': {e}")
            continue

        summarize_scores(base, "Baseline", args.quantiles, args.tail_fracs)
        summarize_scores(guided, "Guided", args.quantiles, args.tail_fracs)
        threshold_stats(base, metric, args.thresholds, name="Baseline")
        threshold_stats(guided, metric, args.thresholds, name="Guided")
        compare_tails(metric, base, guided, args.quantiles, args.tail_fracs)
        unpaired_test(metric, base, guided)

        if args.pair_key:
            paired_test(df_base, df_guided, metric, args.pair_key)


if __name__ == "__main__":
    main()
