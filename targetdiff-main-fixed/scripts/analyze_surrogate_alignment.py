#!/usr/bin/env python3
"""
Analyze how well the guidance surrogate aligns with Vina on evaluated molecules.

What this script measures
-------------------------
- It loads each run's `eval_results/metrics_-1.pt`.
- For every evaluated molecule in that file, it recomputes the guidance surrogate
  score on the exact `(pred_pos, pred_v)` saved during evaluation.
- It then correlates that surrogate score against:
  - `vina.score_only`
  - `vina.minimize`
  - `vina.dock`

Sign convention
---------------
- The guidance wrapper returns `surrogate_score = -predicted_affinity`.
- Sampling maximizes `surrogate_score`.
- More negative Vina is better.

Therefore, if the surrogate is aligned with Vina:
- `corr(surrogate_score, vina_*)` should be NEGATIVE
  because higher surrogate score should correspond to lower/more negative Vina.
- Equivalently, `corr(surrogate_affinity, vina_*)` should be POSITIVE,
  where `surrogate_affinity = -surrogate_score`.

How to interpret the output
---------------------------
- Strongly negative score-correlation (or strongly positive affinity-correlation):
  the surrogate ranks molecules in a Vina-consistent way.
- Near-zero correlation:
  the surrogate is weak / not very predictive on generated molecules.
- Positive score-correlation:
  the surrogate is misaligned in ranking; increasing the guided score tends to
  go with worse Vina values.

Rule of thumb
-------------
- |corr| < 0.10 : essentially no useful ranking signal
- 0.10-0.30     : weak signal
- > 0.30        : meaningful alignment
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from scripts.score_guidance_surrogate import (
    infer_diffusion_ckpt,
    infer_guidance_ckpt,
    load_guidance_wrapper,
    load_yaml,
)


def resolve_run_dirs(path_str):
    path = Path(path_str).expanduser().resolve()

    # Single run directory
    if (path / "sample.pt").is_file() and (path / "eval_results" / "metrics_-1.pt").is_file():
        return [path]

    # Sweep-style directory
    run_dirs = []
    for pattern in ("mode_*", "sampling_run_*"):
        run_dirs.extend(
            sorted(
                d for d in path.glob(pattern)
                if d.is_dir()
                and (d / "sample.pt").is_file()
                and (d / "eval_results" / "metrics_-1.pt").is_file()
            )
        )

    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for run_dir in run_dirs:
        if run_dir not in seen:
            deduped.append(run_dir)
            seen.add(run_dir)

    if deduped:
        return deduped

    raise FileNotFoundError(
        f"Could not find any runs under '{path}'. "
        "Pass a single run dir containing sample.pt + eval_results/metrics_-1.pt, "
        "or a sweep dir containing mode_*/sampling_run_* subdirectories."
    )


def safe_corr(x, y, corr_type):
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if corr_type == "pearson":
        return float(pearsonr(x, y)[0])
    if corr_type == "spearman":
        return float(spearmanr(x, y).correlation)
    raise ValueError(f"Unknown corr_type: {corr_type}")


def load_wrapper_for_run(run_dir, guidance_ckpt_override, guidance_arch, diffusion_ckpt_override, device):
    sample_cfg = load_yaml(run_dir / "sample.yml")
    sample_obj = torch.load(run_dir / "sample.pt", map_location="cpu", weights_only=False)
    wrapper, ligand_atom_mode, inferred_arch = load_guidance_wrapper(
        sample_obj=sample_obj,
        sample_cfg=sample_cfg,
        guidance_ckpt=infer_guidance_ckpt(sample_cfg, guidance_ckpt_override),
        guidance_arch=guidance_arch,
        diffusion_ckpt=infer_diffusion_ckpt(sample_cfg, diffusion_ckpt_override),
        device=device,
    )
    return wrapper, ligand_atom_mode, inferred_arch


def score_metrics_results(run_dir, wrapper):
    metrics_pt = run_dir / "eval_results" / "metrics_-1.pt"
    obj = torch.load(metrics_pt, map_location="cpu", weights_only=False)
    results = obj["all_results"]
    sample_obj = torch.load(run_dir / "sample.pt", map_location="cpu", weights_only=False)
    protein_pos = sample_obj["data"].protein_pos
    protein_batch = torch.zeros(protein_pos.size(0), dtype=torch.long)

    rows = []
    with torch.no_grad():
        for i, record in enumerate(results):
            pos = torch.tensor(record["pred_pos"], dtype=torch.float32)
            v = torch.tensor(record["pred_v"], dtype=torch.long)
            batch = torch.zeros(pos.size(0), dtype=torch.long)
            surrogate_score = float(
                wrapper(
                    ligand_pos=pos,
                    ligand_v=v,
                    batch_ligand=batch,
                    batch_protein=protein_batch,
                    protein_pos=protein_pos,
                ).item()
            )

            rows.append({
                "eval_index": i,
                "smiles": record.get("smiles"),
                "surrogate_score": surrogate_score,
                "surrogate_affinity": -surrogate_score,
                "vina_score_only": float(record["vina"]["score_only"][0]["affinity"]),
                "vina_minimize": float(record["vina"]["minimize"][0]["affinity"]),
                "vina_dock": float(record["vina"]["dock"][0]["affinity"]),
            })
    return rows


def summarize_alignment(rows, run_name, guidance_arch, ligand_atom_mode):
    s_score = np.array([r["surrogate_score"] for r in rows], dtype=float)
    s_aff = np.array([r["surrogate_affinity"] for r in rows], dtype=float)

    metrics = {
        "score_only": np.array([r["vina_score_only"] for r in rows], dtype=float),
        "minimize": np.array([r["vina_minimize"] for r in rows], dtype=float),
        "dock": np.array([r["vina_dock"] for r in rows], dtype=float),
    }

    out = {
        "run_name": run_name,
        "n_eval": len(rows),
        "guidance_arch": guidance_arch,
        "ligand_atom_mode": ligand_atom_mode,
        "surrogate_score_mean": float(np.mean(s_score)) if len(s_score) else float("nan"),
        "surrogate_affinity_mean": float(np.mean(s_aff)) if len(s_aff) else float("nan"),
    }

    for metric_name, vina_vals in metrics.items():
        out[f"pearson_score_vs_{metric_name}"] = safe_corr(s_score, vina_vals, "pearson")
        out[f"spearman_score_vs_{metric_name}"] = safe_corr(s_score, vina_vals, "spearman")
        out[f"pearson_affinity_vs_{metric_name}"] = safe_corr(s_aff, vina_vals, "pearson")
        out[f"spearman_affinity_vs_{metric_name}"] = safe_corr(s_aff, vina_vals, "spearman")

    return out


def interpretation_line(row, metric_name):
    corr = row[f"spearman_score_vs_{metric_name}"]
    if math.isnan(corr):
        quality = "undefined"
        note = "not enough variation to estimate correlation"
    elif corr <= -0.30:
        quality = "good"
        note = "surrogate ranking is meaningfully aligned"
    elif corr < -0.10:
        quality = "weak"
        note = "surrogate has some alignment, but not strong"
    elif corr < 0.10:
        quality = "poor"
        note = "surrogate provides little ranking signal"
    else:
        quality = "misaligned"
        note = "higher surrogate score tends to go with worse Vina"
    return quality, note


def main():
    parser = argparse.ArgumentParser(
        description="Measure per-run alignment between the guidance surrogate and Vina on evaluated molecules."
    )
    parser.add_argument(
        "path",
        help="A single run dir or a sweep dir containing mode_*/sampling_run_* subdirectories",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Summary CSV path. Default: <path>/surrogate_alignment.csv for sweep dirs, or <run>/surrogate_alignment.csv for a single run.",
    )
    parser.add_argument(
        "--write_rows",
        action="store_true",
        help="Also write per-molecule rescored rows to <run>/surrogate_alignment_rows.csv",
    )
    parser.add_argument("--guidance_ckpt", default=None, help="Optional override for guidance checkpoint")
    parser.add_argument("--guidance_arch", default="auto", help="auto, ligand_context, or equivariant")
    parser.add_argument("--diffusion_ckpt", default=None, help="Optional override for diffusion checkpoint")
    parser.add_argument("--device", default="cpu", help="Use cpu by default for analysis")
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(args.path)
    base_path = Path(args.path).expanduser().resolve()
    if args.out_csv is not None:
        out_csv = Path(args.out_csv).expanduser().resolve()
    else:
        if len(run_dirs) == 1 and run_dirs[0] == base_path:
            out_csv = run_dirs[0] / "surrogate_alignment.csv"
        else:
            out_csv = base_path / "surrogate_alignment.csv"

    summaries = []
    for run_dir in run_dirs:
        wrapper, ligand_atom_mode, guidance_arch = load_wrapper_for_run(
            run_dir=run_dir,
            guidance_ckpt_override=args.guidance_ckpt,
            guidance_arch=args.guidance_arch,
            diffusion_ckpt_override=args.diffusion_ckpt,
            device=args.device,
        )
        rows = score_metrics_results(run_dir, wrapper)
        summary = summarize_alignment(rows, run_dir.name, guidance_arch, ligand_atom_mode)
        summaries.append(summary)

        if args.write_rows:
            rows_csv = run_dir / "surrogate_alignment_rows.csv"
            with rows_csv.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "eval_index",
                        "smiles",
                        "surrogate_score",
                        "surrogate_affinity",
                        "vina_score_only",
                        "vina_minimize",
                        "vina_dock",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

    fieldnames = list(summaries[0].keys()) if summaries else []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {out_csv}")
    print()
    print("Interpretation guide:")
    print("- Look at `spearman_score_vs_dock` first.")
    print("- Expected sign for good alignment: NEGATIVE.")
    print("- Near zero means the surrogate is weak on generated molecules.")
    print("- Positive means ranking misalignment: higher guided score tends to come with worse Vina.")
    print()
    for summary in summaries:
        quality, note = interpretation_line(summary, "dock")
        print(
            f"{summary['run_name']}: "
            f"spearman_score_vs_dock={summary['spearman_score_vs_dock']:.4f} "
            f"pearson_score_vs_dock={summary['pearson_score_vs_dock']:.4f} "
            f"=> {quality} ({note})"
        )


if __name__ == "__main__":
    main()
