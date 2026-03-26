#!/usr/bin/env python3
"""
Analyze how the guidance surrogate behaves along saved diffusion trajectories.

What this script measures
-------------------------
- It loads one or more TargetDiff run directories containing `sample.pt`.
- It rescoring selected saved trajectory states with the same guidance surrogate
  used during sampling.
- It summarizes whether the surrogate objective improves during denoising,
  especially across the guided window.

How to interpret the key outputs
--------------------------------
- `surrogate_affinity_mean`: lower / more negative is better.
- `guidance_active`: whether the saved step falls inside the configured
  guidance window for that run.
- `delta_from_pre_guidance_mean`: negative means the run improved the surrogate
  relative to the last saved step before guidance turned on.
- Comparing a guided run against an unguided baseline:
  - if the guided run becomes more negative during the guided window and ends
    more negative than baseline, guidance is helping the surrogate objective.
  - if it improves relative to its own pre-guidance step but still ends worse
    than baseline, the gradient signal may be locally helpful but not enough to
    beat the unguided sampler.
  - if it does not improve during the guided window, guidance is too weak, too
    noisy, or otherwise ineffective in practice.

Important caveat
----------------
These scores are on the surrogate itself, not on Vina. They tell you whether
the guidance update is moving samples in the surrogate's preferred direction,
not whether that movement improves the downstream objective.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from scripts.score_guidance_surrogate import (
    infer_diffusion_ckpt,
    infer_guidance_ckpt,
    load_guidance_wrapper,
    load_yaml,
)


def resolve_run_dirs(path_strs):
    run_dirs = []
    for path_str in path_strs:
        path = Path(path_str).expanduser().resolve()
        if (path / "sample.pt").is_file():
            run_dirs.append(path)
            continue
        for pattern in ("mode_*", "sampling_run_*"):
            run_dirs.extend(sorted(d for d in path.glob(pattern) if (d / "sample.pt").is_file()))
    deduped = []
    seen = set()
    for run_dir in run_dirs:
        if run_dir not in seen:
            deduped.append(run_dir)
            seen.add(run_dir)
    if not deduped:
        raise FileNotFoundError("Could not find any run directories with sample.pt")
    return deduped


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
    return wrapper, sample_obj, sample_cfg, ligand_atom_mode, inferred_arch


def infer_guidance_start_traj_index(num_steps, guidance_start_step):
    if guidance_start_step is None:
        return None
    if isinstance(guidance_start_step, (float, int)) and 0 < float(guidance_start_step) < 1:
        return max(0, min(num_steps - 1, num_steps - 1 - int(float(guidance_start_step) * num_steps)))
    try:
        return max(0, min(num_steps - 1, num_steps - 1 - int(guidance_start_step)))
    except Exception:
        return None


def choose_step_indices(num_steps, guidance_start_step, requested_steps):
    if requested_steps:
        return sorted(set(max(0, min(num_steps - 1, int(s))) for s in requested_steps))

    candidates = {
        0,
        max(0, num_steps // 4),
        max(0, num_steps // 2),
        max(0, (3 * num_steps) // 4),
        num_steps - 1,
    }
    gidx = infer_guidance_start_traj_index(num_steps, guidance_start_step)
    if gidx is not None:
        candidates.update({
            max(0, gidx - 1),
            gidx,
            min(num_steps - 1, gidx + 1),
            min(num_steps - 1, gidx + 20),
            min(num_steps - 1, gidx + 50),
            min(num_steps - 1, gidx + 80),
        })
    return sorted(candidates)


def batch_score_states(wrapper, protein_pos, pos_list, v_list, batch_size, device):
    protein_pos = protein_pos.to(device)
    num_pocket_atoms = protein_pos.size(0)
    scores = []
    with torch.no_grad():
        for start in range(0, len(pos_list), batch_size):
            pos_chunks = []
            v_chunks = []
            batch_chunks = []
            protein_chunks = []
            protein_batch_chunks = []
            stop = min(len(pos_list), start + batch_size)
            for graph_idx, sample_idx in enumerate(range(start, stop)):
                pos = torch.as_tensor(pos_list[sample_idx], dtype=torch.float32, device=device)
                v = torch.as_tensor(v_list[sample_idx], dtype=torch.long, device=device)
                pos_chunks.append(pos)
                v_chunks.append(v)
                batch_chunks.append(torch.full((pos.size(0),), graph_idx, dtype=torch.long, device=device))
                protein_chunks.append(protein_pos)
                protein_batch_chunks.append(
                    torch.full((num_pocket_atoms,), graph_idx, dtype=torch.long, device=device)
                )

            ligand_pos = torch.cat(pos_chunks, dim=0)
            ligand_v = torch.cat(v_chunks, dim=0)
            batch_ligand = torch.cat(batch_chunks, dim=0)
            protein_pos_batch = torch.cat(protein_chunks, dim=0)
            batch_protein = torch.cat(protein_batch_chunks, dim=0)

            batch_scores = wrapper(
                ligand_pos=ligand_pos,
                ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                batch_protein=batch_protein,
                protein_pos=protein_pos_batch,
            )
            scores.extend(batch_scores.detach().cpu().tolist())
    return np.asarray(scores, dtype=float)


def summarize_values(values):
    return {
        "mean": float(np.mean(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q01": float(np.quantile(values, 0.01)),
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze surrogate scores along saved diffusion trajectories.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories or parent directories containing mode_*/sampling_run_* runs",
    )
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated trajectory indices to score. Default: auto-selected checkpoints around the guidance window.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--guidance_ckpt", default=None)
    parser.add_argument("--guidance_arch", default="auto")
    parser.add_argument("--diffusion_ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_csv", default=None, help="Default: <root>/trajectory_surrogate.csv")
    parser.add_argument("--summary_csv", default=None, help="Default: <root>/trajectory_surrogate_summary.csv")
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(args.paths)
    root = Path(args.paths[0]).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else root / "trajectory_surrogate.csv"
    summary_csv = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else root / "trajectory_surrogate_summary.csv"
    )
    requested_steps = [int(x) for x in args.steps.split(",") if x.strip()] if args.steps.strip() else []

    per_step_rows = []
    summary_rows = []

    for run_dir in run_dirs:
        wrapper, sample_obj, sample_cfg, ligand_atom_mode, guidance_arch = load_wrapper_for_run(
            run_dir=run_dir,
            guidance_ckpt_override=args.guidance_ckpt,
            guidance_arch=args.guidance_arch,
            diffusion_ckpt_override=args.diffusion_ckpt,
            device=args.device,
        )
        protein_pos = sample_obj["data"].protein_pos
        pos_trajs = sample_obj["pred_ligand_pos_traj"]
        v_trajs = sample_obj["pred_ligand_v_traj"]
        num_samples = len(pos_trajs)
        num_steps = int(pos_trajs[0].shape[0])
        guidance_start_step = (((sample_cfg or {}).get("sample") or {}).get("guidance_start_step"))
        guidance_start_traj_idx = infer_guidance_start_traj_index(num_steps, guidance_start_step)
        step_indices = choose_step_indices(num_steps, guidance_start_step, requested_steps)

        baseline_means = {}
        for step_idx in step_indices:
            pos_list = [traj[step_idx] for traj in pos_trajs]
            v_list = [traj[step_idx] for traj in v_trajs]
            score_vals = batch_score_states(
                wrapper=wrapper,
                protein_pos=protein_pos,
                pos_list=pos_list,
                v_list=v_list,
                batch_size=args.batch_size,
                device=args.device,
            )
            affinity_vals = -score_vals
            stats = summarize_values(affinity_vals)
            row = {
                "run_name": run_dir.name,
                "guidance_arch": guidance_arch,
                "ligand_atom_mode": ligand_atom_mode,
                "num_samples": num_samples,
                "step_index": int(step_idx),
                "reverse_t": int(num_steps - 1 - step_idx),
                "guidance_active": bool(guidance_start_traj_idx is not None and step_idx >= guidance_start_traj_idx),
                "guidance_start_step": guidance_start_step,
                "guidance_start_traj_idx": guidance_start_traj_idx,
                "surrogate_affinity_mean": stats["mean"],
                "surrogate_affinity_q05": stats["q05"],
                "surrogate_affinity_q01": stats["q01"],
            }
            per_step_rows.append(row)
            baseline_means[step_idx] = stats["mean"]

        first_idx = step_indices[0]
        last_idx = step_indices[-1]
        pre_guidance_idx = None
        if guidance_start_traj_idx is not None and guidance_start_traj_idx > 0:
            pre_guidance_idx = guidance_start_traj_idx - 1
            if pre_guidance_idx not in baseline_means:
                pos_list = [traj[pre_guidance_idx] for traj in pos_trajs]
                v_list = [traj[pre_guidance_idx] for traj in v_trajs]
                score_vals = batch_score_states(
                    wrapper=wrapper,
                    protein_pos=protein_pos,
                    pos_list=pos_list,
                    v_list=v_list,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                baseline_means[pre_guidance_idx] = float(np.mean(-score_vals))

        summary = {
            "run_name": run_dir.name,
            "guidance_arch": guidance_arch,
            "ligand_atom_mode": ligand_atom_mode,
            "num_samples": num_samples,
            "num_steps": num_steps,
            "guidance_start_step": guidance_start_step,
            "guidance_start_traj_idx": guidance_start_traj_idx,
            "first_scored_step": first_idx,
            "last_scored_step": last_idx,
            "surrogate_affinity_mean_first": baseline_means[first_idx],
            "surrogate_affinity_mean_final": baseline_means[last_idx],
            "delta_from_first_mean": baseline_means[last_idx] - baseline_means[first_idx],
            "surrogate_affinity_mean_pre_guidance": baseline_means.get(pre_guidance_idx, float("nan")),
            "delta_from_pre_guidance_mean": (
                baseline_means[last_idx] - baseline_means[pre_guidance_idx]
                if pre_guidance_idx is not None and pre_guidance_idx in baseline_means
                else float("nan")
            ),
        }
        summary_rows.append(summary)

    write_csv(out_csv, per_step_rows)
    write_csv(summary_csv, summary_rows)

    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_csv}")
    print()
    print("Interpretation guide:")
    print("- More negative `surrogate_affinity_mean` is better.")
    print("- `delta_from_pre_guidance_mean < 0` means the run improved the surrogate during the guided window.")
    print("- Compare the final row against your unguided baseline to see whether that improvement beats unguided denoising.")
    for row in summary_rows:
        delta = row["delta_from_pre_guidance_mean"]
        if math.isnan(delta):
            note = "no separate pre-guidance step available"
        elif delta < 0:
            note = "improved during guided window"
        elif delta > 0:
            note = "worsened during guided window"
        else:
            note = "no net change during guided window"
        print(
            f"{row['run_name']}: "
            f"final_affinity_mean={row['surrogate_affinity_mean_final']:.4f} "
            f"delta_from_pre_guidance_mean={delta:.4f} "
            f"=> {note}"
        )


if __name__ == "__main__":
    main()
