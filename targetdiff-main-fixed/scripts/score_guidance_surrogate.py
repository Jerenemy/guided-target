import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.guidance import (
    EquivariantAffinityModel,
    GuidedAffinityWrapper,
    GuidedLigandContextWrapper,
    LigandContextAffinityModel,
    extract_guidance_state_dict,
    infer_guidance_architecture,
    infer_ligand_context_model_kwargs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_run_dir(path_str):
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        if path.name != "sample.pt":
            raise ValueError(f"Expected a run directory or sample.pt, got file '{path}'")
        run_dir = path.parent
        sample_pt = path
    else:
        run_dir = path
        sample_pt = run_dir / "sample.pt"
    sample_yml = run_dir / "sample.yml"
    if not sample_pt.is_file():
        raise FileNotFoundError(f"Missing sample.pt at {sample_pt}")
    return run_dir, sample_pt, sample_yml


def load_yaml(path):
    if not path.is_file():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def resolve_repo_path(path_str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = (REPO_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def infer_diffusion_ckpt(config_dict, explicit_path=None):
    if explicit_path:
        return resolve_repo_path(explicit_path)
    ckpt = (((config_dict or {}).get("model") or {}).get("checkpoint"))
    if not ckpt:
        raise ValueError("Could not infer diffusion checkpoint. Pass --diffusion_ckpt.")
    return resolve_repo_path(ckpt)


def infer_guidance_ckpt(config_dict, explicit_path=None):
    if explicit_path:
        return resolve_repo_path(explicit_path)
    ckpt = (((config_dict or {}).get("sample") or {}).get("guidance_ckpt"))
    if not ckpt:
        raise ValueError("Could not infer guidance checkpoint. Pass --guidance_ckpt.")
    return resolve_repo_path(ckpt)


def infer_guidance_arch(config_dict, explicit_arch, state_dict):
    if explicit_arch and explicit_arch != "auto":
        return explicit_arch
    cfg_arch = (((config_dict or {}).get("sample") or {}).get("guidance_arch"))
    if cfg_arch and cfg_arch != "auto":
        return cfg_arch
    return infer_guidance_architecture(state_dict)


def load_guidance_wrapper(sample_obj, sample_cfg, guidance_ckpt, guidance_arch, diffusion_ckpt, device):
    diff_ckpt = torch.load(diffusion_ckpt, map_location="cpu")
    ligand_atom_mode = diff_ckpt["config"].data.transform.ligand_atom_mode

    ckpt_aff = torch.load(guidance_ckpt, map_location="cpu")
    state_aff = extract_guidance_state_dict(ckpt_aff)
    guidance_arch = infer_guidance_arch(sample_cfg, guidance_arch, state_aff)

    protein_pos = sample_obj["data"].protein_pos
    protein_element = sample_obj["data"].protein_element
    sample_cfg_dict = (sample_cfg or {}).get("sample") or {}

    if guidance_arch == "equivariant":
        aff_model = EquivariantAffinityModel(max_z=100).to(device)
        aff_model.load_state_dict(state_aff)
        wrapper = GuidedAffinityWrapper(
            affinity_model=aff_model,
            pocket_pos=protein_pos,
            pocket_z=protein_element,
            ligand_atom_mode=ligand_atom_mode,
            device=device,
        )
    elif guidance_arch == "ligand_context":
        model_kwargs = infer_ligand_context_model_kwargs(state_aff)
        aff_model = LigandContextAffinityModel(**model_kwargs).to(device)
        aff_model.load_state_dict(state_aff)
        wrapper = GuidedLigandContextWrapper(
            affinity_model=aff_model,
            pocket_pos=protein_pos,
            pocket_z=protein_element,
            ligand_atom_mode=ligand_atom_mode,
            r_ligand=sample_cfg_dict.get("guidance_ligand_radius", 5.0),
            r_cross=sample_cfg_dict.get("guidance_cross_radius", 6.0),
            device=device,
        )
    else:
        raise ValueError(f"Unknown guidance architecture '{guidance_arch}'")

    aff_model.eval()
    wrapper.eval()
    return wrapper, ligand_atom_mode, guidance_arch


def batched_surrogate_scores(wrapper, pred_pos_list, pred_v_list, protein_pos, batch_size, device):
    rows = []
    protein_pos = protein_pos.to(device)
    num_pocket_atoms = protein_pos.size(0)
    with torch.no_grad():
        for start in range(0, len(pred_pos_list), batch_size):
            pos_chunks = []
            v_chunks = []
            batch_chunks = []
            protein_chunks = []
            protein_batch_chunks = []
            graph_ids = []
            stop = min(len(pred_pos_list), start + batch_size)
            for graph_idx, sample_idx in enumerate(range(start, stop)):
                pos_np = pred_pos_list[sample_idx]
                v_np = pred_v_list[sample_idx]
                pos = torch.as_tensor(pos_np, dtype=torch.float32, device=device)
                v = torch.as_tensor(v_np, dtype=torch.long, device=device)
                pos_chunks.append(pos)
                v_chunks.append(v)
                batch_chunks.append(torch.full((pos.size(0),), graph_idx, dtype=torch.long, device=device))
                protein_chunks.append(protein_pos)
                protein_batch_chunks.append(
                    torch.full((num_pocket_atoms,), graph_idx, dtype=torch.long, device=device)
                )
                graph_ids.append(sample_idx)

            ligand_pos = torch.cat(pos_chunks, dim=0)
            ligand_v = torch.cat(v_chunks, dim=0)
            batch_ligand = torch.cat(batch_chunks, dim=0)
            protein_pos_batch = torch.cat(protein_chunks, dim=0)
            batch_protein = torch.cat(protein_batch_chunks, dim=0)
            score = wrapper(
                ligand_pos=ligand_pos,
                ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                batch_protein=batch_protein,
                protein_pos=protein_pos_batch,
            )
            score = score.detach().cpu().numpy()

            for local_idx, sample_idx in enumerate(graph_ids):
                surrogate_score = float(score[local_idx])
                rows.append({
                    "index": sample_idx,
                    "n_atoms": int(len(pred_v_list[sample_idx])),
                    "surrogate_score": surrogate_score,
                    "surrogate_affinity": -surrogate_score,
                })
    return rows


def quantile(xs, q):
    if not xs:
        return float("nan")
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def summarize_rows(rows, run_dir, guidance_ckpt, guidance_arch, ligand_atom_mode):
    affinities = [r["surrogate_affinity"] for r in rows]
    scores = [r["surrogate_score"] for r in rows]
    return {
        "run_dir": str(run_dir),
        "guidance_ckpt": str(guidance_ckpt),
        "guidance_arch": guidance_arch,
        "ligand_atom_mode": ligand_atom_mode,
        "num_samples": len(rows),
        "surrogate_affinity_mean": float(np.mean(affinities)) if affinities else float("nan"),
        "surrogate_affinity_q01": quantile(affinities, 0.01),
        "surrogate_affinity_q05": quantile(affinities, 0.05),
        "surrogate_affinity_q10": quantile(affinities, 0.10),
        "surrogate_affinity_min": float(np.min(affinities)) if affinities else float("nan"),
        "surrogate_score_mean": float(np.mean(scores)) if scores else float("nan"),
        "surrogate_score_q90": quantile(scores, 0.90),
        "surrogate_score_q95": quantile(scores, 0.95),
        "surrogate_score_q99": quantile(scores, 0.99),
        "surrogate_score_max": float(np.max(scores)) if scores else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Score a TargetDiff run with the guidance surrogate model.")
    parser.add_argument("run", help="Run directory or sample.pt")
    parser.add_argument("--out_csv", default=None, help="Output CSV path (default: <run>/surrogate_scores.csv)")
    parser.add_argument("--summary_json", default=None, help="Optional summary JSON path")
    parser.add_argument("--guidance_ckpt", default=None, help="Guidance checkpoint path. Defaults to sample.yml if present.")
    parser.add_argument("--guidance_arch", default="auto", help="Guidance architecture: auto, ligand_context, or equivariant")
    parser.add_argument("--diffusion_ckpt", default=None, help="Diffusion checkpoint path used to recover ligand_atom_mode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of graphs per scoring batch")
    args = parser.parse_args()

    run_dir, sample_pt, sample_yml = resolve_run_dir(args.run)
    sample_cfg = load_yaml(sample_yml)
    guidance_ckpt = infer_guidance_ckpt(sample_cfg, args.guidance_ckpt)
    diffusion_ckpt = infer_diffusion_ckpt(sample_cfg, args.diffusion_ckpt)

    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "surrogate_scores.csv"
    summary_json = Path(args.summary_json) if args.summary_json else run_dir / "surrogate_scores_summary.json"

    sample_obj = torch.load(sample_pt, map_location="cpu", weights_only=False)
    wrapper, ligand_atom_mode, guidance_arch = load_guidance_wrapper(
        sample_obj=sample_obj,
        sample_cfg=sample_cfg,
        guidance_ckpt=guidance_ckpt,
        guidance_arch=args.guidance_arch,
        diffusion_ckpt=diffusion_ckpt,
        device=args.device,
    )

    rows = batched_surrogate_scores(
        wrapper=wrapper,
        pred_pos_list=sample_obj["pred_ligand_pos"],
        pred_v_list=sample_obj["pred_ligand_v"],
        protein_pos=sample_obj["data"].protein_pos,
        batch_size=args.batch_size,
        device=args.device,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "n_atoms", "surrogate_score", "surrogate_affinity"])
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows, run_dir, guidance_ckpt, guidance_arch, ligand_atom_mode)
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Scored {len(rows)} samples from {sample_pt}")
    print(f"Wrote {out_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
