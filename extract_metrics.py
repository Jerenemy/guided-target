#!/usr/bin/env python3
import argparse, csv, json, os, sys
from collections import defaultdict

import torch

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    Chem = None  # SDF writing will be disabled if RDKit isn't available


def safe_torch_load(path):
    """Works with both older and newer torch (weights_only arg added in 2.6)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_first_affinity(v):
    """Return a float affinity from a vina sub-result that may be a list/dict/None."""
    if v is None:
        return None
    # common shapes:
    # - list of dicts: [{'affinity': -7.8, ...}, ...]
    # - dict with 'affinity'
    if isinstance(v, list) and v:
        item = v[0]
        if isinstance(item, dict) and 'affinity' in item:
            return float(item['affinity'])
    if isinstance(v, dict) and 'affinity' in v:
        return float(v['affinity'])
    # dict of modes: {'score_only': [...], 'minimize': [...], 'dock': [...]}
    # (handled elsewhere)
    return None


def unpack_vina(vina_obj):
    """Normalize vina results into three columns."""
    score_only = minimize = dock = None
    if vina_obj is None:
        return score_only, minimize, dock

    # If it's a dict with modes:
    if isinstance(vina_obj, dict):
        score_only = get_first_affinity(vina_obj.get('score_only'))
        minimize   = get_first_affinity(vina_obj.get('minimize'))
        dock       = get_first_affinity(vina_obj.get('dock'))
        return score_only, minimize, dock

    # If it's just a list (qvina style):
    score_only = get_first_affinity(vina_obj)
    return score_only, minimize, dock


def main():
    ap = argparse.ArgumentParser(description="Extract per-molecule info from metrics_-1.pt")
    ap.add_argument("metrics_pt", help="Path to metrics_-1.pt")
    ap.add_argument("--csv", required=True, help="Output CSV path")
    ap.add_argument("--sdf", default=None, help="(Optional) Output SDF path with properties")
    args = ap.parse_args()

    data = safe_torch_load(args.metrics_pt)
    results = data.get("all_results", [])
    if not results:
        print("No results found in PT file (key 'all_results' missing or empty).", file=sys.stderr)
        sys.exit(1)

    # Prepare CSV
    fieldnames = [
        "index",
        "smiles",
        "ligand_filename",
        "qed",
        "sa",
        # ring-size counts (3..9)
        "ring_3", "ring_4", "ring_5", "ring_6", "ring_7", "ring_8", "ring_9",
        # vina
        "vina_score_only",
        "vina_minimize",
        "vina_dock",
    ]

    n_written_csv = 0
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Optional SDF writer
        sdf_writer = None
        if args.sdf:
            if Chem is None:
                print("RDKit not available; SDF will not be written.", file=sys.stderr)
            else:
                os.makedirs(os.path.dirname(os.path.abspath(args.sdf)), exist_ok=True)
                sdf_writer = Chem.SDWriter(args.sdf)

        for i, r in enumerate(results):
            smiles = r.get("smiles")
            ligfile = r.get("ligand_filename", None)

            chem = r.get("chem_results", {}) or {}
            qed = chem.get("qed", None)
            sa  = chem.get("sa", None)
            ring_counter = chem.get("ring_size", {}) or {}

            # normalize rings to ints
            rings = defaultdict(int)
            # ring_counter may be a Counter or dict with str/int keys
            for k, v in dict(ring_counter).items():
                try:
                    rings[int(k)] += int(v)
                except Exception:
                    pass

            vina_score_only, vina_minimize, vina_dock = unpack_vina(r.get("vina"))

            row = {
                "index": i,
                "smiles": smiles,
                "ligand_filename": ligfile,
                "qed": None if qed is None else float(qed),
                "sa": None if sa is None else float(sa),
                "ring_3": rings[3],
                "ring_4": rings[4],
                "ring_5": rings[5],
                "ring_6": rings[6],
                "ring_7": rings[7],
                "ring_8": rings[8],
                "ring_9": rings[9],
                "vina_score_only": vina_score_only,
                "vina_minimize": vina_minimize,
                "vina_dock": vina_dock,
            }
            writer.writerow(row)
            n_written_csv += 1

            # SDF row (with properties)
            if sdf_writer is not None and Chem is not None:
                mol = r.get("mol")
                if mol is not None:
                    try:
                        m = Chem.Mol(mol)  # RDKit copies the molecule
                        # attach props as strings
                        m.SetProp("index", str(i))
                        if smiles is not None: m.SetProp("SMILES", smiles)
                        if ligfile is not None: m.SetProp("ligand_filename", str(ligfile))
                        if qed is not None: m.SetProp("QED", f"{float(qed):.6f}")
                        if sa is not None: m.SetProp("SA", f"{float(sa):.6f}")
                        m.SetProp("ring_sizes", json.dumps({k: rings[k] for k in range(3,10)}))
                        if vina_score_only is not None: m.SetProp("Vina_ScoreOnly", f"{vina_score_only:.6f}")
                        if vina_minimize   is not None: m.SetProp("Vina_Minimize",  f"{vina_minimize:.6f}")
                        if vina_dock       is not None: m.SetProp("Vina_Dock",      f"{vina_dock:.6f}")
                        sdf_writer.write(m)
                    except Exception as e:
                        # Skip SDF write for this mol but keep going
                        print(f"[warn] SDF write failed for index {i}: {e}", file=sys.stderr)

        if sdf_writer is not None:
            sdf_writer.close()

    # Print the global stability block too (handy to keep with the CSV)
    stability = data.get("stability")
    if stability:
        print("Global stability/validity:", json.dumps(stability, indent=2))

    print(f"Wrote {n_written_csv} rows to {args.csv}")
    if args.sdf and Chem is not None:
        print(f"Also wrote SDF to {args.sdf}")


if __name__ == "__main__":
    main()
