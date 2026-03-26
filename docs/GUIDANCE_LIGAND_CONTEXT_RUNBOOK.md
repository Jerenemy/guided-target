# Ligand-Context Guidance Runbook

This note describes the new ligand-context guidance integration, what changed in TargetDiff, and how to run guided sampling with the new pretrained guidance checkpoint.

## What Changed

The old guidance path used an equivariant affinity model that:

- built a single ligand+pocket graph,
- used atomic number plus geometry,
- pooled over all nodes,
- and predicted a scalar affinity from that pooled graph representation.

The new guidance path keeps the ligand+pocket graph, but changes the guidance model to a ligand-context architecture:

- ligand and pocket atoms are both present during message passing,
- ligand nodes receive messages from nearby ligand atoms and nearby pocket atoms,
- edges have explicit types:
  - `ligand_ligand`
  - `pocket_to_ligand`
- each edge includes an RBF embedding of pairwise distance,
- the final readout pools over ligand nodes only,
- the pooled ligand representation goes through an MLP head to predict affinity.

This makes the expert more suitable for guidance because:

- pocket context affects ligand states,
- the fixed pocket does not dominate the graph-level representation,
- the scalar output still depends smoothly on ligand coordinates,
- the runtime graph can be rebuilt directly during sampling.

## Files Updated

The integration was added in these files:

- [`targetdiff-main-fixed/models/guidance.py`](targetdiff-main-fixed/models/guidance.py)
  - added `LigandContextAffinityModel`
  - added `GuidedLigandContextWrapper`
  - added runtime edge construction for ligand-ligand and pocket-to-ligand edges
  - added automatic checkpoint architecture detection
- [`targetdiff-main-fixed/scripts/sample_for_pocket.py`](targetdiff-main-fixed/scripts/sample_for_pocket.py)
  - now loads either the old equivariant checkpoint or the new ligand-context checkpoint
  - now passes `ligand_atom_mode` through guidance and reconstruction
- [`targetdiff-main-fixed/configs/sampling.yml`](targetdiff-main-fixed/configs/sampling.yml)
  - default guidance checkpoint now points to the new model
  - added new guidance config fields for architecture and radii

## New Guidance Checkpoint

The new pretrained checkpoint is expected at:

- [`targetdiff-main-fixed/pretrained_models/ligand_context_score_only.pt`](targetdiff-main-fixed/pretrained_models/ligand_context_score_only.pt)

The loader can infer the architecture automatically from the checkpoint state dict.

## Guidance Config

The updated default sampling config is:

```yaml
sample:
  guidance_ckpt: "./pretrained_models/ligand_context_score_only.pt"
  guidance_arch: auto
  guidance_ligand_radius: 5.0
  guidance_cross_radius: 6.0
  guidance_scale: 3.0
  guidance_start_step: 0.1
  guidance_scale_mode: fixed
  guidance_log_stats: true
  use_guidance: true
```

Notes:

- `guidance_arch: auto` will detect the checkpoint type automatically.
- `guidance_ligand_radius` is the ligand-ligand spatial edge cutoff.
- `guidance_cross_radius` is the pocket-to-ligand context cutoff.
- `guidance_start_step: 0.1` means guidance is only applied in the final ~10% of reverse steps.

## How To Run Guided Sampling

Important:

- run the command from the `targetdiff-main-fixed` directory
- the checkpoint paths in the config are relative to that directory
- use the `targetdiff_m` environment

### 1. Activate the environment

```bash
conda activate targetdiff_m
```

### 2. Move into the TargetDiff repo

```bash
cd /home/jzay/Desktop/mol_gen/targetdiff-main-fixed
```

### 3. Run guided sampling

```bash
python scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path /absolute/path/to/your_pocket.pdb \
  --device cuda:0 \
  --batch_size 32 \
  --num_samples 100 \
  --result_path ./outputs_ligand_context
```

Example:

```bash
python scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path /home/jzay/Desktop/mol_gen/y220c_pocket10.pdb \
  --device cuda:0 \
  --batch_size 32 \
  --num_samples 100 \
  --result_path ./outputs_y220c_ligand_context
```

## Recommended Smoke Test

Before a full run, do a small test:

```bash
python scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path /home/jzay/Desktop/mol_gen/y220c_pocket10.pdb \
  --device cuda:0 \
  --batch_size 4 \
  --num_samples 4 \
  --result_path ./outputs_smoke_ligand_context
```

This should produce:

- `sample.pt`
- `guidance_stats.json` if guidance logging is enabled
- `sdf/` with reconstructed molecules

## Expected Outputs

For a successful run, the result directory should contain:

- `sample.yml`
- `sample.pt`
- `guidance_stats.json` if `guidance_log_stats: true`
- `sdf/*.sdf`

## How To Switch Back To The Old Guidance Model

If you want to run the older equivariant guidance model instead, edit [`targetdiff-main-fixed/configs/sampling.yml`](targetdiff-main-fixed/configs/sampling.yml) like this:

```yaml
sample:
  guidance_ckpt: "./pretrained_models/affinity_latest.pt"
  guidance_arch: equivariant
```

Then run the same sampling command.

## Practical Notes

- The new ligand-context model was smoke-tested in the `targetdiff_m` environment:
  - the checkpoint loads,
  - the wrapper builds a runtime graph,
  - the output is differentiable with respect to ligand coordinates.
- The notebook environment under `moment/.venv` is not sufficient for runtime TargetDiff testing because it is missing `torch_scatter`.
- Reconstruction in `sample_for_pocket.py` now uses the diffusion checkpoint’s `ligand_atom_mode` instead of hardcoding `add_aromatic`.

## Suggested Next Step

After the smoke test passes, run a normal guided sampling job and compare it against:

- unguided samples,
- the old equivariant guidance model,
- and the new ligand-context guidance model

using the same pocket, seed, and number of samples.
