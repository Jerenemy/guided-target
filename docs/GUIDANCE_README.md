# Guidance Implementation (TargetDiff + Affinity Model)

This document explains exactly how classifier-free guidance is realized: the equivariant affinity model, the pocket-aware wrapper, how the gradient is injected into diffusion sampling, and how to configure and run it.

## Files and Roles
- `targetdiff-main-fixed/models/guidance.py`
  - `EquivariantAffinityModel`: SE(3)-equivariant graph network used as the guidance scorer.
  - `GuidedAffinityWrapper`: adapts the scorer to TargetDiff by handling pocket centering, batching, and ligand atom-type decoding.
- `targetdiff-main-fixed/models/molopt_score_model.py` (`ScorePosNet3D.sample_diffusion`): diffusion sampler that consumes the guidance gradient.
- `targetdiff-main-fixed/scripts/sample_diffusion.py`: threads guidance arguments into the sampler.
- `targetdiff-main-fixed/scripts/sample_for_pocket.py`: runnable entry point that builds the pocket graph, loads checkpoints, and emits guided samples plus optional guidance stats.
- `targetdiff-main-fixed/configs/sampling.yml`: config toggles and magnitudes for guidance.
- `targetdiff-main-fixed/scripts/guidance_training/guidance_wrapper_fixed.ipynb`: reference notebook matching the fixed wrapper/model used here.

## Affinity Model (guidance scorer)
Path: `targetdiff-main-fixed/models/guidance.py`
- Inputs: concatenated ligand+pocket graph (`torch_geometric.data.Data`) with fields `pos`, `z` (atomic numbers), `batch`, and `node_type` (1 = ligand, 0 = pocket).
- Embedding: atomic numbers → 16-dim scalars; position enters as a 1x1o vector irrep; concatenated to `16x0e + 1x1o`.
- Backbone: three `EquivariantMPBlock` layers, each with gated tensor-product messages over a radius graph (`r=5.0 Å`), LayerNorm on aggregated messages, and residual connections.
- Readout: project to scalar irreps, mean-pool per graph, then MLP (`128 → 1`) to produce one affinity score per complex. Lower (more negative Vina-like) is better.
- *Intuition:* The model learns “how good is this ligand in this pocket” directly on 3D geometry, respecting rotations/translations; its single scalar drives the guidance gradient.

## GuidedAffinityWrapper (pocket + batching glue)
Path: `targetdiff-main-fixed/models/guidance.py`
- Centers the provided pocket coordinates once at init: `pocket_pos_centered = pocket_pos - pocket_pos.mean(0)`.
- Stores true pocket atomic numbers (`pocket_z`); if absent, fills with sentinel 99.
- Forward signature: `wrapper(ligand_pos, ligand_v, batch_ligand, batch_protein, protein_pos=None)`.
  - Decodes ligand type indices (`ligand_v`) to atomic numbers with `utils.transforms.get_atomic_number_from_index(..., mode='add_aromatic')`.
  - If `protein_pos/batch_protein` for all graphs are provided, they are used; otherwise the centered pocket is repeated for each ligand graph, with batches `[0, 0, ..., 1, 1, ...]`.
  - Concatenates ligand+pocket tensors and feeds them to the affinity model.
- Returns `-affinity` so that *increasing* the guidance objective corresponds to improved binding (the sampler ascends this score).
- *Intuition:* Keep the pocket fixed and correctly batched, translate ligand atom-type logits into real atomic numbers, and hand a consistent ligand+pocket frame to the scorer so its gradient points toward better binders.

## Guidance Flow Inside Sampling
Path: `targetdiff-main-fixed/models/molopt_score_model.py`, method `ScorePosNet3D.sample_diffusion`
1. Center protein and ligand (`center_pos`); store offsets.
   - *Intuition:* Align both parts so the pocket coordinates match the wrapper’s frame; translation is removed, then restored at the end.
2. Iterate timesteps `t` from high → 0 over `[T-num_steps, T)`.
   - *Intuition:* Walk the reverse diffusion chain, deciding at each step how much to denoise and how much to follow guidance.
3. Predict `pos0_from_e`; build posterior stats: `pos_model_mean`, `sigma = exp(0.5 * posterior_logvar)`.
   - *Intuition:* Compute where the model thinks the clean ligand should be and how uncertain it is at this step.
4. If guidance active and `t <= guidance_start_step`:
   - Clone `ligand_pos` with `requires_grad=True`.
   - Evaluate guidance score per graph.
   - Backprop to get `grad = ∇_{x_t} score_sum`; remove per-graph translation.
   - Map gradient to drift:
     - `var`: `guidance_delta = guidance_scale * sigma^2 * grad`
     - `sigma`: `guidance_scale * sigma * grad`
     - `fixed`: `guidance_scale * grad`
   - Sample diffusion noise: `noise_term = nonzero_mask * (sigma * noise_scale) * N(0, I)`.
   - Update: `ligand_pos_next = pos_model_mean + guidance_delta + noise_term`.
   - *Intuition:* Treat the affinity score as an energy; its gradient nudges atoms toward better binding while diffusion noise keeps diversity. Scaling by `sigma`/`sigma^2` tempers guidance when uncertainty is high.
5. If guidance inactive: `ligand_pos_next = pos_model_mean + nonzero_mask * (sigma * noise_scale) * N(0, I)`.
   - *Intuition:* Fall back to pure diffusion when guidance is off or past the allowed window.
6. Sample atom types via the diffusion head; record trajectories; restore the saved offset to positions at the end.
   - *Intuition:* Positions are guided; atom identities still come from the learned categorical process. Offsets put coordinates back into the original PDB frame.
7. Log guided-step stats when enabled.
   - *Intuition:* Diagnostics show whether guidance meaningfully overpowers noise (`guidance_over_noise`), and how strong the gradients/scores are at each step.

## Sampling Entry Point
Path: `targetdiff-main-fixed/scripts/sample_for_pocket.py`
- Loads diffusion checkpoint (`config.model.checkpoint`) and affinity checkpoint (`config.sample.guidance_ckpt`).
- Builds pocket graph from `--pdb_path`, featurizes protein atoms, and (optionally) sets `config.sample.num_samples` from CLI.
- Constructs `GuidedAffinityWrapper` when `sample.use_guidance` is true and passes it into `sample_diffusion_ligand`.
- Outputs:
  - `sample.pt` containing positions, atom types, and trajectories; optional `guidance_stats`.
  - Reconstructed SDFs under `result_path/sdf`.
  - If `guidance_log_stats=true`, writes `guidance_stats.json` with the per-step metrics above.

## Configuration Reference (`targetdiff-main-fixed/configs/sampling.yml`)
- `sample.use_guidance` (bool): turn guidance on/off.
- `sample.guidance_ckpt` (path): affinity model checkpoint to load.
- `sample.guidance_scale` (float): multiplier on the guidance drift.
- `sample.guidance_start_step` (int or 0–1 float): max timestep where guidance is applied; floats are interpreted as a fraction of the executed sampling window `[T-num_steps, T)`.
- `sample.guidance_scale_mode` (`var` | `sigma` | `fixed`): whether to multiply the gradient by `sigma^2`, `sigma`, or nothing.
- `sample.noise_scale` (float): scales diffusion noise; <1 increases guidance influence.
- `sample.guidance_log_stats` (bool): enable collecting/saving guidance diagnostics.
- Other sampling knobs (`num_steps`, `sample_num_atoms`, etc.) behave as in the base sampler.

## Minimal Command
```bash
cd targetdiff-main-fixed
python3 scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path path/to/pocket.pdb \
  --result_path outputs_guided \
  --device cuda:0
```
Tune `guidance_scale`, `guidance_start_step`, `guidance_scale_mode`, and `noise_scale` in the config to control when and how strongly the affinity model steers the diffusion trajectories.
