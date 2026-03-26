# Guidance Model Training Pipeline

This document explains how the guidance model is trained in the current implementation, what the input and output datasets are, where the dataset-creation pipeline lives, and what should be changed so the expert integrates properly into TargetDiff.

## Short Answer

- Yes, the dataset-creation pipeline is present in the repo.
- Yes, for the current TargetDiff runs that use `--docking_mode vina_dock`, the replacement label you want (`score_only`) is already computed and stored alongside `dock`.
- That means you usually do **not** need to reconstruct molecules or rerun docking from scratch just to switch from `vina_dock` to `vina_score_only`.
- The current expert-training notebook still defaults to `VINA_MODE = "dock"`, which is the wrong target for direct pose-gradient guidance.

## Current Pipeline Overview

The current guidance-training flow is:

1. Sample ligands with TargetDiff and save `sample.pt`.
2. Evaluate those sampled ligands with Vina and save `metrics_-1.pt`.
3. Optionally export `metrics_extracted.csv` / `metrics_extracted.sdf`.
4. Train the guidance model by loading `metrics_-1.pt` files directly into the notebook dataset loader.

The important point is that the training notebook does **not** build the dataset from raw PDB/SDF plus fresh docking. It builds the dataset from already-evaluated TargetDiff outputs stored in `metrics_*.pt`.

## Step 1: Sampling Output

The TargetDiff sampling entry point is:

- [`targetdiff-main-fixed/scripts/sample_for_pocket.py`](targetdiff-main-fixed/scripts/sample_for_pocket.py)

This produces:

- `sample.pt`

That file contains:

- the pocket data;
- final sampled ligand positions/types;
- full position/type trajectories across denoising steps.

Relevant code:

- saving `sample.pt`: [`sample_for_pocket.py:159-163`](targetdiff-main-fixed/scripts/sample_for_pocket.py#L159)

## Step 2: Evaluation Output

The evaluation step is:

- [`targetdiff-main-fixed/scripts/evaluate_diffusion.py`](targetdiff-main-fixed/scripts/evaluate_diffusion.py)

For each reconstructed molecule, it stores a result record with:

- `mol`
- `smiles`
- `pred_pos`
- `pred_v`
- `chem_results`
- `vina`

Relevant code:

- result record construction: [`evaluate_diffusion.py:161-169`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L161)
- writing `metrics_-1.pt`: [`evaluate_diffusion.py:226-231`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L226)

### What Vina labels are computed

If evaluation is run with `--docking_mode vina_dock` or `vina_score`, the script computes:

- `score_only`
- `minimize`
- optionally `dock` when using `vina_dock`

Relevant code:

- `score_only` and `minimize` are always computed for `vina_score` and `vina_dock`: [`evaluate_diffusion.py:124-140`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L124)
- `dock` is added only when `docking_mode == 'vina_dock'`: [`evaluate_diffusion.py:142-144`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L142)

So the `vina` field in `metrics_-1.pt` looks like:

```python
{
    "score_only": [...],
    "minimize": [...],
    "dock": [...],   # only for vina_dock
}
```

## Step 3: CSV/SDF Export

There is also a small export script:

- [`extract_metrics.py`](extract_metrics.py)

It reads `metrics_-1.pt` and writes:

- `metrics_extracted.csv`
- optional `metrics_extracted.sdf`

Relevant code:

- expected input file: [`extract_metrics.py:59-67`](extract_metrics.py#L59)
- exported columns: [`extract_metrics.py:72-85`](extract_metrics.py#L72)
- Vina columns are `vina_score_only`, `vina_minimize`, `vina_dock`

## Step 4: Guidance Training Input Dataset

The current training notebook is:

- [`guidance_wrapper_fixed.ipynb`](guidance_wrapper_fixed.ipynb)

Its dataset class is:

- `LigandPocketFromPT`

Relevant code:

- dataset definition: [`guidance_wrapper_fixed.ipynb:135-217`](guidance_wrapper_fixed.ipynb#L135)

### Input dataset used by the notebook

The notebook expects either:

- `PT_DIR`: a directory of `metrics_*.pt` files, or
- `PT_PATH`: a single `metrics_*.pt` file

Relevant code:

- path setup: [`guidance_wrapper_fixed.ipynb:235-260`](guidance_wrapper_fixed.ipynb#L235)

Each training example is constructed from one record in `all_results`:

- the ligand coordinates come from `rec["pred_pos"]`
- the label comes from `rec["vina"][vina_mode][0]["affinity"]`
- the pocket comes from a fixed pocket PDB loaded separately

Relevant code:

- record loading: [`guidance_wrapper_fixed.ipynb:184-199`](guidance_wrapper_fixed.ipynb#L184)

### Current default label

The notebook currently sets:

- `VINA_MODE = "dock"`

Relevant code:

- [`guidance_wrapper_fixed.ipynb:239`](guidance_wrapper_fixed.ipynb#L239)

That means the expert is currently trained on:

- input: current generated pose `pred_pos`
- target: docked-pose affinity

This is the main issue.

## Input Dataset And Output Dataset

### Input dataset to guidance training

On disk:

- one or more `metrics_*.pt` files produced by `evaluate_diffusion.py`

Logical structure:

- top-level key: `all_results`
- each entry contains:
  - current sampled pose `pred_pos`
  - reconstructed RDKit molecule `mol`
  - Vina results `vina`
  - optional chemistry annotations `chem_results`

### Output dataset / artifacts from guidance training

In memory:

- PyG `Data` objects with fields:
  - `pos`
  - `z`
  - `y`
  - `node_type`
  - optional `qed`, `sa`, `logp`

Relevant code:

- PyG object construction: [`guidance_wrapper_fixed.ipynb:190-217`](guidance_wrapper_fixed.ipynb#L190)

On disk after training:

- `checkpoints/latest.pt`

Relevant code:

- checkpoint save: [`guidance_wrapper_fixed.ipynb:505-547`](guidance_wrapper_fixed.ipynb#L505)

## Dataset-Creation Pipeline: Is It Already There?

Yes.

The full current pipeline already exists:

- sampling: [`sample_for_pocket.py`](targetdiff-main-fixed/scripts/sample_for_pocket.py)
- evaluation + Vina labeling: [`evaluate_diffusion.py`](targetdiff-main-fixed/scripts/evaluate_diffusion.py)
- CSV export: [`extract_metrics.py`](extract_metrics.py)
- training dataset loader: [`guidance_wrapper_fixed.ipynb`](guidance_wrapper_fixed.ipynb)

What is **missing** is a dedicated, script-based relabeling utility for guidance training. Right now the notebook just reads whatever `vina_mode` you point it at.

## Was The Desired Replacement Label Already Computed?

Yes, for the current runs that used `vina_dock`.

Evidence in code:

- your run script uses `--docking_mode vina_dock`: [`run_eval_extract_targetdiff.sh:26-29`](run_eval_extract_targetdiff.sh#L26)
- `vina_dock` evaluation stores `score_only`, `minimize`, and `dock`: [`evaluate_diffusion.py:136-144`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L136)

Evidence from local outputs:

- I checked `targetdiff-main-fixed/output/y220c_guided/sampling_run_0/eval_results/metrics_-1.pt`
- its `vina` dict contains:
  - `score_only`
  - `minimize`
  - `dock`
- I also checked `targetdiff-main-fixed/output/y220c_guided/sampling_run_0/eval_results/metrics_extracted.csv`
- its columns include:
  - `vina_score_only`
  - `vina_minimize`
  - `vina_dock`

So for those runs, the replacement label is already available.

## Do We Need To Relabel Or Reconstruct The Dataset?

### Best case

If your existing dataset comes from `metrics_-1.pt` files created with `--docking_mode vina_dock` or `vina_score`, you do **not** need to reconstruct the molecules or rerun docking.

You can simply:

1. reuse the same `metrics_-1.pt` files;
2. switch the notebook label mode from `dock` to `score_only`;
3. retrain the expert.

### When you do need to rebuild

You need to regenerate dataset files if:

- the old runs were evaluated with `qvina` or `none`, so `score_only` was never computed;
- the old `metrics_*.pt` files are not readable in the current environment;
- you want a cleaner dataset definition, such as one that strips RDKit objects and stores only the training-relevant fields.

### Important local note

There are historical files under:

- `targetdiff-main-fixed/scripts/guidance_training/metrics_dir/`

Those files exist, but in the current `targetdiff_m` environment they did not load cleanly because of pickle/RDKit compatibility issues. That means:

- the pipeline itself is present;
- but some old cached training data may be brittle across environments.

In practice, regenerating `metrics_-1.pt` from `sample.pt` with the current environment may be easier than trying to preserve older caches.

## How To Fix The Training So It Integrates Properly With TargetDiff

### Minimal fix

Keep the current dataset pipeline, but change the training label from:

- `vina["dock"]`

to:

- `vina["score_only"]`

That requires only a small change in the training notebook:

- change `VINA_MODE = "dock"` to `score_only`: [`guidance_wrapper_fixed.ipynb:239`](guidance_wrapper_fixed.ipynb#L239)

This is the lowest-friction path and is compatible with the `metrics_-1.pt` files you are already generating from current TargetDiff runs.

### Better integration fix

Move the notebook logic into a small script that:

1. reads `metrics_-1.pt`;
2. extracts only the fields needed for expert training;
3. writes a clean training dataset without pickled RDKit objects;
4. lets you choose the label mode explicitly (`score_only`, `minimize`, `dock`);
5. defaults to `score_only`.

That would make the expert-training dataset:

- easier to reload across environments;
- easier to relabel;
- easier to version and audit.

### Best theoretical fix

If you want the expert to behave like proper diffusion guidance rather than reward guidance:

1. build training examples from noisy `x_t` states, not just final `pred_pos`;
2. pass timestep/noise level into the expert;
3. train against a target attached to the current noisy state;
4. use that expert with variance-scaled guidance in the sampler.

That is a larger change. The minimal TargetDiff-compatible fix is still to swap `dock` for `score_only`.

## Recommended Next Step

The most practical next step is:

1. stop using `vina_dock` as the expert target;
2. reuse the existing `metrics_-1.pt` files from `vina_dock` runs;
3. retrain the expert with `vina["score_only"]`;
4. only regenerate data for runs that do not already contain `score_only` or for historical caches that no longer load cleanly.

## Bottom Line

The current guidance-training dataset is built from evaluated TargetDiff outputs, not from a separate raw-data pipeline. The relabeling path is already available for current runs, because `score_only` is stored in the same `metrics_-1.pt` files that currently feed the notebook. So the quickest correct fix is to reuse those files and change the expert target from `dock` to `score_only`, rather than reconstructing the whole dataset from scratch.
