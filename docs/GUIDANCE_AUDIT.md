# Guidance Audit And Fix Plan

This note summarizes the current guidance implementation, the main mistakes or inconsistencies I found, and the best fix path.

Note: links into notebooks point to raw `.ipynb` JSON lines, because the training code currently lives in notebooks rather than `.py` modules.

## Executive Summary

- The current sampler is not doing classifier-free guidance. It is doing external expert-gradient guidance from a separately trained affinity predictor.
- The highest-risk issue is in expert training: the fixed training notebook feeds the model the current generated pose `pred_pos`, but by default supervises it with `vina["dock"]`, which is the score of a re-docked pose rather than the current pose.
- Two additional implementation issues are likely to hurt guidance quality:
  - the sampler removes rigid-body translation from the expert gradient;
  - the guidance wrapper hardcodes ligand atom decoding to `add_aromatic`.
- Guidance is scheduled late by default: with the current config, it only acts in the final ~10% of reverse denoising steps.
- There is theoretical cause for concern if the goal is proper conditional guidance. The expert is not trained as `log p(cond | x_t)` and is not noise- or timestep-conditioned.
- I did not find a checkpoint/runtime architecture mismatch. The shipped `affinity_latest.pt` loads cleanly into the runtime [`EquivariantAffinityModel`](targetdiff-main-fixed/models/guidance.py#L163).

## Linked Findings

### 1. The method is mislabeled as classifier-free guidance

Evidence:

- The README says this is classifier-free guidance: [`GUIDANCE_README.md:3`](GUIDANCE_README.md#L3)
- The sampler actually:
  - runs the diffusion model once: [`molopt_score_model.py:684`](targetdiff-main-fixed/models/molopt_score_model.py#L684)
  - evaluates a separate guidance model: [`molopt_score_model.py:724`](targetdiff-main-fixed/models/molopt_score_model.py#L724)
  - takes `autograd.grad` with respect to ligand coordinates: [`molopt_score_model.py:733`](targetdiff-main-fixed/models/molopt_score_model.py#L733)
  - adds that drift into the reverse step: [`molopt_score_model.py:751`](targetdiff-main-fixed/models/molopt_score_model.py#L751)

Why this matters:

- The documentation currently describes the wrong guidance family, which makes the design harder to reason about and compare against the diffusion literature.

Recommended fix:

- Rename the method everywhere from "classifier-free guidance" to "expert-gradient guidance", "classifier guidance", or "energy guidance with an auxiliary affinity model".
- Update the README first, then update config comments and notebook text to match the actual algorithm.

### 2. The expert is trained on the wrong target for a pose-gradient method

Evidence:

- The fixed training dataset takes the current generated coordinates from `pred_pos`: [`guidance_wrapper_fixed.ipynb:186-190`](guidance_wrapper_fixed.ipynb#L186)
- The label is taken from `vina[vina_mode][0]["affinity"]`: [`guidance_wrapper_fixed.ipynb:188-199`](guidance_wrapper_fixed.ipynb#L188)
- The fixed notebook default is `VINA_MODE = "dock"`: [`guidance_wrapper_fixed.ipynb:239`](guidance_wrapper_fixed.ipynb#L239)
- During evaluation, `pred_pos` is stored directly in the result record: [`evaluate_diffusion.py:161-170`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L161)
- The dock score is computed separately by Vina: [`evaluate_diffusion.py:136-144`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L136)
- In `dock` mode, Vina performs a search and returns the best docked pose/energy, not the score of the current pose: [`docking_vina.py:128-150`](targetdiff-main-fixed/utils/evaluation/docking_vina.py#L128)

Why this matters:

- The expert sees pose `x`, but the label is the docking score of a different pose produced after Vina search.
- That breaks the interpretation of the learned scalar as a local energy of the current coordinates.
- Once you use `grad_x score(x)` at sampling time, this mismatch becomes a first-order problem rather than just a noisy regression target.

Best fix:

- Retrain the expert on `vina["score_only"]`, not `vina["dock"]`, when the model input is `pred_pos`.
- `score_only` is the clean pose-conditioned target among the available Vina outputs. It matches the object you actually need for gradient guidance: a scalar attached to the current geometry.

Second-best fix:

- If you insist on using `dock`, then train on the returned docked pose rather than `pred_pos`.
- Even that is still less coherent for sampling-time gradients than `score_only`, because the optimizer/search step is not part of the differentiable model seen at runtime.

### 3. The sampler removes rigid-body translation from the expert gradient

Evidence:

- The guidance gradient is mean-centered per ligand graph before it is applied: [`molopt_score_model.py:735-737`](targetdiff-main-fixed/models/molopt_score_model.py#L735)

Why this matters:

- For docking-style guidance, rigid-body translation is often exactly what the expert should influence.
- Removing the center-of-mass component prevents the expert from directly moving the ligand as a whole within the pocket.
- This is especially suspicious because the current guidance model is supposed to judge ligand-pocket geometry, not just internal ligand shape.

Recommended fix:

- Add a config flag such as `guidance_remove_translation: false`.
- Default it to `false`.
- Only keep the mean-subtraction path as an optional experiment.

### 4. Ligand atom decoding is hardcoded to `add_aromatic`

Evidence:

- The guidance wrapper decodes sampled ligand types with `mode='add_aromatic'`: [`guidance.py:30-33`](targetdiff-main-fixed/models/guidance.py#L30)
- Reconstruction in the pocket sampler also hardcodes `add_aromatic`: [`sample_for_pocket.py:140-142`](targetdiff-main-fixed/scripts/sample_for_pocket.py#L140)
- The diffusion checkpoint separately carries `ligand_atom_mode`: [`sample_for_pocket.py:59-60`](targetdiff-main-fixed/scripts/sample_for_pocket.py#L59)

Why this matters:

- The current repo default uses `add_aromatic`, so this works today.
- But the implementation is brittle and can silently become wrong if a checkpoint uses `basic` or `full`.

Recommended fix:

- Thread `ligand_atom_mode` from the diffusion checkpoint into:
  - [`GuidedAffinityWrapper`](targetdiff-main-fixed/models/guidance.py#L8)
  - guided reconstruction in [`sample_for_pocket.py`](targetdiff-main-fixed/scripts/sample_for_pocket.py#L140)
- Do not hardcode the decoding mode in runtime code.

### 5. The training split is optimistic for validation

Evidence:

- The fixed notebook uses a plain random split over the pooled dataset: [`guidance_wrapper_fixed.ipynb:276-284`](guidance_wrapper_fixed.ipynb#L276)

Why this matters:

- If the dataset contains many similar samples from the same pocket or the same generation/evaluation run, train and validation are strongly correlated.
- That does not prevent the model from working as a pocket-specific expert, but it does make validation loss look better than true held-out performance.

Recommended fix:

- Split by source `.pt` file, pocket, or generation batch instead of by individual sample.
- If you want a stronger estimate, also consider scaffold-based grouping over generated ligands.

### 6. Guidance is scheduled late by default

Evidence:

- The sampler iterates timesteps in reverse: [`molopt_score_model.py:681-682`](targetdiff-main-fixed/models/molopt_score_model.py#L681)
- Fractional `guidance_start_step` values are converted into a timestep index over the executed window: [`molopt_score_model.py:663-671`](targetdiff-main-fixed/models/molopt_score_model.py#L663)
- Guidance is only active when `i <= guidance_start_idx`: [`molopt_score_model.py:719-720`](targetdiff-main-fixed/models/molopt_score_model.py#L719)
- The current config sets `guidance_start_step: 0.1`: [`sampling.yml:17`](targetdiff-main-fixed/configs/sampling.yml#L17)

What this means:

- Since reverse diffusion runs from large timestep to small timestep, smaller `i` means later denoising.
- Therefore `guidance_start_step: 0.1` means guidance acts only in the final roughly 10% of reverse steps.
- If `num_steps=1000`, guidance starts around step `100` and continues down to `0`.

Critique:

- Late-only guidance is a sensible defensive choice given the current expert, because the expert was not trained on very noisy states and is less likely to behave sensibly early in the chain.
- However, the name `guidance_start_step` is easy to misread as “start guiding early,” when in practice smaller values push guidance later.
- The current default also uses `guidance_scale_mode: fixed`, so once guidance turns on, it is not tempered by `sigma` or `sigma^2`: [`sampling.yml:18-19`](targetdiff-main-fixed/configs/sampling.yml#L18), [`molopt_score_model.py:739-745`](targetdiff-main-fixed/models/molopt_score_model.py#L739)

Recommended fix:

- Rename this config to something like `guidance_max_timestep` or `guidance_end_fraction` to match its actual meaning.
- Keep the default guidance window late unless and until the expert is retrained on noisy `x_t` states.
- If the goal is guidance closer to the standard DDPM derivation, prefer `guidance_scale_mode: var` rather than `fixed`.

### 7. This is not proper `log p(cond | x_t)` guidance, so there is real distribution-shift risk

Evidence:

- The expert is trained by plain MSE regression on a raw scalar affinity target: [`guidance_wrapper_fixed.ipynb:492-495`](guidance_wrapper_fixed.ipynb#L492), [`guidance_wrapper_fixed.ipynb:530-533`](guidance_wrapper_fixed.ipynb#L530)
- The runtime guidance model does not take timestep `t` as an input:
  - wrapper signature: [`guidance.py:23`](targetdiff-main-fixed/models/guidance.py#L23)
  - affinity model forward: [`guidance.py:196-227`](targetdiff-main-fixed/models/guidance.py#L196)
- During sampling, the gradient is taken with respect to the current intermediate ligand coordinates `x_t`: [`molopt_score_model.py:722-733`](targetdiff-main-fixed/models/molopt_score_model.py#L722)
- The expert training data comes from evaluated sample trajectories with default `eval_step=-1`, i.e. final denoised poses rather than generic noisy intermediate states: [`evaluate_diffusion.py:39`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L39), [`evaluate_diffusion.py:81`](targetdiff-main-fixed/scripts/evaluate_diffusion.py#L81)
- The default runtime update uses `guidance_scale_mode: fixed`, not the variance-scaled form that would most closely match classifier guidance: [`sampling.yml:18-19`](targetdiff-main-fixed/configs/sampling.yml#L18), [`molopt_score_model.py:739-745`](targetdiff-main-fixed/models/molopt_score_model.py#L739)

Why this matters:

- In the diffusion-guidance derivation, the theoretically clean term is proportional to `∇_{x_t} log p(cond | x_t)`.
- The current expert is not trained to estimate that quantity:
  - it predicts raw affinity, not a conditional log-probability;
  - it is not conditioned on timestep/noise level;
  - it is trained on final poses, not on the full `x_t` distribution encountered during sampling.
- Because of that, the guidance term is better interpreted as an arbitrary learned energy/reward gradient than as principled conditional guidance.
- That means there is genuine risk of pushing samples off the realistic support of the unguided model, especially if guidance is made stronger or applied earlier.

Mitigating factors:

- Training on TargetDiff-generated samples at least keeps the expert closer to the unguided model’s pose manifold than training on unrelated external data.
- Late-only scheduling also reduces the mismatch, because the late states are closer to the clean-pose regime the expert was trained on.
- Those help empirically, but they do not restore the theoretical guarantee associated with `∇ log p(cond | x_t)`.

Best fix if you want principled conditional guidance:

- Train a timestep-conditioned expert on noisy states sampled from the same forward diffusion process.
- Use an objective that defines a proper conditional likelihood or conditional log-density.
- Then apply the guidance in variance-scaled form, i.e. the equivalent of `guidance_scale_mode: var`.

Practical fix if you want to keep the current reward-guidance style:

- Treat the method honestly as reward/energy guidance, not conditional guidance.
- Keep the guidance window late.
- Keep the guidance scale conservative.
- Validate aggressively for reconstruction failures, unrealistic geometries, and drift away from the unguided distribution.

## Best Overall Fix

If I were fixing this pipeline for actual guidance quality rather than just cleaning up the code, I would do the following in order:

1. Change expert training to use `vina["score_only"]` as the label for the current `pred_pos`.
2. If the goal is principled conditional guidance, retrain the expert on noisy `x_t` states with timestep conditioning instead of using a clean-pose reward regressor.
3. Remove center-of-mass gradient subtraction by default, or at least make it configurable and disabled by default.
4. Pass `ligand_atom_mode` through the guidance wrapper and reconstruction path instead of hardcoding `add_aromatic`.
5. Rename the method everywhere so the documentation matches the implementation.
6. Tighten the validation split so expert quality is measured more honestly.

The first item is the most important by far. If the expert is not trained on a scalar that is actually attached to the current pose, then using its coordinate gradient during sampling is conceptually inconsistent no matter how well the rest of the implementation is cleaned up. The second item is what matters if you specifically want the guidance term to behave like `∇ log p(cond | x_t)` rather than as an unconstrained reward gradient.

## Concrete Edit Plan

### Documentation

- Update [`GUIDANCE_README.md`](GUIDANCE_README.md) to describe the method as expert-gradient guidance rather than classifier-free guidance.

### Runtime code

- In [`targetdiff-main-fixed/models/guidance.py`](targetdiff-main-fixed/models/guidance.py):
  - add a `ligand_atom_mode` argument to `GuidedAffinityWrapper`;
  - decode ligand atom types using that mode.
- In [`targetdiff-main-fixed/models/molopt_score_model.py`](targetdiff-main-fixed/models/molopt_score_model.py):
  - add a config-driven switch for translation removal;
  - default to keeping the raw expert gradient.
- In [`targetdiff-main-fixed/scripts/sample_for_pocket.py`](targetdiff-main-fixed/scripts/sample_for_pocket.py):
  - pass `ligand_atom_mode` into the wrapper;
  - use the same mode for reconstruction.
- In [`targetdiff-main-fixed/configs/sampling.yml`](targetdiff-main-fixed/configs/sampling.yml):
  - add `guidance_remove_translation: false`;
  - consider renaming `guidance_start_step` to reflect that it gates late timesteps, not early ones;
  - consider adding an explicit `guidance_label_mode: score_only` note in comments so the intended training target is recorded.

### Training code

- In [`guidance_wrapper_fixed.ipynb`](guidance_wrapper_fixed.ipynb):
  - change [`VINA_MODE = "dock"`](guidance_wrapper_fixed.ipynb#L239) to `score_only`;
  - keep the input coordinates tied to the same `pred_pos`;
  - if you want principled diffusion guidance, generate noisy `x_t` training inputs and pass timestep/noise level into the expert;
  - replace the naive random split with a grouped split if the dataset comes from multiple runs or sources.

## Bottom Line

The current system is usable as an expert-gradient guidance prototype, but it should not currently be treated as principled conditional diffusion guidance. Its main conceptual flaws are that the expert is trained on redocked labels while being differentiated with respect to the undocked generated pose, and that it is not trained as a timestep-conditioned `log p(cond | x_t)` model. The best fix is to retrain the expert on `score_only` for the same pose it receives as input, and if conditional-guidance fidelity matters, retrain it on noisy `x_t` states with timestep conditioning before using it as a true guidance term.
