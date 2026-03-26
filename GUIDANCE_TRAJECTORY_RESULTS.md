# Guidance Trajectory Results

This note summarizes the corrected interpretation of the late-guidance sweep for `y220c`, with emphasis on whether the surrogate is valid, whether guidance helps during denoising, and what happens if we evaluate earlier trajectory checkpoints instead of only the final sample.

## 1. Corrected Surrogate Interpretation

The earlier conclusion that the surrogate was "not predicting accurately" was wrong.

The issue was in the post-hoc rescoring code, not in the trained surrogate:

- The old offline scorer mixed absolute saved ligand coordinates with a centered pocket frame.
- After fixing that bookkeeping bug in:
  - [`score_guidance_surrogate.py`](targetdiff-main-fixed/scripts/score_guidance_surrogate.py)
  - [`analyze_surrogate_alignment.py`](targetdiff-main-fixed/scripts/analyze_surrogate_alignment.py)
- the surrogate shows strong alignment with the evaluated Vina labels.

Evidence:

- [`surrogate_alignment.csv`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/surrogate_alignment.csv)
- Example row: `mode_sigma__scale_1p0__start_0p1`
  - `pearson_score_vs_score_only = -0.941`
  - `spearman_score_vs_score_only = -0.920`
  - `pearson_score_vs_dock = -0.931`
  - `spearman_score_vs_dock = -0.903`

Interpretation:

- Higher surrogate score corresponds to lower / better Vina values.
- So the surrogate is not failing because it cannot model the endpoint objective.

## 2. What Guidance Does During Sampling

Trajectory scoring was added in:

- [`analyze_guidance_trajectory.py`](targetdiff-main-fixed/scripts/analyze_guidance_trajectory.py)

Outputs:

- [`trajectory_surrogate.csv`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/trajectory_surrogate.csv)
- [`trajectory_surrogate_summary.csv`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/trajectory_surrogate_summary.csv)

The key result is that every guided run improves the surrogate during the late guided window.

Evidence:

- In [`trajectory_surrogate_summary.csv`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/trajectory_surrogate_summary.csv), all guided rows have `delta_from_pre_guidance_mean < 0`.
- Examples:
  - `fixed 0.10`: `-0.1335`
  - `fixed 0.05`: `-0.1116`
  - `sigma 1.0`: `-0.0546`

Interpretation:

- The guidance sign is correct.
- The live integration is not pushing the surrogate in the wrong direction.
- Guidance is doing real local work during the active late window.

## 3. Strongest Guided Run: Temporary Win, Then Late Collapse

The most informative run in this sweep is:

- [`mode_fixed__scale_0p10__start_0p1`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/mode_fixed__scale_0p10__start_0p1)

From [`trajectory_surrogate.csv`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/trajectory_surrogate.csv):

- At step `898` (just before guidance turns on), surrogate affinity mean is about `-2.9019`
- At step `919`, it improves to about `-3.0893`
- At step `949`, it improves further to about `-3.2196`
- At step `979`, it reaches about `-3.2312`
- At final step `999`, it falls back to about `-3.0354`

Compared to the unguided baseline [`sampling_run_6`](targetdiff-main-fixed/output/y220c/sampling_run_6):

- Baseline at step `949`: about `-3.1231`
- Guided `fixed 0.10` at step `949`: about `-3.2196`
- Baseline at step `979`: about `-3.1271`
- Guided `fixed 0.10` at step `979`: about `-3.2312`
- Baseline final: about `-3.1347`
- Guided `fixed 0.10` final: about `-3.0354`

Interpretation:

- The guided run really does beat baseline on the surrogate around `t ≈ 50..20`.
- The final denoising steps then give back that gain.

## 4. Early Checkpoints Also Improve Real Vina

To test whether the surrogate-only trajectory result was real, I evaluated earlier saved trajectory states with:

- `--eval_step 949`
- `--eval_step 979`

for:

- [`mode_fixed__scale_0p10__start_0p1`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/mode_fixed__scale_0p10__start_0p1)

Outputs:

- [`metrics_949_extracted.csv`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/mode_fixed__scale_0p10__start_0p1/eval_results/metrics_949_extracted.csv)
- [`metrics_979_extracted.csv`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/mode_fixed__scale_0p10__start_0p1/eval_results/metrics_979_extracted.csv)

From [`log.txt`](targetdiff-main-fixed/output/y220c_guided_sweep/20260325_225115/mode_fixed__scale_0p10__start_0p1/eval_results/log.txt):

- Final `eval_step=-1`
  - `mol_stable = 0.800`
  - `vina_score_only mean = -2.976`
  - `vina_minimize mean = -3.414`
  - `vina_dock mean = -4.373`

- Step `949`
  - `mol_stable = 0.080`
  - `vina_score_only mean = -3.031`
  - `vina_minimize mean = -3.356`
  - `vina_dock mean = -4.457`

- Step `979`
  - `mol_stable = 0.080`
  - `vina_score_only mean = -3.063`
  - `vina_minimize mean = -3.456`
  - `vina_dock mean = -4.506`

Interpretation:

- The surrogate win at late intermediate checkpoints is not fake.
- Evaluating earlier checkpoints really does improve Vina, especially at step `979`.
- The problem is that those earlier checkpoints are much less stable.

## 5. Position Updates Cause the Late Loss

I also checked whether the late collapse from step `979` to final step `999` was caused by position updates or atom-type updates.

Result:

- Using step-`979` positions with final atom types gave essentially the same surrogate score as step `979`.
- Using final positions with step-`979` atom types gave essentially the same surrogate score as the final sample.

Interpretation:

- The degradation from `979 -> 999` is positional.
- It is not mainly caused by the unguided discrete atom-type updates.

## 6. What This Means

The current picture is:

- The surrogate itself is good.
- The guidance direction and integration sign are correct.
- Guidance helps during the late active window.
- Stronger fixed guidance can temporarily beat the unguided baseline on the surrogate and also improve Vina.
- The last denoising steps then trade away some of that docking improvement in exchange for much better stability.

So the core issue is not "bad surrogate" and not "wrong sign".

The real issue is a docking-vs-stability tradeoff in the final denoising region.

## 7. Practical Next Step

The most promising change is to stop or taper guidance before the final few denoising steps.

Concretely:

1. Add a `guidance_end_step` or equivalent cutoff.
2. Try windows like:
   - active for `t in [100, 20]`
   - off for `t < 20`
3. Re-run the strongest fixed setting first:
   - `guidance_scale_mode = fixed`
   - `guidance_scale = 0.10`
   - `guidance_start_step = 0.1`
4. Evaluate both:
   - final step
   - best late intermediate step

Rationale:

- The trajectory evidence suggests guidance is genuinely useful before the final cleanup steps.
- The early-checkpoint evaluations suggest those late intermediate states can have better docking than the final sample.
- The right fix is probably not to remove guidance, but to control when it stops.
