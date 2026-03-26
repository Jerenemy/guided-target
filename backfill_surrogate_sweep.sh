#!/usr/bin/env bash
set -euo pipefail

# Post-hoc surrogate scoring for an already-produced sweep directory.
# This is safe to run while a different sweep process is active because it does
# not modify the currently executing sweep script.

PYTHON=${PYTHON:-/home/jzay/miniforge3/envs/targetdiff_m/bin/python}
REPO=${REPO:-/home/jzay/Desktop/mol_gen/targetdiff-main-fixed}
SWEEP_DIR=${SWEEP_DIR:-}
BASELINE_RUN=${BASELINE_RUN:-$REPO/output/y220c/sampling_run_6}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-128}
GUIDANCE_CKPT=${GUIDANCE_CKPT:-}
GUIDANCE_ARCH=${GUIDANCE_ARCH:-auto}

if [[ -z "$SWEEP_DIR" ]]; then
  echo "Set SWEEP_DIR to the sweep directory you want to backfill." >&2
  exit 1
fi

if [[ ! -d "$SWEEP_DIR" ]]; then
  echo "Sweep directory does not exist: $SWEEP_DIR" >&2
  exit 1
fi

if [[ ! -d "$BASELINE_RUN" ]]; then
  echo "Baseline run does not exist: $BASELINE_RUN" >&2
  exit 1
fi

infer_guidance_ckpt() {
  "$PYTHON" - <<PY
from pathlib import Path
import yaml

sweep_dir = Path(r"$SWEEP_DIR")
for run_dir in sorted(sweep_dir.glob("mode_*")):
    cfg_path = run_dir / "sample.yml"
    if not cfg_path.is_file():
        continue
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    ckpt = (((cfg or {}).get("sample") or {}).get("guidance_ckpt"))
    if ckpt:
        print(ckpt)
        break
PY
}

if [[ -z "$GUIDANCE_CKPT" ]]; then
  GUIDANCE_CKPT=$(infer_guidance_ckpt)
fi

if [[ -z "$GUIDANCE_CKPT" ]]; then
  echo "Could not infer GUIDANCE_CKPT from the sweep. Set GUIDANCE_CKPT explicitly." >&2
  exit 1
fi

echo "Sweep dir     : $SWEEP_DIR"
echo "Baseline run  : $BASELINE_RUN"
echo "Guidance ckpt : $GUIDANCE_CKPT"

score_run() {
  local run_dir="$1"
  local out_csv="$run_dir/surrogate_scores.csv"
  local out_json="$run_dir/surrogate_scores_summary.json"

  if [[ -f "$out_csv" ]]; then
    echo "Surrogate scores already exist for $run_dir"
    return
  fi

  (
    cd "$REPO"
    "$PYTHON" -m scripts.score_guidance_surrogate "$run_dir" \
      --guidance_ckpt "$GUIDANCE_CKPT" \
      --guidance_arch "$GUIDANCE_ARCH" \
      --device "$DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --out_csv "$out_csv" \
      --summary_json "$out_json"
  )
}

score_run "$BASELINE_RUN"

for run_dir in "$SWEEP_DIR"/mode_*; do
  [[ -d "$run_dir" ]] || continue
  score_run "$run_dir"

  "$PYTHON" /home/jzay/Desktop/mol_gen/compare_vina_runs.py \
    "$BASELINE_RUN/surrogate_scores.csv" \
    "$run_dir/surrogate_scores.csv" \
    --metrics surrogate_affinity \
    > "$run_dir/compare_surrogate_vs_baseline.txt"
done

SUMMARY_IN="$SWEEP_DIR/summary.csv"
SUMMARY_OUT="$SWEEP_DIR/summary_with_surrogate.csv"

"$PYTHON" - <<PY
from pathlib import Path
import csv
import json

def load_summary(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def load_json(path):
    with path.open() as f:
        return json.load(f)

sweep_dir = Path(r"$SWEEP_DIR")
summary_in = Path(r"$SUMMARY_IN")
summary_out = Path(r"$SUMMARY_OUT")
baseline_json = load_json(Path(r"$BASELINE_RUN") / "surrogate_scores_summary.json")

rows = load_summary(summary_in)
fieldnames = list(rows[0].keys()) + [
    "surrogate_affinity_mean",
    "surrogate_affinity_q05",
    "surrogate_affinity_q01",
    "delta_surrogate_affinity_mean",
    "delta_surrogate_affinity_q05",
    "delta_surrogate_affinity_q01",
]

augmented = []
for row in rows:
    run_name = row["run_name"]
    summary_json = sweep_dir / run_name / "surrogate_scores_summary.json"
    if summary_json.is_file():
      s = load_json(summary_json)
      row["surrogate_affinity_mean"] = s["surrogate_affinity_mean"]
      row["surrogate_affinity_q05"] = s["surrogate_affinity_q05"]
      row["surrogate_affinity_q01"] = s["surrogate_affinity_q01"]
      row["delta_surrogate_affinity_mean"] = s["surrogate_affinity_mean"] - baseline_json["surrogate_affinity_mean"]
      row["delta_surrogate_affinity_q05"] = s["surrogate_affinity_q05"] - baseline_json["surrogate_affinity_q05"]
      row["delta_surrogate_affinity_q01"] = s["surrogate_affinity_q01"] - baseline_json["surrogate_affinity_q01"]
    augmented.append(row)

with summary_out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(augmented)

print(f"Wrote {summary_out}")
PY

echo "Finished surrogate backfill."
echo "Merged summary: $SUMMARY_OUT"
