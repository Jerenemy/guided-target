#!/usr/bin/env bash
set -euo pipefail

# Guidance sweep for TargetDiff.
# This script writes one temp config per run, executes sampling/evaluation/extraction,
# compares each run against an unguided baseline, and appends a summary row.

# ---------------- configurable bits ----------------
PYTHON=${PYTHON:-/home/jzay/miniforge3/envs/targetdiff_m/bin/python}
REPO=${REPO:-/home/jzay/Desktop/mol_gen/targetdiff-main-fixed}
P53_NAME=${P53_NAME:-y220c}
POCKET_PDB=${POCKET_PDB:-/home/jzay/Desktop/mol_gen/data/input/${P53_NAME}/${P53_NAME}_pocket10.pdb}
BASE_CONFIG=${BASE_CONFIG:-$REPO/configs/sampling.yml}
BASELINE_RUN=${BASELINE_RUN:-$REPO/output/${P53_NAME}/sampling_run_6}
SWEEP_ROOT=${SWEEP_ROOT:-$REPO/output/${P53_NAME}_guided_sweep}

NUM_SAMPLES=${NUM_SAMPLES:-100}
BATCH_SIZE=${BATCH_SIZE:-50}
DEVICE=${DEVICE:-cuda:0}
DOCKING_MODE=${DOCKING_MODE:-vina_dock}

# A good first sweep is the gap between:
# - run_16: fixed 3.0 (far too strong)
# - run_17: sigma 0.1 (likely too weak)
START_STEPS=(${START_STEPS:-0.05 0.10})
MODES=(${MODES:-sigma fixed})
SIGMA_SCALES=(${SIGMA_SCALES:-0.5 1.0 2.0 5.0})
FIXED_SCALES=(${FIXED_SCALES:-0.01 0.03 0.05 0.10})
VAR_SCALES=(${VAR_SCALES:-25 50 100})

SKIP_EXISTING=${SKIP_EXISTING:-1}
# ---------------------------------------------------

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

require_file "$PYTHON"
require_file "$REPO"
require_file "$POCKET_PDB"
require_file "$BASE_CONFIG"
require_file "$BASELINE_RUN"

BASELINE_CSV="$BASELINE_RUN/eval_results/metrics_extracted.csv"
require_file "$BASELINE_CSV"

mkdir -p "$SWEEP_ROOT"
STAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="$SWEEP_ROOT/$STAMP"
TMP_CFG_DIR="$SWEEP_DIR/configs"
mkdir -p "$TMP_CFG_DIR"

SUMMARY_CSV="$SWEEP_DIR/summary.csv"
MANIFEST_TXT="$SWEEP_DIR/manifest.txt"

cat > "$MANIFEST_TXT" <<EOF
timestamp=$STAMP
repo=$REPO
pocket_pdb=$POCKET_PDB
base_config=$BASE_CONFIG
baseline_run=$BASELINE_RUN
baseline_csv=$BASELINE_CSV
python=$PYTHON
num_samples=$NUM_SAMPLES
batch_size=$BATCH_SIZE
device=$DEVICE
docking_mode=$DOCKING_MODE
start_steps=${START_STEPS[*]}
modes=${MODES[*]}
sigma_scales=${SIGMA_SCALES[*]}
fixed_scales=${FIXED_SCALES[*]}
var_scales=${VAR_SCALES[*]}
EOF

echo "run_name,mode,scale,start_step,num_samples,mol_stable,atm_stable,recon_success,eval_success,complete,n_eval,vina_score_mean,vina_score_q05,vina_score_q01,vina_min_mean,vina_min_q05,vina_min_q01,vina_dock_mean,vina_dock_q05,vina_dock_q01,delta_dock_mean,delta_dock_q05,delta_dock_q01,guidance_over_noise_mean_nonzero,guidance_over_noise_q50_nonzero,guidance_over_noise_q90_nonzero,guidance_term_norm_mean_nonzero,noise_term_norm_mean_nonzero" > "$SUMMARY_CSV"

pick_scales() {
  local mode="$1"
  case "$mode" in
    sigma) echo "${SIGMA_SCALES[*]}" ;;
    fixed) echo "${FIXED_SCALES[*]}" ;;
    var)   echo "${VAR_SCALES[*]}" ;;
    *)
      echo "Unknown mode: $mode" >&2
      exit 1
      ;;
  esac
}

sanitize() {
  echo "$1" | tr '.-' 'pd'
}

write_config() {
  local out_cfg="$1"
  local mode="$2"
  local scale="$3"
  local start_step="$4"

  "$PYTHON" - <<PY
from pathlib import Path
import yaml

base_cfg = Path(r"$BASE_CONFIG")
out_cfg = Path(r"$out_cfg")
cfg = yaml.safe_load(base_cfg.read_text())

cfg["sample"]["guidance_scale_mode"] = "$mode"
cfg["sample"]["guidance_scale"] = float("$scale")
cfg["sample"]["guidance_start_step"] = float("$start_step")
cfg["sample"]["guidance_log_stats"] = True
cfg["sample"]["use_guidance"] = True
cfg["sample"]["num_samples"] = int("$NUM_SAMPLES")

out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

append_summary_row() {
  local run_name="$1"
  local mode="$2"
  local scale="$3"
  local start_step="$4"
  local run_dir="$5"

  "$PYTHON" - <<PY
from pathlib import Path
import csv
import json
import math
import re

def q(xs, frac):
    if not xs:
        return math.nan
    xs = sorted(xs)
    idx = frac * (len(xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    w = idx - lo
    return xs[lo] * (1 - w) + xs[hi] * w

def mean(xs):
    return sum(xs) / len(xs) if xs else math.nan

log_path = Path(r"$run_dir") / "eval_results" / "log.txt"
csv_path = Path(r"$run_dir") / "eval_results" / "metrics_extracted.csv"
stats_path = Path(r"$run_dir") / "guidance_stats.json"
baseline_csv = Path(r"$BASELINE_CSV")
summary_csv = Path(r"$SUMMARY_CSV")

log_text = log_path.read_text()
metric_pat = {
    "mol_stable": r"mol_stable:\s*([0-9.]+)",
    "atm_stable": r"atm_stable:\s*([0-9.]+)",
    "recon_success": r"recon_success:\s*([0-9.]+)",
    "eval_success": r"eval_success:\s*([0-9.]+)",
    "complete": r"complete:\s*([0-9.]+)",
}
vals = {}
for key, pat in metric_pat.items():
    m = re.search(pat, log_text)
    vals[key] = float(m.group(1)) if m else math.nan

def load_col(path, col):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    xs = []
    for r in rows:
        val = r.get(col)
        if val is None or val == "":
            continue
        xs.append(float(val))
    return xs

score = load_col(csv_path, "vina_score_only")
vmin = load_col(csv_path, "vina_minimize")
dock = load_col(csv_path, "vina_dock")
base_dock = load_col(baseline_csv, "vina_dock")

with stats_path.open() as f:
    stats = json.load(f)
nonzero = [r for r in stats if float(r.get("noise_term_norm_mean", 0.0)) > 0]
gor = [float(r["guidance_over_noise"]) for r in nonzero]
gtn = [float(r["guidance_term_norm_mean"]) for r in nonzero]
ntn = [float(r["noise_term_norm_mean"]) for r in nonzero]

row = [
    "$run_name",
    "$mode",
    "$scale",
    "$start_step",
    str($NUM_SAMPLES),
    vals["mol_stable"],
    vals["atm_stable"],
    vals["recon_success"],
    vals["eval_success"],
    vals["complete"],
    len(dock),
    mean(score), q(score, 0.05), q(score, 0.01),
    mean(vmin), q(vmin, 0.05), q(vmin, 0.01),
    mean(dock), q(dock, 0.05), q(dock, 0.01),
    mean(dock) - mean(base_dock),
    q(dock, 0.05) - q(base_dock, 0.05),
    q(dock, 0.01) - q(base_dock, 0.01),
    mean(gor), q(gor, 0.50), q(gor, 0.90),
    mean(gtn), mean(ntn),
]

with summary_csv.open("a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(row)
PY
}

run_one() {
  local mode="$1"
  local scale="$2"
  local start_step="$3"

  local mode_tag scale_tag start_tag run_name out_dir cfg_path compare_txt
  mode_tag=$(sanitize "$mode")
  scale_tag=$(sanitize "$scale")
  start_tag=$(sanitize "$start_step")
  run_name="mode_${mode_tag}__scale_${scale_tag}__start_${start_tag}"
  out_dir="$SWEEP_DIR/$run_name"
  cfg_path="$TMP_CFG_DIR/${run_name}.yml"
  compare_txt="$out_dir/compare_vs_baseline.txt"

  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/eval_results/metrics_extracted.csv" ]]; then
    echo "Skipping existing run: $run_name"
    return
  fi

  mkdir -p "$out_dir"
  write_config "$cfg_path" "$mode" "$scale" "$start_step"

  echo
  echo "=== Running $run_name ==="
  echo "config=$cfg_path"
  echo "out_dir=$out_dir"

  (
    cd "$REPO"
    "$PYTHON" -m scripts.sample_for_pocket "$cfg_path" \
      --pdb_path "$POCKET_PDB" \
      --result_path "$out_dir" \
      --num_samples "$NUM_SAMPLES" \
      --batch_size "$BATCH_SIZE" \
      --device "$DEVICE"

    "$PYTHON" -m scripts.evaluate_diffusion "$out_dir" \
      --docking_mode "$DOCKING_MODE" \
      --verbose True \
      --protein_pdb "$POCKET_PDB"
  )

  "$PYTHON" /home/jzay/Desktop/mol_gen/extract_metrics.py \
    "$out_dir/eval_results/metrics_-1.pt" \
    --csv "$out_dir/eval_results/metrics_extracted.csv" \
    --sdf "$out_dir/eval_results/metrics_extracted.sdf"

  "$PYTHON" /home/jzay/Desktop/mol_gen/compare_vina_runs.py \
    "$BASELINE_CSV" \
    "$out_dir/eval_results/metrics_extracted.csv" \
    > "$compare_txt"

  append_summary_row "$run_name" "$mode" "$scale" "$start_step" "$out_dir"
}

echo "Sweep dir: $SWEEP_DIR"
echo "Baseline : $BASELINE_CSV"

for mode in "${MODES[@]}"; do
  read -r -a scales <<< "$(pick_scales "$mode")"
  for start_step in "${START_STEPS[@]}"; do
    for scale in "${scales[@]}"; do
      run_one "$mode" "$scale" "$start_step"
    done
  done
done

echo
echo "Sweep complete."
echo "Summary CSV: $SUMMARY_CSV"
