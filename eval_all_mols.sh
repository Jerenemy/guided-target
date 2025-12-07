#!/usr/bin/env bash
set -euo pipefail

# -------- configurable bits --------
P53_NAME="y220c"
RUN_NUM="2"

REPO=/home/jzay/Desktop/mol_gen/targetdiff-main-fixed
PROT_INPUT=/home/jzay/Desktop/mol_gen/data/input/"$P53_NAME"/"$P53_NAME"_pocket10.pdb
# PROT_INPUT=/home/jzay/Desktop/mol_gen/data/input/"$P53_NAME"/"$P53_NAME"_av.pdb

INPUT_DIR="$REPO/output/$P53_NAME/sampling_run_$RUN_NUM"/sdf   # folder with <number>.sdf files
OUT_CSV="$REPO/output/$P53_NAME/sampling_run_$RUN_NUM"/eval_results/kd_predictions.csv             # where to write results
CKPT="$REPO/pretrained_models/egnn_pdbbind_v2016.pt"
# -----------------------------------

which python3 >/dev/null

cd "$REPO"

# Collect and sort numerically all files matching <number>.sdf
shopt -s nullglob
files=("$INPUT_DIR"/*.sdf)
if [ ${#files[@]} -eq 0 ]; then
  echo "No .sdf files found in: $INPUT_DIR"
  exit 1
fi

# Keep only numeric basenames and sort them numerically
mapfile -t BASENAMES < <(printf '%s\n' "${files[@]##*/}" \
  | sed 's/\.sdf$//' \
  | grep -E '^[0-9]+$' \
  | sort -n)

if [ ${#BASENAMES[@]} -eq 0 ]; then
  echo "No files of the form <number>.sdf found in: $INPUT_DIR"
  exit 1
fi

# Write CSV header
echo "index,Kd_M" > "$OUT_CSV"

# Iterate and run predictions
for idx in "${BASENAMES[@]}"; do
  MOL_INPUT="$INPUT_DIR/$idx.sdf"

  # Run the predictor; it prints lots of stuff—capture all
  output="$(
    python -m scripts.property_prediction.inference \
      --ckpt_path "$CKPT" \
      --protein_path "$PROT_INPUT" \
      --ligand_path "$MOL_INPUT" \
      --kind Kd 2>&1
    )" || true

  # Extract the numeric part from a line like: "Prediction: Kd=2.42e-04 m"
  # - robust to extra text before/after, scientific notation, +/- signs
  kd_val="$(printf '%s\n' "$output" \
    | grep -E 'Prediction: *Kd=' \
    | tail -n1 \
    | sed -n 's/.*Prediction: *Kd=\([0-9.eE+-]\+\).*/\1/p')"

  if [[ -z "$kd_val" ]]; then
    echo "Warning: could not parse Kd for $idx.sdf; leaving blank in CSV" >&2
  fi

  echo "$idx,$kd_val" >> "$OUT_CSV"
done

echo "Wrote $(wc -l < "$OUT_CSV") rows (including header) to: $OUT_CSV"
