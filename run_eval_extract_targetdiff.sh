set -euo pipefail

P53_NAME="y220c"

REPO=/home/jzay/Desktop/mol_gen/targetdiff-main-fixed
INPUT=/home/jzay/Desktop/mol_gen/data/input/"$P53_NAME"
BASE_OUTDIR=${OUTDIR:-$REPO/output}/"$P53_NAME"_guided
prefix="sampling_run_"
i=0
while [ -d "$BASE_OUTDIR/${prefix}${i}" ]; do i=$((i+1)); done
OUTDIR="$BASE_OUTDIR/${prefix}${i}"

mkdir -p "$OUTDIR"

which python3

cd "$REPO"

python -m scripts.sample_for_pocket configs/sampling.yml \
   --pdb_path "$INPUT"/"$P53_NAME"_pocket10.pdb \
   --result_path "$OUTDIR" \
   --num_samples 3000



python -m scripts.evaluate_diffusion "$OUTDIR" \
    --docking_mode vina_dock \
    --verbose True \
    --protein_pdb "$INPUT"/"$P53_NAME"_pocket10.pdb

cd ../ 

OUTPUT=$OUTDIR/eval_results

python extract_metrics.py \
  $OUTPUT/metrics_-1.pt \
  --csv $OUTPUT/metrics_extracted.csv \
  --sdf $OUTPUT/metrics_extracted.sdf
