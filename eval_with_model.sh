set -euo pipefail

P53_NAME="y220c"
RUN_NUM="2"
MOL_NAME=best_mol.sdf

REPO=/home/jzay/Desktop/mol_gen/targetdiff-main-fixed
PROT_INPUT=/home/jzay/Desktop/mol_gen/data/input/"$P53_NAME"/"$P53_NAME"_pocket10.pdb
# PROT_INPUT=/home/jzay/Desktop/mol_gen/data/input/"$P53_NAME"/"$P53_NAME"_av.pdb
MOL_INPUT="$REPO"/output/"$P53_NAME"/sampling_run_"$RUN_NUM"/"$MOL_NAME"

which python3

cd "$REPO"

python -m scripts.property_prediction.inference \
  --ckpt_path pretrained_models/egnn_pdbbind_v2016.pt \
  --protein_path $PROT_INPUT \
  --ligand_path $MOL_INPUT \
  --kind Kd