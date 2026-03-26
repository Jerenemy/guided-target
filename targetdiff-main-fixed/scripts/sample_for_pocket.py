import argparse
import os
import shutil

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils.data import PDBProtein
from utils import reconstruct
from rdkit import Chem

from models.guidance import (
    EquivariantAffinityModel,
    GuidedAffinityWrapper,
    GuidedLigandContextWrapper,
    LigandContextAffinityModel,
    extract_guidance_state_dict,
    infer_guidance_architecture,
    infer_ligand_context_model_kwargs,
)


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs_pdb')
    parser.add_argument('--num_samples', type=int)
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # Load pocket
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)
    if args.num_samples:
        config.sample.num_samples = args.num_samples

    #################################### guidance #############################
    # build guidance model if enabled
    guidance_model = None
    if getattr(config.sample, "use_guidance", False):
        ckpt_aff = torch.load(config.sample.guidance_ckpt, map_location=args.device)
        state_aff = extract_guidance_state_dict(ckpt_aff)
        guidance_arch = getattr(config.sample, "guidance_arch", "auto")
        if guidance_arch in (None, "auto"):
            guidance_arch = infer_guidance_architecture(state_aff)

        if guidance_arch == "equivariant":
            aff_model = EquivariantAffinityModel(max_z=100).to(args.device)
            aff_model.load_state_dict(state_aff)
            wrapper_cls = GuidedAffinityWrapper
            wrapper_kwargs = {
                "ligand_atom_mode": ligand_atom_mode,
            }
        elif guidance_arch == "ligand_context":
            model_kwargs = infer_ligand_context_model_kwargs(state_aff)
            aff_model = LigandContextAffinityModel(**model_kwargs).to(args.device)
            aff_model.load_state_dict(state_aff)
            wrapper_cls = GuidedLigandContextWrapper
            wrapper_kwargs = {
                "ligand_atom_mode": ligand_atom_mode,
                "r_ligand": getattr(config.sample, "guidance_ligand_radius", 5.0),
                "r_cross": getattr(config.sample, "guidance_cross_radius", 6.0),
            }
        else:
            raise ValueError(f"Unknown guidance_arch {guidance_arch}")

        aff_model.eval()
        logger.info(f"Using guidance architecture: {guidance_arch}")

        # 2) wrap it with pocket info
        pocket_pos = data.protein_pos.to(args.device)  # this is your pocket from PDB
        guidance_model = wrapper_cls(
            affinity_model=aff_model,
            pocket_pos=pocket_pos,
            pocket_z=data.protein_element,
            device=args.device,
            **wrapper_kwargs,
        )
    #################################### guidance #############################
    
    guidance_return_stats = getattr(config.sample, "guidance_log_stats", False)
    guidance_start_step = getattr(config.sample, "guidance_start_step", None)
    guidance_end_step = getattr(config.sample, "guidance_end_step", None)
    guidance_log = guidance_return_stats or getattr(config.sample, "guidance_log", False)
    guidance_scale_mode = getattr(config.sample, "guidance_scale_mode", "var")
    noise_scale = getattr(config.sample, "noise_scale", 1.0)

    sample_outputs = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        guidance_model=guidance_model,
        guidance_scale=getattr(config.sample, "guidance_scale", 0.0),
        guidance_start_step=guidance_start_step,
        guidance_end_step=guidance_end_step,
        guidance_log=guidance_log,
        guidance_return_stats=guidance_return_stats,
        guidance_scale_mode=guidance_scale_mode,
        noise_scale=noise_scale,
    )
    if guidance_return_stats:
        (all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, guidance_stats) = sample_outputs
    else:
        (all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list) = sample_outputs
    result = {
        'data': data,
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj
    }
    if guidance_return_stats:
        result['guidance_stats'] = guidance_stats
    logger.info('Sample done!')

    # reconstruction
    gen_mols = []
    n_recon_success, n_complete = 0, 0
    for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=ligand_atom_mode)
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=ligand_atom_mode)
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            gen_mols.append(None)
            continue
        n_recon_success += 1

        if '.' in smiles:
            gen_mols.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)
    result['mols'] = gen_mols
    logger.info('Reconstruction done!')
    logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'sample.pt'))
    # optional: dump guidance stats for post-hoc inspection
    if guidance_return_stats and guidance_stats:
        import json
        stats_path = os.path.join(result_path, "guidance_stats.json")
        with open(stats_path, "w") as f:
            json.dump(guidance_stats, f, indent=2)
        logger.info(f"Saved guidance stats to {stats_path}")
    mols_save_path = os.path.join(result_path, f'sdf')
    os.makedirs(mols_save_path, exist_ok=True)
    for idx, mol in enumerate(gen_mols):
        if mol is not None:
            sdf_writer = Chem.SDWriter(os.path.join(mols_save_path, f'{idx:03d}.sdf'))
            sdf_writer.write(mol)
            sdf_writer.close()
    logger.info(f'Results are saved in {result_path}')
