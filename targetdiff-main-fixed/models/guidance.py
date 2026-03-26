import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.utils import scatter
import utils.transforms as trans  # for get_atomic_number_from_index

LIGAND_NODE = 1
POCKET_NODE = 0

EDGE_LIGAND_LIGAND = 0
EDGE_POCKET_TO_LIGAND = 1

class GuidedAffinityWrapper(torch.nn.Module):
    def __init__(self, affinity_model, pocket_pos, pocket_z=None, device='cpu', z_pocket_val=99, ligand_atom_mode='add_aromatic'):
        super().__init__()
        self.affinity_model = affinity_model
        self.ligand_atom_mode = ligand_atom_mode
        pocket_pos = pocket_pos.to(device)
        # keep pocket coordinates in the same centered frame as the sampler
        pocket_center = pocket_pos.mean(dim=0, keepdim=True)
        self.register_buffer("pocket_pos_centered", pocket_pos - pocket_center)
        if pocket_z is None:
            pocket_z = torch.full((pocket_pos.size(0),), z_pocket_val, dtype=torch.long, device=device)
        else:
            pocket_z = pocket_z.to(device).long()
        self.register_buffer("pocket_z", pocket_z)
        self.num_pocket_atoms = pocket_pos.size(0)

    def forward(self, ligand_pos, ligand_v, batch_ligand, batch_protein, protein_pos=None):
        # ligand_pos: [N_lig, 3]  (requires_grad True in sampler)
        # ligand_v:   [N_lig]     (indices in [0, num_classes))
        device = ligand_pos.device
        num_graphs = batch_ligand.max().item() + 1

        # 1) map TargetDiff index -> atomic number using your transforms
        atomic_numbers = trans.get_atomic_number_from_index(
            ligand_v.detach().cpu(), mode=self.ligand_atom_mode
        )  # returns list of ints
        z_lig = torch.tensor(atomic_numbers, dtype=torch.long, device=device)  # [N_lig]

        # 2) pocket positions / batches in the same frame as ligand_pos
        use_provided_protein = (
            protein_pos is not None
            and batch_protein is not None
            and protein_pos.size(0) == self.num_pocket_atoms * num_graphs
        )
        if use_provided_protein:
            pocket_pos = protein_pos
            pocket_batch = batch_protein
            z_pocket = self.pocket_z.repeat(num_graphs).to(device)
        else:
            pocket_pos = self.pocket_pos_centered.repeat(num_graphs, 1)
            pocket_batch = torch.arange(num_graphs, device=device).repeat_interleave(self.num_pocket_atoms)
            z_pocket = self.pocket_z.repeat(num_graphs).to(device)

        # 3) concat ligand + pocket for the affinity model
        pos = torch.cat([ligand_pos, pocket_pos], dim=0)
        z = torch.cat([z_lig, z_pocket], dim=0)
        batch = torch.cat([batch_ligand, pocket_batch], dim=0)
        node_type = torch.cat([
            torch.ones(ligand_pos.size(0), dtype=torch.long, device=device),
            torch.zeros(pocket_pos.size(0), dtype=torch.long, device=device),
        ])

        data = Data(pos=pos, z=z, batch=batch, node_type=node_type)

        affinity = self.affinity_model(data)  # [B]
        # better binder = more negative Vina, so maximize -affinity
        return -affinity


def extract_guidance_state_dict(ckpt_or_state):
    if isinstance(ckpt_or_state, dict) and "model" in ckpt_or_state:
        return ckpt_or_state["model"]
    return ckpt_or_state


def infer_guidance_architecture(state_dict):
    keys = set(state_dict.keys())
    if "node_type_emb.weight" in keys and any(k.endswith("rbf.centers") for k in keys):
        return "ligand_context"
    if any("tp_msg" in k for k in keys):
        return "equivariant"
    raise ValueError("Could not infer guidance architecture from checkpoint state_dict")


def infer_ligand_context_model_kwargs(state_dict):
    layer_ids = sorted({
        int(k.split(".")[1])
        for k in state_dict.keys()
        if k.startswith("layers.") and len(k.split(".")) > 2 and k.split(".")[1].isdigit()
    })
    n_layers = len(layer_ids)
    hidden = state_dict["z_emb.weight"].shape[1]
    num_rbf = state_dict["layers.0.rbf.centers"].numel()
    cutoff = float(state_dict["layers.0.rbf.centers"][-1].item()) if num_rbf > 0 else 6.0
    return {
        "max_z": state_dict["z_emb.weight"].shape[0],
        "hidden": hidden,
        "n_layers": n_layers,
        "num_rbf": num_rbf,
        "cutoff": cutoff,
        # Dropout does not affect eval-mode inference; keep runtime default aligned with training notebook.
        "dropout": 0.1,
    }


def build_context_edges(pos, node_type, batch=None, r_ligand=5.0, r_cross=6.0):
    """
    Runtime-compatible directed context graph:
      - ligand -> ligand spatial edges within r_ligand
      - pocket -> ligand context edges within r_cross
    Pocket nodes act as context sources and are not updated by default.
    """
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    edge_src = []
    edge_dst = []
    edge_type = []

    for b in batch.unique():
        mask_b = batch == b
        idx = mask_b.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        sub_pos = pos[idx]
        sub_type = node_type[idx]

        lig_local = (sub_type == LIGAND_NODE).nonzero(as_tuple=True)[0]
        poc_local = (sub_type == POCKET_NODE).nonzero(as_tuple=True)[0]

        if lig_local.numel() > 0:
            lig_pos = sub_pos[lig_local]
            dist_ll = torch.cdist(lig_pos, lig_pos, p=2)
            keep_ll = dist_ll <= r_ligand
            keep_ll = keep_ll & ~torch.eye(lig_local.numel(), dtype=torch.bool, device=pos.device)
            src_ll, dst_ll = keep_ll.nonzero(as_tuple=True)
            if src_ll.numel() > 0:
                edge_src.append(idx[lig_local[src_ll]])
                edge_dst.append(idx[lig_local[dst_ll]])
                edge_type.append(torch.full((src_ll.numel(),), EDGE_LIGAND_LIGAND, dtype=torch.long, device=pos.device))

        if lig_local.numel() > 0 and poc_local.numel() > 0:
            lig_pos = sub_pos[lig_local]
            poc_pos = sub_pos[poc_local]
            dist_pl = torch.cdist(poc_pos, lig_pos, p=2)
            keep_pl = dist_pl <= r_cross
            src_pl, dst_pl = keep_pl.nonzero(as_tuple=True)
            if src_pl.numel() > 0:
                edge_src.append(idx[poc_local[src_pl]])
                edge_dst.append(idx[lig_local[dst_pl]])
                edge_type.append(torch.full((src_pl.numel(),), EDGE_POCKET_TO_LIGAND, dtype=torch.long, device=pos.device))

    if len(edge_src) == 0:
        return (
            torch.empty(2, 0, dtype=torch.long, device=pos.device),
            torch.empty(0, dtype=torch.long, device=pos.device),
        )

    edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
    edge_type = torch.cat(edge_type, dim=0)
    return edge_index, edge_type


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf=32, cutoff=6.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max((centers[1] - centers[0]).item() ** 2, 1e-6) if num_rbf > 1 else 1.0

    def forward(self, dist):
        diff = dist.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * diff.pow(2))


class ContextMessageBlock(nn.Module):
    def __init__(self, hidden, num_edge_types=2, num_rbf=32, cutoff=6.0, dropout=0.1):
        super().__init__()
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden + hidden + num_rbf, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, pos, edge_index, edge_type, node_type):
        if edge_index.numel() == 0:
            return h

        src, dst = edge_index
        rel = pos[src] - pos[dst]
        dist = rel.norm(dim=-1)
        radial = self.rbf(dist)
        edge_kind = self.edge_type_emb(edge_type)

        m_in = torch.cat([h[src], h[dst], edge_kind, radial], dim=-1)
        m_ij = self.msg_mlp(m_in)
        m_i = scatter(m_ij, dst, dim=0, dim_size=h.size(0), reduce="mean")

        upd = self.upd_mlp(torch.cat([h, m_i], dim=-1))
        lig_mask = node_type == LIGAND_NODE
        out = h.clone()
        out[lig_mask] = self.norm(h[lig_mask] + upd[lig_mask])
        return out


class LigandContextAffinityModel(nn.Module):
    def __init__(self, max_z=100, hidden=128, n_layers=4, num_rbf=32, cutoff=6.0, dropout=0.1):
        super().__init__()
        self.z_emb = nn.Embedding(max_z, hidden)
        self.node_type_emb = nn.Embedding(2, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([
            ContextMessageBlock(hidden=hidden, num_edge_types=2, num_rbf=num_rbf, cutoff=cutoff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        pos = data.pos
        z = data.z
        node_type = data.node_type.long()
        edge_index = data.edge_index
        edge_type = data.edge_type.long()

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        h = self.z_emb(z) + self.node_type_emb(node_type)
        h = self.input_norm(h)

        for layer in self.layers:
            h = layer(h, pos, edge_index, edge_type, node_type)

        lig_mask = (node_type == LIGAND_NODE).float()
        lig_counts = scatter(lig_mask, batch, dim=0, reduce="sum").clamp(min=1.0)
        g = scatter(h * lig_mask.unsqueeze(-1), batch, dim=0, reduce="sum")
        g = g / lig_counts.unsqueeze(-1)
        affinity = self.head(g).squeeze(-1)
        return affinity


class GuidedLigandContextWrapper(nn.Module):
    def __init__(
        self,
        affinity_model,
        pocket_pos,
        pocket_z,
        ligand_atom_mode='add_aromatic',
        r_ligand=5.0,
        r_cross=6.0,
        device='cpu',
    ):
        super().__init__()
        self.affinity_model = affinity_model
        self.ligand_atom_mode = ligand_atom_mode
        self.r_ligand = r_ligand
        self.r_cross = r_cross

        pocket_pos = pocket_pos.to(device)
        pocket_center = pocket_pos.mean(dim=0, keepdim=True)
        self.register_buffer("pocket_pos_centered", pocket_pos - pocket_center)
        self.register_buffer("pocket_z", pocket_z.to(device).long())
        self.num_pocket_atoms = pocket_pos.size(0)

    def forward(self, ligand_pos, ligand_v, batch_ligand, batch_protein, protein_pos=None):
        device = ligand_pos.device
        num_graphs = batch_ligand.max().item() + 1

        atomic_numbers = trans.get_atomic_number_from_index(
            ligand_v.detach().cpu(), mode=self.ligand_atom_mode
        )
        z_lig = torch.tensor(atomic_numbers, dtype=torch.long, device=device)

        use_provided_protein = (
            protein_pos is not None
            and batch_protein is not None
            and protein_pos.size(0) == self.num_pocket_atoms * num_graphs
        )
        if use_provided_protein:
            pocket_pos = protein_pos
            pocket_batch = batch_protein
        else:
            pocket_pos = self.pocket_pos_centered.repeat(num_graphs, 1)
            pocket_batch = torch.arange(num_graphs, device=device).repeat_interleave(self.num_pocket_atoms)

        z_pocket = self.pocket_z.repeat(num_graphs).to(device)

        pos = torch.cat([ligand_pos, pocket_pos], dim=0)
        z = torch.cat([z_lig, z_pocket], dim=0)
        batch = torch.cat([batch_ligand, pocket_batch], dim=0)
        node_type = torch.cat([
            torch.ones(ligand_pos.size(0), dtype=torch.long, device=device),
            torch.zeros(pocket_pos.size(0), dtype=torch.long, device=device),
        ])
        edge_index, edge_type = build_context_edges(
            pos=pos,
            node_type=node_type,
            batch=batch,
            r_ligand=self.r_ligand,
            r_cross=self.r_cross,
        )

        data = Data(
            pos=pos,
            z=z,
            batch=batch,
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
        )
        affinity = self.affinity_model(data)
        return -affinity

from e3nn import o3
from e3nn.nn import Gate

# Node feature irreps: 32 scalars + 32 vectors
IRREPS_NODE   = o3.Irreps("32x0e + 32x1o")

# For the Gate inside blocks: split into
#   32 scalars  |  32 gate scalars  | 32 vectors
IRREPS_GATE_IN = o3.Irreps("32x0e + 32x0e + 32x1o")

# Per-node scalar channels used for graph readout
IRREPS_OUT    = o3.Irreps("32x0e")

# Spherical harmonics of degree 1 -> one 1o irrep
IRREPS_SH     = o3.Irreps("1x1o")

HIDDEN = 128   # hidden width for atom encoder / readout
DROPOUT = 0.1

class EquivariantMPBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # Mix node features into scalars / gate scalars / vectors
        self.lin_node = o3.Linear(
            irreps_in=IRREPS_NODE,
            irreps_out=IRREPS_GATE_IN,
        )

        self.gate = Gate(
            irreps_scalars="32x0e",
            act_scalars=[F.relu],          # broadcast over multiplicity
            irreps_gates="32x0e",
            act_gates=[torch.sigmoid],
            irreps_gated="32x1o",
        )

        # Tensor product between node features and spherical harmonics on edges
        # message_{i<-j} = TP( x_j , Y^1(r_ij) )  -> IRREPS_NODE
        self.tp_msg = o3.FullyConnectedTensorProduct(
            irreps_in1=IRREPS_NODE,
            irreps_in2=IRREPS_SH,
            irreps_out=IRREPS_NODE,
        )

        # Optional extra mixing on aggregated messages
        self.lin_msg = o3.Linear(
            irreps_in=IRREPS_NODE,
            irreps_out=IRREPS_NODE,
        )

        self.norm = nn.LayerNorm(IRREPS_NODE.dim)


    def forward(self, x, pos, edge_index):
        """
        x:   [N, IRREPS_NODE.dim] node features
        pos: [N, 3] positions
        edge_index: [2, E] with edges (src=j, dst=i)
        """
        row, col = edge_index  # row = dst (i), col = src (j)

        # Prepare node features for gating
        h = self.lin_node(x)          # [N, IRREPS_GATE_IN.dim]
        h = self.gate(h)              # [N, IRREPS_NODE.dim] again

        # Relative vectors per edge
        rel = pos[col] - pos[row]     # [E, 3]

        # Spherical harmonics Y^1(r_ij) -> 1x1o, shape [E, 3]
        sh = o3.spherical_harmonics(
            l=1,
            x=rel,
            normalize=True,
            normalization='component',
        )  # [E, 3]

        # Messages from j to i
        m_ij = self.tp_msg(h[col], sh)   # [E, IRREPS_NODE.dim]

        # Aggregate messages at destination nodes
        # m_i = scatter(m_ij, row, dim=0, dim_size=x.shape[0], reduce="sum")
        # m_i = scatter(m_ij, row, dim=0, dim_size=x.shape[0], reduce="mean")

        # after computing m_ij
        deg = scatter(torch.ones_like(row, dtype=m_ij.dtype), row, dim=0, dim_size=x.size(0), reduce="sum").clamp(min=1.0)
        m_i = scatter(m_ij, row, dim=0, dim_size=x.size(0), reduce="sum")
        m_i = m_i / deg.unsqueeze(-1)  # or / deg.sqrt().unsqueeze(-1)

        
        # Mix + residual
        m_i = self.lin_msg(m_i)
        m_i = self.norm(m_i) # ADD LAYER NORM

        x_out = x + m_i

        return x_out

class EquivariantAffinityModel(nn.Module):
    def __init__(self, max_z: int = 100, n_layers: int = 3, radius: float = 5.0):
        super().__init__()

        # Embed atomic number -> scalar channels
        self.emb = nn.Embedding(max_z, 16)

        # First linear: (16x0e + 1x1o) -> IRREPS_NODE
        self.irreps_in = o3.Irreps("16x0e + 1x1o")

        self.lin_in = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=IRREPS_NODE,
        )

        # Stack of message passing blocks
        self.layers = nn.ModuleList(
            [EquivariantMPBlock() for _ in range(n_layers)]
        )

        # Map node features to scalar channels, then graph-level MLP head
        self.lin_out = o3.Linear(
            irreps_in=IRREPS_NODE,
            irreps_out=IRREPS_OUT,
        )
        self.readout = nn.Sequential(
            nn.Linear(IRREPS_OUT.dim, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )
        self.radius = radius

    def forward(self, data):
        pos = data.pos          # [N, 3]
        z = data.z              # [N]
        batch = data.batch      # [N]
        edge_index = getattr(data, "edge_index", None)  # [2, E] or None

        # Build a radius graph on-the-fly if not provided
        if edge_index is None:
            edge_index = build_radius_edge_index(pos, batch=batch, r=self.radius).to(pos.device)

        # Center each molecule (translation invariance)
        # should i be centering? it's supposed to be equivariant, not invariant
        center = scatter(pos, batch, dim=0, reduce="mean")   # [B, 3]
        pos_rel = (pos - center[batch])

        # Input features: scalars from embedding + vector from position
        scalars = self.emb(z)                                # [N, 16]
        x_in = torch.cat([scalars, pos_rel], dim=-1)         # [N, 16+3]

        # Map to hidden irreps
        x = self.lin_in(x_in)                                # [N, IRREPS_NODE.dim]

        # Message passing layers
        for layer in self.layers:
            x = layer(x, pos_rel, edge_index)

        # Per-node scalar channels -> pool over nodes for graph representation
        node_scalars = self.lin_out(x)                        # [N, IRREPS_OUT.dim]
        g = scatter(node_scalars, batch, dim=0, reduce="mean")  # [B, IRREPS_OUT.dim]

        affinity = self.readout(g).squeeze(-1)                # [B]
        return affinity



def build_radius_edge_index(pos: torch.Tensor, batch: torch.Tensor = None, r: float = 5.0, loop: bool = False):
    """
    Pure PyTorch radius graph that respects batch.
    pos: [N, 3]
    batch: [N] or None
    returns edge_index: [2, E]
    """
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    edge_src = []
    edge_dst = []

    for b in batch.unique():
        mask = batch == b
        idx = mask.nonzero(as_tuple=True)[0]
        sub_pos = pos[idx]

        dist = torch.cdist(sub_pos, sub_pos, p=2)
        sub_mask = dist <= r
        if not loop:
            sub_mask = sub_mask & ~torch.eye(len(idx), dtype=torch.bool, device=pos.device)

        src, dst = sub_mask.nonzero(as_tuple=True)
        edge_src.append(idx[src])
        edge_dst.append(idx[dst])

    edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
    return edge_index
