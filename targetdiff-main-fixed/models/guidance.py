import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.utils import scatter
import utils.transforms as trans  # for get_atomic_number_from_index

class GuidedAffinityWrapper(torch.nn.Module):
    def __init__(self, affinity_model, pocket_pos, pocket_z=None, device='cpu', z_pocket_val=99):
        super().__init__()
        self.affinity_model = affinity_model
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
            ligand_v.detach().cpu(), mode='add_aromatic'
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
