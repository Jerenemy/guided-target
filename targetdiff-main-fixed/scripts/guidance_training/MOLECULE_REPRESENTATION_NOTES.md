# Molecule Representation And Message Passing Notes

This note summarizes how the current guidance / affinity model represents a ligand-pocket complex, what information actually reaches the model, what the message-passing block does, and what I would change first.

The most important distinction is:

- The dataset code sometimes builds richer per-atom features.
- The current equivariant guidance model mostly ignores those richer features and uses only atomic number plus geometry.

## 1. What A Single Training Example Looks Like

In the current notebook-style guidance pipeline, each sample is one combined graph containing:

- ligand atoms from the generated pose `pred_pos`
- pocket atoms from the fixed PDB pocket
- a scalar target such as `vina["score_only"][0]["affinity"]`

Relevant code:

- ligand + pocket dataset assembly: [`predict_binding_affinity.py`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L224)
- notebook mirror of the same logic: [`guidance_wrapper_fixed.ipynb`](/home/jzay/moment/predict_binding_affinity/guidance_wrapper_fixed.ipynb)

### Effective node fields used by the current notebook guidance model

For the current `guidance_wrapper_fixed.ipynb` path, the actual `Data` object passed into the model contains:

- `pos`: concatenated ligand and pocket coordinates
- `z`: concatenated ligand and pocket atomic numbers
- `node_type`: `1` for ligand atoms, `0` for pocket atoms
- `y`: scalar affinity target

The notebook helper does this directly:

- pocket positions and atomic numbers: [`guidance_wrapper_fixed.ipynb`](/home/jzay/moment/predict_binding_affinity/guidance_wrapper_fixed.ipynb)
- dataset concatenation into `Data(pos, z, y, node_type)`: [`guidance_wrapper_fixed.ipynb`](/home/jzay/moment/predict_binding_affinity/guidance_wrapper_fixed.ipynb)

In the exported Python version, the dataset also computes extra fields like `x`, `qed`, `sa`, and `logp`:

- pocket residue/element one-hot features: [`predict_binding_affinity.py:12`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L12)
- ligand handcrafted atom features: [`predict_binding_affinity.py:49`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L49)
- combined `x` feature tensor: [`predict_binding_affinity.py:286`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L286)

But the current equivariant guidance model does not consume `data.x`. It uses only:

- `data.z`
- `data.pos`
- `data.batch`
- optionally `data.edge_index`

See:

- current notebook/exported equivariant model: [`predict_binding_affinity.py:727`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L727)
- deployed TargetDiff runtime model: [`guidance.py:163`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L163)

## 2. What The Node Features Really Are

### Ligand atoms

Effective ligand representation in the current equivariant model:

- atomic number embedding via `nn.Embedding(max_z, 16)`
- relative 3D position vector after centering

Code:

- ligand atomic number extraction in notebook dataset: [`guidance_wrapper_fixed.ipynb`](/home/jzay/moment/predict_binding_affinity/guidance_wrapper_fixed.ipynb)
- atomic number embedding: [`guidance.py:168`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L168)
- concatenation of embedding with position vector: [`guidance.py:207`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L207)

What is not currently used by the equivariant model:

- aromatic flag
- formal charge
- degree
- hybridization
- donor/acceptor indicators
- ligand bond order

The older Python dataset code computes some of these, but the deployed model drops them before message passing.

### Pocket atoms

Effective pocket representation in the current equivariant model:

- atomic number embedding
- relative 3D position vector after centering

Runtime TargetDiff guidance does pass real protein atomic numbers:

- wrapper construction with `pocket_z=data.protein_element`: [`sample_for_pocket.py:92`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/scripts/sample_for_pocket.py#L92)

The training/export code also computes richer pocket features:

- residue-type one-hot
- element one-hot

But those are not used by the current equivariant model.

## 3. How Edges Are Built

### Notebook/runtime guidance path

The notebook guidance model and the deployed runtime model both default to a plain radius graph over all atoms:

- if two atoms are within radius `r`, connect them
- no explicit distinction between ligand-ligand, ligand-pocket, or pocket-pocket edge types
- no bond-order information

Code:

- notebook/runtime radius graph helper: [`guidance.py:228`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L228)

### Older exported Python path

The older `predict_binding_affinity.py` dataset is somewhat richer:

- ligand-ligand edges with `r_ligand`
- ligand-pocket and pocket-ligand edges with `r_cross`
- pocket-pocket edges with `r_pocket`

Code:

- masked edge construction: [`predict_binding_affinity.py:181`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L181)
- four edge sets combined: [`predict_binding_affinity.py:317`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L317)

But even there, the edge type itself is not encoded as an edge feature, so once edges are merged the model still does not explicitly know whether a given edge is:

- covalent ligand structure
- ligand-pocket contact
- pocket internal edge

## 4. What Message Passing Does Right Now

### Current equivariant block

The current guidance model uses an `e3nn` block with:

- scalar atom embedding + vector position as input irreps
- a gated nonlinear mixing step
- degree-1 spherical harmonics on relative edge vectors
- a tensor product to build messages
- degree-normalized aggregation
- linear mix + layer norm + residual

Code:

- irreps definitions: [`guidance.py:69`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L69)
- block definition: [`guidance.py:84`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L84)

Concretely, for each edge it does:

1. transform node features into scalar/vector channels
2. compute relative vector `r_ij = pos_j - pos_i`
3. compute `Y^1(r_ij)` spherical harmonics
4. tensor-product neighbor features with `Y^1(r_ij)`
5. aggregate messages at each node
6. add the result back as a residual update

### Important limitation: there is no radial distance encoding

This is a major weakness of the current equivariant block.

The block uses:

- edge direction through spherical harmonics
- edge existence through the cutoff graph

But it does not use the actual edge length as a learned feature. Inside the cutoff, a very short clash and a comfortable contact can look nearly the same except for direction.

There is no radial basis or distance-conditioned weight network in:

- [`guidance.py:135`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L135)
- [`guidance.py:143`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L143)

For a Vina-like target, this is a serious representational gap because sterics and contact quality depend strongly on distance magnitude, not just direction.

## 5. How The Graph Is Read Out

After message passing, the current equivariant model:

- projects each node to scalar channels
- mean-pools over all nodes in the graph
- runs a small MLP to predict affinity

Code:

- scalar projection: [`guidance.py:185`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L185)
- all-node mean pooling: [`guidance.py:224`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L224)

This is one of the biggest practical problems in the current implementation.

In your data:

- the pocket has about 265 atoms
- ligands average only about 11 to 12 atoms

So a plain mean over all nodes makes the graph representation dominated by the fixed pocket. That encourages the model to collapse toward a nearly constant prediction, which is exactly the failure mode your `score_only` parity plot suggests.

The older non-equivariant message-passing model in the same repo already has the better pattern:

- ligand-only pooling using `node_type == 1`

Code:

- ligand-only readout: [`predict_binding_affinity.py:592`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L592)

## 6. Representation Problems In The Current Setup

### Problem 1: richer atom features are built, then discarded

The code computes more chemistry than the current equivariant model uses.

Examples:

- ligand handcrafted features are built in [`predict_binding_affinity.py:49`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L49)
- pocket residue/element one-hot features are built in [`predict_binding_affinity.py:12`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L12)

But the deployed model reads only `z` and `pos`.

### Problem 2: no explicit distance magnitude in the equivariant messages

For a docking-style target, this is a major omission. Steric clashes, hydrogen-bond distances, and contact quality are distance-sensitive.

### Problem 3: readout is dominated by the fixed pocket

All-node mean pooling makes the representation too similar across samples.

### Problem 4: no explicit distinction between bond edges and nonbonded contacts

The model currently treats all radius edges as the same kind of relation unless you manually separate them outside the block.

That makes it harder to learn:

- internal ligand geometry constraints
- pocket internal structure
- ligand-pocket interaction rules

### Problem 5: `node_type` is present but mostly unused

`node_type` is already available in the graph, but the current equivariant model does not use it in message passing or readout.

### Problem 6: training and deployment use only a stripped-down ligand type signal

At runtime, ligand classes from TargetDiff are converted back to atomic numbers:

- [`guidance.py:27`](/home/jzay/Desktop/mol_gen/targetdiff-main-fixed/models/guidance.py#L27)

That means aromatic-state information encoded in the diffusion atom vocabulary is largely collapsed before the affinity model sees it.

## 7. What I Recommend Changing First

These are in priority order.

### 1. Change the readout to ligand-only pooling

Recommendation:

- pool only over ligand nodes for the final graph representation
- optionally concatenate a separate pooled pocket summary if you want the head to see both

Why:

- this directly fixes the pocket-dominance problem
- it matches the fact that the target changes mainly with ligand pose, not with the fixed pocket identity across samples

Best minimal change:

- replace all-node mean pooling with the ligand-only pattern already used in [`predict_binding_affinity.py:592`](/home/jzay/moment/predict_binding_affinity/predict_binding_affinity.py#L592)

### 2. Add radial distance features to the equivariant block

Recommendation:

- use an RBF embedding of edge length
- feed that into a small MLP that modulates message weights
- or move to an EGNN / PaiNN / SchNet-style geometric layer with explicit radial dependence

Why:

- Vina-like energies depend strongly on distance magnitude
- direction-only messages are too weak for sterics and contact quality

This is the most important representational change after the readout fix.

### 3. Use `node_type` explicitly inside the model

Recommendation:

- add a learned ligand-vs-pocket embedding
- optionally use separate input projections for ligand and pocket atoms

Why:

- the same atomic number should not necessarily mean the same thing in ligand and protein context
- a carbon in the ligand and a carbon in the pocket play different roles

### 4. Preserve more chemistry than just atomic number

Recommendation:

- ligand features: atomic number, aromaticity, formal charge, hybridization, degree, H-count, donor/acceptor flags
- pocket features: atomic number plus residue-type or atom-name embeddings if available

Why:

- `score_only` is sensitive to chemistry, not just element identity
- collapsing everything to atomic number throws away useful signal you already know how to compute

### 5. Separate edge types

Recommendation:

- distinguish at least:
  - ligand covalent bonds
  - ligand-ligand nonbonded radius edges
  - ligand-pocket contact edges
  - pocket-pocket structural edges

Why:

- these interactions have different semantics
- the model should not learn them from scratch from a single undifferentiated radius graph

If you want a simple version:

- keep covalent ligand bond edges explicit
- keep ligand-pocket contact edges explicit
- drop pocket-pocket message passing entirely at first

### 6. Align training and runtime representations

Recommendation:

- train the expert on the same feature set it will see inside TargetDiff
- if runtime uses only `z` and `pos`, either improve both together or make the training model match deployment exactly

Why:

- otherwise improvements in the notebook may not carry over to guided sampling

### 7. Do not switch to a plain GAT for this task

Recommendation:

- keep a geometry-aware model, not a topology-only GAT

Why:

- guidance needs gradients with respect to coordinates
- a plain GAT that ignores `pos` is a poor choice for coordinate-gradient guidance

## 8. A Good Near-Term Design

If I were making the smallest change set that is likely to help, I would do this:

1. keep the current ligand+pocket graph setup
2. switch the readout to ligand-only pooling
3. add a learned ligand/pocket type embedding
4. add radial distance embeddings to every edge
5. add explicit edge-type embeddings for ligand bond vs ligand-pocket contact vs pocket-pocket
6. keep the output scalar invariant

That gives you a model that is still compatible with sampling-time gradients, but is much better matched to a `score_only` objective.

## 9. Bottom Line

The current implementation is not failing because the idea of an equivariant guidance model is wrong. It is failing because the effective representation is too stripped down:

- almost all chemistry beyond atomic number is discarded
- distance magnitude is missing from the equivariant message function
- the readout averages over a graph dominated by the fixed pocket

Fix those three things first:

1. ligand-only readout
2. radial distance features
3. explicit ligand/pocket and edge-type information

That is the best path to making `score_only` learnable enough to be useful inside TargetDiff.
