"""SynFrag inference: SMILES in, synthetic accessibility score out.

Wraps the released SynFrag AttentiveFP predictor (Zhang et al., J. Chem. Inf.
Model. 2026, 66, 2997-3012), taken from https://github.com/simmzx/SynFrag at
commit 2c6cacec198c2746cade22643330d38cc19c62a4. ``synfrag/AttfpMPNN.py`` is the
authors' file, vendored byte-identical, and ``synfrag_default.pth`` is their
released checkpoint. The score is ``1 - model output``, exactly as in the authors'
``synfrag.py``: high means easy to synthesize.

Three upstream problems are worked around here, in the wrapper only:

1. The vendored file calls ``dgl.function.copy_edge`` and ``src_mul_edge``, which
   DGL removed in 0.9. Both were renamed, not changed (``copy_e``, ``u_mul_e``),
   so aliases are registered below before the vendored module is imported.

2. The checkpoint is pickled from CUDA tensors and ``from_pretrained_all`` loads it
   without ``map_location``, so it fails on a CPU-only machine. The state dict is
   loaded here with ``map_location="cpu"`` and ``strict=True`` (the upstream call is
   strict as well).

3. Upstream ``predict_batch`` wraps the whole batch in one ``try`` and returns
   ``None`` for every molecule in it when any one of them fails to featurize, with
   only a printed warning. Here a failed batch is retried molecule by molecule, so
   a single bad row costs only that row. The model has no batch-dependent layer in
   eval mode, so a molecule scores the same alone or in a batch.
"""

import os
from pathlib import Path

import numpy as np
import torch

import dgl.function as _fn

# DGL >= 0.9 renamed these; the vendored file still uses the old names.
for _old, _new in (("copy_edge", "copy_e"), ("src_mul_edge", "u_mul_e")):
    if not hasattr(_fn, _old):
        setattr(_fn, _old, getattr(_fn, _new))

import dgl  # noqa: E402
from rdkit import Chem  # noqa: E402

from synfrag.AttfpMPNN import AttentiveFPPredictor  # noqa: E402

# Same hyperparameters as load_model() in the authors' synfrag.py.
NODE_FEAT_SIZE = 30
EDGE_FEAT_SIZE = 11
BATCH_SIZE = 32

# model/checkpoints/ is gitignored: the 12.8 MB checkpoint is hosted on eosvc and
# restored into this directory at pack time. Path stays relative to this file.
CHECKPOINT_PATH = Path(
    os.environ.get(
        "SYNFRAG_CHECKPOINT",
        Path(__file__).resolve().parents[2] / "checkpoints" / "synfrag_default.pth",
    )
)

_model = None
_featurizer = None


def _load():
    """Build the model and featurizer on CPU once per process."""
    global _model, _featurizer
    if _model is not None:
        return _model, _featurizer

    import deepchem as dc

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("SynFrag checkpoint not found: %s" % CHECKPOINT_PATH)

    model = AttentiveFPPredictor(
        node_feat_size=NODE_FEAT_SIZE,
        edge_feat_size=EDGE_FEAT_SIZE,
        num_layers=4,
        num_timesteps=1,
        graph_feat_size=300,
        n_tasks=1,
        dropout=0,
    )
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    torch.set_num_threads(1)

    _model = model
    _featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    return _model, _featurizer


def _standardize(smiles):
    """Canonical RDKit SMILES, or None. Mirrors the authors' standardize_smiles."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def _score(model, featurizer, smiles_list):
    """Score canonical SMILES in one batch; raises if any molecule fails."""
    graphs = featurizer.featurize(smiles_list)
    dgl_graphs = [g.to_dgl_graph(self_loop=True) for g in graphs]
    batch_graph = dgl.batch(dgl_graphs)
    with torch.no_grad():
        pred = model(batch_graph, batch_graph.ndata["x"], batch_graph.edata["edge_attr"])
    return (1.0 - pred.detach().cpu().flatten()).tolist()


def predict(smiles_list):
    """Return a float array of SynFrag scores, NaN where a SMILES cannot be scored."""
    model, featurizer = _load()
    scores = np.full(len(smiles_list), np.nan, dtype=np.float64)

    valid = []
    for i, smi in enumerate(smiles_list):
        std = _standardize(smi)
        if std is not None:
            valid.append((i, std))

    for start in range(0, len(valid), BATCH_SIZE):
        chunk = valid[start:start + BATCH_SIZE]
        idx = [i for i, _ in chunk]
        smis = [s for _, s in chunk]
        try:
            scores[idx] = _score(model, featurizer, smis)
        except Exception:
            # One molecule broke the batch (upstream would lose all of them).
            for i, s in chunk:
                try:
                    scores[i] = _score(model, featurizer, [s])[0]
                except Exception:
                    pass
    return scores
