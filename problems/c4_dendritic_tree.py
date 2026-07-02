"""
C4 — Physically-motivated network: synthetic dendritic-tree topology.

Models a neural dendrite with branching factor decreasing with depth:
  - Depth 0 (soma):   1 root node, branching factor 3
  - Depth 1:          3 nodes, branching factor 2
  - Depth 2:          6 nodes, branching factor 1
  - Depth 3:          6 terminal nodes (synaptic inputs)

Total: 16 vertices, 15 edges. Source decays from soma consistent
with expected transport/diffusion behavior in neural dendrites.

Option (b) from the prompt: synthetic dendritic-tree topology.
"""
import torch
import math

ALPHA = 1.5
LAMBDA = 0.5

# Build dendritic tree topology
# Depth 0: v0 (soma, root)
# Depth 1: v1, v2, v3 (primary branches)
# Depth 2: v4-v9 (secondary branches)
# Depth 3: v10-v15 (terminals)
EDGE_DEFS = [
    # Primary branches (soma → depth 1), longer and thicker
    (0, 1, 2.0),    # e0
    (0, 2, 1.8),    # e1
    (0, 3, 1.9),    # e2
    # Secondary branches (depth 1 → depth 2), medium length
    (1, 4, 1.2),    # e3
    (1, 5, 1.1),    # e4
    (2, 6, 1.3),    # e5
    (2, 7, 1.0),    # e6
    (3, 8, 1.2),    # e7
    (3, 9, 1.1),    # e8
    # Terminals (depth 2 → depth 3), short
    (4, 10, 0.7),   # e9
    (5, 11, 0.6),   # e10
    (6, 12, 0.8),   # e11
    (7, 13, 0.7),   # e12
    (8, 14, 0.6),   # e13
    (9, 15, 0.5),   # e14
]
NUM_EDGES = len(EDGE_DEFS)

# Soma is boundary (degree 1 in undirected sense? No, degree 3).
# Terminals v10-v15 are boundaries (degree 1).
# v0 has degree 3 => internal.
# All other internal nodes: v1-v9 (degree >= 2).
BC_DEFS = {
    10: {'type': 'Dirichlet', 'value': 0.0},
    11: {'type': 'Dirichlet', 'value': 0.0},
    12: {'type': 'Dirichlet', 'value': 0.0},
    13: {'type': 'Dirichlet', 'value': 0.0},
    14: {'type': 'Dirichlet', 'value': 0.0},
    15: {'type': 'Dirichlet', 'value': 0.0},
}

# Depth mapping for each edge (used for distance-based source decay)
_EDGE_DEPTH = [0]*3 + [1]*6 + [2]*6

def exact_sol(x, edge_i):
    return None

def source_fn(x, edge_i):
    """Source concentrated at soma (depth 0), decaying with dendritic depth.
    Models signal propagation from cell body to synaptic terminals."""
    depth = _EDGE_DEPTH[edge_i]
    L = EDGE_DEFS[edge_i][2]
    amplitude = 3.0 * math.exp(-1.5 * depth)
    return amplitude * torch.sin(math.pi * x / L)

def reaction(x, edge_i):
    """Mild reaction with depth-dependent increase (membrane leakage)."""
    depth = _EDGE_DEPTH[edge_i]
    return (0.5 + 0.3 * depth) * torch.ones_like(x)

def k_volt(edge_i, x, z):
    return LAMBDA * torch.exp(-0.5 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    """Cross-edge coupling decays with edge-index distance."""
    weight = 1.0 / (1.0 + abs(edge_i - edge_j))
    return (1.0 - LAMBDA) * weight * 0.3 * torch.cos(x) * torch.sin(z)
