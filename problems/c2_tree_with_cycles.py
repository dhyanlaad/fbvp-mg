"""
C2 — Graph with cycles: C1's binary tree plus 2 extra edges closing cycles.

Adds edges: v3→v5 (leaf-to-leaf cross-link), v7→v8 (terminal back to root).
This creates an explicitly non-acyclic graph with 9 edges.

After adding cycles:
  v8: deg 2 (e0 out, e8 in) -> internal
  v3: deg 2 (e3 in, e7 out) -> internal
  v5: deg 2 (e5 in, e7 in) -> internal  
  v7: deg 2 (e6 in, e8 out) -> internal
  
We need at least one boundary vertex, so we add a pendant edge
from v0 to a new leaf v9 to create a boundary.

Boundary vertices: v9
Internal vertices: v0, v1, v2, v3, v4, v5, v7, v8
"""
import torch
import math

ALPHA = 1.5
LAMBDA = 0.5

EDGE_DEFS = [
    # Original tree edges (same as C1)
    (8, 0, 1.0),   # e0: root v8 → v0
    (0, 1, 1.2),   # e1: v0 → v1
    (0, 2, 1.0),   # e2: v0 → v2
    (1, 3, 0.9),   # e3: v1 → v3
    (1, 4, 1.1),   # e4: v1 → v4
    (2, 5, 0.7),   # e5: v2 → v5
    (4, 7, 0.8),   # e6: v4 → v7
    # Cycle-closing edges
    (3, 5, 1.3),   # e7: v3 → v5 (leaf-to-leaf cycle)
    (7, 8, 0.9),   # e8: v7 → v8 (terminal back to root cycle)
    # Pendant for boundary
    (0, 9, 0.6),   # e9: v0 → v9 (new boundary leaf)
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    9: {'type': 'Dirichlet', 'value': 1.0},
}

def exact_sol(x, edge_i):
    return None

def source_fn(x, edge_i):
    L = EDGE_DEFS[edge_i][2]
    return torch.sin(math.pi * x / L) + 0.5

def reaction(x, edge_i):
    return 1.0 + 0.3 * x

def k_volt(edge_i, x, z):
    return LAMBDA * torch.exp(-0.5 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    weight = 1.0 / (1.0 + abs(edge_i - edge_j))
    return (1.0 - LAMBDA) * weight * torch.cos(x) * torch.sin(z)
