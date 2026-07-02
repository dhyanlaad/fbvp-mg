"""
C1 — Binary tree, depth 3 (7 edges).

Topology (lambda=0.5 fixed):
  v8 (root, boundary)
   |
   e0 (L=1.0)
   |
  v0 (hub)
  ├── e1 (L=1.2) → v1
  │   ├── e3 (L=0.9) → v3 (leaf)
  │   └── e4 (L=1.1) → v4
  │       └── e6 (L=0.8) → v7 (leaf, depth 3)
  └── e2 (L=1.0) → v2
      ├── e5 (L=0.7) → v5 (leaf)
      └── (terminal at v6 via e5)

Boundary vertices: v8 (root), v3, v5, v6, v7
Internal vertices: v0 (deg 3), v1 (deg 3), v2 (deg 3), v4 (deg 2)
"""
import torch
import math

ALPHA = 1.5
LAMBDA = 0.5

EDGE_DEFS = [
    (8, 0, 1.0),   # e0: root v8 → hub v0
    (0, 1, 1.2),   # e1: v0 → v1
    (0, 2, 1.0),   # e2: v0 → v2
    (1, 3, 0.9),   # e3: v1 → v3 (leaf)
    (1, 4, 1.1),   # e4: v1 → v4
    (2, 5, 0.7),   # e5: v2 → v5 (leaf)
    (4, 7, 0.8),   # e6: v4 → v7 (leaf, depth 3)
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    8: {'type': 'Dirichlet', 'value': 1.0},
    3: {'type': 'Dirichlet', 'value': 0.0},
    5: {'type': 'Dirichlet', 'value': 0.0},
    7: {'type': 'Neumann',   'value': 0.0},
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
