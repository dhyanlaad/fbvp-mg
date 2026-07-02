"""
B1 — Volterra-dominant (lambda=1, c_ij=0 off-diagonal).

Topology: 8-edge graph shared across all B problems.
  v0 (degree 4, hub), v1 (degree 3), v2 (degree 3), v7 (degree 2)
  Boundaries: v3, v4, v5, v6 (degree 1)

  v3 --e0-- v0 --e2-- v1 --e5-- v7 --e6-- v2 --e7-- v6
            |         |                    |
  v4 --e1--+    v5 --e4             e3 ---+

lambda=1: Full Volterra, no Fredholm coupling. Each edge decouples.
No exact solution — use explicit forcing.
"""
import torch
import math

ALPHA = 1.5

EDGE_DEFS = [
    (3, 0, 1.0),   # e0
    (4, 0, 1.2),   # e1
    (0, 1, 1.0),   # e2
    (0, 2, 1.5),   # e3
    (5, 1, 0.8),   # e4
    (1, 7, 1.0),   # e5
    (7, 2, 0.9),   # e6
    (2, 6, 1.1),   # e7
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    3: {'type': 'Dirichlet', 'value': 0.0},
    4: {'type': 'Dirichlet', 'value': 0.0},
    5: {'type': 'Dirichlet', 'value': 0.0},
    6: {'type': 'Neumann',   'value': 0.0},
}

def exact_sol(x, edge_i):
    return None

def source_fn(x, edge_i):
    """Smooth sinusoidal forcing on each edge."""
    L = EDGE_DEFS[edge_i][2]
    return torch.sin(math.pi * x / L)

def reaction(x, edge_i):
    return 1.0 + 0.5 * x**2

def k_volt(edge_i, x, z):
    """Full Volterra kernel (lambda=1)."""
    return torch.exp(-0.5 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    """Zero Fredholm — fully decoupled (lambda=1, c_ij=0 off-diagonal)."""
    return torch.zeros_like(x * z)
