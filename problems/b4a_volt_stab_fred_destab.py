"""
B4a — Volterra stabilizing (k_1 < 0, damping), Fredholm destabilizing (large positive c_ij).

Same topology as other B problems. Stress test for competing sign/scales.
"""
import torch
import math

ALPHA = 1.5

EDGE_DEFS = [
    (3, 0, 1.0),
    (4, 0, 1.2),
    (0, 1, 1.0),
    (0, 2, 1.5),
    (5, 1, 0.8),
    (1, 7, 1.0),
    (7, 2, 0.9),
    (2, 6, 1.1),
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
    L = EDGE_DEFS[edge_i][2]
    return torch.sin(math.pi * x / L)

def reaction(x, edge_i):
    return 1.0 + 0.5 * x**2

def k_volt(edge_i, x, z):
    """Stabilizing (negative) Volterra kernel — acts as damping."""
    return -2.0 * torch.exp(-0.5 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    """Destabilizing (large positive) Fredholm coupling."""
    return 5.0 * torch.cos(x) * torch.sin(z)
