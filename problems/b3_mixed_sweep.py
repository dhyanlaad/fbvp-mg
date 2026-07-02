"""
B3 — Mixed regime (lambda=0.5).

Same topology as B1/B2. Both Volterra and Fredholm active,
each scaled by their respective lambda factor.
Default lambda=0.5; for the sweep {1, 0.75, 0.5, 0.25, 0} adjust LAMBDA.
"""
import torch
import math

ALPHA = 1.5
LAMBDA = 0.5  # Mixing parameter; adjust for sweep experiments

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

_D_EDGE = [
    [0, 1, 1, 1, 2, 2, 2, 2],
    [1, 0, 1, 1, 2, 2, 2, 2],
    [1, 1, 0, 1, 1, 1, 2, 2],
    [1, 1, 1, 0, 2, 2, 1, 1],
    [2, 2, 1, 2, 0, 1, 2, 3],
    [2, 2, 1, 2, 1, 0, 1, 2],
    [2, 2, 2, 1, 2, 1, 0, 1],
    [2, 2, 2, 1, 3, 2, 1, 0],
]
_C_FRED = [[1.0 / (1.0 + _D_EDGE[i][j]) for j in range(8)] for i in range(8)]

def exact_sol(x, edge_i):
    return None

def source_fn(x, edge_i):
    L = EDGE_DEFS[edge_i][2]
    return torch.sin(math.pi * x / L)

def reaction(x, edge_i):
    return 1.0 + 0.5 * x**2

def k_volt(edge_i, x, z):
    """Volterra kernel scaled by lambda."""
    return LAMBDA * torch.exp(-0.5 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    """Fredholm kernel scaled by (1 - lambda)."""
    return (1.0 - LAMBDA) * _C_FRED[edge_i][edge_j] * torch.cos(x) * torch.sin(z)
