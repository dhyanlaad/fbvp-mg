"""
B2 — Fredholm-dominant (lambda=0, k_1=0, dense c_ij).

Same topology as B1. c_ij = 1/(1 + d_ij) where d_ij is shortest-path
graph distance in the line graph of the topology.

Source: localized Gaussian bump on edge 4 only, zero elsewhere.
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

# Precomputed shortest-path distances in the line graph of the topology
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
    """Localized Gaussian bump on edge 4 only."""
    if edge_i == 4:
        L = EDGE_DEFS[4][2]
        mu = L / 2.0
        sigma = L / 6.0
        return 5.0 * torch.exp(-((x - mu)**2) / (2 * sigma**2))
    return torch.zeros_like(x)

def reaction(x, edge_i):
    return 1.0 + 0.5 * x**2

def k_volt(edge_i, x, z):
    """Zero Volterra (lambda=0)."""
    return torch.zeros_like(x * z)

def k_fred(edge_i, edge_j, x, z):
    """Dense Fredholm coupling: c_ij / (1 + d_ij)."""
    return _C_FRED[edge_i][edge_j] * torch.cos(x) * torch.sin(z)
