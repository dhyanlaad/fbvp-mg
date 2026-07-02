"""
A2 — Star graph, manufactured solution with limited regularity at a vertex.

Same topology/kernels as A1 but edge 0 uses y_0(x) = x^(alpha+0.5) * (L - x)^2 + b
which has limited regularity near x = 0. Other edges remain smooth polynomials.

source_fn returns None => auto-computed from exact_sol by the solver.
"""
import torch

ALPHA = 1.5

EDGE_DEFS = [
    (1, 0, 1.0),   # e0: v1 -> v0
    (0, 2, 1.2),   # e1: v0 -> v2
    (0, 3, 0.8),   # e2: v0 -> v3
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    1: {'type': 'Dirichlet', 'value': 1.0},
    2: {'type': 'Dirichlet', 'value': 1.0},
    3: {'type': 'Dirichlet', 'value': 1.0},
}

_LENGTHS = [1.0, 1.2, 0.8]
_A_SMOOTH = [None, 0.8, 1.5]
_B = 1.0

_C_FRED = [
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.1],
    [0.2, 0.1, 0.6],
]

def exact_sol(x, edge_i):
    L = _LENGTHS[edge_i]
    if edge_i == 0:
        # Limited regularity: x^(alpha + 0.5) vanishes at x=0,
        # (L-x)^2 vanishes at x=L, so y(0)=b, y(L)=b => continuity holds.
        return torch.pow(x.clamp(min=1e-30), ALPHA + 0.5) * (L - x)**2 + _B
    else:
        a = _A_SMOOTH[edge_i]
        return a * x**2 * (L - x)**2 + _B

def source_fn(x, edge_i):
    return None

def reaction(x, edge_i):
    return torch.ones_like(x)

def k_volt(edge_i, x, z):
    return x - z

def k_fred(edge_i, edge_j, x, z):
    return _C_FRED[edge_i][edge_j] * torch.ones_like(x * z)
