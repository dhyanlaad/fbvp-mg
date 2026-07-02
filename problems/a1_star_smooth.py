"""
A1 — Star graph, 3 edges, smooth manufactured solution.

Topology: Star with central vertex v0, boundary vertices v1, v2, v3.
  v1 --(e0, L=1.0)-- v0 --(e1, L=1.2)-- v2
                       |
                  (e2, L=0.8)
                       |
                       v3

Manufactured solution: y_i(x) = a_i * x^2 * (l_i - x)^2 + b_i
  with b_i = 1.0 for all edges (ensures continuity at v0).
  Derivatives vanish at both endpoints => Kirchhoff automatically satisfied.

Kernels:
  Volterra: k_1(x, xi) = x - xi
  Fredholm: k_2(i,j)(x, xi) = c_ij (constant, dense, symmetric)

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
_A = [1.0, 0.8, 1.5]
_B = 1.0

# Fredholm coupling matrix c_ij, dense, symmetric, values in [0.1, 1.0]
_C_FRED = [
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.1],
    [0.2, 0.1, 0.6],
]

def exact_sol(x, edge_i):
    L = _LENGTHS[edge_i]
    a = _A[edge_i]
    return a * x**2 * (L - x)**2 + _B

def source_fn(x, edge_i):
    return None

def reaction(x, edge_i):
    return torch.ones_like(x)

def k_volt(edge_i, x, z):
    """Volterra kernel: k_1(x, xi) = x - xi"""
    return x - z

def k_fred(edge_i, edge_j, x, z):
    """Fredholm kernel: constant c_ij"""
    return _C_FRED[edge_i][edge_j] * torch.ones_like(x * z)
