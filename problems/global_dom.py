import torch
import math

ALPHA = 1.5

EDGE_DEFS = [
    (1, 0, 1.0),
    (2, 0, 1.0),
    (0, 3, 1.0),
    (0, 4, 1.0),
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    1: {'type': 'Dirichlet', 'value': 0.0},
    2: {'type': 'Dirichlet', 'value': 0.0},
    3: {'type': 'Neumann', 'value': -math.pi / 2.0},
    4: {'type': 'Neumann', 'value': 2.0}
}

def exact_sol(x, edge_i):
    if edge_i == 0:
        return torch.sin(math.pi * x / 2.0)
    elif edge_i == 1:
        return x**2
    elif edge_i == 2:
        return torch.cos(math.pi * x / 2.0)
    elif edge_i == 3:
        return 1.0 + 2.0 * x
    return None

def source_fn(x, edge_i):
    return None

def reaction(x, edge_i):
    """Reaction coefficient R(x), made small for global domination"""
    return 0.01 * torch.ones_like(x)

def k_volt(edge_i, x, z):
    """Volterra kernel V(x, z), small for global interactions"""
    return 0.01 * torch.exp(x - z)

def k_fred(edge_i, edge_j, x, z):
    """Fredholm kernel F_ij(x, z), made large to maximize global interactions across edges"""
    return 100.0 * torch.cos(x) * torch.sin(z)
