import torch
import math

ALPHA = 1.65 

EDGE_DEFS = [
    (1, 2, 1.0),
    (2, 3, 1.5),
    (2, 4, 1.2),
    (4, 5, 2.0),
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    1: {'type': 'Dirichlet', 'value': 2.0},
    3: {'type': 'Neumann', 'value': 0.0},
    5: {'type': 'Dirichlet', 'value': -1.0}
}

def exact_sol(x, edge_i):
    """
    Omitted for blind testing.
    """
    return None

def source_fn(x, edge_i):
    """
    Explicit forcing function f(x) provided to drive the system 
    in the absence of an exact solution.
    """
    if edge_i == 0:
        return torch.sin(math.pi * x)
    elif edge_i == 1:
        return torch.exp(-x)
    elif edge_i == 2:
        return 2.0 * x
    elif edge_i == 3:
        return torch.cos(x)
    return torch.zeros_like(x)

def reaction(x, edge_i):
    """
    Spatially dependent reaction coefficient, creating a parabolic well.
    """
    return 1.0 + 0.5 * x**2

def k_volt(edge_i, x, z):
    """
    Volterra kernel V(x, z) utilizing exponential decay based on distance.
    """
    return torch.exp(-0.5 * torch.abs(x - z))

def k_fred(edge_i, edge_j, x, z):
    """
    Fredholm kernel F_ij(x, z) with a dynamic cross-edge weighting scalar 
    to test inter-edge coupling strength.
    """
    weight = 1.0 / (edge_i + edge_j + 1.0)
    return weight * torch.sin(x) * torch.cos(z)