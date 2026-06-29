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
    """
    Manufactured exact solution for the star graph.
    Designed to perfectly satisfy:
    1. Continuity at central junction (x=1 for inflows, x=0 for outflows):
       u_0(1) = u_1(1) = u_2(0) = u_3(0) = 1.0
    2. Kirchhoff at central junction:
       Incoming flux: u_0'(1) + u_1'(1) = 0 + 2 = 2
       Outgoing flux: u_2'(0) + u_3'(0) = 0 + 2 = 2
       Net flux = 2 - 2 = 0
    3. Boundary conditions:
       u_0(0) = 0 (Dirichlet at v1)
       u_1(0) = 0 (Dirichlet at v2)
       u_2'(1) = -pi/2 (Neumann at v3)
       u_3'(1) = 2 (Neumann at v4)
    """
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
    """
    Returns None so that train.py automatically calculates the exact source term
    dynamically using the continuous autograd fractional derivative evaluator.
    """
    return None

def reaction(x, edge_i):
    """Reaction coefficient R(x)"""
    return torch.ones_like(x)

def k_volt(edge_i, x, z):
    """Volterra kernel V(x, z)"""
    return torch.exp(x - z)

def k_fred(edge_i, edge_j, x, z):
    """Fredholm kernel F_ij(x, z)"""
    return torch.cos(x) * torch.sin(z)
