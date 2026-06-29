import torch
import math

ALPHA = 1.5

EDGE_DEFS = [(0, 1, 1.0), (1, 1, 2.0)]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    0: {'type': 'Neumann', 'value': 0.0}
}

def exact_sol(x, edge_i):
    """
    Manufactured solution for the tadpole graph.
    Tail (edge 0): 2x^3 - 3x^2 + 2
    Loop (edge 1): cos(pi * x)

    This ensures:
    1. Continuity at junction (v=1): u_0(1) = 1, u_1(0) = 1, u_1(2) = 1
    2. Kirchhoff at junction (v=1): flux is u_0'(1) - u_1'(0) + u_1'(2) = 0 - 0 + 0 = 0
    3. Neumann at tail end (v=0): u_0'(0) = 0
    """
    if edge_i == 0:
        return 2.0 * (x**3) - 3.0 * (x**2) + 2.0
    elif edge_i == 1:
        return torch.cos(math.pi * x)

def k_volt(edge_i, x, zeta):
    return torch.zeros_like(x * zeta)

def k_fred(edge_i, edge_j, x, zeta):
    return torch.zeros_like(x * zeta)

def reaction(x, edge_i):
    return torch.ones_like(x)

def source_fn(x, edge_i):
    return None
