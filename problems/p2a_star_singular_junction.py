import torch

ALPHA = 1.5

# 3 edges, a simple star graph. 
# Node 0 is the center, nodes 1, 2, 3 are leaves
EDGE_DEFS = [
    (0, 1, 1.0), # Edge 0
    (0, 2, 1.0), # Edge 1
    (0, 3, 1.0), # Edge 2
]

NUM_EDGES = len(EDGE_DEFS)

# Boundary conditions at the leaves (nodes 1, 2, 3)
BC_DEFS = {
    1: {'type': 'Dirichlet', 'value': 0.0},
    2: {'type': 'Dirichlet', 'value': 0.0},
    3: {'type': 'Dirichlet', 'value': 0.0}
}

def exact_sol(x, edge_i):
    """
    Limited regularity exact solution: y_i(x) = x^{1.5} * (L - x)^2.
    At x=0 (the central vertex):
    y(0) = 0 (Continuity is satisfied: 0 = 0 = 0)
    y'(0) = 0 (Kirchhoff is satisfied: 0 + 0 + 0 = 0)
    
    The second derivative y''(x) diverges as x->0, creating a singularity.
    """
    L = EDGE_DEFS[edge_i][2]
    return (x**1.5) * ((L - x)**2)

def source_fn(x, edge_i):
    """
    Will be computed automatically by applying the operator to exact_sol.
    """
    return None

def reaction(x, edge_i):
    """
    Standard reaction term: r(x) = 1.0
    """
    return torch.ones_like(x)

def k_volt(edge_i, x, xi):
    """
    Volterra kernel: k_V = 0.5 * e^{-(x-xi)}
    """
    return 0.5 * torch.exp(-(x - xi))

def k_fred(edge_i, edge_j, x, xi):
    """
    Fredholm kernel: Weak coupling across all edges
    """
    return 0.1 * torch.sin(x) * torch.cos(xi)
