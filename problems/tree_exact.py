import torch
import math

ALPHA = 1.5

EDGE_DEFS = [
    (0, 1, 1.0),  # e0: root -> left child
    (0, 2, 1.0),  # e1: root -> right child
    (1, 3, 0.8),  # e2: left child -> leaf
    (1, 4, 0.8),  # e3: left child -> leaf
    (2, 5, 0.8),  # e4: right child -> leaf
    (2, 6, 0.8),  # e5: right child -> leaf
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    3: {'type': 'Dirichlet', 'value': 4.08},
    4: {'type': 'Dirichlet', 'value': 2.32},
    5: {'type': 'Dirichlet', 'value': 0.72},
    6: {'type': 'Dirichlet', 'value': -7.12},
}

# Coefficients for y_i(x) = A_i x^2 + B_i x + C_i
# Derived to strictly satisfy continuity and Kirchhoff at v0, v1, v2
_A = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
_B = [1.0, -1.0, 1.0, 2.0, 1.0, -4.0]
_C = [0.0, 0.0, 2.0, 2.0, -2.0, -2.0]

def exact_sol(x, edge_i):
    if isinstance(x, torch.Tensor):
        return _A[edge_i] * x**2 + _B[edge_i] * x + _C[edge_i]
    else:
        return _A[edge_i] * x**2 + _B[edge_i] * x + _C[edge_i]

def source_fn(x, edge_i):
    return None

def reaction(x, edge_i):
    return 1.0 + 0.3 * x

def k_volt(edge_i, x, z):
    return 0.5 * torch.exp(-0.5 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    weight = 0.5 / (1.0 + abs(edge_i - edge_j))
    return weight * torch.cos(x) * torch.sin(z)

if __name__ == "__main__":
    def y(i, x): return _A[i]*x**2 + _B[i]*x + _C[i]
    def dy(i, x): return 2*_A[i]*x + _B[i]
    
    print("--- Tree Exact Constraints Check ---")
    print(f"Cont v0: {y(0, 0.0) - y(1, 0.0):.2e}")
    print(f"Cont v1 (0=2): {y(0, 1.0) - y(2, 0.0):.2e}")
    print(f"Cont v1 (0=3): {y(0, 1.0) - y(3, 0.0):.2e}")
    print(f"Cont v2 (1=4): {y(1, 1.0) - y(4, 0.0):.2e}")
    print(f"Cont v2 (1=5): {y(1, 1.0) - y(5, 0.0):.2e}")
    
    print(f"Kirch v0: {dy(0, 0.0) + dy(1, 0.0):.2e}")
    print(f"Kirch v1: {-dy(0, 1.0) + dy(2, 0.0) + dy(3, 0.0):.2e}")
    print(f"Kirch v2: {-dy(1, 1.0) + dy(4, 0.0) + dy(5, 0.0):.2e}")
    print("If all are ~1e-16, constraints are perfectly satisfied.")
