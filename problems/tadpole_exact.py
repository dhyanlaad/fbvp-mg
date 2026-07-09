import torch
import math

ALPHA = 1.5

EDGE_DEFS = [
    (0, 1, 1.0),   # e0: triangle
    (1, 2, 1.0),   # e1: triangle
    (2, 0, 1.0),   # e2: triangle
    (0, 3, 1.3),   # e3: tail, leaf vertex 3
]
NUM_EDGES = len(EDGE_DEFS)

# y3(x) = x^2 + 4x. At leaf 3 (x=1.3): y3(1.3) = 6.89
BC_DEFS = {
    3: {'type': 'Dirichlet', 'value': 6.89}
}

# Coefficients for y_i(x) = A_i x^2 + B_i x + C_i
# Derived to satisfy continuity and Kirchhoff flux at all internal vertices (v0, v1, v2)
_A = [1.0, -1.0, 2.0, 1.0]
_B = [-4.0/3.0, 2.0/3.0, -4.0/3.0, 4.0]
_C = [0.0, -1.0/3.0, -2.0/3.0, 0.0]

def exact_sol(x, edge_i):
    if isinstance(x, torch.Tensor):
        return _A[edge_i] * x**2 + _B[edge_i] * x + _C[edge_i]
    else:
        return _A[edge_i] * x**2 + _B[edge_i] * x + _C[edge_i]

def source_fn(x, edge_i):
    return None

def reaction(x, edge_i):
    return 1.0 + 0.5 * x**2

def k_volt(edge_i, x, z):
    return 0.5 * torch.exp(-0.5 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    weight = 0.5 / (1.0 + abs(edge_i - edge_j))
    return weight * torch.cos(x) * torch.sin(z)

if __name__ == "__main__":
    def y(i, x): return _A[i]*x**2 + _B[i]*x + _C[i]
    def dy(i, x): return 2*_A[i]*x + _B[i]
    
    print("--- Tadpole Exact Constraints Check ---")
    print(f"Cont v1: {y(0, 1.0) - y(1, 0.0):.2e}")
    print(f"Cont v2: {y(1, 1.0) - y(2, 0.0):.2e}")
    print(f"Cont v0 (0=2): {y(0, 0.0) - y(2, 1.0):.2e}")
    print(f"Cont v0 (0=3): {y(0, 0.0) - y(3, 0.0):.2e}")
    print(f"Kirch v1: {-dy(0, 1.0) + dy(1, 0.0):.2e}")
    print(f"Kirch v2: {-dy(1, 1.0) + dy(2, 0.0):.2e}")
    print(f"Kirch v0: {dy(0, 0.0) - dy(2, 1.0) + dy(3, 0.0):.2e}")
    print("If all are ~1e-16, constraints are perfectly satisfied.")
