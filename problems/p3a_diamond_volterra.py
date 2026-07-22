import torch

ALPHA = 1.5

# H-graph: 4 edges, 6 nodes
#
#   0 ---e0--- 1 ---e1--- 2
#              |
#             e2
#              |
#              3 ---e3--- 4
#
# Internal vertices: 1 (degree 3), 3 (degree 2)
# Boundary vertices: 0, 2, 4

EDGE_DEFS = [
    (0, 1, 1.0),  # e0: left arm
    (1, 2, 1.0),  # e1: right arm
    (1, 3, 0.8),  # e2: vertical bridge
    (3, 4, 1.0),  # e3: bottom arm
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    0: {'type': 'Dirichlet', 'value': 1.0},
    2: {'type': 'Dirichlet', 'value': -0.5},
    4: {'type': 'Dirichlet', 'value': 2.0},
}

# Exact solutions: y_i(x) = A_i x^2 + B_i x + C_i
# Coefficients derived to satisfy continuity and Kirchhoff at nodes 1 and 3.
_A = [1.0, 2.0, 1.0, -0.14]
_B = [-3.0, -1.5, 0.5, 2.1]
_C = [1.0, -1.0, -1.0, 0.04]

def exact_sol(x, edge_i):
    return _A[edge_i] * x**2 + _B[edge_i] * x + _C[edge_i]

def source_fn(x, edge_i):
    return None

def reaction(x, edge_i):
    return 0.5 + 0.2 * x

def k_volt(edge_i, x, z):
    """
    Strong Volterra kernel: magnitude ~3.0-4.5.
    Edge-dependent exponential decay with high amplitude.
    """
    scale = 3.0 + 0.5 * edge_i
    return scale * torch.exp(-(x - z))

def k_fred(edge_i, edge_j, x, z):
    """
    Weak Fredholm kernel: magnitude ~0.05.
    Deliberately small cross-edge coupling so the Volterra term dominates.
    """
    weight = 0.05 / (1.0 + abs(edge_i - edge_j))
    return weight * torch.cos(x) * torch.sin(z)


if __name__ == "__main__":
    def y(i, x): return _A[i]*x**2 + _B[i]*x + _C[i]
    def dy(i, x): return 2*_A[i]*x + _B[i]

    print("--- H-graph Volterra-Dominant: Constraints Check ---")
    print(f"BC node 0: y_0(0) = {y(0, 0.0):.6f} (expect 1.0)")
    print(f"BC node 2: y_1(1) = {y(1, 1.0):.6f} (expect -0.5)")
    print(f"BC node 4: y_3(1) = {y(3, 1.0):.6f} (expect 2.0)")
    print()
    print(f"Cont node 1 (e0=e1): {y(0, 1.0) - y(1, 0.0):.2e}")
    print(f"Cont node 1 (e0=e2): {y(0, 1.0) - y(2, 0.0):.2e}")
    print(f"Cont node 3 (e2=e3): {y(2, 0.8) - y(3, 0.0):.2e}")
    print()
    print(f"Kirch node 1: {-dy(0, 1.0) + dy(1, 0.0) + dy(2, 0.0):.2e}")
    print(f"Kirch node 3: {-dy(2, 0.8) + dy(3, 0.0):.2e}")
    print("If all are ~0, constraints are perfectly satisfied.")
