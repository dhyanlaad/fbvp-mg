import torch

ALPHA = 1.5

# Branched path graph: 5 edges, 6 nodes
#
#   4 ---e3--- 1 ---e0--- 0 ---e1--- 2 ---e2--- 5
#                                     |
#                                    e4
#                                     |
#                                     3
#
# Internal vertices: 0 (deg 2), 1 (deg 2), 2 (deg 3)
# Boundary vertices: 3, 4, 5

EDGE_DEFS = [
    (1, 0, 1.0),  # e0: node 1 → node 0
    (0, 2, 1.0),  # e1: node 0 → node 2
    (2, 5, 1.0),  # e2: node 2 → node 5 (leaf)
    (1, 4, 0.8),  # e3: node 1 → node 4 (leaf)
    (2, 3, 0.8),  # e4: node 2 → node 3 (leaf)
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    5: {'type': 'Dirichlet', 'value': 3.0},
    4: {'type': 'Dirichlet', 'value': -1.0},
    3: {'type': 'Dirichlet', 'value': 1.5},
}

# Exact solutions: y_i(x) = A_i x^2 + B_i x + C_i
# Coefficients derived to satisfy continuity and Kirchhoff at nodes 0, 1, 2.
_A = [-1.0, 1.0, 0.0, 0.9375, -2.03125]
_B = [2.0, 0.0, 1.0, -2.0, 1.0]
_C = [0.0, 1.0, 2.0, 0.0, 2.0]

def exact_sol(x, edge_i):
    return _A[edge_i] * x**2 + _B[edge_i] * x + _C[edge_i]

def source_fn(x, edge_i):
    return None

def reaction(x, edge_i):
    return 0.3 + 0.1 * x

def k_volt(edge_i, x, z):
    """
    Weak Volterra kernel: magnitude ~0.05.
    Deliberately small so the Fredholm term dominates.
    """
    return 0.05 * torch.exp(-2.0 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    """
    Strong Fredholm kernel: magnitude ~2.0–4.0.
    Dense cross-edge coupling with large amplitude.
    """
    base = 2.0
    adjacency = 1.0 / (1.0 + 0.3 * abs(edge_i - edge_j))
    return base * adjacency * (1.0 + 0.5 * torch.cos(x - z))


if __name__ == "__main__":
    def y(i, x): return _A[i]*x**2 + _B[i]*x + _C[i]
    def dy(i, x): return 2*_A[i]*x + _B[i]

    print("--- Branched-path Fredholm-Dominant: Constraints Check ---")
    print(f"BC node 5: y_2(1.0)  = {y(2, 1.0):.6f} (expect 3.0)")
    print(f"BC node 4: y_3(0.8)  = {y(3, 0.8):.6f} (expect -1.0)")
    print(f"BC node 3: y_4(0.8)  = {y(4, 0.8):.6f} (expect 1.5)")
    print()
    print(f"Cont node 1 (e0=e3): {y(0, 0.0) - y(3, 0.0):.2e}")
    print(f"Cont node 0 (e0=e1): {y(0, 1.0) - y(1, 0.0):.2e}")
    print(f"Cont node 2 (e1=e2): {y(1, 1.0) - y(2, 0.0):.2e}")
    print(f"Cont node 2 (e1=e4): {y(1, 1.0) - y(4, 0.0):.2e}")
    print()
    print(f"Kirch node 1: {dy(0, 0.0) + dy(3, 0.0):.2e}")
    print(f"Kirch node 0: {-dy(0, 1.0) + dy(1, 0.0):.2e}")
    print(f"Kirch node 2: {-dy(1, 1.0) + dy(2, 0.0) + dy(4, 0.0):.2e}")
    print("If all are ~0, constraints are perfectly satisfied.")
