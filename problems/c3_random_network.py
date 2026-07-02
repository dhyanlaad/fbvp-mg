"""
C3 — Larger random network: Barabasi-Albert graph (n=20, m=2).

Generated at import time with a fixed random seed for reproducibility.
Edge lengths ~ Uniform(0.5, 2.0). No closed-form reference — use
finest-mesh solution as pseudo-reference with Richardson extrapolation.

Pendant edges are added to ensure at least one boundary vertex exists
(BA graphs with m>=2 have minimum degree 2).
"""
import torch
import math
import random

ALPHA = 1.5
LAMBDA = 0.5

# --- Deterministic Barabasi-Albert generation (seed=42) ---
def _generate_ba_graph(n, m, seed=42):
    """Generate a Barabasi-Albert preferential attachment graph."""
    rng = random.Random(seed)
    
    # Start with a complete graph on m+1 nodes
    adj = {i: set() for i in range(m + 1)}
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i].add(j)
            adj[j].add(i)
    
    edges = []
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            edges.append((i, j))
    
    # Degree list for preferential attachment
    degree = [m] * (m + 1)
    
    for new_node in range(m + 1, n):
        adj[new_node] = set()
        targets = set()
        pool = []
        for v in range(new_node):
            pool.extend([v] * degree[v])
        
        while len(targets) < m and pool:
            chosen = rng.choice(pool)
            if chosen not in targets:
                targets.add(chosen)
        
        degree.append(0)
        for t in targets:
            adj[new_node].add(t)
            adj[t].add(new_node)
            edges.append((min(new_node, t), max(new_node, t)))
            degree[new_node] += 1
            degree[t] += 1
    
    return edges, adj, degree

def _assign_lengths(edges, seed=42):
    """Assign edge lengths ~ Uniform(0.5, 2.0)."""
    rng = random.Random(seed + 1)
    return [(u, v, round(rng.uniform(0.5, 2.0), 2)) for u, v in edges]

_raw_edges, _adj, _degrees = _generate_ba_graph(20, 2)

# Add pendant edges to create boundary vertices (BA m=2 has min degree 2)
_max_node = max(max(u, v) for u, v in _raw_edges)
_pendant_edges = [
    (_max_node + 1, 0, 0.8),   # pendant from hub node 0
    (_max_node + 2, 5, 0.7),   # pendant from node 5
]
_degrees.extend([1, 1])  # pendant nodes are degree 1

_all_edges = _assign_lengths(_raw_edges) + _pendant_edges
EDGE_DEFS = _all_edges
NUM_EDGES = len(EDGE_DEFS)

# Boundary vertices: the pendant leaf nodes
BC_DEFS = {
    _max_node + 1: {'type': 'Dirichlet', 'value': 1.0},
    _max_node + 2: {'type': 'Dirichlet', 'value': 0.0},
}

def exact_sol(x, edge_i):
    return None

def source_fn(x, edge_i):
    L = EDGE_DEFS[edge_i][2]
    return torch.sin(math.pi * x / L) + 0.5

def reaction(x, edge_i):
    return 1.0 + 0.2 * x

def k_volt(edge_i, x, z):
    return LAMBDA * torch.exp(-0.3 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    weight = 1.0 / (1.0 + abs(edge_i - edge_j))
    return (1.0 - LAMBDA) * weight * 0.5 * torch.cos(x) * torch.sin(z)
