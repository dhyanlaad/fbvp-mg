import torch
import numpy as np
from scipy.interpolate import interp1d

ALPHA = 1.5

# Tadpole Topology
# e0, e1, e2 form a cycle/triangle (v0 -> v1 -> v2 -> v0)
# e3 is a tail/pendant starting from v0 and ending at a leaf v3
EDGE_DEFS = [
    (0, 1, 1.0),   # e0: triangle
    (1, 2, 1.0),   # e1: triangle
    (2, 0, 1.0),   # e2: triangle
    (0, 3, 1.3),   # e3: tail, leaf vertex 3
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    3: {'type': 'Dirichlet', 'value': 0.0}
}

# Generate Whittle-Matérn random fields for each edge
# We precompute a single realization so it stays constant across solver runs
np.random.seed(42)
nu = 1.5  # smoothness
length_scale = 0.2

def matern_kernel(dist, nu, length_scale):
    # For nu=1.5: (1 + sqrt(3)*dist/l) * exp(-sqrt(3)*dist/l)
    r = np.sqrt(3) * dist / length_scale
    return (1.0 + r) * np.exp(-r)

_sources = {}
for i, (_, _, L) in enumerate(EDGE_DEFS):
    pts = np.linspace(0, L, 200)
    dist = np.abs(pts[:, None] - pts[None, :])
    cov = matern_kernel(dist, nu, length_scale)
    cov += 1e-8 * np.eye(200) # numerical stability
    
    L_chol = np.linalg.cholesky(cov)
    z = np.random.randn(200)
    f_vals = L_chol @ z
    
    _sources[i] = interp1d(pts, f_vals, kind='cubic', bounds_error=False, fill_value="extrapolate")

def exact_sol(x, edge_i):
    # No analytical exact solution available for GRF source
    return None

def source_fn(x, edge_i):
    is_tensor = isinstance(x, torch.Tensor)
    x_val = x.detach().cpu().numpy() if is_tensor else x
    scalar = np.isscalar(x_val) or (isinstance(x_val, np.ndarray) and x_val.ndim == 0)
    x_arr = np.array([x_val]) if scalar else x_val
    
    y_arr = _sources[edge_i](x_arr)
    
    if is_tensor:
        res = torch.tensor(y_arr, dtype=torch.float32, device=x.device)
        return res[0] if scalar else res
    else:
        return float(y_arr[0]) if scalar else y_arr

def reaction(x, edge_i):
    return 1.0 + 0.5 * x**2

def k_volt(edge_i, x, z):
    return 0.5 * torch.exp(-0.5 * (x - z))

def k_fred(edge_i, edge_j, x, z):
    weight = 0.5 / (1.0 + abs(edge_i - edge_j))
    return weight * torch.cos(x) * torch.sin(z)
