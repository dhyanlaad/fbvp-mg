import sys
import os
import math
import numpy as np
from scipy.integrate import quad

K_VAL = 6
M_VAL = 4

# Setup paths so imports resolve
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wavelets.solver import WaveletSolverMG
import torch
import scipy.special

def get_jacobi_quadrature(N_q, alpha, device='cpu'):
    a_scipy = 0.0
    b_scipy = 1.0 - alpha
    pts, wts = scipy.special.roots_jacobi(N_q, a_scipy, b_scipy)
    return (torch.tensor(pts, dtype=torch.float32, device=device),
            torch.tensor(wts, dtype=torch.float32, device=device))

def autograd_caputo_derivative(func, x, alpha, quad_pts, quad_wts):
    """
    Computes D^alpha u(x) exactly using Autograd and Gauss-Jacobi Quadrature.
    """
    mask = (x > 1e-7).flatten()
    x_safe = x[mask]
    
    if len(x_safe) == 0:
        return torch.zeros_like(x)
        
    N_q = len(quad_pts)
    N_safe = len(x_safe)
    
    xi = (x_safe / 2.0) * (1.0 - quad_pts.unsqueeze(0))
    xi_flat = xi.flatten().unsqueeze(1)
    xi_flat.requires_grad_(True)
    
    u_xi = func(xi_flat)
    
    du_dxi = torch.autograd.grad(
        u_xi, xi_flat,
        grad_outputs=torch.ones_like(u_xi),
        create_graph=True,
        allow_unused=True
    )[0]
    
    if du_dxi is None:
        du_dxi = torch.zeros_like(u_xi)
        
    if not du_dxi.requires_grad:
        d2u_dxi2 = torch.zeros_like(du_dxi)
    else:
        d2u_dxi2 = torch.autograd.grad(
            du_dxi, xi_flat,
            grad_outputs=torch.ones_like(du_dxi),
            create_graph=True,
            allow_unused=True
        )[0]
        
    if d2u_dxi2 is None:
        d2u_dxi2 = torch.zeros_like(du_dxi)
        
    d2u_dxi2 = d2u_dxi2.view(N_safe, N_q)
    
    integral = torch.matmul(d2u_dxi2, quad_wts)
    
    prefix = (1.0 / scipy.special.gamma(2.0 - alpha)) * ((x_safe.flatten() / 2.0) ** (2.0 - alpha))
    frac_safe = prefix * integral
    
    frac_d = torch.zeros_like(x).flatten()
    frac_d[mask] = frac_safe
    
    return frac_d.unsqueeze(1)

def run_wavelets(problem_mod, k_val=K_VAL, m_val=M_VAL):
    ALPHA = problem_mod.ALPHA
    EDGE_DEFS = problem_mod.EDGE_DEFS
    BC_DEFS = problem_mod.BC_DEFS
    
    QUAD_PTS, QUAD_WTS = get_jacobi_quadrature(30, ALPHA)

    def exact_sol(x, edge_i):
        res = problem_mod.exact_sol(torch.tensor(x), edge_i)
        return res.item() if isinstance(res, torch.Tensor) else res

    def reaction(x, edge_i):
        res = problem_mod.reaction(torch.tensor(x), edge_i)
        return res.item() if isinstance(res, torch.Tensor) else res

    def k_volt(x, xi, edge_i):
        res = problem_mod.k_volt(edge_i, torch.tensor(x), torch.tensor(xi))
        return res.item() if isinstance(res, torch.Tensor) else res

    def k_fred(x, xi, edge_i, edge_j):
        res = problem_mod.k_fred(edge_i, edge_j, torch.tensor(x), torch.tensor(xi))
        return res.item() if isinstance(res, torch.Tensor) else res

    def compute_f(x, edge_i, gamma):
        user_f = problem_mod.source_fn(torch.tensor(x), edge_i)
        if user_f is not None:
            return user_f.item() if isinstance(user_f, torch.Tensor) else user_f

        x_t = torch.tensor([[x]], dtype=torch.float32)
        def func(xi_t):
            return problem_mod.exact_sol(xi_t, edge_i)
        frac_d_t = autograd_caputo_derivative(func, x_t, gamma, QUAD_PTS, QUAD_WTS)
        frac_d = frac_d_t.item()
        
        react = reaction(x, edge_i) * exact_sol(x, edge_i)
        
        volterra, _ = quad(lambda xi: k_volt(x, xi, edge_i) * exact_sol(xi, edge_i), 0, x, limit=100)
        
        fredholm = 0.0
        for j in range(len(EDGE_DEFS)):
            L_j = EDGE_DEFS[j][2]
            int_val, _ = quad(lambda xi: k_fred(x, xi, edge_i, j) * exact_sol(xi, j), 0, L_j, limit=100)
            fredholm += int_val
            
        return frac_d + react - volterra - fredholm

    solver = WaveletSolverMG(EDGE_DEFS, BC_DEFS, ALPHA, reaction, k_volt, k_fred)
    
    print(f"[ Wavelet Solver | gamma: {ALPHA} | k: {k_val} | M: {m_val} ]")
    
    approx_func, collocation_pts = solver.solve(k_val, m_val, compute_f)
    
    print(f"\nEdge | x          | Exact           | Approx          | Error")
    print("------------------------------------------------------------------")
    
    max_error = 0
    has_exact = True
    for edge_i in range(len(EDGE_DEFS)):
        for x_val in collocation_pts[edge_i]:
            exact_val = exact_sol(x_val, edge_i)
            approx_val = approx_func(edge_i, x_val)
            if exact_val is not None:
                error = abs(exact_val - approx_val)
                if error > max_error:
                    max_error = error
                print(f"{edge_i:<5} | {x_val:<10.4f} | {exact_val:<15.6e} | {approx_val:<15.6e} | {error:<15.6e}")
            else:
                has_exact = False
                print(f"{edge_i:<4} | {x_val:<10.4f} | N/A             | {approx_val:<15.6e} | N/A")
            
    print("------------------------------------------------------------------")
    if has_exact:
        print(f"Max absolute error across entire graph: {max_error:.6e}")
    else:
        print("Max absolute error across entire graph: N/A")
    
    import json
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exports_dir = os.path.join(script_dir, 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    with open(os.path.join(exports_dir, 'topology.json'), 'w') as f:
        json.dump(EDGE_DEFS, f)
        
    results = {}
    for edge_i in range(len(EDGE_DEFS)):
        L = EDGE_DEFS[edge_i][2]
        x_dense = np.linspace(0, L, 100)
        u_pred = approx_func(edge_i, x_dense)
        u_exact = [exact_sol(x, edge_i) for x in x_dense]
        
        results[str(edge_i)] = {
            'x': torch.tensor(x_dense, dtype=torch.float32),
            'u_pred': torch.tensor(u_pred, dtype=torch.float32),
        }
        if u_exact[0] is not None:
            results[str(edge_i)]['u_exact'] = torch.tensor(np.array(u_exact), dtype=torch.float32)
        
    torch.save(results, os.path.join(exports_dir, 'predictions.pt'))
    print(f"\nData successfully exported to {exports_dir} for visualization!")

if __name__ == "__main__":
    import argparse
    import importlib
    import glob
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', type=str, required=True, help='Problem ID (e.g. b4a)')
    parser.add_argument('-k', '--k_val', type=int, default=K_VAL, help='Resolution level k for Wavelet solver')
    parser.add_argument('-m', '--m_val', type=int, default=M_VAL, help='Degree M for Wavelet solver')
    args = parser.parse_args()
    
    problem_id = os.path.basename(args.problem).replace('.py', '')
    files = glob.glob(f"problems/{problem_id}*.py")
    if not files:
        raise ValueError(f"No problem found matching {args.problem}")
    
    mod_name = files[0].split('/')[-1].replace('.py', '')
    problem_mod = importlib.import_module(f"problems.{mod_name}")
    
    run_wavelets(problem_mod, k_val=args.k_val, m_val=args.m_val)
