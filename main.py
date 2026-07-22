import sys
import os
import re
import importlib
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Ensure both project root and fpinn directory are in python path
# This allows fpinn imports (which assume fpinn/ is the working directory) to resolve correctly
project_root = os.path.abspath(os.path.dirname(__file__))
fpinn_dir = os.path.join(project_root, "fpinn")
sys.path.insert(0, fpinn_dir)
sys.path.insert(0, project_root)

import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--problem', type=str, required=False, help="Problem ID (e.g. p1a)")
parser.add_argument('--mode', type=str, choices=['fpinn', 'wavelet', 'compare', 'sweep'], required=True, help="Execution mode")
parser.add_argument('-k', '--k_val', type=int, default=6, help="Resolution level k for Wavelet solver")
parser.add_argument('-m', '--m_val', type=int, default=4, help="Degree M for Wavelet solver")
args = parser.parse_args()

# Map mode to internal execution flags
args.skip_fpinn = (args.mode == 'wavelet')
args.skip_wavelet = (args.mode == 'fpinn')
args.viz3d = args.mode if args.mode in ['fpinn', 'wavelet'] else None

def run_sweep():
    import subprocess
    import shutil

    ALPHA_SWEEP = [1.1, 1.3, 1.5, 1.7, 1.9]
    WAVELET_SWEEP = [(k, m) for m in [2, 3, 4] for k in range(1, 7)]

    def run_cmd(cmd):
        print(f"  >> {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        if result.returncode != 0:
            print(f"  !! Error: {result.stderr[-500:] if result.stderr else 'unknown'}")
        return result.stdout

    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    problem_files = glob.glob(os.path.join(project_root, "problems", "*.py"))
    problem_names = [os.path.basename(p).replace('.py', '') for p in problem_files]
    problem_names = sorted([p for p in problem_names if not p.startswith('__') and p != 'temp_benchmark'])

    print(f"Found {len(problem_names)} problems: {problem_names}")
    print(f"Alpha sweep: {ALPHA_SWEEP}")
    print(f"Wavelet sweep: {len(WAVELET_SWEEP)} configurations")
    total_runs = len(problem_names) * len(ALPHA_SWEEP)
    run_idx = 0

    for prob in problem_names:
        print(f"\n{'='*60}\n  PROBLEM: {prob}\n{'='*60}")
        with open(os.path.join(project_root, f"problems/{prob}.py"), "r") as f:
            original_code = f.read()

        for alpha in ALPHA_SWEEP:
            run_idx += 1
            print(f"\n--- [{run_idx}/{total_runs}] {prob} | alpha={alpha} ---")

            temp_code = re.sub(r'ALPHA\s*=\s*[0-9\.]+', f'ALPHA = {alpha}', original_code)
            temp_path = os.path.join(project_root, "problems", "temp_benchmark.py")
            with open(temp_path, "w") as f:
                f.write(temp_code)

            out_dir = os.path.join(project_root, f"results/{prob}_alpha_{alpha}")
            os.makedirs(out_dir, exist_ok=True)

            stats = [f"Problem: {prob}", f"Alpha: {alpha}", f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", "=" * 50]

            print("  [1/3] FPINN training...")
            t0 = time.time()
            fpinn_out = run_cmd([sys.executable, "main.py", "--mode", "fpinn", "-p", "temp_benchmark"])
            stats.append(f"\nFPINN Training Time: {time.time() - t0:.2f}s")

            exports = os.path.join(project_root, "exports")
            for src, dst in [("loss_plot.png", "fpinn_loss.png"), ("predictions_plot.png", "fpinn_2d.png"), ("sol_3d_fpinn.png", "fpinn_3d.png")]:
                src_path = os.path.join(exports, src)
                if os.path.exists(src_path):
                    shutil.copy(src_path, os.path.join(out_dir, dst))

            print("  [2/3] Wavelet sweep...")
            stats.append(f"\nWavelet Sweep Results:\n{'k':>3} {'M':>3} {'Time(s)':>10} {'Cond':>12} {'MaxError':>15}\n" + "-" * 50)

            for k, m in WAVELET_SWEEP:
                t0 = time.time()
                wave_out = run_cmd([sys.executable, "main.py", "--mode", "wavelet", "-p", "temp_benchmark", "-k", str(k), "-m", str(m)])
                wave_time = time.time() - t0
                
                cond, err = "N/A", "N/A"
                for line in wave_out.split('\n'):
                    if "Condition number:" in line: cond = line.split(":")[-1].strip()
                    if "Max absolute error" in line: err = line.split(":")[-1].strip()
                stats.append(f"{k:>3} {m:>3} {wave_time:>10.2f} {cond:>12} {err:>15}")

            if os.path.exists(os.path.join(exports, "sol_3d_wavelet.png")):
                shutil.copy(os.path.join(exports, "sol_3d_wavelet.png"), os.path.join(out_dir, f"wavelet_3d.png"))

            print("  [3/3] Unified comparison...")
            best_k, best_m = WAVELET_SWEEP[-1]
            comp_out = run_cmd([sys.executable, "main.py", "--mode", "compare", "-p", "temp_benchmark", "-k", str(best_k), "-m", str(best_m)])

            if os.path.exists(os.path.join(exports, "comparison_plot.png")):
                shutil.copy(os.path.join(exports, "comparison_plot.png"), os.path.join(out_dir, "compare_unified.png"))

            stats.append(f"\nUnified Comparison (k={best_k}, M={best_m}):")
            for line in comp_out.split('\n'):
                if "FPINN:" in line and "skipped" not in line: stats.append(f"  Compare FPINN Time: {line.split(':')[-1].strip()}")
                if "Wavelet:" in line and "skipped" not in line: stats.append(f"  Compare Wavelet Time: {line.split(':')[-1].strip()}")
                if "Ratio:" in line: stats.append(f"  Speedup Ratio: {line.split(':')[-1].strip()}")

            with open(os.path.join(out_dir, "stats.txt"), "w") as f:
                f.write("\n".join(stats) + "\n")

    if os.path.exists(os.path.join(project_root, "problems", "temp_benchmark.py")):
        os.remove(os.path.join(project_root, "problems", "temp_benchmark.py"))
    print(f"\n{'='*60}\n  BENCHMARK COMPLETE\n{'='*60}")


if args.mode == 'sweep':
    run_sweep()
    sys.exit(0)

if not args.problem:
    raise ValueError("The --problem argument is required unless running in --mode sweep")
    
# Strip out 'problems/' prefix or '.py' extension if the user used bash autocomplete
problem_id = os.path.basename(args.problem).replace('.py', '')

files = glob.glob(f"problems/{problem_id}*.py")
if not files:
    raise ValueError(f"No problem found matching {args.problem}")
problem_name = files[0].split('/')[-1].replace('.py', '')

print(f"Detected active problem: {problem_name}")

# 2. Import the active problem module
problem_mod = importlib.import_module(f"problems.{problem_name}")

# Import solvers and utilities
from fpinn.train import train as run_fpinn
from wavelets.solver import WaveletSolverMG
from fpinn.math_ops.fractional_operators import get_jacobi_quadrature, autograd_caputo_derivative, l21sigma_weights, fdv

# --- Set up Wavelet solver wrappers for PyTorch/SciPy compatibility ---
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

# Set up Jacobi quadrature for Caputo derivative evaluation (only for MMS source function auto-computation)
QUAD_PTS, QUAD_WTS = get_jacobi_quadrature(30, problem_mod.ALPHA, 'cpu')

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
    for j in range(len(problem_mod.EDGE_DEFS)):
        L_j = problem_mod.EDGE_DEFS[j][2]
        int_val, _ = quad(lambda xi: k_fred(x, xi, edge_i, j) * exact_sol(xi, j), 0, L_j, limit=100)
        fredholm += int_val
        
    return frac_d + react - volterra - fredholm

def compute_numerical_residual(problem_mod, u_preds_dict):
    alpha = problem_mod.ALPHA
    residuals = {}
    
    for edge_i, data in u_preds_dict.items():
        x = data['x']
        u = data['u_pred']
        
        W = l21sigma_weights(x, alpha)
        frac_d = fdv(u, W)
        react = problem_mod.reaction(x, edge_i) * u
        f_val = problem_mod.source_fn(x, edge_i)
        
        residuals[edge_i] = {
            'x': x,
            'frac_d': frac_d,
            'react': react,
            'f_val': f_val if f_val is not None else 0.0,
            'volt': torch.zeros_like(x),
            'fred': torch.zeros_like(x)
        }
        
    for i, data_i in u_preds_dict.items():
        x_i = data_i['x']
        u_i = data_i['u_pred']
        
        volt_arr = torch.zeros_like(x_i)
        for idx in range(1, len(x_i)):
            s = x_i[:idx+1]
            u_s = u_i[:idx+1]
            k_v = problem_mod.k_volt(i, x_i[idx], s)
            volt_arr[idx] = torch.trapz(k_v * u_s, s)
        residuals[i]['volt'] = volt_arr
        
        fred_arr = torch.zeros_like(x_i)
        for j, data_j in u_preds_dict.items():
            x_j = data_j['x']
            u_j = data_j['u_pred']
            for idx in range(len(x_i)):
                k_f = problem_mod.k_fred(i, j, x_i[idx], x_j)
                fred_arr[idx] += torch.trapz(k_f * u_j, x_j)
        residuals[i]['fred'] = fred_arr
        
        R = residuals[i]['frac_d'] + residuals[i]['react'] - residuals[i]['volt'] - residuals[i]['fred'] - residuals[i]['f_val']
        residuals[i]['R'] = torch.abs(R)
        
    return residuals


def main():
    fpinn_time = None
    wavelet_time = None

    # 1. Run the FPINN Solver
    if not args.skip_fpinn:
        print("\n[ Running FPINN (Neural Network) Solver ]")
        t0 = time.perf_counter()
        run_fpinn(problem_mod)
        fpinn_time = time.perf_counter() - t0
        print(f"\nFPINN wall-clock time: {fpinn_time:.2f}s")
    else:
        print("\n[ Skipping FPINN (Neural Network) Solver ]")

    # 2. Run the Wavelet Solver
    wavelet_results = {}
    if not args.skip_wavelet:
        print("\n[ Running Wavelet Solver ]")
        t0 = time.perf_counter()
        solver = WaveletSolverMG(
            problem_mod.EDGE_DEFS, 
            problem_mod.BC_DEFS, 
            problem_mod.ALPHA, 
            reaction, 
            k_volt, 
            k_fred
        )
        approx_func, collocation_pts = solver.solve(args.k_val, args.m_val, compute_f)
        wavelet_time = time.perf_counter() - t0
        print(f"\nWavelet wall-clock time: {wavelet_time:.2f}s")

        # Evaluate Wavelet predictions on a dense grid
        for edge_i in range(len(problem_mod.EDGE_DEFS)):
            L = problem_mod.EDGE_DEFS[edge_i][2]
            x_dense = np.linspace(0, L, 100)
            u_pred = approx_func(edge_i, x_dense)
            wavelet_results[edge_i] = {
                'x': x_dense,
                'u_pred': u_pred
            }
    else:
        print("\n[ Skipping Wavelet Solver ]")

    # 3. Load FPINN predictions
    # FPINN train.py saves exports relative to the working directory it was started from.
    # We look in both local exports and fpinn/exports
    fpinn_pred_path = os.path.join(project_root, "exports", "predictions.pt")
    if not os.path.exists(fpinn_pred_path):
        fpinn_pred_path = os.path.join(fpinn_dir, "exports", "predictions.pt")
    
    print(f"Loading FPINN results from {fpinn_pred_path}...")
    fpinn_results = torch.load(fpinn_pred_path, map_location='cpu', weights_only=False)

    # 4. Determine Validation Mode and Compute Tensors
    num_edges = len(problem_mod.EDGE_DEFS)
    has_exact = exact_sol(0.0, 0) is not None
    
    fpinn_res_data = {}
    wavelet_res_data = {}
    for i in range(num_edges):
        # Extract FPINN to pure 1D tensors
        data_fpinn = fpinn_results[str(i)] if str(i) in fpinn_results else fpinn_results[i]
        fpinn_res_data[i] = {
            'x': data_fpinn['x'].flatten() if isinstance(data_fpinn['x'], torch.Tensor) else torch.tensor(data_fpinn['x']).flatten(),
            'u_pred': data_fpinn['u_pred'].flatten() if isinstance(data_fpinn['u_pred'], torch.Tensor) else torch.tensor(data_fpinn['u_pred']).flatten()
        }
        
        if not args.skip_wavelet and i in wavelet_results:
            w_x = wavelet_results[i]['x']
            w_u = wavelet_results[i]['u_pred']
            wavelet_res_data[i] = {
                'x': w_x.flatten() if isinstance(w_x, torch.Tensor) else torch.tensor(w_x, dtype=torch.float32).flatten(),
                'u_pred': w_u.flatten() if isinstance(w_u, torch.Tensor) else torch.tensor(w_u, dtype=torch.float32).flatten()
            }

    if not has_exact:
        print("\nComputing continuous a posteriori residuals for validation...")
        fpinn_residuals = compute_numerical_residual(problem_mod, fpinn_res_data)
        wavelet_residuals = compute_numerical_residual(problem_mod, wavelet_res_data) if not args.skip_wavelet else None
    
    # 5. Plot both side-by-side in a 2-row grid
    fig, axes = plt.subplots(2, num_edges, figsize=(5 * num_edges, 10), squeeze=False)
    
    for i in range(num_edges):
        ax1 = axes[0][i]
        ax2 = axes[1][i]
        
        # --- ROW 1: Solutions ---
        # Plot Exact Solution (if available)
        L = problem_mod.EDGE_DEFS[i][2]
        x_plot = np.linspace(0, L, 100)
        
        y_exact = []
        if has_exact:
            for val in x_plot:
                res = exact_sol(val, i)
                y_exact.append(res)
            ax1.plot(x_plot, y_exact, label='Exact Solution', linestyle='-', color='black', linewidth=2)
        
        # Plot FPINN predictions
        fpinn_x = fpinn_res_data[i]['x'].numpy()
        fpinn_pred = fpinn_res_data[i]['u_pred'].numpy()
        ax1.plot(fpinn_x, fpinn_pred, label='FPINN (Neural Net)', linestyle='--', color='red', linewidth=2)
        
        # Plot Wavelet predictions
        if not args.skip_wavelet and i in wavelet_res_data:
            wavelet_x = wavelet_res_data[i]['x'].numpy()
            wavelet_pred = wavelet_res_data[i]['u_pred'].numpy()
            ax1.plot(wavelet_x, wavelet_pred, label='Wavelet Solver', linestyle=':', color='blue', linewidth=2.5)
        
        ax1.set_title(f'Edge {i}')
        ax1.set_xlabel('s (Physical Coordinate)')
        ax1.set_ylabel('y(s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- ROW 2: Validation Metric ---
        if has_exact:
            # Absolute Error Mode
            fpinn_exact_vals = [exact_sol(v, i) for v in fpinn_x]
            fpinn_err = np.abs(fpinn_pred - np.array(fpinn_exact_vals))
            ax2.plot(fpinn_x, fpinn_err, label='FPINN Error', linestyle='--', color='red', linewidth=2)
            
            if not args.skip_wavelet and i in wavelet_res_data:
                wv_exact_vals = [exact_sol(v, i) for v in wavelet_x]
                wv_err = np.abs(wavelet_pred - np.array(wv_exact_vals))
                ax2.plot(wavelet_x, wv_err, label='Wavelet Error', linestyle=':', color='blue', linewidth=2)
                
            ax2.set_xlabel('s (Physical Coordinate)')
            ax2.set_ylabel('Absolute Error')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Continuous Residual Mode
            r_x = fpinn_residuals[i]['x'].numpy()
            r_fpinn = fpinn_residuals[i]['R'].numpy()
            trim = max(1, len(r_x) // 20) # Avoid visual noise from L2-1sigma boundary artifacts
            
            ax2.plot(r_x[trim:], r_fpinn[trim:], label='FPINN Residual', linestyle='--', color='red', linewidth=2)
            
            if not args.skip_wavelet and wavelet_residuals is not None:
                r_w = wavelet_residuals[i]['x'].numpy()
                r_wv = wavelet_residuals[i]['R'].numpy()
                ax2.plot(r_w[trim:], r_wv[trim:], label='Wavelet Residual', linestyle=':', color='blue', linewidth=2)
                
            ax2.set_xlabel('s (Physical Coordinate)')
            ax2.set_ylabel('|Residual|')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
    fig.suptitle(f'Problem: {problem_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save output to both root and fpinn exports folder for convenience
    os.makedirs(os.path.join(project_root, 'exports'), exist_ok=True)
    out_path = os.path.join(project_root, 'exports', 'comparison_plot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot successfully saved to {out_path}!")

    # Print timing summary
    print("\n" + "=" * 50)
    print("  TIMING SUMMARY")
    print("=" * 50)
    if fpinn_time is not None:
        print(f"  FPINN:    {fpinn_time:>10.2f}s")
    else:
        print(f"  FPINN:    skipped")
    if wavelet_time is not None:
        print(f"  Wavelet:  {wavelet_time:>10.2f}s")
    else:
        print(f"  Wavelet:  skipped")
    if fpinn_time is not None and wavelet_time is not None:
        ratio = fpinn_time / wavelet_time if wavelet_time > 0 else float('inf')
        faster = "Wavelet" if ratio > 1 else "FPINN"
        print(f"  Ratio:    {ratio:>10.2f}x ({faster} is faster)")
    print("=" * 50)

    if args.viz3d:
        print(f"Opening 3D visualization for {args.viz3d}...")
        results_to_plot = fpinn_results if args.viz3d == 'fpinn' else wavelet_results
        
        import networkx as nx
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        G = nx.MultiGraph()
        for i, (u, v, L) in enumerate(problem_mod.EDGE_DEFS):
            G.add_edge(u, v, key=i, length=L)
        
        pos = nx.kamada_kawai_layout(G, weight='length')
        
        fig3d = plt.figure(figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection='3d')

        for edge_id, (u, v, L) in enumerate(problem_mod.EDGE_DEFS):
            if str(edge_id) in results_to_plot:
                data = results_to_plot[str(edge_id)]
            elif edge_id in results_to_plot:
                data = results_to_plot[edge_id]
            else:
                continue
                
            x_param = data['x']
            if isinstance(x_param, torch.Tensor): x_param = x_param.detach().cpu().numpy()
            x_param = x_param.flatten()
            
            z_val = data['u_pred']
            if isinstance(z_val, torch.Tensor): z_val = z_val.detach().cpu().numpy()
            z_val = z_val.flatten()
            
            t = x_param / L
            
            if u != v:
                pt_u = pos[u]
                pt_v = pos[v]
                X = pt_u[0] + t * (pt_v[0] - pt_u[0])
                Y = pt_u[1] + t * (pt_v[1] - pt_u[1])
            else:
                pt_u = pos[u]
                com = np.mean([p for p in pos.values()], axis=0)
                dir_vec = pt_u - com
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-5:
                    dir_vec = np.array([1.0, 0.0])
                else:
                    dir_vec = dir_vec / norm
                    
                radius = L / (2 * np.pi)
                center = pt_u + dir_vec * radius
                
                angle_start = np.arctan2(-dir_vec[1], -dir_vec[0])
                
                angles = angle_start + t * 2 * np.pi
                X = center[0] + radius * np.cos(angles)
                Y = center[1] + radius * np.sin(angles)

            ax3d.plot(X, Y, z_val, color='black', linewidth=2)
            ax3d.plot(X, Y, 0, color='black', linewidth=1)
            
            verts = list(zip(X, Y, z_val)) + list(zip(X[::-1], Y[::-1], np.zeros_like(X)))
            poly = Poly3DCollection([verts], facecolors='gray', alpha=0.2, edgecolors='none')
            ax3d.add_collection3d(poly)

        for v_id, pt in pos.items():
            ax3d.scatter(pt[0], pt[1], 0, color='black', s=50, depthshade=False, zorder=5)

        ax3d.set_title(f"3D Topology View ({args.viz3d.upper()})")
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('u(x)')
        ax3d.set_box_aspect((2, 2, 1))
        
        ax3d.view_init(elev=25, azim=45)

        plt.tight_layout()
        out_3d_path = os.path.join(project_root, 'exports', f'sol_3d_{args.viz3d}.png')
        plt.savefig(out_3d_path, dpi=300, bbox_inches='tight')
        print(f"3D visualization saved to {out_3d_path}")

if __name__ == "__main__":
    main()
