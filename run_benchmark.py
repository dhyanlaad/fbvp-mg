#!/usr/bin/env python3
"""
Automated benchmarking script for FPINN and Wavelet solvers.
Sweeps across all problems, alpha values, and wavelet (k, M) configurations.
Outputs structured results to results/<problem>_alpha_<alpha>/.
"""
import os
import glob
import re
import subprocess
import shutil
import time
import sys

PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'bin', 'python')

ALPHA_SWEEP = [1.1, 1.3, 1.5, 1.7, 1.9]
WAVELET_SWEEP = [(k, m) for m in [2, 3, 4] for k in range(1, 7)]

def run_cmd(cmd):
    """Run a command and return stdout. Print errors to stderr."""
    print(f"  >> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  !! Error: {result.stderr[-500:] if result.stderr else 'unknown'}")
    return result.stdout

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
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
        print(f"\n{'='*60}")
        print(f"  PROBLEM: {prob}")
        print(f"{'='*60}")

        with open(os.path.join(project_root, f"problems/{prob}.py"), "r") as f:
            original_code = f.read()

        for alpha in ALPHA_SWEEP:
            run_idx += 1
            print(f"\n--- [{run_idx}/{total_runs}] {prob} | alpha={alpha} ---")

            # Create dynamic temporary problem with swapped ALPHA
            temp_code = re.sub(r'ALPHA\s*=\s*[0-9\.]+', f'ALPHA = {alpha}', original_code)
            temp_path = os.path.join(project_root, "problems", "temp_benchmark.py")
            with open(temp_path, "w") as f:
                f.write(temp_code)

            out_dir = os.path.join(project_root, f"results/{prob}_alpha_{alpha}")
            os.makedirs(out_dir, exist_ok=True)

            stats = []
            stats.append(f"Problem: {prob}")
            stats.append(f"Alpha: {alpha}")
            stats.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            stats.append("=" * 50)

            # --- 1. FPINN ---
            print("  [1/3] FPINN training...")
            t0 = time.time()
            fpinn_out = run_cmd([PYTHON, "main.py", "--mode", "fpinn", "-p", "temp_benchmark"])
            fpinn_time = time.time() - t0
            stats.append(f"\nFPINN Training Time: {fpinn_time:.2f}s")

            run_cmd([PYTHON, "fpinn/viz.py"])
            run_cmd([PYTHON, "fpinn/viz_3d.py"])

            exports = os.path.join(project_root, "exports")
            for src, dst in [("loss_plot.png", "fpinn_loss.png"),
                             ("predictions_plot.png", "fpinn_2d.png"),
                             ("sol_3d.png", "fpinn_3d.png")]:
                src_path = os.path.join(exports, src)
                if os.path.exists(src_path):
                    shutil.copy(src_path, os.path.join(out_dir, dst))

            # --- 2. Wavelet Sweep ---
            print("  [2/3] Wavelet sweep...")
            stats.append(f"\nWavelet Sweep Results:")
            stats.append(f"{'k':>3} {'M':>3} {'Time(s)':>10} {'Cond':>12} {'MaxError':>15}")
            stats.append("-" * 50)

            for k, m in WAVELET_SWEEP:
                t0 = time.time()
                wave_out = run_cmd([PYTHON, "main.py", "--mode", "wavelet", "-p", "temp_benchmark", "-k", str(k), "-m", str(m)])
                wave_time = time.time() - t0

                run_cmd([PYTHON, "wavelets/viz.py"])
                run_cmd([PYTHON, "wavelets/viz_3d.py"])

                # Copy plots
                wav_plots = os.path.join(project_root, "wavelets", "plots")
                wav_exports = os.path.join(project_root, "wavelets", "exports")
                if os.path.exists(os.path.join(wav_plots, "sol.png")):
                    shutil.copy(os.path.join(wav_plots, "sol.png"), os.path.join(out_dir, f"wavelet_k{k}_M{m}_2d.png"))
                if os.path.exists(os.path.join(wav_plots, "sol_3d.png")):
                    shutil.copy(os.path.join(wav_plots, "sol_3d.png"), os.path.join(out_dir, f"wavelet_k{k}_M{m}_3d.png"))

                # Parse condition number and max error
                cond = "N/A"
                err = "N/A"
                for line in wave_out.split('\n'):
                    if "Condition number:" in line:
                        cond = line.split(":")[-1].strip()
                    if "Max absolute error" in line:
                        err = line.split(":")[-1].strip()

                stats.append(f"{k:>3} {m:>3} {wave_time:>10.2f} {cond:>12} {err:>15}")

            # --- 3. Compare.py ---
            print("  [3/3] Unified comparison...")
            best_k, best_m = WAVELET_SWEEP[-1]
            comp_out = run_cmd([PYTHON, "main.py", "--mode", "compare", "-p", "temp_benchmark",
                                "-k", str(best_k), "-m", str(best_m)])

            if os.path.exists(os.path.join(exports, "comparison_plot.png")):
                shutil.copy(os.path.join(exports, "comparison_plot.png"),
                            os.path.join(out_dir, "compare_unified.png"))

            # Parse compare.py timing
            stats.append(f"\nUnified Comparison (k={best_k}, M={best_m}):")
            for line in comp_out.split('\n'):
                if "FPINN:" in line and "skipped" not in line:
                    stats.append(f"  Compare FPINN Time: {line.split(':')[-1].strip()}")
                if "Wavelet:" in line and "skipped" not in line:
                    stats.append(f"  Compare Wavelet Time: {line.split(':')[-1].strip()}")
                if "Ratio:" in line:
                    stats.append(f"  Speedup Ratio: {line.split(':')[-1].strip()}")

            # Write stats
            with open(os.path.join(out_dir, "stats.txt"), "w") as f:
                f.write("\n".join(stats) + "\n")

            print(f"  -> Results saved to {out_dir}")

    # Cleanup temp file
    temp_path = os.path.join(project_root, "problems", "temp_benchmark.py")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
