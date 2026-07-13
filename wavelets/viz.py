import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exports_dir = os.path.join(script_dir, 'exports')
    pred_path = os.path.join(exports_dir, 'predictions.pt')
    
    if not os.path.exists(pred_path):
        print(f"Error: Exported data not found at {pred_path}. Run main.py first to generate it.")
        return

    print("Loading data from exports/...")
    results = torch.load(pred_path, map_location='cpu', weights_only=False)
    
    num_edges = len(results)
    fig, axes = plt.subplots(1, num_edges, figsize=(5 * num_edges, 5), squeeze=False)
    
    for i in range(num_edges):
        ax = axes[0][i]
        
        data = results[str(i)] if str(i) in results else results[i]
        x_dense = data['x'].numpy().flatten()
        pred = data['u_pred'].numpy().flatten()
        
        if 'u_exact' in data:
            exact = data['u_exact'].numpy().flatten()
            ax.plot(x_dense, exact, label='True Solution', linestyle='-', color='blue', linewidth=2)
            
        ax.plot(x_dense, pred, label='Wavelet Approx', linestyle='--', color='red', linewidth=2)
        
        ax.set_title(f'Edge {i}')
        ax.set_xlabel('s (Physical Coordinate)')
        ax.set_ylabel('y(s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'sol.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"2D Plot saved to {out_path}")

if __name__ == "__main__":
    main()
