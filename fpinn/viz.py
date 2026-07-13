import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import json
import os

def smooth(scalars, weight=0.99):
    """Exponential moving average for smoothing loss curves."""
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    if not os.path.exists('exports/predictions.pt') or not os.path.exists('exports/history.json'):
        print("Error: Exported data not found.")
        return

    print("Loading data from exports/...")
    results = torch.load('exports/predictions.pt', weights_only=False)
    with open('exports/history.json', 'r') as f:
        history = json.load(f)

    num_edges = len(results)

    nrows = 1
    ncols = num_edges
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for i in range(num_edges):
        ax = axes[0][i]
        x     = results[i]['x'].cpu().detach().numpy().flatten()
        pred  = results[i]['u_pred'].cpu().detach().numpy().flatten()

        if 'u_exact' in results[i]:
            exact = results[i]['u_exact'].cpu().detach().numpy().flatten()
            ax.plot(x, exact, label='True Solution', linestyle='-',  color='blue', linewidth=2)
            ax.plot(x, pred,  label='NN Approximation', linestyle='--', color='red',  linewidth=2)
        else:
            ax.plot(x, pred, label='NN Approximation', linestyle='-', color='red', linewidth=2)

        ax.set_title(f'Edge {i}')
        ax.set_xlabel('x')
        ax.set_ylabel('y(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exports/predictions_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved to exports/predictions_plot.png")

    fig_loss, ax_loss = plt.subplots(figsize=(6, 5))
    
    t_raw, e_raw, n_raw = history['total'], history['edges'], history['nodes']
    
    ax_loss.semilogy(t_raw, color='black', alpha=0.15)
    ax_loss.semilogy(e_raw, color='blue', alpha=0.15)
    ax_loss.semilogy(n_raw, color='orange', alpha=0.15)
    
    ax_loss.semilogy(smooth(t_raw), label='Total Loss (smoothed)', color='black', linewidth=2)
    ax_loss.semilogy(smooth(e_raw), label='Edges (ODE)', color='blue', linestyle='--', linewidth=2)
    ax_loss.semilogy(smooth(n_raw), label='Nodes (Kirchhoff+Cont)', color='orange', linestyle='--', linewidth=2)
    
    ax_loss.set_title('Training Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exports/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig_loss)
    print("Plot saved to exports/loss_plot.png")

if __name__ == "__main__":
    main()
