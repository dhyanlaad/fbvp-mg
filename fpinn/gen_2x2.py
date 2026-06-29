#!/usr/bin/env python3
"""Generate a 2x2 prediction plot from saved exports."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import math

results = torch.load('exports/predictions.pt', map_location='cpu', weights_only=False)
num_edges = len(results)
nrows = math.ceil(num_edges / 2)
ncols = 2

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

for i in range(num_edges):
    ax = axes[i // ncols][i % ncols]
    x = results[i]['x'].cpu().detach().numpy().flatten()
    pred = results[i]['u_pred'].cpu().detach().numpy().flatten()
    if 'u_exact' in results[i]:
        exact = results[i]['u_exact'].cpu().detach().numpy().flatten()
        ax.plot(x, exact, label='True Solution', linestyle='-', color='blue', linewidth=2)
        ax.plot(x, pred, label='NN Approximation', linestyle='--', color='red', linewidth=2)
    else:
        ax.plot(x, pred, label='NN Approximation', linestyle='-', color='red', linewidth=2)
    ax.set_title(f'Edge {i}')
    ax.set_xlabel('x')
    ax.set_ylabel('y(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

for j in range(num_edges, nrows * ncols):
    axes[j // ncols][j % ncols].set_visible(False)

plt.tight_layout()
plt.savefig('exports/predictions_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("2x2 plot saved to exports/predictions_plot.png")
