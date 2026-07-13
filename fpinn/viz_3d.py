import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import numpy as np
import json
import networkx as nx

def main():
    if not os.path.exists('exports/predictions.pt') or not os.path.exists('exports/topology.json'):
        print("Error: Exported data not found. Run train.py first to generate the topology and predictions.")
        return

    print("Loading data from exports/...")
    results = torch.load('exports/predictions.pt', map_location='cpu', weights_only=False)
    
    with open('exports/topology.json', 'r') as f:
        edge_defs = json.load(f)

    G = nx.MultiGraph()
    for i, (u, v, L) in enumerate(edge_defs):
        G.add_edge(u, v, key=i, length=L)
    
    pos = nx.kamada_kawai_layout(G, weight='length')
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for edge_id, (u, v, L) in enumerate(edge_defs):
        if str(edge_id) in results:
            data = results[str(edge_id)]
        elif edge_id in results:
            data = results[edge_id]
        else:
            continue
            
        x_param = data['x'].detach().cpu().numpy().flatten()
        z_val = data['u_pred'].detach().cpu().numpy().flatten()
        
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

        ax.plot(X, Y, z_val, color='black', linewidth=2)
        ax.plot(X, Y, 0, color='black', linewidth=1)
        
        if 'u_exact' in data:
            z_ex = data['u_exact'].detach().cpu().numpy().flatten()
            ax.plot(X, Y, z_ex, color='blue', linewidth=1, linestyle='--', alpha=0.7)
            
        verts = list(zip(X, Y, z_val)) + list(zip(X[::-1], Y[::-1], np.zeros_like(X)))
        poly = Poly3DCollection([verts], facecolors='gray', alpha=0.2, edgecolors='none')
        ax.add_collection3d(poly)

    for v_id, pt in pos.items():
        ax.scatter(pt[0], pt[1], 0, color='black', s=50, depthshade=False, zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x)')
    ax.set_box_aspect((2, 2, 1))
    
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig('exports/sol_3d.png', dpi=300, bbox_inches='tight')
    print("3D Plot saved to exports/sol_3d.png")

if __name__ == "__main__":
    main()