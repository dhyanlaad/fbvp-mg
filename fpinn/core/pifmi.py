import torch
import torch.nn as nn
import numpy as np
import copy

def grid(N, length=1.0, beta=1.0, device='cpu'):
    """
    Generates a structured collocation grid for an edge.
    
    Args:
        N (int): Number of points.
        length (float): Length of the spatial domain.
        beta (float): Polynomial stretching factor (beta > 1 clusters points near x=0).
        device (str): Computation device ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: Column vector of spatial coordinates with `requires_grad=True`.
    """
    indices = np.linspace(0, 1, N)
    x_grid = length * (indices ** beta)

    x_tensor = torch.tensor(x_grid, dtype=torch.float32, device=device).view(-1, 1)
    x_tensor.requires_grad_()

    return x_tensor

class EdgeNet(nn.Module):
    """
    A purely topological neural network representing a single graph edge.
    This acts as the unconstrained latent function `raw(x)` before any boundary
    or continuity constraints are mathematically enforced.
    """
    def __init__(self, layers=[1, 50, 50, 50, 50, 50, 50, 1]):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        ])
        self._init_weights()

    def _init_weights(self):
        for layer in self.linears:
            nn.init.xavier_uniform_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)

    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        return self.linears[-1](x)

class MetricGraphNet(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for Fractional BVP on Metric Graphs.
    
    This architecture utilizes a rigorous mathematical ansatz that automatically
    satisfies exact boundary conditions (Dirichlet, Neumann) and completely
    hardcodes internal node continuity to strictly prevent vertex tearing.
    """
    def __init__(self, num_edges, lengths, bnd_verts=[], int_verts=[], vtx_info={}, bc_defs={},
                 layers=[1, 50, 50, 50, 50, 50, 50, 1],
                 share_init=False, alpha=None):
        super().__init__()
        
        self.alpha = alpha
        
        # Instantiate EdgeNets
        if share_init and num_edges > 1:
            prototype = EdgeNet(layers)
            self.nets = nn.ModuleList([prototype] +
                [copy.deepcopy(prototype) for _ in range(num_edges - 1)])
        else:
            self.nets = nn.ModuleList([EdgeNet(layers) for _ in range(num_edges)])
            
        # Singular basis weights
        if alpha is not None:
            self.w_sing = nn.ParameterDict({
                str(e): nn.Parameter(torch.tensor(0.0)) for e in range(num_edges)
            })

        self.register_buffer('lengths', torch.tensor(lengths, dtype=torch.float32))

        # Learnable global anchors for internal junctions to enforce strict continuity
        self.int_v_vals = nn.ParameterDict({
            str(v): nn.Parameter(torch.tensor(0.0)) for v in int_verts
        })

        # Arrays to store the specific mathematical boundary rules per edge
        self.leaf_0_type = [None] * num_edges
        self.leaf_L_type = [None] * num_edges
        
        self.leaf_0_val  = [0.0] * num_edges
        self.leaf_L_val  = [0.0] * num_edges
        
        self.leaf_0_vid  = [None] * num_edges
        self.leaf_L_vid  = [None] * num_edges
        
        # 1. Map true boundary conditions
        for v in bnd_verts:
            if v not in bc_defs:
                raise ValueError(f"Missing boundary condition for leaf vertex {v}")
            bc_type = bc_defs[v]['type'].capitalize()
            bc_val  = bc_defs[v]['value']
            
            for edge_id, is_start in vtx_info[v]:
                if is_start:
                    self.leaf_0_type[edge_id] = bc_type
                    self.leaf_0_val[edge_id] = bc_val
                else:
                    self.leaf_L_type[edge_id] = bc_type
                    self.leaf_L_val[edge_id] = bc_val

        # 2. Map internal junctions as dynamic Dirichlet anchors
        # By forcing internal junctions to act like Dirichlet boundaries whose values
        # are tied to a globally shared learnable parameter, we mathematically guarantee
        # continuity across the graph.
        for v in int_verts:
            for edge_id, is_start in vtx_info[v]:
                if is_start:
                    self.leaf_0_type[edge_id] = 'Dirichlet'
                    self.leaf_0_vid[edge_id] = str(v)
                else:
                    self.leaf_L_type[edge_id] = 'Dirichlet'
                    self.leaf_L_vid[edge_id] = str(v)

    def forward(self, x, edge_idx):
        """
        Forward pass applying the structural mathematical ansatz.
        """
        x_in = x.unsqueeze(1) if x.dim() == 1 else x
        L = self.lengths[edge_idx]
        
        T0 = self.leaf_0_type[edge_idx]
        TL = self.leaf_L_type[edge_idx]
        
        # Dynamically fetch target values: static constants for real boundaries,
        # or learnable tensors for internal junctions
        v0 = self.int_v_vals[self.leaf_0_vid[edge_idx]] if self.leaf_0_vid[edge_idx] is not None else self.leaf_0_val[edge_idx]
        vL = self.int_v_vals[self.leaf_L_vid[edge_idx]] if self.leaf_L_vid[edge_idx] is not None else self.leaf_L_val[edge_idx]
        
        need_N0 = (T0 == 'Dirichlet' and TL != 'Dirichlet')
        need_NL = (TL == 'Dirichlet' and T0 != 'Dirichlet')
        need_dN0 = (T0 == 'Neumann')
        need_dNL = (TL == 'Neumann')
        
        N0 = 0.0
        NL = 0.0
        dN0 = 0.0
        dNL = 0.0
        
        grad_enabled = torch.is_grad_enabled()
        
        # Precompute EdgeNet evaluations at boundary points for Neumann compensation
        with torch.enable_grad():
            if need_N0 or need_dN0:
                x0 = torch.zeros((1, 1), dtype=torch.float32, device=x.device, requires_grad=need_dN0)
                out0 = self.nets[edge_idx](x0)
                if need_N0: N0 = out0.squeeze()
                if need_dN0:
                    dN0 = torch.autograd.grad(out0, x0, grad_outputs=torch.ones_like(out0), create_graph=grad_enabled)[0].squeeze()
                    if not grad_enabled: dN0 = dN0.detach()
                    
            if need_NL or need_dNL:
                xL = torch.full((1, 1), L.item(), dtype=torch.float32, device=x.device, requires_grad=need_dNL)
                outL = self.nets[edge_idx](xL)
                if need_NL: NL = outL.squeeze()
                if need_dNL:
                    dNL = torch.autograd.grad(outL, xL, grad_outputs=torch.ones_like(outL), create_graph=grad_enabled)[0].squeeze()
                    if not grad_enabled: dNL = dNL.detach()
            
        raw = self.nets[edge_idx](x_in)
        
        x_scaled = x_in / L
        x_val = x_in
        
        if T0 == 'Dirichlet' and TL == 'Dirichlet':
            y = v0 * (1 - x_scaled) + vL * x_scaled + x_scaled * (1 - x_scaled) * raw
        elif T0 == 'Dirichlet' and TL == 'Neumann':
            y = raw - N0 + v0 - (dNL - vL) * x_val
        elif T0 == 'Neumann' and TL == 'Dirichlet':
            y = raw - NL + vL - (dN0 - v0) * (x_val - L)
        elif T0 == 'Neumann' and TL == 'Neumann':
            y = raw - (dN0 - v0) * (x_val - (x_val**2)/(2*L)) - (dNL - vL) * ((x_val**2)/(2*L))
        elif T0 == 'Dirichlet' and TL is None:
            y = raw - N0 + v0
        elif T0 is None and TL == 'Dirichlet':
            y = raw - NL + vL
        elif T0 == 'Neumann' and TL is None:
            y = raw - (dN0 - v0) * x_val
        elif T0 is None and TL == 'Neumann':
            y = raw - (dNL - vL) * x_val
        else:
            y = raw

        if self.alpha is not None:
            # Inject explicit singular basis function outside the ansatz envelope
            w = self.w_sing[str(edge_idx)]
            y = y + w * (x_val ** self.alpha) * ((L - x_val) ** 2)

        if x.dim() == 1:
            y = y.squeeze(1)

        return y
