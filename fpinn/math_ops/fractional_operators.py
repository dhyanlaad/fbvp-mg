import torch
import scipy.special

def l21sigma_weights(x_grid, alpha):
    """
    L2-1sigma discretization weights for the Caputo fractional derivative of order alpha in (1, 2).
    """
    N = len(x_grid)
    device = x_grid.device
    W = torch.zeros((N, N), dtype=torch.float32, device=device)

    gamma_tail = scipy.special.gamma(3 - alpha)

    for i in range(1, N):
        for j in range(1, i + 1):
            h_j = x_grid[j] - x_grid[j - 1]

            d_prev = (x_grid[i] - x_grid[j - 1]) ** (2 - alpha)
            d_curr = (x_grid[i] - x_grid[j]) ** (2 - alpha) if j < i else 0.0

            val = (d_prev - d_curr) / (gamma_tail * (h_j ** 2))

            W[i, j] += val
            W[i, j - 1] -= 2 * val
            if j > 1:
                W[i, j - 2] += val

    return W

def fdv(u_pred, W_weights):
    return torch.matmul(W_weights, u_pred)

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
