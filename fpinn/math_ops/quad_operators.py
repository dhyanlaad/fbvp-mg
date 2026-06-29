import torch

def trap_weights(x_grid):
    """Trapezoidal quadrature weights for an arbitrary 1-D grid."""
    weights = torch.zeros_like(x_grid)

    dx = x_grid[1:] - x_grid[:-1]

    weights[0]    = dx[0] / 2.0
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    weights[-1]   = dx[-1] / 2.0

    return weights

def precompute_volt_matrix(x_grid, kern_fn, weights):
    """Precomputes the masked Volterra kernel matrix."""
    x_mat    = x_grid.view(-1, 1)
    zeta_mat = x_grid.view(1, -1)
    K_mat    = kern_fn(x_mat, zeta_mat)

    mask  = torch.tril(torch.ones_like(K_mat))
    W_mat = weights.view(1, -1) * mask

    dx = x_grid[1:] - x_grid[:-1]
    dx_diag = torch.zeros_like(x_grid).view(-1)
    dx_diag[:-1] = dx.view(-1) / 2.0
    W_mat = W_mat - torch.diag(dx_diag)

    return K_mat * W_mat

def eval_volt_precomputed(u_pred, K_mat_masked):
    """Volterra integral using precomputed kernel matrix."""
    integrand = u_pred.view(-1)
    integral_approx = torch.matmul(K_mat_masked, integrand)
    return integral_approx.view(-1, 1)

def precompute_fred_matrices(x_grids, kern_fn, weights_list, num_edges):
    """Precomputes the dense Fredholm kernel matrices for all edges."""
    K_fred = [[None for _ in range(num_edges)] for _ in range(num_edges)]
    for i in range(num_edges):
        for j in range(num_edges):
            x_mat    = x_grids[i].view(-1, 1)
            zeta_mat = x_grids[j].view(1, -1)
            K_mat    = kern_fn(i, j, x_mat, zeta_mat)
            K_fred[i][j] = K_mat * weights_list[j].view(1, -1)
    return K_fred

def eval_fred_precomputed(u_preds, K_fred_i, num_edges):
    """Graph-level Fredholm integral using precomputed kernel matrices."""
    N_i = K_fred_i[0].shape[0]
    result = torch.zeros((N_i, 1), dtype=u_preds[0].dtype, device=u_preds[0].device)
    
    for j in range(num_edges):
        integrand = u_preds[j].view(-1)
        integral  = torch.matmul(K_fred_i[j], integrand)
        result = result + integral.view(-1, 1)
        
    return result
