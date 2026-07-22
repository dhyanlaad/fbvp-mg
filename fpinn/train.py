import sys
import os
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
import math
import torch
import torch.optim as optim

from core.graph import (build_graph, get_edge_list, get_vertex_info,
                   get_internal_vertices, get_boundary_vertices)
from core.pifmi import MetricGraphNet, grid
from math_ops.fractional_operators import (l21sigma_weights, fdv,
                                           get_jacobi_quadrature, autograd_caputo_derivative)
from math_ops.quad_operators import (trap_weights, 
                                     precompute_volt_matrix, eval_volt_precomputed,
                                     precompute_fred_matrices, eval_fred_precomputed)

def compute_loss(problem_mod, model, x_grids, quad_pts, quad_wts, q_weights, f_vecs, lengths,
                 vtx_info, int_verts, bnd_verts, G, num_edges, device,
                 lam_ode, lam_kirch, epoch,
                 K_volt_matrices, K_fred_matrices):
    """
    Evaluates the total physics loss of the FPINN.
    
    Because internal junction continuity and boundary conditions are explicitly
    guaranteed by the MetricGraphNet ansatz, this function only penalizes:
    1. ODE Physics (Fractional + Reaction + Volterra + Fredholm vs Forcing)
    2. Kirchhoff Flux (Net derivative fluxes at internal junctions must balance to 0)
    
    Returns: (total_loss, loss_ode, loss_kirch)
    """

    u_preds = [model(x_grids[i], i) for i in range(num_edges)]

    fs = 1.0
    
    # 1. Fractional Integro-Differential Equation (ODE) Loss
    loss_ode = torch.tensor(0.0, device=device)
    for i in range(num_edges):
        func = lambda x, ei=i: model(x, ei)
        frac_d = autograd_caputo_derivative(func, x_grids[i], problem_mod.ALPHA, quad_pts, quad_wts)
        react  = problem_mod.reaction(x_grids[i], i) * u_preds[i]

        volt = eval_volt_precomputed(u_preds[i], K_volt_matrices[i])
        fred = eval_fred_precomputed(u_preds, K_fred_matrices[i], num_edges)

        residual = frac_d + react - volt - fs * fred - f_vecs[i]
        
        f_mean_sq = torch.mean(f_vecs[i] ** 2)
        scale = f_mean_sq if f_mean_sq > 1e-4 else 1.0
        loss_ode = loss_ode + torch.mean(residual ** 2) / scale
    loss_ode = loss_ode / num_edges

    # 2. Kirchhoff Flux Law (Neumann interaction at internal junctions)
    loss_kirch = torch.tensor(0.0, device=device)
    for v in int_verts:
        flux_sum = torch.tensor(0.0, device=device)
        for edge_id, is_start in vtx_info[v]:
            if is_start:
                x_bnd = torch.zeros(1, 1, dtype=torch.float32,
                                    device=device, requires_grad=True)
            else:
                x_bnd = torch.full((1, 1), lengths[edge_id],
                                   dtype=torch.float32,
                                   device=device, requires_grad=True)

            y_bnd  = model(x_bnd, edge_id)
            dy_dx  = torch.autograd.grad(
                y_bnd, x_bnd,
                grad_outputs=torch.ones_like(y_bnd),
                create_graph=True
            )[0]

            sign = 1.0 if is_start else -1.0
            flux_sum = flux_sum + sign * dy_dx.squeeze()

        loss_kirch = loss_kirch + flux_sum ** 2

    # Total loss
    loss = (lam_ode * loss_ode + lam_kirch * loss_kirch)

    return loss, loss_ode, loss_kirch

def train(problem_mod):
    """
    Main training execution loop for the FPINN metric graph architecture.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.init()
        torch.zeros(1, device=device)  # force context creation

    # Topology Analysis
    G         = build_graph(problem_mod.EDGE_DEFS)
    edge_list = get_edge_list(G)
    vtx_info  = get_vertex_info(G)
    int_verts = get_internal_vertices(G)
    bnd_verts = get_boundary_vertices(G)
    num_edges = len(edge_list)

    N_coll = 125
    lr     = 1e-3

    lam_ode       = 10.0
    lam_kirch     = 1000.0

    adam_max_epochs = 20000
    lbfgs_max_iter = 10000

    # Grid Construction
    x_grids   = []
    q_weights = []
    f_vecs    = []
    lengths   = []
    
    quad_pts, quad_wts = get_jacobi_quadrature(30, problem_mod.ALPHA, device)
    
    for eid, start, end, length in edge_list:
        xg = grid(N=N_coll, length=length, beta=1.0, device=device)
        x_grids.append(xg)
        q_weights.append(trap_weights(xg).detach())
        lengths.append(length)

    # Integral Operator Caching
    with torch.no_grad():
        K_volt_matrices = []
        for i in range(num_edges):
            kern_i = lambda x, z, ei=i: problem_mod.k_volt(ei, x, z)
            K_volt_i = precompute_volt_matrix(x_grids[i], kern_i, q_weights[i])
            K_volt_matrices.append(K_volt_i)

        K_fred_matrices = precompute_fred_matrices(x_grids, problem_mod.k_fred, q_weights, num_edges)

    # Forcing Vector Generation
    for i in range(num_edges):
        user_f = problem_mod.source_fn(x_grids[i], i)
        if user_f is not None:
            f_vecs.append(user_f.detach())
        else:
            u_ex_all = [problem_mod.exact_sol(x_grids[j], j) for j in range(num_edges)]
            if any(u is None for u in u_ex_all):
                raise ValueError(f"source_fn returned None on edge {i}, but exact_sol is also missing. Cannot compute source term.")
            
            func = lambda x, ei=i: problem_mod.exact_sol(x, ei)
            frac_d = autograd_caputo_derivative(func, x_grids[i], problem_mod.ALPHA, quad_pts, quad_wts).detach()
            react  = problem_mod.reaction(x_grids[i], i) * u_ex_all[i]
            
            volt   = eval_volt_precomputed(u_ex_all[i], K_volt_matrices[i])
            fred   = eval_fred_precomputed(u_ex_all, K_fred_matrices[i], num_edges)
            
            f_vec = frac_d + react - volt - fred
            f_vecs.append(f_vec.detach())

    # Model Initialisation
    model = MetricGraphNet(num_edges, lengths=lengths,
                           bnd_verts=bnd_verts, 
                           int_verts=int_verts,
                           vtx_info=vtx_info,
                           bc_defs=problem_mod.BC_DEFS,
                           share_init=True).to(device)

    loss_kwargs = dict(
        problem_mod=problem_mod, model=model, x_grids=x_grids, quad_pts=quad_pts, quad_wts=quad_wts,
        q_weights=q_weights, f_vecs=f_vecs, lengths=lengths,
        vtx_info=vtx_info, int_verts=int_verts, bnd_verts=bnd_verts,
        G=G, num_edges=num_edges, device=device,
        lam_ode=lam_ode,
        lam_kirch=lam_kirch,
        K_volt_matrices=K_volt_matrices,
        K_fred_matrices=K_fred_matrices
    )

    # Use standard learning rate for all parameters. 
    # We rely on L-BFGS to correctly scale the gradients 
    # via the inverse Hessian to perfectly align the junction values.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("\nAdam optimisation")

    history = {'total': [], 'edges': [], 'nodes': []}
    
    best_loss = float('inf')
    patience_counter = 0
    adam_patience = 500
    adam_min_delta = 1e-10

    for epoch in range(1, adam_max_epochs + 1):
        optimizer.zero_grad()

        loss, l_ode, l_kirch = compute_loss(
            **loss_kwargs, epoch=epoch)

        history['total'].append(loss.item())
        history['edges'].append(l_ode.item())
        history['nodes'].append(l_kirch.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 1000 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{adam_max_epochs} | Loss: {loss.item():.4e} "
                  f"| ODE: {l_ode.item():.4e} "
                  f"| Kirch: {l_kirch.item():.4e} "
                  f"| fred_s: 1.00")
                  
        if epoch > 1000:
            # Keep cosine schedule scaled to new adam max
            T_max = 18000
            T_cur = min(epoch, T_max)
            lr_min = 1e-4
            lr_max = 1e-3
            cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * T_cur / T_max))
            optimizer.param_groups[0]['lr'] = cosine_lr
            
            # Early stopping check
            loss_val = loss.item()
            if loss_val < best_loss - adam_min_delta:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= adam_patience:
                print(f"\nEarly stopping Adam at epoch {epoch} (no improvement > {adam_min_delta} for {adam_patience} epochs).")
                break

    # L-BFGS Refinement
    print("\nL-BFGS refinement")
    lbfgs_epoch = epoch

    lbfgs_optimizer = optim.LBFGS(
        model.parameters(),
        max_iter=20,  # Smaller block size for even more frequent restarts
        tolerance_grad=1e-13,
        tolerance_change=1e-16,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    lbfgs_iter = 0

    def closure():
        nonlocal lbfgs_iter
        lbfgs_optimizer.zero_grad()
        loss, l_ode, l_kirch = compute_loss(
            **loss_kwargs, epoch=lbfgs_epoch)
        loss.backward()

        history['total'].append(loss.item())
        history['edges'].append(l_ode.item())
        history['nodes'].append(l_kirch.item())

        lbfgs_iter += 1
        if lbfgs_iter % 100 == 0 or lbfgs_iter == 1:
            print(f"L-BFGS eval {lbfgs_iter:4d} "
                  f"| Loss: {loss.item():.4e} "
                  f"| ODE: {l_ode.item():.4e} "
                  f"| Kirch: {l_kirch.item():.4e}")
        return loss

    prev_loss = float('inf')
    print("\nL-BFGS refinement")
    while lbfgs_iter < lbfgs_max_iter:
        loss_val = lbfgs_optimizer.step(closure).item()
        
        # Absolute convergence check to break loop if fully converged
        if abs(prev_loss - loss_val) < 1e-14:
            print(f"L-BFGS converged absolutely at eval {lbfgs_iter}.")
            break
            
        prev_loss = loss_val

    print("\n[ Training Complete ]")
    model.eval()

    results = {}
    global_l2_sq = 0.0
    global_norm_sq = 0.0
    global_max_error = 0.0
    has_exact = False

    with torch.no_grad():
        for i in range(num_edges):
            u_pred = model(x_grids[i], i).cpu()
            u_ex   = problem_mod.exact_sol(x_grids[i], i)

            entry = {
                'x': x_grids[i].cpu(),
                'u_pred': u_pred,
            }

            if u_ex is not None:
                has_exact = True
                u_ex = u_ex.cpu() if isinstance(u_ex, torch.Tensor) else u_ex
                abs_error = torch.abs(u_pred - u_ex)
                max_error = torch.max(abs_error).item()
                global_max_error = max(global_max_error, max_error)
                
                mae = torch.mean(abs_error).item()
                mse = torch.mean((u_pred - u_ex)**2).item()
                
                w = q_weights[i].cpu()
                l2_sq = torch.sum(w * (u_pred - u_ex)**2).item()
                norm_sq = torch.sum(w * u_ex**2).item()
                global_l2_sq += l2_sq
                global_norm_sq += norm_sq
                
                l2_error = l2_sq**0.5
                u_ex_l2_norm = norm_sq**0.5
                rel_l2_error = (l2_error / u_ex_l2_norm) if u_ex_l2_norm > 0 else 0.0

                print(f"Edge {i:2d} | L_inf: {max_error:.4e} | L2: {l2_error:.4e} | Rel L2: {rel_l2_error:.4e} | MSE: {mse:.4e} | MAE: {mae:.4e}")
                entry['u_exact'] = u_ex
            else:
                print(f"Edge {i:2d} | No analytic solution - skipping error report")

            results[i] = entry

        if has_exact:
            global_l2 = global_l2_sq**0.5
            global_rel_l2 = (global_l2 / (global_norm_sq**0.5)) if global_norm_sq > 0 else 0.0
            print("--------------------------------------------------------------------------------")
            print(f"Global  | L_inf: {global_max_error:.4e} | L2: {global_l2:.4e} | Rel L2: {global_rel_l2:.4e}")
            print("--------------------------------------------------------------------------------")

    import json
    import os
    os.makedirs('exports', exist_ok=True)

    torch.save(model.state_dict(), 'exports/weights.pth')
    print("Model weights saved to exports/weights.pth")

    with open('exports/history.json', 'w') as f:
        json.dump(history, f, indent=4)
    print("Training history saved to exports/history.json")

    with open('exports/topology.json', 'w') as f:
        json.dump(problem_mod.EDGE_DEFS, f, indent=4)
    print("Topology saved to exports/topology.json")

    torch.save(results, 'exports/predictions.pt')
    print("Predictions saved to exports/predictions.pt")

    return results, G, history


