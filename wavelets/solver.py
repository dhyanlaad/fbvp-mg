import numpy as np
import math
import scipy.special as sp
from scipy.integrate import IntegrationWarning, quad
import warnings

warnings.filterwarnings("ignore", category=IntegrationWarning)

def psi(x, i, k, M):
    idx = i - 1
    n = (idx // M) + 1
    m = idx % M
    a = (n - 1) / (2**(k - 1))
    b = n / (2**(k - 1))
    if a <= x < b or (x == 1.0 and b == 1.0):
        coef = (2**(k / 2)) / np.sqrt((2**(m - 1)) * math.factorial(m) * np.sqrt(np.pi))
        H_m = sp.eval_hermite(m, 2**k * x - 2 * n + 1)
        return coef * H_m
    return 0.0

def R(x, i, k, M):
    if x == 0.0: return 0.0
    res, _ = quad(lambda s: (x - s) * psi(s, i, k, M), 0, x, limit=100)
    return res

def R_prime(x, i, k, M):
    if x == 0.0: return 0.0
    res, _ = quad(lambda s: psi(s, i, k, M), 0, x, limit=100)
    return res

def Z(x, i, gamma, k, M):
    alpha = 2.0 - gamma
    if x == 0: return 0.0
    res, _ = quad(lambda s: ((x - s)**(alpha - 1)) * psi(s, i, k, M), 0, x, limit=100)
    return res / math.gamma(alpha)

class WaveletSolverMG:
    def __init__(self, edge_defs, bc_defs, alpha, r_func, k_volt, k_fred):
        self.edge_defs = edge_defs
        self.bc_defs = bc_defs
        self.gamma = alpha
        self.r_func = r_func
        self.k_volt = k_volt
        self.k_fred = k_fred
        self.num_edges = len(edge_defs)
        self.lengths = [length for _, _, length in self.edge_defs]
        
        self.nodes = {}
        for edge_i, (v_start, v_end, length) in enumerate(self.edge_defs):
            if v_start not in self.nodes: self.nodes[v_start] = {'in': [], 'out': []}
            if v_end not in self.nodes: self.nodes[v_end] = {'in': [], 'out': []}
            self.nodes[v_start]['out'].append(edge_i)
            self.nodes[v_end]['in'].append(edge_i)
            
        self.internal_nodes = []
        self.boundary_nodes = []
        for v, conn in self.nodes.items():
            deg = len(conn['in']) + len(conn['out'])
            if deg > 1:
                self.internal_nodes.append(v)
            else:
                self.boundary_nodes.append(v)
                
    def solve(self, k, M, f_funcs):
        N_w = (2**(k - 1)) * M
        
        num_unknowns = self.num_edges * (N_w + 2)
        
        num_bc = len(self.boundary_nodes)
        num_cont = sum([len(self.nodes[v]['in']) + len(self.nodes[v]['out']) - 1 for v in self.internal_nodes])
        num_kirch = len(self.internal_nodes)
        
        expected_eqs = self.num_edges * N_w + num_bc + num_cont + num_kirch
        if expected_eqs != num_unknowns:
            raise ValueError(f"Topology mismatch: expected {num_unknowns} equations, got {expected_eqs} instead.")
            
        x_q = np.array([(q - 0.5) / N_w for q in range(1, N_w + 1)])
        collocation_pts = {edge_i: x_q * self.lengths[edge_i] for edge_i in range(self.num_edges)}
        
        A = np.zeros((num_unknowns, num_unknowns))
        B = np.zeros(num_unknowns)
        
        print(f"Precomputing basis integrals for N_w = {N_w}.")
        R_cache = {}
        Z_cache = {}
        for i in range(1, N_w + 1):
            for x in x_q:
                R_cache[(x, i)] = R(x, i, k, M)
                Z_cache[(x, i)] = Z(x, i, self.gamma, k, M)
        R_1 = {i: R(1.0, i, k, M) for i in range(1, N_w + 1)}
        R_prime_1 = {i: R_prime(1.0, i, k, M) for i in range(1, N_w + 1)}
        
        print(f"{num_bc} boundaries and {len(self.internal_nodes)} junction(s).")
        print("Assembling matrix system.")
        row = 0
        
        for edge_i in range(self.num_edges):
            L_i = self.lengths[edge_i]
            for x_val in x_q:
                s_q = x_val * L_i
                B[row] = f_funcs(s_q, edge_i, self.gamma)
                col_start = edge_i * (N_w + 2)
                
                A[row, col_start + 0] = self.r_func(s_q, edge_i) - quad(lambda xi: self.k_volt(s_q, xi, edge_i) * 1.0, 0, s_q, limit=100)[0] - quad(lambda xi: self.k_fred(s_q, xi, edge_i, edge_i) * 1.0, 0, L_i, limit=100)[0]

                A[row, col_start + 1] = self.r_func(s_q, edge_i) * s_q - quad(lambda xi: self.k_volt(s_q, xi, edge_i) * xi, 0, s_q, limit=100)[0] - quad(lambda xi: self.k_fred(s_q, xi, edge_i, edge_i) * xi, 0, L_i, limit=100)[0]

                for m in range(1, N_w + 1):
                    A[row, col_start + 1 + m] = (L_i ** (-self.gamma)) * Z_cache[(x_val, m)] + self.r_func(s_q, edge_i) * R_cache[(x_val, m)] - quad(lambda xi: self.k_volt(s_q, xi, edge_i) * R(xi / L_i, m, k, M), 0, s_q, limit=100)[0] - quad(lambda xi: self.k_fred(s_q, xi, edge_i, edge_i) * R(xi / L_i, m, k, M), 0, L_i, limit=100)[0]
                    
                for edge_j in range(self.num_edges):
                    if edge_j == edge_i: continue
                    L_j = self.lengths[edge_j]
                    col_start_j = edge_j * (N_w + 2)
                    A[row, col_start_j + 0] -= quad(lambda xi: self.k_fred(s_q, xi, edge_i, edge_j) * 1.0, 0, L_j, limit=100)[0]
                    A[row, col_start_j + 1] -= quad(lambda xi: self.k_fred(s_q, xi, edge_i, edge_j) * xi, 0, L_j, limit=100)[0]
                    for m in range(1, N_w + 1):
                        A[row, col_start_j + 1 + m] -= quad(lambda xi: self.k_fred(s_q, xi, edge_i, edge_j) * R(xi / L_j, m, k, M), 0, L_j, limit=100)[0]

                row += 1

        for v in self.boundary_nodes:
            bc = self.bc_defs[v]
            btype = bc['type']
            bval = bc['value']
            
            if len(self.nodes[v]['out']) == 1:
                edge_i = self.nodes[v]['out'][0]
                x_eval = 0.0
                sign = -1.0
            else:
                edge_i = self.nodes[v]['in'][0]
                x_eval = 1.0
                sign = 1.0
                
            L_i = self.lengths[edge_i]
            col_start = edge_i * (N_w + 2)
            
            if btype == 'Dirichlet':
                if x_eval == 0.0:
                    A[row, col_start + 0] = 1.0
                else:
                    A[row, col_start + 0] = 1.0
                    A[row, col_start + 1] = L_i
                    for m in range(1, N_w + 1): A[row, col_start + 1 + m] = R_1[m]
            elif btype == 'Neumann':
                if x_eval == 0.0:
                    A[row, col_start + 1] = sign * 1.0
                else:
                    A[row, col_start + 1] = sign * 1.0
                    for m in range(1, N_w + 1): A[row, col_start + 1 + m] = sign * R_prime_1[m] / L_i
            B[row] = bval
            row += 1

        for v in self.internal_nodes:
            conn_edges = []
            conn_types = [] 
            for edge_i in self.nodes[v]['in']:
                conn_edges.append(edge_i)
                conn_types.append('in')
            for edge_i in self.nodes[v]['out']:
                conn_edges.append(edge_i)
                conn_types.append('out')

            base_edge = conn_edges[0]
            base_type = conn_types[0]
            
            def set_y(r, edge, etype, coeff):
                L_e = self.lengths[edge]
                col = edge * (N_w + 2)
                if etype == 'out':
                    A[r, col + 0] += coeff * 1.0
                else:
                    A[r, col + 0] += coeff * 1.0
                    A[r, col + 1] += coeff * L_e
                    for m in range(1, N_w + 1): A[r, col + 1 + m] += coeff * R_1[m]

            for i in range(1, len(conn_edges)):
                set_y(row, base_edge, base_type, 1.0)
                set_y(row, conn_edges[i], conn_types[i], -1.0)
                B[row] = 0.0
                row += 1
                
            for i in range(len(conn_edges)):
                edge_i = conn_edges[i]
                etype = conn_types[i]
                L_i = self.lengths[edge_i]
                col = edge_i * (N_w + 2)
                if etype == 'out':
                    A[row, col + 1] += -1.0
                else:
                    A[row, col + 1] += 1.0
                    for m in range(1, N_w + 1): A[row, col + 1 + m] += R_prime_1[m] / L_i
            B[row] = 0.0
            row += 1

        print("Inverting matrix.")
        U = np.linalg.solve(A, B)
        
        def get_approx(edge_i, x):
            L_i = self.lengths[edge_i]
            col_start = edge_i * (N_w + 2)
            if not isinstance(x, np.ndarray):
                x = np.array([x])
            
            res = np.zeros_like(x, dtype=float)
            for j, s_val in enumerate(x):
                val = U[col_start] + U[col_start + 1] * s_val
                for m in range(1, N_w + 1):
                    val += U[col_start + 1 + m] * R(s_val / L_i, m, k, M)
                res[j] = val
                
            return res if len(res) > 1 else res[0]

        return get_approx, collocation_pts
