import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from problems.star_singular_pendant import exact_sol, ALPHA
from fpinn.math_ops.fractional_operators import get_jacobi_quadrature, autograd_caputo_derivative

QUAD_PTS, QUAD_WTS = get_jacobi_quadrature(30, ALPHA)

def func(x):
    return exact_sol(x, 0)

x_vals = np.linspace(0.01, 0.999, 500)
x_t = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1)
frac_d = autograd_caputo_derivative(func, x_t, ALPHA, QUAD_PTS, QUAD_WTS).detach().numpy()

plt.plot(x_vals, frac_d)
plt.title("Numerical Fractional Derivative of x^2(1-x)^1.5")
plt.savefig("frac_d_test.png")
print(f"Max frac_d: {np.max(frac_d)}, Min frac_d: {np.min(frac_d)}")
