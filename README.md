# Fractional Boundary Value Problems on Metric Graphs (FBVP-MG)

This repository contains a comprehensive benchmarking suite for solving Fractional Boundary Value Problems (FBVPs) defined on complex metric graph topologies. The project implements and compares two distinct numerical methodologies: a high-precision **Wavelet-based Collocation Solver** and a **Fractional Physics-Informed Neural Network (FPINN)**.

## Mathematical Formulation

The problems solved are integro-differential equations of fractional order $\alpha \in (1, 2)$ on a metric graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. On each edge $e \in \mathcal{E}$, the solution $u_e(x)$ satisfies:

$$
D^\alpha_{0+} u_e(x) + r_e(x) u_e(x) - \int_0^x k_V(x, \xi) u_e(\xi) d\xi - \sum_{j \in \mathcal{E}} \int_0^{L_j} k_{F, j}(x, \xi) u_j(\xi) d\xi = f_e(x)
$$

where:
*   $D^\alpha_{0+}$ is the Caputo fractional derivative.
*   $r_e(x)$ is a reaction coefficient.
*   $k_V$ is a Volterra kernel (memory/history term).
*   $k_F$ is a Fredholm kernel (cross-edge coupling).
*   $f_e(x)$ is the source term (including complex, non-analytical sources like Whittle-Matérn random fields).

The system is closed by standard continuity and Kirchhoff flux conditions at the internal junctions, and Dirichlet/Neumann conditions at the boundaries.

## Methodology 1: Fractional Physics-Informed Neural Networks (FPINN)

The FPINN methodology employs a novel structural ansatz (`MetricGraphNet`) that strictly hardcodes boundary and continuity constraints, preventing "vertex tearing" during training. 

*   **Ansatz Enforcement**: Internal junctions are mapped to learnable global variables. Edge-specific neural networks ($\mathcal{N}_e(x)$) are wrapped in interpolation formulas that perfectly satisfy Dirichlet and Neumann boundaries algebraically.
*   **Optimization**: The Kirchhoff flux and the fractional ODE residual are enforced softly via the loss function. The Caputo derivative is evaluated using L1/L2 interpolation quadrature inside the computational graph, while integral operators are evaluated using precomputed dense matrix interactions.

## Methodology 2: Wavelet Collocation Solver

The wavelet solver uses a pseudo-spectral collocation approach leveraging Hermite wavelets.

*   **Basis Expansion**: The solution on each edge is expanded into a series of scaling functions (polynomials up to degree 1) and localized wavelets defined by resolution level $k$ and polynomial degree $M$.
*   **Analytical Integration**: The fractional derivatives of the wavelet basis are integrated analytically where possible. 
*   **Composite Quadrature**: To handle the highly localized support of wavelets at high resolution ($k$), the solver dynamically subdivides the domain and employs a composite Gauss-Legendre quadrature. This guarantees numerical stability and prevents under-resolution of narrow support bands.
*   **Hard Constraints**: The junction continuity, Kirchhoff flux, and boundary conditions are explicitly assembled into the global algebraic system $A\mathbf{U} = \mathbf{B}$, ensuring perfect constraint satisfaction up to machine precision.

## Benchmarking Suite

The `run_benchmark.py` orchestration script dynamically sweeps through:
*   **Fractional orders**: $\alpha \in \{1.1, 1.3, 1.5, 1.7, 1.9\}$
*   **Wavelet resolutions**: $k \in [1, 6]$, $M \in \{2, 3, 4\}$
*   **Topologies**: Star graphs, tadpole loops, binary trees, H-graphs (diamond), and branched paths.

The script automatically generates high-quality 2D profiles, 3D topology visualizations, and performance statistics (condition numbers, error rates, timings).
