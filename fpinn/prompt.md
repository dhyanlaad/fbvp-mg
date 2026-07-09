# Antigravity Task: 3 Problem-Definition Files for the Metric-Graph IDE Solver

## Reference format

The solver already consumes graph-problem files shaped like this (`global_dom.py`, included as
context — do not modify it, just match its conventions):

```python
import torch
import math

ALPHA = 1.5

EDGE_DEFS = [
    (1, 0, 1.0),   # (tail_vertex, head_vertex, length) — x=0 is the tail end, x=length the head end
    (2, 0, 1.0),
    (0, 3, 1.0),
    (0, 4, 1.0),
]
NUM_EDGES = len(EDGE_DEFS)

BC_DEFS = {
    1: {'type': 'Dirichlet', 'value': 0.0},
    2: {'type': 'Dirichlet', 'value': 0.0},
    3: {'type': 'Neumann', 'value': -math.pi / 2.0},
    4: {'type': 'Neumann', 'value': 2.0}
}

def exact_sol(x, edge_i):   # None if no closed form for that edge
    ...

def source_fn(x, edge_i):   # None => solver derives f_i from exact_sol via the operator itself
    return None

def reaction(x, edge_i):
    ...

def k_volt(edge_i, x, z):
    ...

def k_fred(edge_i, edge_j, x, z):
    ...
```

**Inferred conventions — confirm these against the solver, but design all three files
consistently with them:**

- `EDGE_DEFS` indices double as `edge_i`/`edge_j` in every other function (0-indexed, in list
  order).
- **Exactly one `BC_DEFS` entry per degree-1 (leaf) vertex.** Internal vertices (degree ≥ 2) get
  continuity + Kirchhoff automatically from the solver and must **not** appear in `BC_DEFS`.
  (Handshake check: for `|E|` edges and `L` leaves, junctions absorb `2|E| - L` of the `2|E|`
  free constants, leaving exactly `L` — one per leaf. Verified against `global_dom.py`: 4 edges,
  4 leaves, 4 BC entries.)
- **Continuity**: at every internal vertex, all incident edges' values at that vertex's end of
  the edge must be numerically equal.
- **Kirchhoff**: at every internal vertex, `Σ (outward-normal derivative) = 0`, where the
  outward-normal derivative of an edge at a vertex is `+dy/dx` evaluated at `x=0` if the vertex
  is the edge's *tail*, or `-dy/dx` evaluated at `x=length` if the vertex is the edge's *head*.
- **`BC_DEFS` values are read straight off `exact_sol`** at the leaf: `Dirichlet.value =
  y(leaf_end)`, `Neumann.value = dy/dx(leaf_end)` (with the same outward-normal sign convention
  above).
- When `exact_sol` returns a real function (not `None`) for every edge, `source_fn` must return
  `None` for every edge — the solver applies the operator to `exact_sol` itself to get the
  manufactured forcing. **Do not hand-derive `f_i(x)` yourself for the two exact-solution
  problems below** — just make sure `exact_sol` is continuity/Kirchhoff-consistent, since the
  solver won't correct for an inconsistent one.
- When there's no closed form (Problem 3), `exact_sol` returns `None` for every edge and
  `source_fn` must return an actual function.

## Task: produce exactly 3 files

### 1. `tadpole_exact.py`
Tadpole graph — a 3-edge triangle (loop) with a 1-edge tail:
```
EDGE_DEFS = [
    (0, 1, 1.0),   # e0: triangle
    (1, 2, 1.0),   # e1: triangle
    (2, 0, 1.0),   # e2: triangle
    (0, 3, 1.3),   # e3: tail, leaf vertex 3
]
```
`v0` is the junction (degree 3: e0 tail, e2 head, e3 tail); `v1, v2` are degree-2 internal
vertices on the loop; `v3` is the only leaf → exactly **one** `BC_DEFS` entry (vertex 3).

Build `exact_sol(x, edge_i)` for all 4 edges (pick simple closed forms — sin/cos/polynomial,
mixed style like `global_dom.py`) such that:
- `y0(0) == y2(1.0) == y3(0)` (junction value at v0)
- `y0(1.0) == y1(0)` (junction value at v1)
- `y1(1.0) == y2(0)` (junction value at v2)
- Kirchhoff at v0: `y0'(0) - y2'(1.0) + y3'(0) == 0`
- Kirchhoff at v1: `-y0'(1.0) + y1'(0) == 0`
- Kirchhoff at v2: `-y1'(1.0) + y2'(0) == 0`

Then set the single `BC_DEFS[3]` entry (Dirichlet or Neumann, your choice) from `exact_sol`'s
value/derivative at `x=1.3` on edge 3. `source_fn` returns `None` for all 4 edges. Use
`reaction`, `k_volt`, `k_fred` in the same style as `global_dom.py` (pick any physically
reasonable coefficients — this problem isn't targeting a dominance regime, so no need to bias
them small/large).

**Before finalizing, numerically verify all 6 constraints above to ~1e-10** (evaluate the
chosen closed forms and their derivatives at the junction points) and print a short pass/fail
diagnostic — don't just eyeball it.

### 2. `tree_exact.py`
Small binary tree, depth 2:
```
EDGE_DEFS = [
    (0, 1, 1.0),  # e0: root -> left child
    (0, 2, 1.0),  # e1: root -> right child
    (1, 3, 0.8),  # e2: left child -> leaf
    (1, 4, 0.8),  # e3: left child -> leaf
    (2, 5, 0.8),  # e4: right child -> leaf
    (2, 6, 0.8),  # e5: right child -> leaf
]
```
`v0` degree 2 (root, both edges are tails), `v1, v2` degree 3 (one head + two tails each),
`v3..v6` degree-1 leaves → **four** `BC_DEFS` entries (vertices 3, 4, 5, 6).

Same requirements as the tadpole file: build `exact_sol` for all 6 edges satisfying continuity
+ Kirchhoff at `v0, v1, v2`, derive the 4 leaf `BC_DEFS` entries from it, `source_fn` returns
`None` everywhere, same numeric self-check before finalizing, `reaction`/`k_volt`/`k_fred` in
the established style.

### 3. `random_network_numeric.py`
No exact solution — purely numerical demo on a larger, more realistic graph.
- Generate a Barabási–Albert graph, `n=22`, `m=2`, with `networkx`, **fixed seed** (print the
  seed). Convert to `EDGE_DEFS` (assign each generated edge a random length `~U(0.5, 2.0)`,
  same seed).
- `exact_sol(x, edge_i)` returns `None` for every edge.
- `source_fn(x, edge_i)` returns an actual randomly generated smooth function per edge — e.g. a
  short sum of 2–3 random-frequency/random-phase sine terms with bounded amplitude (fixed seed,
  same one as the graph). Do not use `torch.randn` directly inside the function (that would
  resample every call); pre-generate the random coefficients once at module load time and close
  over them.
- `reaction`, `k_volt`, `k_fred`: randomize coefficients within stable ranges at module load
  (same seed), reusing the *shape* of `global_dom.py`'s functions (small local reaction/Volterra
  term, but don't force a dominance regime — sample coefficients broadly, e.g. `U(0.01, 1.0)`
  for local terms and `U(0.5, 5.0)` for the Fredholm coupling amplitude) rather than hand-fixing
  0.01/100.0.
- Print the seed and the generated graph's edge count once at the top, so the run is
  reproducible and self-documenting.

## Deliverables
```
tadpole_exact.py
tree_exact.py
random_network_numeric.py
```
Each file must be importable standalone (only `torch`, `math`, and — for file 3 — `networkx`
and `random`/`numpy` as dependencies) and expose exactly the same six names as
`global_dom.py`: `ALPHA`, `EDGE_DEFS`, `NUM_EDGES`, `BC_DEFS`, `exact_sol`, `source_fn`,
`reaction`, `k_volt`, `k_fred`.