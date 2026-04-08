# elqcp

Solvers for the **Extended Linear-Quadratic Optimal Control Problem (ELQCP)**, implemented in C.

Based on:
> Jørgensen, J. B. (2004). *Moving Horizon Estimation and Control*. PhD thesis, Technical University of Denmark.
> Jørgensen, J. B. et al. (2012). *Numerical Methods for Solution of the ELQCP*. IFAC NMPC.

---

## Problem

The ELQCP (Problem 4.3.1, eq. 4.46–4.47) is:

$$\min_{\{x_{k+1},\, u_k\}_{k=0}^{N-1}} \phi = \sum_{k=0}^{N-1} \ell_k(x_k, u_k) + \ell_N(x_N)$$

subject to

$$x_{k+1} = A_k x_k + B_k u_k + b_k, \quad k = 0, \ldots, N-1$$

with stage costs

$$\ell_k(x, u) = \tfrac{1}{2} x^\top Q_k x + x^\top M_k u + \tfrac{1}{2} u^\top R_k u + q_k^\top x + r_k^\top u + f_k$$

$$\ell_N(x) = \tfrac{1}{2} x^\top P_N x + p_N^\top x + \gamma_N$$

The initial state $x_0$ is a fixed parameter, not a decision variable.

---

## Solvers

### 1. Riccati Recursion (`src/riccati.c`)

Implements **Algorithm 1** (§4.3.1, Proposition 4.3.5).  
Exploits the block-tridiagonal structure of the KKT matrix via dynamic programming.

**Complexity:** $O(N \cdot (n_x^2 n_u + n_u^3))$ — **linear in $N$**.

**Backward sweep** (k = N−1 down to 0):
```
S   = Aᵀ P
Re  = R + BᵀPB          (nu×nu, positive definite)
Y   = (M + SB)ᵀ         (nu×nx)
s   = P b
c   = s + p
d   = r + Bᵀ c
Re  = L Lᵀ              (Cholesky)
L Z = Y                 (forward substitution → Z: nu×nx)
L z = d                 (forward substitution → z: nu)
P  ← Q + SA − ZᵀZ      (symmetrized)
γ  ← γ + f + p'b + ½s'b − ½z'z
p  ← q + Aᵀc − Zᵀz
```

**Forward sweep** (k = 0 to N−1):
```
y = Zₖ xₖ + zₖ
Lₖᵀ uₖ = −y            (backward substitution)
x_{k+1} = A xₖ + B uₖ + b
```

---

### 2. Sparse KKT Solver (`src/kkt_sparse_solver.c`) — recommended

Formulates the ELQCP as an equality-constrained QP (§4.4.1):

$$\begin{pmatrix} G & -C \\ -C^\top & 0 \end{pmatrix} \begin{pmatrix} y \\ \pi \end{pmatrix} = -\begin{pmatrix} g \\ h \end{pmatrix}$$

with $y = [u_0,\, x_1,\, u_1,\, \ldots,\, u_{N-1},\, x_N]$.

The KKT matrix has **block-banded, symmetric indefinite** structure with $O(N(n_x+n_u)^2)$ nonzeros — linear in $N$. It is assembled in sparse COO format, converted to CSC, and solved with **UMFPACK** (SuiteSparse sparse LU), as recommended by Jørgensen et al. (2012), eq. (43)/(46):

> *"(43) and (46) are symmetric indefinite and may be solved by direct sparse algorithms such as MA57, PARDISO, and SuiteSparse."*

**Complexity:** $O(N \cdot (n_x + n_u)^3)$ — **linear in $N$** for fixed system dimensions.

**Requires:** SuiteSparse (`brew install suite-sparse` / `apt-get install libsuitesparse-dev`).

---

### 3. Dense KKT Solver (`src/kkt_solver.c`) — reference only

Same KKT formulation but assembled as a **full dense matrix** and solved with **dense LU** (Gaussian elimination with partial pivoting).

**Complexity:** $O\!\left((N(n_x + n_u))^3\right)$ — **cubic in $N$**.

Deliberately ignores all structure. Included only to demonstrate why dense methods are inappropriate here; use the sparse solver in practice.

---

## Repository Structure

```
elqcp/
├── CMakeLists.txt
├── include/
│   ├── elqcp.h          # Problem/solution structs and solver API
│   └── linalg.h         # Dense linear algebra declarations
├── src/
│   ├── linalg.c         # mat_mul, Cholesky, fwd/bwd substitution, LU
│   ├── riccati.c        # elqcp_solve_riccati() + alloc/free helpers
│   ├── kkt_sparse_solver.c  # elqcp_solve_kkt_sparse() — UMFPACK-based
│   └── kkt_solver.c     # elqcp_solve_kkt() — dense LU reference
├── examples/
│   └── compare_solvers.c  # Three-way timing comparison
└── docs/
    ├── jorgensen-2005.pdf
    └── jorgensen-2012.pdf
```

All matrices are stored **row-major**: `A[i*cols + j]` = element (i, j).

---

## Build

**Requirements:** CMake ≥ 3.10, a C99 compiler, math library, and SuiteSparse (for the sparse solver).

```bash
# macOS
brew install suite-sparse

# Ubuntu / Debian
# apt-get install libsuitesparse-dev

mkdir build && cd build
cmake ..
make
```

CMake will report whether UMFPACK was found:
```
-- UMFPACK found — sparse KKT solver enabled
```

This builds:
- `libelqcp_lib.a` — static library (all solvers)
- `compare_solvers` — timing comparison example

---

## Run the Example

```bash
./compare_solvers
```

**Toy system** (2 states, 1 input):

$$A = \begin{pmatrix} 0.9 & 0.1 \\ 0 & 0.8 \end{pmatrix}, \quad B = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad Q = I_2, \quad R = 0.01, \quad x_0 = (1, 0)^\top$$

### Sample output

```
N       Riccati(us)     SpKKT(us)       DenseKKT(us)    Sp/Ricc     Dense/Ricc  u diff (max)
------  --------------  --------------  --------------  ----------  ----------  ------------
5       7.90            499.85          6.35            63.3x       0.8x        8.33e-17
10      7.05            46.30           24.90           6.6x        3.5x        6.42e-17
20      11.95           114.00          170.55          9.5x        14.3x       9.54e-17
50      30.85           236.55          2052.40         7.7x        66.5x       1.28e-16
100     52.40           401.55          16210.95        7.7x        309.4x      6.94e-17
200     104.60          861.15          126349.00       8.2x        1207.9x     2.78e-16
500     255.95          2187.10         2354514.95      8.5x        9199.1x     2.78e-16
1000    502.15          4173.00         skipped         8.3x        ---         3.33e-16
2000    1017.20         8617.70         skipped         8.5x        ---         2.78e-16
```

**Key observations:**

- **Riccati** and **Sparse KKT** both scale **linearly** with $N$ (confirmed for $N = 5$ to $2000$).
- **Dense KKT** scales **cubically**: 9200× slower than Riccati at $N = 500$.
- Sparse KKT is ~8× slower than Riccati due to UMFPACK's symbolic factorization overhead (amortized when the same sparsity pattern is reused across many solves, as in MPC).
- All three solvers agree to **machine precision** ($\approx 10^{-16}$).

The Sp/Ricc ratio at $N = 5$ (63×) is dominated by UMFPACK's one-time symbolic factorization overhead; it stabilizes at ~8× for all larger $N$.

---

## API

```c
#include "elqcp.h"

// Allocate problem (all arrays zero-initialized)
ELQCP *prob = elqcp_alloc(N, nx, nu);

// Fill prob->A[k], prob->B[k], prob->Q[k], prob->R[k], ...
// (see include/elqcp.h for the full struct definition)

// Allocate solution
ELQCPSol *sol = elqcp_sol_alloc(N, nx, nu);

// Solve
double x0[nx] = { ... };
elqcp_solve_riccati(prob, x0, sol);       // O(N)   — structure-exploiting DP
elqcp_solve_kkt_sparse(prob, x0, sol);    // O(N)   — sparse LU via UMFPACK
elqcp_solve_kkt(prob, x0, sol);           // O(N^3) — dense LU, reference only

// Access results: sol->x[k], sol->u[k], sol->phi

// Free
elqcp_sol_free(sol, N);
elqcp_free(prob);
```
