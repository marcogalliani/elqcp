# elqcp

Solvers for the **Extended Linear-Quadratic Optimal Control Problem (ELQCP)**, implemented in C++ using Eigen for efficient sparse and structured linear algebra.

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

### 1. Riccati Recursion (`src/riccati.cpp`)

Implements **Algorithm 1** (§4.3.1, Proposition 4.3.5) in C++ using Eigen's dense linear algebra.
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

### 2. Sparse KKT Solver (`src/kkt_sparse_solver.cpp`) — recommended

Formulates the ELQCP as an equality-constrained QP (§4.4.1):

$$\begin{pmatrix} G & -C \\ -C^\top & 0 \end{pmatrix} \begin{pmatrix} y \\ \pi \end{pmatrix} = -\begin{pmatrix} g \\ h \end{pmatrix}$$

with $y = [u_0,\, x_1,\, u_1,\, \ldots,\, u_{N-1},\, x_N]$.

The KKT matrix has **block-banded, symmetric indefinite** structure with $O(N(n_x+n_u)^2)$ nonzeros — linear in $N$. It is assembled in sparse COO format and solved with **Eigen::SparseLU** (supernodal LU with COLAMD fill-reducing ordering), which correctly handles the symmetric indefinite structure via partial pivoting.

**Note:** The KKT matrix is structurally symmetric but the current solver treats it as general unsymmetric (for portability and stability). See *Optimization Note* below for advanced alternatives.

**Complexity:** $O(N \cdot (n_x + n_u)^3)$ — **linear in $N$** for fixed system dimensions.

---

## Repository Structure

```
elqcp/
├── CMakeLists.txt
├── include/
│   └── elqcp.hpp        # Problem/solution structs and solver API (C++ / Eigen)
├── src/
│   ├── riccati.cpp      # elqcp_solve_riccati() + alloc helper
│   └── kkt_sparse_solver.cpp  # elqcp_solve_kkt_sparse() via Eigen::SparseLU
├── examples/
│   ├── compare_solvers.cpp  # Riccati vs sparse KKT timing benchmark
│   └── plot_scaling.py      # Matplotlib script: timing graphs
└── docs/
    ├── jorgensen-2004.pdf   # Moving Horizon Estimation and Control (PhD thesis)
    ├── jorgensen-2012.pdf   # Numerical Methods for Solution of the ELQCP (IFAC NMPC)
    └── OPTIMIZATION_NOTE.md # Symmetric indefinite solver options
```

All matrices are stored in **Eigen's default Eigen::RowMajor** (for explicit dense matrices) or **Eigen::ColMajor** for sparse matrices.

---

## Build

**Requirements:** CMake ≥ 3.14, a C++17 compiler, and Eigen3 (header-only).

```bash
# macOS
brew install eigen

# Ubuntu / Debian
apt-get install libeigen3-dev

# Build
mkdir build && cd build
cmake ..
make -j4
```

This builds:
- `libelqcp_lib.a` — static library (Riccati and sparse KKT solvers)
- `compare_solvers` — timing/correctness benchmark

---

## Run the Example

```bash
cd build
./compare_solvers
```

**Toy system** (2 states, 1 input):

$$A = \begin{pmatrix} 0.9 & 0.1 \\ 0 & 0.8 \end{pmatrix}, \quad B = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad Q = I_2, \quad R = 0.01, \quad x_0 = (1, 0)^\top$$

### Sample output (timing in microseconds, N=5 to 5000)

```
Extended LQR: solver timing comparison
System: nx=2, nu=1, x0=[1,0]
Sparse KKT solver: Eigen::SparseLU (COLAMDOrdering)

N       Riccati(us)     SpKKT(us)       Sp/Ricc     u diff (max)
------  --------------  --------------  ----------  -----------
5       17              27              1.6x        1.11e-16
10      18              32              1.8x        8.88e-17
20      20              45              2.2x        3.33e-16
50      56              145             2.6x        2.22e-16
100     65              280             4.3x        1.11e-16
200     135             550             4.1x        3.33e-16
500     270             1437            5.3x        2.78e-16
1000    590             2885            4.9x        2.78e-16
2000    1185            5932            5.0x        3.33e-16
5000    3067            14680           4.8x        1.28e-16
```

**Key observations:**

- **Riccati** and **Sparse KKT** both scale **linearly** with $N$ (confirmed for $N = 5$ to $5000$).
- Both solvers agree to **machine precision** ($\approx 10^{-16}$).
- Riccati recursion is 4–5× faster than sparse KKT across all $N$, owing to:
  - Riccati's tighter O(N·nu³) operations on small blocks
  - Sparse KKT's overhead from assembling and factoring the full KKT matrix
  - When the same sparsity pattern is reused across many solves (e.g., MPC), the sparse solver overhead is amortized
- Both show excellent scaling and are suitable for large prediction horizons.

### Python visualization

```bash
python3 plot_scaling.py
```

Generates:
- `scaling.png` — log-log plot of solve time vs. horizon length
- `scaling_ratio.png` — semilog-x plot of SparseLU/Riccati ratio

---

## API

```cpp
#include "elqcp.hpp"
#include <Eigen/Dense>

// Create problem instance
ELQCP prob;
prob.N = 100;
prob.nx = 2;
prob.nu = 1;

// Allocate and fill problem data (Eigen vectors/matrices)
prob.Q.resize(prob.N);
prob.M.resize(prob.N);
prob.R.resize(prob.N);
// ... fill A[k], B[k], Q[k], R[k], q[k], r[k], b[k], f[k] ...
prob.P_N = Eigen::MatrixXd::Identity(2, 2);

// Allocate solution
ELQCPSol sol = elqcp_sol_alloc(prob);

// Initial state (parameter)
Eigen::VectorXd x0(2);
x0 << 1.0, 0.0;

// Solve
double phi_riccati = elqcp_solve_riccati(prob, x0, sol);
double phi_kkt     = elqcp_solve_kkt_sparse(prob, x0, sol);

// Access results
for (int k = 0; k <= prob.N; k++)
    Eigen::VectorXd xk = sol.x[k];  // state at stage k
for (int k = 0; k < prob.N; k++)
    Eigen::VectorXd uk = sol.u[k];  // input at stage k
double phi = sol.opt_val;            // optimal objective value
```

---

## Optimization Note: Symmetric-Indefinite KKT Solvers

The current implementation uses `Eigen::SparseLU` with COLAMD ordering, which treats the **structurally symmetric** KKT matrix as a general unsymmetric matrix. This is robust and requires no external dependencies, but pays a ~2× cost in both memory and flops compared to a dedicated symmetric-indefinite solver.

### When to consider alternatives:

For **large systems** (e.g., $n_x = 20, n_u = 10$) where the KKT matrix is $O(30N) \times O(30N)$ with dense blocks, a true symmetric-indefinite LDL^T factorization (using Bunch-Kaufman pivoting and symmetric AMD ordering) could provide meaningful speedup.

### Available options (and limitations):

| Solver | Library | Advantages | Limitations |
|--------|---------|------------|-----------
| **SparseLU** (current) | Eigen (built-in) | Portable, no dependencies, stable | ~2× overhead from treating as unsymmetric |
| **Pardiso LDL^T** | Intel MKL via `Eigen::PardisoLDLT` | Full Bunch-Kaufman pivoting, best asymptotic complexity | MKL is large (~1 GB), not natively available on Apple Silicon, license restricted |
| **CHOLMOD LDL^T** | SuiteSparse via `Eigen::CholmodDecomposition` | Lightweight (~50 MB), AMD ordering, available on all platforms | Simplicial LDL^T lacks Bunch-Kaufman pivoting — factorization fails on indefinite systems due to zero/negative pivots |
| **UMFPACK** | SuiteSparse via `Eigen::UmfPackLU` | Recommended in Jørgensen et al. (2012) | Unsymmetric solver like SparseLU — no benefit over current choice |

**Recommendation:** The current `Eigen::SparseLU` approach is the right balance of **correctness, portability, and simplicity** for the broadest range of users. If you have a specific large-scale deployment target (e.g., always x86-64 Linux with MKL installed), Pardiso is viable; otherwise, stick with the built-in solver.
