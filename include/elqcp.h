#ifndef ELQCP_H
#define ELQCP_H

/*
 * Extended Linear-Quadratic Optimal Control Problem (ELQCP)
 *
 * Based on Jørgensen (2004), "Moving Horizon Estimation and Control",
 * Chapter 4, Problem 4.3.1 (eq. 4.46–4.47).
 *
 * Problem:
 *   min  phi = sum_{k=0}^{N-1} l_k(x_k, u_k) + l_N(x_N)
 *   s.t. x_{k+1} = A_k x_k + B_k u_k + b_k,  k = 0,...,N-1
 *
 * Stage cost:
 *   l_k(x,u) = 1/2 x'Q_k x + x'M_k u + 1/2 u'R_k u + q_k'x + r_k'u + f_k
 *
 * Terminal cost:
 *   l_N(x) = 1/2 x'P_N x + p_N'x + gamma_N
 *
 * All matrices stored row-major: A[i*cols + j] = A(i,j).
 * x_0 is a parameter (not a decision variable).
 */

/* ---------------------------------------------------------------------------
 * Problem data
 * --------------------------------------------------------------------------- */
typedef struct {
    int N;   /* prediction horizon (number of stages) */
    int nx;  /* state dimension */
    int nu;  /* input dimension */

    /* Stage data: arrays of length N, each entry pointing to a flat matrix/vector */
    double **Q;     /* [N]  nx×nx  symmetric PSD state-cost matrices       */
    double **M_mat; /* [N]  nx×nu  cross-cost matrices                     */
    double **R;     /* [N]  nu×nu  symmetric PD  control-cost matrices     */
    double **q;     /* [N]  nx     linear state-cost vectors                */
    double **r;     /* [N]  nu     linear control-cost vectors              */
    double  *f;     /* [N]         scalar stage costs                      */
    double **A;     /* [N]  nx×nx  state-transition matrices               */
    double **B;     /* [N]  nx×nu  input matrices                          */
    double **b;     /* [N]  nx     affine dynamics terms                   */

    /* Terminal cost */
    double *P_N;     /* nx×nx symmetric PSD terminal cost matrix           */
    double *p_N;     /* nx    linear terminal cost vector                  */
    double  gamma_N; /* scalar terminal cost                               */
} ELQCP;

/* ---------------------------------------------------------------------------
 * Solution
 * --------------------------------------------------------------------------- */
typedef struct {
    double **x;  /* [N+1] pointers to nx-vectors: optimal states x_0,...,x_N */
    double **u;  /* [N]   pointers to nu-vectors: optimal inputs u_0,...,u_{N-1} */
    double   phi; /* optimal objective value */
} ELQCPSol;

/* ---------------------------------------------------------------------------
 * Memory management helpers
 * --------------------------------------------------------------------------- */

/* Allocate an ELQCP problem struct and all its arrays (zero-initialized). */
ELQCP *elqcp_alloc(int N, int nx, int nu);

/* Free all memory associated with an ELQCP problem. */
void elqcp_free(ELQCP *prob);

/* Allocate a solution struct for a given problem. */
ELQCPSol *elqcp_sol_alloc(int N, int nx, int nu);

/* Free a solution struct. */
void elqcp_sol_free(ELQCPSol *sol, int N);

/* ---------------------------------------------------------------------------
 * Solvers
 * --------------------------------------------------------------------------- */

/*
 * Riccati recursion solver (Algorithm 1 / Proposition 4.3.5).
 *
 * Complexity: O(N * (nx^2 * nu + nu^2 * nx + nu^3))  — linear in N.
 *
 * x0: initial state (nx vector).
 * sol: pre-allocated solution struct (sol->x[0] is set to x0).
 * Returns 0 on success.
 */
int elqcp_solve_riccati(const ELQCP *prob, const double *x0, ELQCPSol *sol);

/*
 * KKT / optimality-system solver — DENSE (reference implementation).
 *
 * Assembles the full dense KKT matrix and solves it with dense LU
 * factorization.  Does NOT exploit the O(N) sparsity of the KKT system.
 *
 * Complexity: O((N*(nx+nu))^3)  — cubic in N.
 * Use only for small N or as a reference; prefer elqcp_solve_kkt_sparse.
 *
 * x0: initial state (nx vector).
 * sol: pre-allocated solution struct.
 * Returns 0 on success.
 */
int elqcp_solve_kkt(const ELQCP *prob, const double *x0, ELQCPSol *sol);

/*
 * KKT / optimality-system solver — SPARSE (via UMFPACK / SuiteSparse).
 *
 * Assembles the KKT matrix in sparse COO format (O(N*(nx+nu)²) nonzeros),
 * converts to CSC, and solves with UMFPACK's sparse LU.  Exploits the
 * block-banded sparsity of the optimality system.
 *
 * Complexity: O(N*(nx+nu)³)  — linear in N for fixed system dimensions.
 *
 * Reference: Jørgensen et al. (2012), IFAC NMPC, eq. (43)/(46).
 *
 * x0: initial state (nx vector).
 * sol: pre-allocated solution struct.
 * Returns 0 on success, -1 on failure.
 */
int elqcp_solve_kkt_sparse(const ELQCP *prob, const double *x0, ELQCPSol *sol);

#endif /* ELQCP_H */
