#pragma once

/*
 * Extended Linear-Quadratic Optimal Control Problem (ELQCP) — C++ / Eigen API
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
 * x_0 is a parameter (not a decision variable).
 */

#include <Eigen/Dense>
#include <vector>

/* ---------------------------------------------------------------------------
 * Problem data
 * --------------------------------------------------------------------------- */
struct ELQCP {
    int N;   /* prediction horizon (number of stages) */
    int nx;  /* state dimension                       */
    int nu;  /* input dimension                       */

    /* Stage data: vectors of length N */
    std::vector<Eigen::MatrixXd> Q;  /* [N]  nx×nx  symmetric PSD state-cost     */
    std::vector<Eigen::MatrixXd> M;  /* [N]  nx×nu  cross-cost                   */
    std::vector<Eigen::MatrixXd> R;  /* [N]  nu×nu  symmetric PD  input-cost     */
    std::vector<Eigen::VectorXd> q;  /* [N]  nx     linear state-cost vector      */
    std::vector<Eigen::VectorXd> r;  /* [N]  nu     linear input-cost vector      */
    std::vector<double>          f;  /* [N]         scalar stage cost offsets     */

    /* Dynamics */
    std::vector<Eigen::MatrixXd> A;  /* [N]  nx×nx  state-transition matrices    */
    std::vector<Eigen::MatrixXd> B;  /* [N]  nx×nu  input matrices               */
    std::vector<Eigen::VectorXd> b;  /* [N]  nx     affine dynamics terms         */

    /* Terminal cost */
    Eigen::MatrixXd P_N;    /* nx×nx  symmetric PSD                              */
    Eigen::VectorXd p_N;    /* nx                                                */
    double          gamma_N = 0.0;
};

/* ---------------------------------------------------------------------------
 * Solution
 * --------------------------------------------------------------------------- */
struct ELQCPSol {
    std::vector<Eigen::VectorXd> x;  /* [N+1] optimal states  x_0, ..., x_N    */
    std::vector<Eigen::VectorXd> u;  /* [N]   optimal inputs  u_0, ..., u_{N-1} */
    double opt_val = 0.0;            /* optimal objective value                  */
};

/* Allocate a solution struct with the right sizes for a given problem. */
ELQCPSol elqcp_sol_alloc(const ELQCP& prob);

/* ---------------------------------------------------------------------------
 * Solvers
 * --------------------------------------------------------------------------- */

/*
 * Riccati recursion solver (Proposition 4.3.5, Jørgensen 2004).
 *
 * Backward sweep computes the optimal value function coefficients {P, p, γ};
 * forward sweep recovers the optimal trajectory.
 *
 * Complexity: O(N · (nx² nu + nu² nx + nu³))  — linear in N.
 *
 * Returns the optimal objective value.
 */
double elqcp_solve_riccati(const ELQCP& prob, const Eigen::VectorXd& x0, ELQCPSol& sol);

/*
 * Sparse KKT solver using Eigen::SparseLU.
 *
 * Assembles the symmetric indefinite saddle-point system
 *
 *   [ G   -C ] [ y ]   [ -g ]
 *   [-C'   0 ] [ π ] = [ -h ]
 *
 * in Eigen's SparseMatrix format (O(N·(nx+nu)²) nonzeros — linear in N),
 * then solves with Eigen::SparseLU and a fill-reducing column ordering.
 *
 * ── Why SparseLU rather than SimplicialLDLT ──────────────────────────────────
 * The KKT matrix is symmetric indefinite: the primal block G ≻ 0 gives
 * positive eigenvalues, while the Schur complement −C'G⁻¹C ≺ 0 gives negative
 * eigenvalues.  Eigen's SimplicialLDLT terminates at the first non-positive
 * pivot and is therefore limited to positive-(semi)definite matrices.
 * SparseLU (supernodal LU with partial column pivoting) handles the full
 * indefinite structure correctly, mirroring the role of UMFPACK in the C
 * implementation.
 *
 * Complexity: O(N · (nx+nu)³)  — linear in N for fixed system dimensions.
 *
 * Reference: Jørgensen et al. (2012), IFAC NMPC, eq. (43)/(46).
 *
 * Returns the optimal objective value.
 */
double elqcp_solve_kkt_sparse(const ELQCP& prob, const Eigen::VectorXd& x0, ELQCPSol& sol);
