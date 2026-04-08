#include "elqcp.hpp"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <stdexcept>
#include <vector>

/*
 * Sparse KKT / Optimality-System Solver  (Eigen::SparseLU)
 *
 * Solves the symmetric indefinite saddle-point (KKT) system
 *
 *   [ G   -C ] [ y ]   [ -g ]
 *   [-C'   0 ] [ π ] = [ -h ]
 *
 * Variable ordering (primal vector y, dual vector π):
 *
 *   y = [u_0, x_1, u_1, x_2, …, u_{N-1}, x_N]    dim M  = N·(nu+nx)
 *   π = [π_0, π_1, …, π_{N-1}]                    dim nc = N·nx
 *
 * Block offsets within y:
 *   u_k  starts at  k·(nu+nx)
 *   x_{k+1} starts at  k·(nu+nx) + nu
 *
 * Hessian G (block-diagonal, positive definite):
 *   k=0      → R_0  at (u_0, u_0)
 *   k=1..N-1 → [[Q_k, M_k], [M_k', R_k]]  at (x_k, u_k)
 *   terminal → P_N  at (x_N, x_N)
 *
 * Constraint Jacobian −C (upper-right block of KKT), per constraint k:
 *   −B_k' at (u_k,    π_k)     ← u_k contribution  (B_k stored nx×nu)
 *   +I    at (x_{k+1},π_k)     ← identity (from −(−I))
 *   −A_k' at (x_k,    π_k)     ← k>0 only  (A_k stored nx×nx)
 *
 * The matrix is assembled once as a full Eigen::SparseMatrix<double> and solved
 * with Eigen::SparseLU using COLAMD fill-reducing ordering.
 *
 * Reference: Jørgensen et al. (2012), IFAC NMPC, eq. (43)/(46).
 */

double elqcp_solve_kkt_sparse(const ELQCP& prob, const Eigen::VectorXd& x0, ELQCPSol& sol)
{
    const int N  = prob.N;
    const int nx = prob.nx;
    const int nu = prob.nu;

    const int M   = N * (nu + nx);  /* number of primal variables */
    const int nc  = N * nx;          /* number of equality constraints */
    const int dim = M + nc;          /* total KKT system dimension */

    /* Primal and dual block offsets ---------------------------------------- */
    auto u_off = [&](int k) { return k * (nu + nx); };          /* u_k in y     */
    auto x_off = [&](int k) { return k * (nu + nx) + nu; };     /* x_{k+1} in y */
    auto d_off = [&](int k) { return M + k * nx; };             /* π_k in full w */

    /* -----------------------------------------------------------------------
     * Count nonzeros and reserve triplet storage
     *
     * G block (full, both triangles):
     *   R_0             : nu²
     *   (nx+nu)² each   : (N-1) middle stages
     *   P_N             : nx²
     *
     * ±C blocks (upper-right and lower-left, factor 2):
     *   −B_k' (nu×nx)   : N blocks
     *   +I   (nx)       : N blocks (diagonal)
     *   −A_k' (nx×nx)   : (N-1) blocks (k>0)
     * ----------------------------------------------------------------------- */
    const int nnz_G = nu*nu + (N - 1)*(nx + nu)*(nx + nu) + nx*nx;
    const int nnz_C = N*nu*nx + N*nx + (N > 1 ? (N - 1)*nx*nx : 0);
    const int nnz   = nnz_G + 2 * nnz_C;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    /* -----------------------------------------------------------------------
     * G block — block-diagonal Hessian
     * ----------------------------------------------------------------------- */

    /* k=0: R_0 for u_0 */
    {
        const int ou = u_off(0);
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nu; j++)
                triplets.emplace_back(ou + i, ou + j, prob.R[0](i, j));
    }

    /* k=1..N-1: [[Q_k, M_k], [M_k', R_k]] spanning (x_k, u_k) */
    for (int k = 1; k < N; k++) {
        const int ox = x_off(k - 1);  /* x_k = x_{(k-1)+1} lives here */
        const int ou = u_off(k);

        for (int i = 0; i < nx; i++)  /* Q_k */
            for (int j = 0; j < nx; j++)
                triplets.emplace_back(ox + i, ox + j, prob.Q[k](i, j));

        for (int i = 0; i < nx; i++)  /* M_k and M_k' (symmetric off-diagonal) */
            for (int j = 0; j < nu; j++) {
                triplets.emplace_back(ox + i, ou + j, prob.M[k](i, j));
                triplets.emplace_back(ou + j, ox + i, prob.M[k](i, j));
            }

        for (int i = 0; i < nu; i++)  /* R_k */
            for (int j = 0; j < nu; j++)
                triplets.emplace_back(ou + i, ou + j, prob.R[k](i, j));
    }

    /* Terminal: P_N for x_N */
    {
        const int ox = x_off(N - 1);
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < nx; j++)
                triplets.emplace_back(ox + i, ox + j, prob.P_N(i, j));
    }

    /* -----------------------------------------------------------------------
     * ±C blocks — upper-right (−C) and lower-left (−C') simultaneously
     *
     * The KKT matrix is symmetric so every off-diagonal entry is inserted
     * twice with the same value.
     *
     * Constraint k (k=0..N-1), component j in [0, nx):
     *   Dynamics: x_{k+1,j} − Σ_i B_k(j,i) u_{k,i} [+ k>0: − Σ_i A_k(j,i) x_{k,i}] = b_{k,j}
     *
     *   −C block (upper-right):
     *     −B_k(j,i)  at row (u_{k,i}),   col (π_{k,j})
     *     +1         at row (x_{k+1,j}), col (π_{k,j})
     *     −A_k(j,i)  at row (x_{k,i}),   col (π_{k,j})  [k>0]
     *
     *   −C' block (lower-left): transpose of the above (same values, swapped indices)
     * ----------------------------------------------------------------------- */
    for (int k = 0; k < N; k++) {
        const int ou      = u_off(k);
        const int ox_next = x_off(k);
        const int dc      = d_off(k);

        /* −B_k' entries: value = −B_k(j, i) */
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nx; j++) {
                const double v = -prob.B[k](j, i);
                triplets.emplace_back(ou + i,  dc + j, v);  /* upper-right */
                triplets.emplace_back(dc + j,  ou + i, v);  /* lower-left  */
            }

        /* +I entries (from −(−I) of the x_{k+1} column of C) */
        for (int j = 0; j < nx; j++) {
            triplets.emplace_back(ox_next + j, dc + j, 1.0);
            triplets.emplace_back(dc + j, ox_next + j, 1.0);
        }

        /* −A_k' entries (k > 0): value = −A_k(j, i) */
        if (k > 0) {
            const int ox_k = x_off(k - 1);  /* x_k lives at x_off(k-1) */
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < nx; j++) {
                    const double v = -prob.A[k](j, i);
                    triplets.emplace_back(ox_k + i, dc + j, v);
                    triplets.emplace_back(dc + j, ox_k + i, v);
                }
        }
    }

    /* Build sparse matrix */
    Eigen::SparseMatrix<double> K(dim, dim);
    K.setFromTriplets(triplets.begin(), triplets.end());

    /* -----------------------------------------------------------------------
     * Right-hand side  rhs = [−g; −h]
     *
     * −g (primal part):
     *   At u_0      : −(r_0 + M_0' x_0)    [folded initial state]
     *   At x_k (k≥1): −q_k
     *   At u_k (k≥1): −r_k
     *   At x_N      : −p_N
     *
     * −h (dual part):
     *   At π_0: A_0 x_0 + b_0   (= −h_0, h_0 = −(A_0 x_0 + b_0))
     *   At π_k: b_k             (= −h_k, h_k = −b_k)  for k≥1
     * ----------------------------------------------------------------------- */
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(dim);

    /* Primal: k=0, u_0 */
    rhs.segment(u_off(0), nu) = -(prob.r[0] + prob.M[0].transpose() * x0);

    /* Primal: k=1..N-1 */
    for (int k = 1; k < N; k++) {
        rhs.segment(x_off(k - 1), nx) = -prob.q[k];
        rhs.segment(u_off(k),     nu) = -prob.r[k];
    }

    /* Primal: terminal x_N */
    rhs.segment(x_off(N - 1), nx) = -prob.p_N;

    /* Dual: k=0 */
    rhs.segment(d_off(0), nx) = prob.A[0] * x0 + prob.b[0];

    /* Dual: k=1..N-1 */
    for (int k = 1; k < N; k++)
        rhs.segment(d_off(k), nx) = prob.b[k];

    /* -----------------------------------------------------------------------
     * Sparse LU factorization and solve
     *
     * Eigen::SparseLU uses a supernodal LU decomposition with COLAMD
     * fill-reducing ordering, which is well-suited for symmetric indefinite
     * saddle-point systems such as the KKT matrix assembled here.
     * ----------------------------------------------------------------------- */
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;

    solver.analyzePattern(K);
    solver.factorize(K);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("elqcp_solve_kkt_sparse: SparseLU factorization failed.");

    const Eigen::VectorXd w = solver.solve(rhs);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("elqcp_solve_kkt_sparse: SparseLU solve failed.");

    /* -----------------------------------------------------------------------
     * Extract primal solution
     * ----------------------------------------------------------------------- */
    sol.x[0] = x0;
    for (int k = 0; k < N; k++) {
        sol.u[k]     = w.segment(u_off(k), nu);
        sol.x[k + 1] = w.segment(x_off(k), nx);
    }

    /* -----------------------------------------------------------------------
     * Compute optimal objective value from the recovered trajectory
     * ----------------------------------------------------------------------- */
    double phi = prob.gamma_N;

    /* Stage k=0: x_0 is fixed */
    {
        const Eigen::VectorXd& u0 = sol.u[0];
        phi += 0.5 * x0.dot(prob.Q[0] * x0) + x0.dot(prob.M[0] * u0)
             + 0.5 * u0.dot(prob.R[0] * u0)  + prob.q[0].dot(x0)
             + prob.r[0].dot(u0)              + prob.f[0];
    }

    /* Stages k=1..N-1 */
    for (int k = 1; k < N; k++) {
        const Eigen::VectorXd& xk = sol.x[k];
        const Eigen::VectorXd& uk = sol.u[k];
        phi += 0.5 * xk.dot(prob.Q[k] * xk) + xk.dot(prob.M[k] * uk)
             + 0.5 * uk.dot(prob.R[k] * uk)  + prob.q[k].dot(xk)
             + prob.r[k].dot(uk)              + prob.f[k];
    }

    /* Terminal cost */
    const Eigen::VectorXd& xN = sol.x[N];
    phi += 0.5 * xN.dot(prob.P_N * xN) + prob.p_N.dot(xN);

    sol.opt_val = phi;
    return phi;
}
