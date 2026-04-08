#include "elqcp.hpp"
#include <stdexcept>
#include <string>

/*
 * Riccati recursion solver  (Algorithm 1, §4.3.1, Jørgensen 2004)
 *
 * Notation:
 *   S   = A_k^T P                     [nx×nx]
 *   PB  = P B_k                       [nx×nu]
 *   Re  = R_k + B_k^T PB              [nu×nu]  (positive definite)
 *   MB  = M_k + S B_k                 [nx×nu]
 *   Y   = MB^T                        [nu×nx]
 *   s   = P b_k                       [nx]
 *   c   = s + p                       [nx]
 *   d   = r_k + B_k^T c               [nu]
 *   Cholesky: Re = L L^T
 *   L Z = Y  →  Z [nu×nx]
 *   L z = d  →  z [nu]
 *   P  ← Q_k + S A_k − Z^T Z         (symmetrized)
 *   γ  ← γ + f_k + p'b + ½ s'b − ½ z'z
 *   p  ← q_k + A_k^T c − Z^T z
 *
 * Forward sweep:
 *   y_k = Z_k x_k + z_k
 *   Solve L_k^T u_k = −y_k
 *   x_{k+1} = A_k x_k + B_k u_k + b_k
 */

ELQCPSol elqcp_sol_alloc(const ELQCP& prob)
{
    ELQCPSol sol;
    sol.x.assign(prob.N + 1, Eigen::VectorXd::Zero(prob.nx));
    sol.u.assign(prob.N,     Eigen::VectorXd::Zero(prob.nu));
    return sol;
}

double elqcp_solve_riccati(const ELQCP& prob, const Eigen::VectorXd& x0, ELQCPSol& sol)
{
    const int N = prob.N;

    /* Storage for the factorization sequence {L_k, Z_k, z_k}_{k=0}^{N-1}
     * needed by the forward sweep.                                         */
    std::vector<Eigen::MatrixXd> Lk(N);  /* [nu×nu] lower-triangular Cholesky factor */
    std::vector<Eigen::MatrixXd> Zk(N);  /* [nu×nx] Z = L^{-1} Y                    */
    std::vector<Eigen::VectorXd> zk(N);  /* [nu]    z = L^{-1} d                    */

    /* Initialise value function at the terminal stage */
    Eigen::MatrixXd P = prob.P_N;
    Eigen::VectorXd p = prob.p_N;
    double gamma      = prob.gamma_N;

    /* ---- Backward sweep ---- */
    for (int k = N - 1; k >= 0; k--) {
        const Eigen::MatrixXd& Ak = prob.A[k];
        const Eigen::MatrixXd& Bk = prob.B[k];
        const Eigen::VectorXd& bk = prob.b[k];
        const Eigen::MatrixXd& Qk = prob.Q[k];
        const Eigen::MatrixXd& Mk = prob.M[k];
        const Eigen::MatrixXd& Rk = prob.R[k];
        const Eigen::VectorXd& qk = prob.q[k];
        const Eigen::VectorXd& rk = prob.r[k];

        Eigen::MatrixXd S  = Ak.transpose() * P;      /* nx × nx */
        Eigen::MatrixXd PB = P * Bk;                  /* nx × nu */
        Eigen::MatrixXd Re = Rk + Bk.transpose() * PB;/* nu × nu  (PD) */
        Eigen::MatrixXd MB = Mk + S * Bk;             /* nx × nu */
        Eigen::MatrixXd Y  = MB.transpose();           /* nu × nx */

        Eigen::VectorXd s = P * bk;                   /* nx */
        Eigen::VectorXd c = s + p;                    /* nx */
        Eigen::VectorXd d = rk + Bk.transpose() * c;  /* nu */

        /* Cholesky: Re = L L^T */
        Eigen::LLT<Eigen::MatrixXd> llt(Re);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error(
                "Riccati: Cholesky of Re failed at stage " + std::to_string(k) +
                " — Re is not positive definite.");

        Lk[k] = llt.matrixL();

        /* Z_k = L^{-1} Y  (solve L Z = Y column-by-column) */
        Zk[k] = llt.matrixL().solve(Y);   /* nu × nx */

        /* z_k = L^{-1} d */
        zk[k] = llt.matrixL().solve(d);   /* nu */

        /* γ ← γ + f_k + p'b + ½ s'b − ½ z'z */
        gamma += prob.f[k] + p.dot(bk) + 0.5 * s.dot(bk) - 0.5 * zk[k].squaredNorm();

        /* P ← Q_k + S A_k − Z_k^T Z_k,  then symmetrize to prevent drift */
        Eigen::MatrixXd Pnew = Qk + S * Ak - Zk[k].transpose() * Zk[k];
        P = 0.5 * (Pnew + Pnew.transpose());

        /* p ← q_k + A_k^T c − Z_k^T z_k */
        p = qk + Ak.transpose() * c - Zk[k].transpose() * zk[k];
    }

    /* Optimal value: φ* = ½ x0' P x0 + p' x0 + γ */
    const double opt_val = 0.5 * x0.dot(P * x0) + p.dot(x0) + gamma;

    /* ---- Forward sweep ---- */
    sol.x[0] = x0;
    for (int k = 0; k < N; k++) {
        /* y = Z_k x_k + z_k */
        Eigen::VectorXd y = Zk[k] * sol.x[k] + zk[k];

        /* Solve L_k^T u_k = −y  (L_k^T is upper triangular) */
        sol.u[k] = Lk[k].transpose().triangularView<Eigen::Upper>().solve(-y);

        /* x_{k+1} = A_k x_k + B_k u_k + b_k */
        sol.x[k + 1] = prob.A[k] * sol.x[k] + prob.B[k] * sol.u[k] + prob.b[k];
    }

    sol.opt_val = opt_val;
    return opt_val;
}
