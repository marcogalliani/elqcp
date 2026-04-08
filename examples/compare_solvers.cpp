/*
 * compare_solvers.cpp
 *
 * Timing comparison of two solvers for the ELQCP:
 *
 *   1. Riccati recursion   — O(N)  structure-exploiting dynamic programming
 *   2. Sparse KKT (Eigen)  — O(N)  sparse LU on the block-banded KKT system
 *
 * Both solvers scale linearly in the horizon length N; Riccati is faster in
 * practice because it exploits the control-specific block structure more
 * aggressively, while the sparse KKT solver pays the overhead of SparseLU
 * symbolic and numeric factorization.
 *
 * Toy LTI system (nx=2, nu=1):
 *   x_{k+1} = A x_k + B u_k
 *   A = [[0.9, 0.1], [0, 0.8]],  B = [[0], [1]]
 *   Q = I_2,  R = 0.01,  P_N = I_2,  x0 = [1, 0]
 *
 * Output:
 *   - Human-readable table to stdout
 *   - Machine-readable CSV to timing_results.csv  (for plot_scaling.py)
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "elqcp.hpp"

/* ---- Timing helper -------------------------------------------------------- */
static double wall_time_us()
{
    using Clock = std::chrono::high_resolution_clock;
    using Dur   = std::chrono::duration<double, std::micro>;
    return Dur(Clock::now().time_since_epoch()).count();
}

/* ---- Build the LTI toy problem ------------------------------------------- */
static ELQCP build_problem(int N)
{
    const int nx = 2, nu = 1;

    Eigen::Matrix2d A;
    A << 0.9, 0.1,
         0.0, 0.8;

    Eigen::Matrix<double, 2, 1> B;
    B << 0.0, 1.0;

    Eigen::Matrix2d Q = Eigen::Matrix2d::Identity();
    Eigen::Matrix<double, 1, 1> R;
    R << 0.01;

    ELQCP prob;
    prob.N  = N;
    prob.nx = nx;
    prob.nu = nu;

    prob.Q.assign(N, Q);
    prob.M.assign(N, Eigen::MatrixXd::Zero(nx, nu));
    prob.R.assign(N, R);
    prob.q.assign(N, Eigen::VectorXd::Zero(nx));
    prob.r.assign(N, Eigen::VectorXd::Zero(nu));
    prob.f.assign(N, 0.0);
    prob.A.assign(N, A);
    prob.B.assign(N, B);
    prob.b.assign(N, Eigen::VectorXd::Zero(nx));

    prob.P_N    = Q;
    prob.p_N    = Eigen::VectorXd::Zero(nx);
    prob.gamma_N = 0.0;

    return prob;
}

/* ---- Solution difference helpers ----------------------------------------- */
static double max_u_diff(const ELQCPSol& s1, const ELQCPSol& s2)
{
    double m = 0.0;
    for (std::size_t k = 0; k < s1.u.size(); k++)
        m = std::max(m, (s1.u[k] - s2.u[k]).cwiseAbs().maxCoeff());
    return m;
}

static double max_x_diff(const ELQCPSol& s1, const ELQCPSol& s2)
{
    double m = 0.0;
    for (std::size_t k = 0; k < s1.x.size(); k++)
        m = std::max(m, (s1.x[k] - s2.x[k]).cwiseAbs().maxCoeff());
    return m;
}

/* ========================================================================== */
int main()
{
    const int nx = 2, nu = 1;
    Eigen::Vector2d x0(1.0, 0.0);

    const std::vector<int> horizons = {5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
    const int n_repeats = 20;

    std::cout << "Extended LQR: solver timing comparison\n"
              << "System: nx=" << nx << ", nu=" << nu << ", x0=[1,0]\n"
              << "Sparse KKT solver: Eigen::SparseLU (COLAMDOrdering)\n\n";

    /* Header */
    std::printf("%-6s  %-14s  %-14s  %-10s  %-12s\n",
                "N", "Riccati(us)", "SpKKT(us)", "Sp/Ricc", "u diff (max)");
    std::printf("%-6s  %-14s  %-14s  %-10s  %-12s\n",
                "------", "--------------", "--------------", "----------", "------------");

    /* CSV output */
    std::ofstream csv("timing_results.csv");
    csv << "N,riccati_us,kkt_sparse_us\n";

    for (int N : horizons) {
        ELQCP prob = build_problem(N);

        ELQCPSol sol_r = elqcp_sol_alloc(prob);
        ELQCPSol sol_s = elqcp_sol_alloc(prob);

        /* --- Riccati --- */
        double t0 = wall_time_us();
        for (int rep = 0; rep < n_repeats; rep++)
            elqcp_solve_riccati(prob, x0, sol_r);
        const double t_riccati = (wall_time_us() - t0) / n_repeats;

        /* --- Sparse KKT --- */
        t0 = wall_time_us();
        for (int rep = 0; rep < n_repeats; rep++)
            elqcp_solve_kkt_sparse(prob, x0, sol_s);
        const double t_kkt = (wall_time_us() - t0) / n_repeats;

        const double ratio  = t_kkt / t_riccati;
        const double udiff  = max_u_diff(sol_r, sol_s);

        std::printf("%-6d  %-14.2f  %-14.2f  %-10.1fx  %-12.2e\n",
                    N, t_riccati, t_kkt, ratio, udiff);

        csv << N << "," << std::fixed << std::setprecision(4)
            << t_riccati << "," << t_kkt << "\n";
    }

    csv.close();
    std::cout << "\nTiming data written to timing_results.csv\n";

    /* -----------------------------------------------------------------------
     * Detailed verification for N=10
     * ----------------------------------------------------------------------- */
    std::cout << "\n--- Verification for N=10 ---\n";
    {
        const int N = 10;
        ELQCP prob = build_problem(N);

        ELQCPSol sol_r = elqcp_sol_alloc(prob);
        ELQCPSol sol_s = elqcp_sol_alloc(prob);

        elqcp_solve_riccati(prob, x0, sol_r);
        elqcp_solve_kkt_sparse(prob, x0, sol_s);

        std::printf("Optimal cost:  Riccati = %.10f  |  Sparse KKT = %.10f\n\n",
                    sol_r.opt_val, sol_s.opt_val);

        std::printf("Optimal controls u_k:\n");
        std::printf("  %-4s  %-14s  %-14s\n", "k", "Riccati", "Sparse KKT");
        for (int k = 0; k < N; k++)
            std::printf("  %-4d  %+.8f    %+.8f\n",
                        k, sol_r.u[k](0), sol_s.u[k](0));

        std::printf("\nMax |u_riccati − u_sparse| = %.2e\n", max_u_diff(sol_r, sol_s));
        std::printf("Max |x_riccati − x_sparse| = %.2e\n", max_x_diff(sol_r, sol_s));
    }

    return 0;
}
