/*
 * compare_solvers.c
 *
 * Timing comparison of three solvers for the ELQCP:
 *
 *   1. Riccati recursion       — O(N)    structure-exploiting dynamic programming
 *   2. Sparse KKT (UMFPACK)    — O(N)    sparse LU on the block-banded KKT system
 *   3. Dense  KKT (LU)         — O(N^3)  ignores sparsity; shown only for small N
 *
 * The sparse solver exploits the O(N*(nx+nu)^2) sparsity of the KKT matrix,
 * as recommended by Jørgensen et al. (2012), IFAC NMPC, eqs. (43)/(46).
 * Both 1 and 2 scale linearly in N; the constant factor favours Riccati
 * because it exploits the control-specific block structure more aggressively.
 *
 * Toy LTI system (nx=2, nu=1):
 *   x_{k+1} = A x_k + B u_k
 *   A = [[0.9, 0.1], [0, 0.8]],  B = [[0], [1]]
 *   Q = I_2,  R = 0.01,  P_N = I_2,  x0 = [1, 0]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "elqcp.h"

/* Wall-clock time in seconds */
static double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Build the LTI toy problem for a given horizon N */
static ELQCP *build_problem(int N)
{
    const int nx = 2, nu = 1;
    ELQCP *prob = elqcp_alloc(N, nx, nu);

    double A[4] = {0.9, 0.1, 0.0, 0.8};
    double B[2] = {0.0, 1.0};
    double Q[4] = {1.0, 0.0, 0.0, 1.0};
    double R[1] = {0.01};

    for (int k = 0; k < N; k++) {
        memcpy(prob->A[k], A, sizeof(A));
        memcpy(prob->B[k], B, sizeof(B));
        memcpy(prob->Q[k], Q, sizeof(Q));
        memcpy(prob->R[k], R, sizeof(R));
    }
    memcpy(prob->P_N, Q, sizeof(Q));
    return prob;
}

static double max_u_diff(const ELQCPSol *s1, const ELQCPSol *s2, int N, int nu)
{
    double m = 0.0;
    for (int k = 0; k < N; k++)
        for (int i = 0; i < nu; i++) {
            double d = fabs(s1->u[k][i] - s2->u[k][i]);
            if (d > m) m = d;
        }
    return m;
}

static double max_x_diff(const ELQCPSol *s1, const ELQCPSol *s2, int N, int nx)
{
    double m = 0.0;
    for (int k = 0; k <= N; k++)
        for (int i = 0; i < nx; i++) {
            double d = fabs(s1->x[k][i] - s2->x[k][i]);
            if (d > m) m = d;
        }
    return m;
}

int main(void)
{
    const int nx = 2, nu = 1;
    double x0[2] = {1.0, 0.0};

    int horizons[]  = {5, 10, 20, 50, 100, 200, 500, 1000, 2000};
    int n_horizons  = (int)(sizeof(horizons) / sizeof(horizons[0]));
    const int dense_max_N = 500;  /* dense KKT skipped above this */
    const int n_repeats   = 20;

    printf("Extended LQR: solver timing comparison\n");
    printf("System: nx=%d, nu=%d, x0=[1,0]\n", nx, nu);
#ifdef HAVE_UMFPACK
    printf("Sparse KKT solver: UMFPACK (SuiteSparse)\n");
#else
    printf("Sparse KKT solver: NOT available (build without SuiteSparse)\n");
#endif
    printf("\n");

    /* Header */
    printf("%-6s  %-14s  %-14s  %-14s  %-10s  %-10s  %-12s\n",
           "N", "Riccati(us)", "SpKKT(us)", "DenseKKT(us)",
           "Sp/Ricc", "Dense/Ricc", "u diff (max)");
    printf("%-6s  %-14s  %-14s  %-14s  %-10s  %-10s  %-12s\n",
           "------", "--------------", "--------------", "--------------",
           "----------", "----------", "------------");

    for (int hi = 0; hi < n_horizons; hi++) {
        int N = horizons[hi];

        ELQCP    *prob  = build_problem(N);
        ELQCPSol *sol_r = elqcp_sol_alloc(N, nx, nu);
        ELQCPSol *sol_s = elqcp_sol_alloc(N, nx, nu);
        ELQCPSol *sol_d = elqcp_sol_alloc(N, nx, nu);

        /* --- Riccati --- */
        double t0 = wall_time();
        for (int rep = 0; rep < n_repeats; rep++)
            elqcp_solve_riccati(prob, x0, sol_r);
        double t_riccati = (wall_time() - t0) / n_repeats * 1e6;

        /* --- Sparse KKT --- */
        double t_sparse = -1.0;
#ifdef HAVE_UMFPACK
        t0 = wall_time();
        for (int rep = 0; rep < n_repeats; rep++)
            elqcp_solve_kkt_sparse(prob, x0, sol_s);
        t_sparse = (wall_time() - t0) / n_repeats * 1e6;
#endif

        /* --- Dense KKT (only for small N) --- */
        double t_dense = -1.0;
        if (N <= dense_max_N) {
            t0 = wall_time();
            for (int rep = 0; rep < n_repeats; rep++)
                elqcp_solve_kkt(prob, x0, sol_d);
            t_dense = (wall_time() - t0) / n_repeats * 1e6;
        }

        /* --- Print --- */
        char sp_str[32], dense_str[32], ratio_sp[16], ratio_dense[16], udiff_str[32];

        if (t_sparse >= 0.0)
            snprintf(sp_str,    sizeof(sp_str),    "%.2f", t_sparse);
        else
            snprintf(sp_str,    sizeof(sp_str),    "n/a");

        if (t_dense >= 0.0)
            snprintf(dense_str, sizeof(dense_str), "%.2f", t_dense);
        else
            snprintf(dense_str, sizeof(dense_str), "skipped");

        if (t_sparse >= 0.0)
            snprintf(ratio_sp,    sizeof(ratio_sp),    "%.1fx", t_sparse / t_riccati);
        else
            snprintf(ratio_sp,    sizeof(ratio_sp),    "---");

        if (t_dense >= 0.0)
            snprintf(ratio_dense, sizeof(ratio_dense), "%.1fx", t_dense  / t_riccati);
        else
            snprintf(ratio_dense, sizeof(ratio_dense), "---");

        /* max |u_riccati - u_sparse| */
        if (t_sparse >= 0.0)
            snprintf(udiff_str, sizeof(udiff_str), "%.2e", max_u_diff(sol_r, sol_s, N, nu));
        else
            snprintf(udiff_str, sizeof(udiff_str), "---");

        printf("%-6d  %-14.2f  %-14s  %-14s  %-10s  %-10s  %-12s\n",
               N, t_riccati, sp_str, dense_str, ratio_sp, ratio_dense, udiff_str);

        elqcp_free(prob);
        elqcp_sol_free(sol_r, N);
        elqcp_sol_free(sol_s, N);
        elqcp_sol_free(sol_d, N);
    }

    /* ------------------------------------------------------------------ */
    /* Detailed verification for N=10                                       */
    /* ------------------------------------------------------------------ */
    printf("\n--- Verification for N=10 ---\n");
    {
        int N = 10;
        ELQCP    *prob  = build_problem(N);
        ELQCPSol *sol_r = elqcp_sol_alloc(N, nx, nu);
        ELQCPSol *sol_s = elqcp_sol_alloc(N, nx, nu);
        ELQCPSol *sol_d = elqcp_sol_alloc(N, nx, nu);

        elqcp_solve_riccati(prob, x0, sol_r);
        elqcp_solve_kkt(prob, x0, sol_d);
#ifdef HAVE_UMFPACK
        elqcp_solve_kkt_sparse(prob, x0, sol_s);
#endif

        printf("Optimal cost: Riccati = %.10f  |  Dense KKT = %.10f",
               sol_r->phi, sol_d->phi);
#ifdef HAVE_UMFPACK
        printf("  |  Sparse KKT = %.10f", sol_s->phi);
#endif
        printf("\n\n");

        printf("Optimal controls u_k:\n");
        printf("  %-4s  %-14s  %-14s", "k", "Riccati", "Dense KKT");
#ifdef HAVE_UMFPACK
        printf("  %-14s", "Sparse KKT");
#endif
        printf("\n");
        for (int k = 0; k < N; k++) {
            printf("  %-4d  %+.8f    %+.8f", k, sol_r->u[k][0], sol_d->u[k][0]);
#ifdef HAVE_UMFPACK
            printf("    %+.8f", sol_s->u[k][0]);
#endif
            printf("\n");
        }

        printf("\nMax |u_riccati - u_dense |  = %.2e\n", max_u_diff(sol_r, sol_d, N, nu));
        printf("Max |x_riccati - x_dense |  = %.2e\n", max_x_diff(sol_r, sol_d, N, nx));
#ifdef HAVE_UMFPACK
        printf("Max |u_riccati - u_sparse|  = %.2e\n", max_u_diff(sol_r, sol_s, N, nu));
        printf("Max |x_riccati - x_sparse|  = %.2e\n", max_x_diff(sol_r, sol_s, N, nx));
#endif

        elqcp_free(prob);
        elqcp_sol_free(sol_r, N);
        elqcp_sol_free(sol_s, N);
        elqcp_sol_free(sol_d, N);
    }

    return 0;
}
