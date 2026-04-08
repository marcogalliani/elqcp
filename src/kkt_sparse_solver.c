#include "elqcp.h"
#include "linalg.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <umfpack.h>

/*
 * Sparse KKT / Optimality-System Solver  (UMFPACK sparse LU)
 *
 * Solves the symmetric indefinite saddle-point system
 *
 *   [ G   -C ] [ y ]   [ -g ]
 *   [-C'   0 ] [ π ] = [ -h ]
 *
 * by assembling the KKT matrix in COO (triplet) format, converting to CSC,
 * and calling UMFPACK's unsymmetric multifrontal LU with partial pivoting.
 *
 * ── Why UMFPACK rather than CHOLMOD LDL^T ─────────────────────────────────
 * The KKT matrix is symmetric indefinite: the primal block G ≻ 0 gives
 * positive eigenvalues, while the Schur complement −C^T G^{-1} C ≺ 0 gives
 * negative eigenvalues.  A correct symmetric indefinite factorization
 * requires 2×2 Bunch-Kaufman pivoting (MA57, PARDISO).  CHOLMOD's simplicial
 * LDL^T terminates at any non-positive pivot d ≤ 0 — it is a positive-
 * (semi)definite solver only.  Within SuiteSparse, UMFPACK (general LU with
 * partial pivoting) is therefore the correct solver for this system.
 *
 * ── Sparsity ───────────────────────────────────────────────────────────────
 * The KKT matrix has O(N·(nx+nu)²) nonzeros — linear in N for fixed
 * dimensions.  UMFPACK exploits this sparsity, giving O(N·(nx+nu)³)
 * complexity vs. O((N·(nx+nu))³) for the dense solver.
 *
 * Variable ordering (same as kkt_solver.c):
 *   y = [u_0, x_1, u_1, ..., u_{N-1}, x_N]    M  = N*(nu+nx)
 *   π = [π_1, ..., π_N]                        nc = N*nx
 *
 * Reference: Jørgensen et al. (2012), IFAC NMPC, eq. (43)/(46).
 */
int elqcp_solve_kkt_sparse(const ELQCP *prob, const double *x0, ELQCPSol *sol)
{
    const int N  = prob->N;
    const int nx = prob->nx;
    const int nu = prob->nu;

    const int M   = N * (nu + nx);
    const int nc  = N * nx;
    const int dim = M + nc;

#define OFFSET_U(k)  ((k)*(nu+nx))
#define OFFSET_X(k)  ((k)*(nu+nx) + nu)

    /* ------------------------------------------------------------------ */
    /* Count nonzeros in the full symmetric KKT matrix                     */
    /*                                                                     */
    /* G block (block-diagonal Hessian, full):                             */
    /*   k=0      : R_0                  nu²                               */
    /*   k=1..N-1 : Q_k + M_k + M_k' + R_k    (nx+nu)²                   */
    /*   terminal : P_N                  nx²                               */
    /*                                                                     */
    /* ±C blocks (both upper-right and lower-left):                        */
    /*   B_k' entries  : N·nu·nx  (each stored twice)                     */
    /*   ±I  entries   : N·nx     (each stored twice)                     */
    /*   A_k' entries  : (N-1)·nx²  (each stored twice, k=1..N-1)        */
    /* ------------------------------------------------------------------ */
    int nnz_G = nu*nu + (N - 1)*(nx + nu)*(nx + nu) + nx*nx;
    int nnz_C = N*nu*nx + N*nx + (N > 1 ? (N - 1)*nx*nx : 0);
    int nnz   = nnz_G + 2*nnz_C;   /* factor 2: upper-right + lower-left */

    int    *Ti   = (int    *)malloc((size_t)nnz  * sizeof(int));
    int    *Tj   = (int    *)malloc((size_t)nnz  * sizeof(int));
    double *Tx   = (double *)malloc((size_t)nnz  * sizeof(double));
    double *gvec = (double *)calloc((size_t)M,    sizeof(double));
    double *hvec = (double *)calloc((size_t)nc,   sizeof(double));
    double *rhs  = (double *)calloc((size_t)dim,  sizeof(double));

    if (!Ti || !Tj || !Tx || !gvec || !hvec || !rhs) {
        fprintf(stderr, "KKT sparse solver: allocation failed (N=%d, dim=%d)\n", N, dim);
        free(Ti); free(Tj); free(Tx); free(gvec); free(hvec); free(rhs);
        return -1;
    }

    int nz = 0;

    /* ------------------------------------------------------------------ */
    /* G block — block-diagonal Hessian (full, both triangles)             */
    /* ------------------------------------------------------------------ */

    /* k=0: R_0 at (OFFSET_U(0), OFFSET_U(0)); fold x_0 into g           */
    {
        const int ou = OFFSET_U(0);
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nu; j++) {
                Ti[nz] = ou + i;  Tj[nz] = ou + j;
                Tx[nz] = prob->R[0][i*nu + j];  nz++;
            }
        for (int i = 0; i < nu; i++) {
            gvec[ou + i] = prob->r[0][i];
            for (int jj = 0; jj < nx; jj++)
                gvec[ou + i] += prob->M_mat[0][jj*nu + i] * x0[jj];
        }
    }

    /* k=1..N-1: Q_k, M_k, M_k' (transpose), R_k */
    for (int k = 1; k < N; k++) {
        const int ox = OFFSET_X(k - 1);
        const int ou = OFFSET_U(k);

        for (int i = 0; i < nx; i++)          /* Q_k */
            for (int j = 0; j < nx; j++) {
                Ti[nz] = ox + i;  Tj[nz] = ox + j;
                Tx[nz] = prob->Q[k][i*nx + j];  nz++;
            }
        for (int i = 0; i < nx; i++)          /* M_k */
            for (int j = 0; j < nu; j++) {
                Ti[nz] = ox + i;  Tj[nz] = ou + j;
                Tx[nz] = prob->M_mat[k][i*nu + j];  nz++;
            }
        for (int j = 0; j < nu; j++)          /* M_k' */
            for (int i = 0; i < nx; i++) {
                Ti[nz] = ou + j;  Tj[nz] = ox + i;
                Tx[nz] = prob->M_mat[k][i*nu + j];  nz++;
            }
        for (int i = 0; i < nu; i++)          /* R_k */
            for (int j = 0; j < nu; j++) {
                Ti[nz] = ou + i;  Tj[nz] = ou + j;
                Tx[nz] = prob->R[k][i*nu + j];  nz++;
            }

        for (int i = 0; i < nx; i++) gvec[ox + i] += prob->q[k][i];
        for (int i = 0; i < nu; i++) gvec[ou + i] += prob->r[k][i];
    }

    /* Terminal: P_N */
    {
        const int ox = OFFSET_X(N - 1);
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < nx; j++) {
                Ti[nz] = ox + i;  Tj[nz] = ox + j;
                Tx[nz] = prob->P_N[i*nx + j];  nz++;
            }
        for (int i = 0; i < nx; i++)
            gvec[ox + i] += prob->p_N[i];
    }

    /* ------------------------------------------------------------------ */
    /* ±C blocks — both upper-right (−C) and lower-left (−C') at once     */
    /* ------------------------------------------------------------------ */
    for (int k = 0; k < N; k++) {
        const double *Ak      = prob->A[k];
        const double *Bk      = prob->B[k];
        const double *bk      = prob->b[k];
        const int     cc      = k * nx;
        const int     ou      = OFFSET_U(k);
        const int     ox_next = OFFSET_X(k);

        /* −B_k' upper-right and lower-left (same value, symmetric) */
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nx; j++) {
                double v = -Bk[j*nu + i];
                Ti[nz] = ou + i;      Tj[nz] = M + cc + j;  Tx[nz] = v;  nz++;
                Ti[nz] = M + cc + j;  Tj[nz] = ou + i;      Tx[nz] = v;  nz++;
            }

        /* +I  (from −(−I) in C) */
        for (int j = 0; j < nx; j++) {
            Ti[nz] = ox_next + j;  Tj[nz] = M + cc + j;  Tx[nz] = 1.0;  nz++;
            Ti[nz] = M + cc + j;  Tj[nz] = ox_next + j;  Tx[nz] = 1.0;  nz++;
        }

        /* −A_k' (k > 0) */
        if (k > 0) {
            const int ox_k = OFFSET_X(k - 1);
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < nx; j++) {
                    double v = -Ak[j*nx + i];
                    Ti[nz] = ox_k + i;    Tj[nz] = M + cc + j;  Tx[nz] = v;  nz++;
                    Ti[nz] = M + cc + j;  Tj[nz] = ox_k + i;    Tx[nz] = v;  nz++;
                }
        }

        /* h vector */
        if (k == 0) {
            double *Ax0 = (double *)calloc((size_t)nx, sizeof(double));
            mat_vec(Ax0, Ak, x0, nx, nx);
            for (int j = 0; j < nx; j++)
                hvec[cc + j] = -(Ax0[j] + bk[j]);
            free(Ax0);
        } else {
            for (int j = 0; j < nx; j++)
                hvec[cc + j] = -bk[j];
        }
    }

    for (int i = 0; i < M;  i++) rhs[i]     = -gvec[i];
    for (int j = 0; j < nc; j++) rhs[M + j] = -hvec[j];
    free(gvec); free(hvec);

    /* ------------------------------------------------------------------ */
    /* Convert COO → CSC                                                   */
    /* ------------------------------------------------------------------ */
    int    *Ap = (int    *)malloc((size_t)(dim + 1) * sizeof(int));
    int    *Ai = (int    *)malloc((size_t)nnz       * sizeof(int));
    double *Ax = (double *)malloc((size_t)nnz       * sizeof(double));
    if (!Ap || !Ai || !Ax) {
        fprintf(stderr, "KKT sparse solver: CSC allocation failed\n");
        free(Ti); free(Tj); free(Tx); free(rhs);
        free(Ap); free(Ai); free(Ax);
        return -1;
    }

    if (umfpack_di_triplet_to_col(dim, dim, nz, Ti, Tj, Tx,
                                   Ap, Ai, Ax, NULL) != UMFPACK_OK) {
        fprintf(stderr, "KKT sparse solver: triplet_to_col failed\n");
        free(Ti); free(Tj); free(Tx); free(rhs);
        free(Ap); free(Ai); free(Ax);
        return -1;
    }
    free(Ti); free(Tj); free(Tx);

    /* ------------------------------------------------------------------ */
    /* Sparse LU (UMFPACK) + solve                                         */
    /* ------------------------------------------------------------------ */
    void *Symbolic = NULL, *Numeric = NULL;
    int   status;

    status = umfpack_di_symbolic(dim, dim, Ap, Ai, Ax, &Symbolic, NULL, NULL);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "KKT sparse solver: symbolic factorization failed (%d)\n", status);
        free(Ap); free(Ai); free(Ax); free(rhs);
        return -1;
    }

    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, NULL, NULL);
    umfpack_di_free_symbolic(&Symbolic);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "KKT sparse solver: numeric factorization failed (%d)\n", status);
        free(Ap); free(Ai); free(Ax); free(rhs);
        return -1;
    }

    double *sol_vec = (double *)malloc((size_t)dim * sizeof(double));
    status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, sol_vec, rhs,
                               Numeric, NULL, NULL);
    umfpack_di_free_numeric(&Numeric);
    free(Ap); free(Ai); free(Ax); free(rhs);

    if (status != UMFPACK_OK) {
        fprintf(stderr, "KKT sparse solver: solve failed (%d)\n", status);
        free(sol_vec);
        return -1;
    }

    /* ------------------------------------------------------------------ */
    /* Extract primal solution                                             */
    /* ------------------------------------------------------------------ */
    memcpy(sol->x[0], x0, (size_t)nx * sizeof(double));
    for (int k = 0; k < N; k++) {
        memcpy(sol->u[k],   sol_vec + OFFSET_U(k), (size_t)nu * sizeof(double));
        memcpy(sol->x[k+1], sol_vec + OFFSET_X(k), (size_t)nx * sizeof(double));
    }

    {
        double phi = prob->gamma_N;
        double *Px = (double *)malloc((size_t)nx * sizeof(double));
        mat_vec(Px, prob->P_N, sol->x[N], nx, nx);
        phi += 0.5*dot(sol->x[N], Px, nx) + dot(prob->p_N, sol->x[N], nx);
        free(Px);
        for (int k = 0; k < N; k++) {
            double *Qx = (double *)malloc((size_t)nx * sizeof(double));
            double *Mu = (double *)malloc((size_t)nx * sizeof(double));
            double *Ru = (double *)malloc((size_t)nu * sizeof(double));
            mat_vec(Qx, prob->Q[k],     sol->x[k], nx, nx);
            mat_vec(Mu, prob->M_mat[k], sol->u[k], nx, nu);
            mat_vec(Ru, prob->R[k],     sol->u[k], nu, nu);
            phi += 0.5*dot(sol->x[k], Qx, nx)
                 + dot(sol->x[k], Mu, nx)
                 + 0.5*dot(sol->u[k], Ru, nu)
                 + dot(prob->q[k], sol->x[k], nx)
                 + dot(prob->r[k], sol->u[k], nu)
                 + prob->f[k];
            free(Qx); free(Mu); free(Ru);
        }
        sol->phi = phi;
    }

    free(sol_vec);

#undef OFFSET_U
#undef OFFSET_X

    return 0;
}
