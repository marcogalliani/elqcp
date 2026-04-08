#include "elqcp.h"
#include "linalg.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------------------------------------------------------------------------
 * Memory helpers
 * --------------------------------------------------------------------------- */

ELQCP *elqcp_alloc(int N, int nx, int nu)
{
    ELQCP *prob = (ELQCP *)calloc(1, sizeof(ELQCP));
    prob->N  = N;
    prob->nx = nx;
    prob->nu = nu;

    prob->Q     = (double **)calloc((size_t)N, sizeof(double *));
    prob->M_mat = (double **)calloc((size_t)N, sizeof(double *));
    prob->R     = (double **)calloc((size_t)N, sizeof(double *));
    prob->q     = (double **)calloc((size_t)N, sizeof(double *));
    prob->r     = (double **)calloc((size_t)N, sizeof(double *));
    prob->f     = (double  *)calloc((size_t)N, sizeof(double));
    prob->A     = (double **)calloc((size_t)N, sizeof(double *));
    prob->B     = (double **)calloc((size_t)N, sizeof(double *));
    prob->b     = (double **)calloc((size_t)N, sizeof(double *));

    for (int k = 0; k < N; k++) {
        prob->Q[k]     = (double *)calloc((size_t)(nx * nx), sizeof(double));
        prob->M_mat[k] = (double *)calloc((size_t)(nx * nu), sizeof(double));
        prob->R[k]     = (double *)calloc((size_t)(nu * nu), sizeof(double));
        prob->q[k]     = (double *)calloc((size_t)nx, sizeof(double));
        prob->r[k]     = (double *)calloc((size_t)nu, sizeof(double));
        prob->A[k]     = (double *)calloc((size_t)(nx * nx), sizeof(double));
        prob->B[k]     = (double *)calloc((size_t)(nx * nu), sizeof(double));
        prob->b[k]     = (double *)calloc((size_t)nx, sizeof(double));
    }

    prob->P_N = (double *)calloc((size_t)(nx * nx), sizeof(double));
    prob->p_N = (double *)calloc((size_t)nx, sizeof(double));
    prob->gamma_N = 0.0;

    return prob;
}

void elqcp_free(ELQCP *prob)
{
    if (!prob) return;
    for (int k = 0; k < prob->N; k++) {
        free(prob->Q[k]);
        free(prob->M_mat[k]);
        free(prob->R[k]);
        free(prob->q[k]);
        free(prob->r[k]);
        free(prob->A[k]);
        free(prob->B[k]);
        free(prob->b[k]);
    }
    free(prob->Q);
    free(prob->M_mat);
    free(prob->R);
    free(prob->q);
    free(prob->r);
    free(prob->f);
    free(prob->A);
    free(prob->B);
    free(prob->b);
    free(prob->P_N);
    free(prob->p_N);
    free(prob);
}

ELQCPSol *elqcp_sol_alloc(int N, int nx, int nu)
{
    ELQCPSol *sol = (ELQCPSol *)calloc(1, sizeof(ELQCPSol));
    sol->x = (double **)calloc((size_t)(N + 1), sizeof(double *));
    sol->u = (double **)calloc((size_t)N, sizeof(double *));
    for (int k = 0; k <= N; k++)
        sol->x[k] = (double *)calloc((size_t)nx, sizeof(double));
    for (int k = 0; k < N; k++)
        sol->u[k] = (double *)calloc((size_t)nu, sizeof(double));
    sol->phi = 0.0;
    return sol;
}

void elqcp_sol_free(ELQCPSol *sol, int N)
{
    if (!sol) return;
    for (int k = 0; k <= N; k++) free(sol->x[k]);
    for (int k = 0; k < N;  k++) free(sol->u[k]);
    free(sol->x);
    free(sol->u);
    free(sol);
}

/* ---------------------------------------------------------------------------
 * Riccati recursion solver  (Algorithm 1, §4.3.1, Jørgensen 2004)
 *
 * Notation translation (book → code):
 *   Book uses x_{k+1} = A_k' x_k + B_k' u_k + b_k  (primed convention).
 *   Code uses x_{k+1} = A   x_k + B   u_k + b       (standard convention).
 *   So book's A_k = code's A^T, and book's B_k = code's B^T.
 *
 * Backward sweep recurrences (from eq. 4.100–4.103, translated):
 *   S   = A^T P                        [nx×nx]
 *   Re  = R + (B^T P) B = R + S^T B   ... but simpler: Re = R + B^T*(P*B)
 *         Actually: Re = R_k + B_k P B_k' = R + B^T P B  (B^T is nu×nx)
 *   Y   = (M + S B)^T                  [nu×nx]  (book: Y = (M_k + S B_k')')
 *   s   = P b                          [nx]
 *   c   = s + p                        [nx]
 *   d   = r + B^T c                    [nu]
 *   Cholesky: Re = L L^T
 *   Solve: L Z = Y  →  Z [nu×nx]
 *   Solve: L z = d  →  z [nu]
 *   P  ← Q + S A - Z^T Z              (symmetrized)
 *   γ  ← γ + f + p'b + ½ s'b - ½ z'z
 *   p  ← q + A^T c - Z^T z
 *
 * Forward sweep (eq. 4.105–4.107):
 *   y = Z_k x_k + z_k
 *   Solve: L_k^T u_k = -y
 *   x_{k+1} = A x_k + B u_k + b
 * --------------------------------------------------------------------------- */
int elqcp_solve_riccati(const ELQCP *prob, const double *x0, ELQCPSol *sol)
{
    const int N  = prob->N;
    const int nx = prob->nx;
    const int nu = prob->nu;

    /* Allocate storage for the factorization sequence {L_k, Z_k, z_k}_{k=0}^{N-1} */
    double **Lk = (double **)malloc((size_t)N * sizeof(double *));
    double **Zk = (double **)malloc((size_t)N * sizeof(double *));
    double **zk = (double **)malloc((size_t)N * sizeof(double *));
    for (int k = 0; k < N; k++) {
        Lk[k] = (double *)malloc((size_t)(nu * nu) * sizeof(double));
        Zk[k] = (double *)malloc((size_t)(nu * nx) * sizeof(double));
        zk[k] = (double *)malloc((size_t)nu * sizeof(double));
    }

    /* Temporaries */
    double *P   = (double *)malloc((size_t)(nx * nx) * sizeof(double));
    double *p   = (double *)malloc((size_t)nx * sizeof(double));
    double *S   = (double *)malloc((size_t)(nx * nx) * sizeof(double));
    double *PB  = (double *)malloc((size_t)(nx * nu) * sizeof(double)); /* P*B */
    double *Re  = (double *)malloc((size_t)(nu * nu) * sizeof(double));
    double *MB  = (double *)malloc((size_t)(nx * nu) * sizeof(double)); /* M + S*B */
    double *Y   = (double *)malloc((size_t)(nu * nx) * sizeof(double)); /* (M+S*B)^T */
    double *s   = (double *)malloc((size_t)nx * sizeof(double));
    double *c   = (double *)malloc((size_t)nx * sizeof(double));
    double *d   = (double *)malloc((size_t)nu * sizeof(double));
    double *Pnew= (double *)malloc((size_t)(nx * nx) * sizeof(double));
    double *SA  = (double *)malloc((size_t)(nx * nx) * sizeof(double)); /* S*A */
    double *ZtZ = (double *)malloc((size_t)(nx * nx) * sizeof(double)); /* Z^T Z */
    double *ZtZ_k= (double*)malloc((size_t)(nx * nx) * sizeof(double));
    double *Atc = (double *)malloc((size_t)nx * sizeof(double));
    double *Ztz = (double *)malloc((size_t)nx * sizeof(double));

    /* Initialize: P ← P_N, p ← p_N, γ ← γ_N */
    memcpy(P, prob->P_N, (size_t)(nx * nx) * sizeof(double));
    memcpy(p, prob->p_N, (size_t)nx * sizeof(double));
    double gamma = prob->gamma_N;

    /* ---- Backward sweep ---- */
    for (int k = N - 1; k >= 0; k--) {
        const double *Qk = prob->Q[k];
        const double *Mk = prob->M_mat[k];
        const double *Rk = prob->R[k];
        const double *qk = prob->q[k];
        const double *rk = prob->r[k];
        const double  fk = prob->f[k];
        const double *Ak = prob->A[k];
        const double *Bk = prob->B[k];
        const double *bk = prob->b[k];

        /* S = A^T P  (A: nx×nx, P: nx×nx → S: nx×nx) */
        mat_mul_tn(S, Ak, P, nx, nx, nx);

        /* PB = P * B  (P: nx×nx, B: nx×nu → PB: nx×nu) */
        mat_mul(PB, P, Bk, nx, nx, nu);

        /* Re = R + B^T * PB  (B^T: nu×nx, PB: nx×nu → B^T PB: nu×nu) */
        memcpy(Re, Rk, (size_t)(nu * nu) * sizeof(double));
        mat_mul_tn(ZtZ_k, Bk, PB, nu, nx, nu); /* reuse ZtZ_k as temp */
        for (int i = 0; i < nu * nu; i++) Re[i] += ZtZ_k[i];

        /* MB = M + S*B  (M: nx×nu, S: nx×nx, B: nx×nu → S*B: nx×nu) */
        mat_mul(MB, S, Bk, nx, nx, nu);
        for (int i = 0; i < nx * nu; i++) MB[i] += Mk[i];

        /* Y = MB^T  (MB: nx×nu → Y: nu×nx) */
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < nu; j++)
                Y[j*nx + i] = MB[i*nu + j];

        /* s = P * b */
        mat_vec(s, P, bk, nx, nx);

        /* c = s + p */
        for (int i = 0; i < nx; i++) c[i] = s[i] + p[i];

        /* d = r + B^T * c  (B^T: nu×nx, c: nx → B^T c: nu) */
        memcpy(d, rk, (size_t)nu * sizeof(double));
        mat_vec_t(Ztz, Bk, c, nu, nx); /* reuse Ztz as temp: B^T c */
        for (int i = 0; i < nu; i++) d[i] += Ztz[i];

        /* Cholesky: Re = L L^T, store in Lk[k] */
        if (cholesky(Lk[k], Re, nu) != 0) {
            fprintf(stderr, "Riccati: Cholesky failed at k=%d\n", k);
            /* cleanup and return error */
            for (int i = 0; i < N; i++) { free(Lk[i]); free(Zk[i]); free(zk[i]); }
            free(Lk); free(Zk); free(zk);
            free(P); free(p); free(S); free(PB); free(Re); free(MB); free(Y);
            free(s); free(c); free(d); free(Pnew); free(SA); free(ZtZ); free(ZtZ_k);
            free(Atc); free(Ztz);
            return -1;
        }

        /* Solve L Z = Y  (L: nu×nu lower triangular, Y: nu×nx → Z: nu×nx) */
        fwd_sub_mat(Zk[k], Lk[k], Y, nu, nx);

        /* Solve L z = d  (z: nu) */
        fwd_sub(zk[k], Lk[k], d, nu);

        /* Z^T Z  (Z: nu×nx → Z^T Z: nx×nx) */
        mat_mul_tn(ZtZ, Zk[k], Zk[k], nx, nu, nx);

        /* SA = S * A  (S: nx×nx, A: nx×nx) */
        mat_mul(SA, S, Ak, nx, nx, nx);

        /* P ← Q + SA - Z^T Z, then symmetrize */
        for (int i = 0; i < nx * nx; i++)
            Pnew[i] = Qk[i] + SA[i] - ZtZ[i];
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < nx; j++)
                P[i*nx + j] = 0.5 * (Pnew[i*nx + j] + Pnew[j*nx + i]);

        /* γ ← γ + f + p'b + ½ s'b - ½ z'z */
        gamma += fk + dot(p, bk, nx) + 0.5*dot(s, bk, nx) - 0.5*dot(zk[k], zk[k], nu);

        /* p ← q + A^T c - Z^T z  (A^T c: nx, Z^T z: nx) */
        mat_vec_t(Atc, Ak, c, nx, nx);
        mat_vec_t(Ztz, Zk[k], zk[k], nx, nu);
        for (int i = 0; i < nx; i++)
            p[i] = qk[i] + Atc[i] - Ztz[i];
    }

    /* Optimal value: φ* = ½ x0' P x0 + p' x0 + γ */
    double *Px0 = (double *)malloc((size_t)nx * sizeof(double));
    mat_vec(Px0, P, x0, nx, nx);
    sol->phi = 0.5 * dot(x0, Px0, nx) + dot(p, x0, nx) + gamma;
    free(Px0);

    /* ---- Forward sweep ---- */
    memcpy(sol->x[0], x0, (size_t)nx * sizeof(double));

    double *y = (double *)malloc((size_t)nu * sizeof(double));
    for (int k = 0; k < N; k++) {
        const double *Ak = prob->A[k];
        const double *Bk = prob->B[k];
        const double *bk = prob->b[k];

        /* y = Z_k x_k + z_k */
        mat_vec(y, Zk[k], sol->x[k], nu, nx);
        for (int i = 0; i < nu; i++) y[i] += zk[k][i];

        /* Solve L_k^T u_k = -y  (backward sub with lower triangular L) */
        double *neg_y = (double *)malloc((size_t)nu * sizeof(double));
        for (int i = 0; i < nu; i++) neg_y[i] = -y[i];
        bwd_sub(sol->u[k], Lk[k], neg_y, nu);
        free(neg_y);

        /* x_{k+1} = A x_k + B u_k + b */
        mat_vec(sol->x[k+1], Ak, sol->x[k], nx, nx);
        double *Bu = (double *)malloc((size_t)nx * sizeof(double));
        mat_vec(Bu, Bk, sol->u[k], nx, nu);
        for (int i = 0; i < nx; i++)
            sol->x[k+1][i] += Bu[i] + bk[i];
        free(Bu);
    }
    free(y);

    /* Free temporaries */
    for (int k = 0; k < N; k++) { free(Lk[k]); free(Zk[k]); free(zk[k]); }
    free(Lk); free(Zk); free(zk);
    free(P); free(p); free(S); free(PB); free(Re); free(MB); free(Y);
    free(s); free(c); free(d); free(Pnew); free(SA); free(ZtZ); free(ZtZ_k);
    free(Atc); free(Ztz);

    return 0;
}
