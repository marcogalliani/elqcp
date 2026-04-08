#include "elqcp.h"
#include "linalg.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * KKT / Optimality-System Solver
 *
 * Treats the ELQCP (eq. 4.46) as a general equality-constrained QP (eq. 4.128/4.132):
 *
 *   min_{y}  phi = 1/2 y' G y + g' y + rho
 *   s.t.     C' y = h
 *
 * Decision variables (x_0 is a fixed parameter):
 *   y = [u_0, x_1, u_1, x_2, ..., u_{N-1}, x_N]     length M = N*(nu+nx)
 *
 * Equality constraints (dynamics, N*nx equations):
 *   B_k u_k - x_{k+1} = -(A_k x_k + b_k)   k = 0,...,N-1
 *   (with x_0 = given, folded into RHS for k=0)
 *
 * Block layout of y for stage k:
 *   offset_u[k] = k*(nu+nx)           (u_k block, length nu)
 *   offset_x[k] = k*(nu+nx) + nu      (x_{k+1} block, length nx)
 *
 * KKT system (Corollary 4.4.4 / eq. 4.123):
 *   [ G   -C ] [ y ]   [  g ]
 *   [-C'   0 ] [ π ] = [ -h ]   (i.e. solve for (y, π))
 *
 * We assemble the full dense KKT matrix of size (M + N*nx) × (M + N*nx)
 * and solve with LU factorization.  This is O((N*(nx+nu))^3) — intentionally
 * not exploiting banded structure — to show the advantage of Riccati recursion.
 */
int elqcp_solve_kkt(const ELQCP *prob, const double *x0, ELQCPSol *sol)
{
    const int N  = prob->N;
    const int nx = prob->nx;
    const int nu = prob->nu;

    /* Total primal variables: y = [u_0,x_1,u_1,x_2,...,u_{N-1},x_N] */
    const int M  = N * (nu + nx);      /* primal vars */
    const int nc = N * nx;             /* constraints */
    const int dim = M + nc;            /* KKT system size */

    /* Helper: index of u_k in y  */
#define OFFSET_U(k)  ((k)*(nu+nx))
    /* Helper: index of x_{k+1} in y  (x_1 at k=0, ..., x_N at k=N-1) */
#define OFFSET_X(k)  ((k)*(nu+nx) + nu)
    /* Helper: index of constraint for stage k in π */
#define OFFSET_C(k)  (M + (k)*nx)

    /* Allocate G (M×M), C (M×nc), g (M), h (nc), KKT (dim×dim), rhs (dim) */
    double *G   = (double *)calloc((size_t)(M  * M),   sizeof(double));
    double *C   = (double *)calloc((size_t)(M  * nc),  sizeof(double));
    double *gvec= (double *)calloc((size_t)M,           sizeof(double));
    double *hvec= (double *)calloc((size_t)nc,          sizeof(double));
    double *KKT = (double *)calloc((size_t)(dim * dim), sizeof(double));
    double *rhs = (double *)calloc((size_t)dim,         sizeof(double));
    int    *piv = (int    *)malloc((size_t)dim * sizeof(int));

    if (!G || !C || !gvec || !hvec || !KKT || !rhs || !piv) {
        fprintf(stderr, "KKT solver: allocation failed (N=%d, dim=%d)\n", N, dim);
        free(G); free(C); free(gvec); free(hvec); free(KKT); free(rhs); free(piv);
        return -1;
    }

    /* ------------------------------------------------------------------ */
    /* Assemble G (block-diagonal Hessian) and g (linear cost gradient)   */
    /* ------------------------------------------------------------------ */
    /*
     * Block structure of G (ordering: u_0, x_1, u_1, x_2, ..., u_{N-1}, x_N):
     *
     * Stage k=0: cost 1/2 u_0' R_0 u_0 + x_0'M_0 u_0 + ...
     *   G[u_0, u_0] += R_0
     *   g[u_0]      += (M_0' x_0 + r_0)    (x_0 is fixed → enters g)
     *
     * Stage k=1,...,N-1:
     *   G[x_k, x_k] += Q_k
     *   G[x_k, u_k] += M_k  (and symmetric)
     *   G[u_k, u_k] += R_k
     *   g[x_k]      += q_k
     *   g[u_k]      += r_k
     *
     * Terminal stage:
     *   G[x_N, x_N] += P_N
     *   g[x_N]      += p_N
     */

    /* Stage k=0: only u_0 is a decision variable (x_0 fixed) */
    {
        const int ou = OFFSET_U(0);
        const double *R0 = prob->R[0];
        const double *M0 = prob->M_mat[0];
        const double *r0 = prob->r[0];

        /* G[u_0, u_0] += R_0 */
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nu; j++)
                G[(ou+i)*M + (ou+j)] += R0[i*nu + j];

        /* g[u_0] += M_0' x_0 + r_0  (M_0: nx×nu, so M_0' x_0: nu) */
        for (int i = 0; i < nu; i++) {
            double s = r0[i];
            for (int jj = 0; jj < nx; jj++)
                s += M0[jj*nu + i] * x0[jj];
            gvec[ou + i] += s;
        }
    }

    /* Stages k=1,...,N-1: x_k and u_k are both decision variables */
    for (int k = 1; k < N; k++) {
        const int ox = OFFSET_X(k-1);  /* x_k is at position OFFSET_X(k-1) in y */
        const int ou = OFFSET_U(k);
        const double *Qk = prob->Q[k];
        const double *Mk = prob->M_mat[k];
        const double *Rk = prob->R[k];
        const double *qk = prob->q[k];
        const double *rk = prob->r[k];

        /* G[x_k, x_k] += Q_k */
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < nx; j++)
                G[(ox+i)*M + (ox+j)] += Qk[i*nx + j];

        /* G[x_k, u_k] += M_k  and  G[u_k, x_k] += M_k' */
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < nu; j++) {
                G[(ox+i)*M + (ou+j)] += Mk[i*nu + j];
                G[(ou+j)*M + (ox+i)] += Mk[i*nu + j];
            }

        /* G[u_k, u_k] += R_k */
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nu; j++)
                G[(ou+i)*M + (ou+j)] += Rk[i*nu + j];

        /* g[x_k] += q_k,  g[u_k] += r_k */
        for (int i = 0; i < nx; i++) gvec[ox+i] += qk[i];
        for (int i = 0; i < nu; i++) gvec[ou+i] += rk[i];
    }

    /* Terminal cost on x_N (at OFFSET_X(N-1)) */
    {
        const int ox = OFFSET_X(N-1);
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < nx; j++)
                G[(ox+i)*M + (ox+j)] += prob->P_N[i*nx + j];
        for (int i = 0; i < nx; i++)
            gvec[ox+i] += prob->p_N[i];
    }

    /* ------------------------------------------------------------------ */
    /* Assemble C (constraint Jacobian, M×nc) and h (rhs)                 */
    /* ------------------------------------------------------------------ */
    /*
     * Constraint for stage k:  A_k x_k + B_k u_k - x_{k+1} = -b_k
     *
     * In terms of decision variables y:
     *   k=0: B_0 u_0 - x_1 = -(A_0 x_0 + b_0)  (x_0 fixed → RHS)
     *   k>0: A_k x_k + B_k u_k - x_{k+1} = -b_k
     *
     * C is M×nc.  Constraint k occupies columns [k*nx, (k+1)*nx) of C.
     * The KKT system uses -C on the upper-right block, so C[y_i, c_j] is
     * the coefficient of y_i in constraint c_j.
     *
     * For constraint k (columns c = k*nx .. (k+1)*nx - 1):
     *   C[u_k, c]    += B_k'  (each column c_j of B_k appears in row u_k)
     *   C[x_{k+1},c] -= I     (x_{k+1} coefficient)
     *   C[x_k, c]   += A_k'  (if k>0, x_k is a decision variable)
     */
    for (int k = 0; k < N; k++) {
        const double *Ak = prob->A[k];
        const double *Bk = prob->B[k];
        const double *bk = prob->b[k];
        const int cc = k * nx;    /* column offset in C */
        const int ou = OFFSET_U(k);
        const int ox_next = OFFSET_X(k);   /* x_{k+1} */

        /* C[u_k, cc:cc+nx] += B_k'   (B_k: nx×nu → B_k': nu×nx) */
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nx; j++)
                C[(ou+i)*nc + (cc+j)] += Bk[j*nu + i];

        /* C[x_{k+1}, cc:cc+nx] -= I */
        for (int j = 0; j < nx; j++)
            C[(ox_next+j)*nc + (cc+j)] -= 1.0;

        /* C[x_k, cc:cc+nx] += A_k'  (only if k>0, since x_0 is not a var) */
        if (k > 0) {
            const int ox_k = OFFSET_X(k-1);   /* x_k */
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < nx; j++)
                    C[(ox_k+i)*nc + (cc+j)] += Ak[j*nx + i];
        }

        /* RHS h[cc:cc+nx]:
         *   k=0: h = -(A_0 x_0 + b_0)
         *   k>0: h = -b_k
         */
        if (k == 0) {
            double *Ax0 = (double *)calloc((size_t)nx, sizeof(double));
            mat_vec(Ax0, Ak, x0, nx, nx);
            for (int j = 0; j < nx; j++)
                hvec[cc+j] = -(Ax0[j] + bk[j]);
            free(Ax0);
        } else {
            for (int j = 0; j < nx; j++)
                hvec[cc+j] = -bk[j];
        }
    }

    /* ------------------------------------------------------------------ */
    /* Assemble KKT matrix (dim×dim) and RHS                              */
    /* Layout: [ G   -C ] [ y ]   [  g ]                                  */
    /*         [-C'   0 ] [ π ] = [ -h ]                                  */
    /* ------------------------------------------------------------------ */
    /* Upper-left: G */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++)
            KKT[i*dim + j] = G[i*M + j];

    /* Upper-right: -C  (KKT[i, M+j] = -C[i,j]) */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < nc; j++)
            KKT[i*dim + (M+j)] = -C[i*nc + j];

    /* Lower-left: -C'  (KKT[M+j, i] = -C[i,j]) */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < nc; j++)
            KKT[(M+j)*dim + i] = -C[i*nc + j];

    /* Lower-right: 0 (already zero from calloc) */

    /* RHS: KKT [y;π] = [-g; -h]
     * Upper block: Gy - Cπ = -g  → rhs_upper = -g
     * Lower block: -C'y = -h     → rhs_lower = -h  (constraint is C'y = h)
     */
    for (int i = 0; i < M;  i++) rhs[i]   = -gvec[i];
    for (int j = 0; j < nc; j++) rhs[M+j] = -hvec[j];

    /* ------------------------------------------------------------------ */
    /* Solve KKT system with dense LU                                      */
    /* ------------------------------------------------------------------ */
    if (lu_factor(KKT, piv, dim) != 0) {
        fprintf(stderr, "KKT solver: LU factorization failed (singular)\n");
        free(G); free(C); free(gvec); free(hvec); free(KKT); free(rhs); free(piv);
        return -1;
    }

    double *sol_vec = (double *)malloc((size_t)dim * sizeof(double));
    lu_solve(sol_vec, KKT, piv, rhs, dim);

    /* ------------------------------------------------------------------ */
    /* Extract primal solution                                             */
    /* ------------------------------------------------------------------ */
    memcpy(sol->x[0], x0, (size_t)nx * sizeof(double));

    for (int k = 0; k < N; k++) {
        /* u_k */
        const int ou = OFFSET_U(k);
        memcpy(sol->u[k], sol_vec + ou, (size_t)nu * sizeof(double));

        /* x_{k+1} */
        const int ox = OFFSET_X(k);
        memcpy(sol->x[k+1], sol_vec + ox, (size_t)nx * sizeof(double));
    }

    /* Optimal value: phi = 1/2 y' G_orig y + g' y + rho
     * (We recompute from original G/g since KKT was factored in-place.) */
    /* Use x_{k+1} = A x_k + B u_k + b to just compute phi from definition */
    {
        double phi = prob->gamma_N;
        /* Terminal */
        double *Px = (double *)malloc((size_t)nx * sizeof(double));
        mat_vec(Px, prob->P_N, sol->x[N], nx, nx);
        phi += 0.5*dot(sol->x[N], Px, nx) + dot(prob->p_N, sol->x[N], nx);
        /* Stages */
        for (int k = 0; k < N; k++) {
            double *Qx = (double *)malloc((size_t)nx * sizeof(double));
            double *Mu = (double *)malloc((size_t)nx * sizeof(double));
            double *Ru = (double *)malloc((size_t)nu * sizeof(double));
            mat_vec(Qx, prob->Q[k], sol->x[k], nx, nx);
            mat_vec(Mu, prob->M_mat[k], sol->u[k], nx, nu);
            mat_vec(Ru, prob->R[k], sol->u[k], nu, nu);
            phi += 0.5*dot(sol->x[k], Qx, nx)
                 + dot(sol->x[k], Mu, nx)
                 + 0.5*dot(sol->u[k], Ru, nu)
                 + dot(prob->q[k], sol->x[k], nx)
                 + dot(prob->r[k], sol->u[k], nu)
                 + prob->f[k];
            free(Qx); free(Mu); free(Ru);
        }
        sol->phi = phi;
        free(Px);
    }

    free(G); free(C); free(gvec); free(hvec); free(KKT); free(rhs);
    free(piv); free(sol_vec);

#undef OFFSET_U
#undef OFFSET_X
#undef OFFSET_C

    return 0;
}
