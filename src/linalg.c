#include "linalg.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* C = A * B,  A: n×m, B: m×p */
void mat_mul(double *C, const double *A, const double *B, int n, int m, int p)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double s = 0.0;
            for (int k = 0; k < m; k++)
                s += A[i*m + k] * B[k*p + j];
            C[i*p + j] = s;
        }
    }
}

/* C = A^T * B,  A stored as m×n (so A^T is n×m), B: m×p, C: n×p */
void mat_mul_tn(double *C, const double *A, const double *B, int n, int m, int p)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double s = 0.0;
            for (int k = 0; k < m; k++)
                s += A[k*n + i] * B[k*p + j];
            C[i*p + j] = s;
        }
    }
}

/* C = A * B^T,  A: n×m, B stored as p×m, C: n×p */
void mat_mul_nt(double *C, const double *A, const double *B, int n, int m, int p)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double s = 0.0;
            for (int k = 0; k < m; k++)
                s += A[i*m + k] * B[j*m + k];
            C[i*p + j] = s;
        }
    }
}

/* y = A * x,  A: n×m */
void mat_vec(double *y, const double *A, const double *x, int n, int m)
{
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < m; j++)
            s += A[i*m + j] * x[j];
        y[i] = s;
    }
}

/* y = A^T * x,  A stored as m×n, result y: n */
void mat_vec_t(double *y, const double *A, const double *x, int n, int m)
{
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < m; j++)
            s += A[j*n + i] * x[j];
        y[i] = s;
    }
}

double dot(const double *a, const double *b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

/* Cholesky: A = L L^T, L lower triangular */
int cholesky(double *L, const double *A, int n)
{
    memset(L, 0, (size_t)(n * n) * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[i*n + j];
            for (int k = 0; k < j; k++)
                s -= L[i*n + k] * L[j*n + k];
            if (i == j) {
                if (s <= 0.0) return -1;
                L[i*n + j] = sqrt(s);
            } else {
                L[i*n + j] = s / L[j*n + j];
            }
        }
    }
    return 0;
}

/* Solve L * x = b (forward substitution), L lower triangular n×n */
void fwd_sub(double *x, const double *L, const double *b, int n)
{
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++)
            s -= L[i*n + j] * x[j];
        x[i] = s / L[i*n + i];
    }
}

/* Solve L^T * x = b (backward substitution using lower triangular L) */
void bwd_sub(double *x, const double *L, const double *b, int n)
{
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int j = i + 1; j < n; j++)
            s -= L[j*n + i] * x[j];  /* L^T[i,j] = L[j,i] */
        x[i] = s / L[i*n + i];
    }
}

/* Solve L * X = B column-by-column, B: n×m, X: n×m */
void fwd_sub_mat(double *X, const double *L, const double *B, int n, int m)
{
    double *col = (double *)malloc((size_t)n * sizeof(double));
    double *xcol = (double *)malloc((size_t)n * sizeof(double));
    for (int j = 0; j < m; j++) {
        /* extract column j of B */
        for (int i = 0; i < n; i++)
            col[i] = B[i*m + j];
        fwd_sub(xcol, L, col, n);
        /* write column j of X */
        for (int i = 0; i < n; i++)
            X[i*m + j] = xcol[i];
    }
    free(col);
    free(xcol);
}

/* LU factorization with partial pivoting (in-place) */
int lu_factor(double *A, int *piv, int n)
{
    for (int k = 0; k < n; k++) {
        /* find pivot */
        int p = k;
        double maxval = fabs(A[k*n + k]);
        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i*n + k]) > maxval) {
                maxval = fabs(A[i*n + k]);
                p = i;
            }
        }
        piv[k] = p;
        if (maxval == 0.0) return -1;

        /* swap rows k and p */
        if (p != k) {
            for (int j = 0; j < n; j++) {
                double tmp = A[k*n + j];
                A[k*n + j] = A[p*n + j];
                A[p*n + j] = tmp;
            }
        }

        /* eliminate below */
        double akk = A[k*n + k];
        for (int i = k + 1; i < n; i++) {
            A[i*n + k] /= akk;
            for (int j = k + 1; j < n; j++)
                A[i*n + j] -= A[i*n + k] * A[k*n + j];
        }
    }
    return 0;
}

/* Solve A*x = b using LU factorization from lu_factor() */
void lu_solve(double *x, const double *LU, const int *piv, const double *b, int n)
{
    /* copy b and apply row permutations */
    double *y = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++)
        y[i] = b[i];
    for (int k = 0; k < n; k++) {
        int p = piv[k];
        double tmp = y[k];
        y[k] = y[p];
        y[p] = tmp;
    }

    /* forward substitution with unit lower triangular L */
    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++)
            y[i] -= LU[i*n + j] * y[j];

    /* backward substitution with upper triangular U */
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++)
            y[i] -= LU[i*n + j] * y[j];
        y[i] /= LU[i*n + i];
    }

    for (int i = 0; i < n; i++)
        x[i] = y[i];
    free(y);
}
