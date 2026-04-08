#ifndef LINALG_H
#define LINALG_H

/*
 * Basic dense linear algebra routines.
 * All matrices are stored row-major: A[i*cols + j] = A(i,j).
 */

/* C = A * B,  A: n×m, B: m×p, C: n×p */
void mat_mul(double *C, const double *A, const double *B, int n, int m, int p);

/* C = A^T * B,  A: m×n stored, B: m×p, C: n×p */
void mat_mul_tn(double *C, const double *A, const double *B, int n, int m, int p);

/* C = A * B^T,  A: n×m, B: p×m stored, C: n×p */
void mat_mul_nt(double *C, const double *A, const double *B, int n, int m, int p);

/* y = A * x,  A: n×m */
void mat_vec(double *y, const double *A, const double *x, int n, int m);

/* y = A^T * x,  A: m×n stored, result y: n */
void mat_vec_t(double *y, const double *A, const double *x, int n, int m);

/* dot product of two length-n vectors */
double dot(const double *a, const double *b, int n);

/*
 * Cholesky factorization: A = L * L^T (A symmetric positive definite)
 * Stores lower triangular factor in L (same size n×n, upper part zeroed).
 * Returns 0 on success, -1 if not positive definite.
 */
int cholesky(double *L, const double *A, int n);

/*
 * Forward substitution: solve L * x = b, L lower triangular, n×n.
 * Result written to x.
 */
void fwd_sub(double *x, const double *L, const double *b, int n);

/*
 * Backward substitution: solve L^T * x = b, L lower triangular, n×n.
 * (Equivalently solves upper triangular system U*x=b with U=L^T.)
 * Result written to x.
 */
void bwd_sub(double *x, const double *L, const double *b, int n);

/*
 * Solve L * X = B column-by-column (forward substitution for matrix RHS).
 * L: n×n lower triangular, B: n×m, X: n×m (output).
 */
void fwd_sub_mat(double *X, const double *L, const double *B, int n, int m);

/*
 * LU factorization with partial pivoting (in-place).
 * A: n×n, overwritten with LU factors (L lower unit triangular, U upper).
 * piv: int[n] pivot indices.
 * Returns 0 on success, -1 if singular.
 */
int lu_factor(double *A, int *piv, int n);

/*
 * Solve A*x = b using a pre-computed LU factorization from lu_factor().
 * LU: n×n factored matrix, piv: int[n] pivots, b: n rhs, x: n solution.
 */
void lu_solve(double *x, const double *LU, const int *piv, const double *b, int n);

#endif /* LINALG_H */
