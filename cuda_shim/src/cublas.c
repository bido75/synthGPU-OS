/*
 * SynthGPU CUDA Shim — cuBLAS Implementation
 * ============================================
 * Routes GEMM calls to OpenBLAS via cblas_sgemm.
 * This handles ~90% of transformer compute (every matmul).
 *
 * Column-major ↔ row-major translation:
 *   cuBLAS uses Fortran column-major convention.
 *   cblas_sgemm / numpy use C row-major convention.
 *   Fix: swap A↔B and transa↔transb.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "synthgpu_cuda.h"

#define CUBLAS_STATUS_SUCCESS         0
#define CUBLAS_STATUS_NOT_INITIALIZED 1
#define CUBLAS_STATUS_ALLOC_FAILED    3
#define CUBLAS_STATUS_INVALID_VALUE   7
#define CUBLAS_STATUS_NOT_SUPPORTED  15

/* cuBLAS operation codes (match spec values) */
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1
#define CUBLAS_OP_C 2

/* cuBLAS data type codes */
#define CUDA_R_32F 0
#define CUDA_R_64F 1

/* ── OpenBLAS CBLAS interface ────────────────────────────────────── */
typedef enum { CblasRowMajor=101, CblasColMajor=102 }  CBLAS_ORDER;
typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;

extern void cblas_sgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    int, int, int, float, const float*, int, const float*, int,
    float, float*, int);
extern void cblas_dgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    int, int, int, double, const double*, int, const double*, int,
    double, double*, int);

/* ── cuBLAS handle ───────────────────────────────────────────────── */
typedef struct {
    int   valid;
    void *stream;
    long  gemm_count;
} cublasCtx;

int cublasCreate_v2(void **handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    cublasCtx *ctx = (cublasCtx *)calloc(1, sizeof(cublasCtx));
    if (!ctx) return CUBLAS_STATUS_ALLOC_FAILED;
    ctx->valid = 1;
    *handle = ctx;
    fprintf(stderr, "[SynthGPU] cublasCreate → ready (OpenBLAS backend)\n");
    return CUBLAS_STATUS_SUCCESS;
}

int cublasDestroy_v2(void *handle) {
    free(handle);
    return CUBLAS_STATUS_SUCCESS;
}

int cublasSetStream_v2(void *handle, void *stream) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    ((cublasCtx *)handle)->stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}

int cublasGetStream_v2(void *handle, void **stream) {
    if (!handle || !stream) return CUBLAS_STATUS_INVALID_VALUE;
    *stream = ((cublasCtx *)handle)->stream;
    return CUBLAS_STATUS_SUCCESS;
}

int cublasSetMathMode(void *handle, int mode) {
    (void)handle; (void)mode;
    return CUBLAS_STATUS_SUCCESS;
}

int cublasGetMathMode(void *handle, int *mode) {
    (void)handle;
    if (mode) *mode = 0;
    return CUBLAS_STATUS_SUCCESS;
}

int cublasGetVersion_v2(void *handle, int *version) {
    (void)handle;
    if (!version) return CUBLAS_STATUS_INVALID_VALUE;
    *version = 120200;
    return CUBLAS_STATUS_SUCCESS;
}

/* ── cublasSgemm — single-precision GEMM ───────────────────────── */
int cublasSgemm_v2(void *handle, int transa, int transb,
                   int m, int n, int k,
                   const float *alpha,
                   const float *A, int lda,
                   const float *B, int ldb,
                   const float *beta,
                   float *C, int ldc)
{
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!A || !B || !C || !alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;

    /*
     * cuBLAS column-major → cblas row-major:
     *   C_col = op(A_col) * op(B_col)
     * is the same matrix result as:
     *   C_row = op(B_row) * op(A_row)
     * so we swap A↔B and m↔n.
     */
    CBLAS_TRANSPOSE ta = (transa == CUBLAS_OP_T) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (transb == CUBLAS_OP_T) ? CblasTrans : CblasNoTrans;

    cblas_sgemm(CblasRowMajor, tb, ta,
                n, m, k,
                *alpha, B, ldb, A, lda,
                *beta,  C, ldc);

    if (handle) ((cublasCtx *)handle)->gemm_count++;
    return CUBLAS_STATUS_SUCCESS;
}

/* ── cublasDgemm — double-precision GEMM ───────────────────────── */
int cublasDgemm_v2(void *handle, int transa, int transb,
                   int m, int n, int k,
                   const double *alpha,
                   const double *A, int lda,
                   const double *B, int ldb,
                   const double *beta,
                   double *C, int ldc)
{
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!A || !B || !C || !alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;

    CBLAS_TRANSPOSE ta = (transa == CUBLAS_OP_T) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (transb == CUBLAS_OP_T) ? CblasTrans : CblasNoTrans;

    cblas_dgemm(CblasRowMajor, tb, ta,
                n, m, k,
                *alpha, B, ldb, A, lda,
                *beta,  C, ldc);

    return CUBLAS_STATUS_SUCCESS;
}

/* ── cublasGemmEx — mixed-precision GEMM ──────────────────────── */
int cublasGemmEx(void *handle, int transa, int transb,
                 int m, int n, int k,
                 const void *alpha,
                 const void *A, int Atype, int lda,
                 const void *B, int Btype, int ldb,
                 const void *beta,
                 void *C, int Ctype, int ldc,
                 int computeType, int algo)
{
    (void)computeType;
    (void)algo;
    if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F)
        return cublasSgemm_v2(handle, transa, transb, m, n, k,
                              (const float *)alpha,
                              (const float *)A, lda,
                              (const float *)B, ldb,
                              (const float *)beta,
                              (float *)C, ldc);
    if (Atype == CUDA_R_64F && Btype == CUDA_R_64F && Ctype == CUDA_R_64F)
        return cublasDgemm_v2(handle, transa, transb, m, n, k,
                              (const double *)alpha,
                              (const double *)A, lda,
                              (const double *)B, ldb,
                              (const double *)beta,
                              (double *)C, ldc);
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ── Batched variants ────────────────────────────────────────────── */
int cublasSgemmBatched(void *handle, int transa, int transb,
                       int m, int n, int k,
                       const float *alpha,
                       const float *const Aarray[], int lda,
                       const float *const Barray[], int ldb,
                       const float *beta,
                       float *Carray[], int ldc,
                       int batchCount)
{
    for (int i = 0; i < batchCount; i++)
        cublasSgemm_v2(handle, transa, transb, m, n, k,
                       alpha, Aarray[i], lda, Barray[i], ldb,
                       beta, Carray[i], ldc);
    return CUBLAS_STATUS_SUCCESS;
}

int cublasSgemmStridedBatched(void *handle, int transa, int transb,
                               int m, int n, int k,
                               const float *alpha,
                               const float *A, int lda, long long sA,
                               const float *B, int ldb, long long sB,
                               const float *beta,
                               float *C, int ldc, long long sC,
                               int batchCount)
{
    for (int i = 0; i < batchCount; i++)
        cublasSgemm_v2(handle, transa, transb, m, n, k,
                       alpha, A + i * sA, lda, B + i * sB, ldb,
                       beta,  C + i * sC, ldc);
    return CUBLAS_STATUS_SUCCESS;
}

int cublasGemmStridedBatchedEx(void *handle, int transa, int transb,
                                int m, int n, int k,
                                const void *alpha,
                                const void *A, int Atype, int lda, long long sA,
                                const void *B, int Btype, int ldb, long long sB,
                                const void *beta,
                                void *C, int Ctype, int ldc, long long sC,
                                int batchCount, int computeType, int algo)
{
    (void)Btype; (void)Ctype; (void)computeType;
    (void)algo;
    if (Atype != CUDA_R_32F) return CUBLAS_STATUS_NOT_SUPPORTED;
    for (int i = 0; i < batchCount; i++)
        cublasSgemm_v2(handle, transa, transb, m, n, k,
                       (const float *)alpha,
                       (const float *)A + i * sA, lda,
                       (const float *)B + i * sB, ldb,
                       (const float *)beta,
                       (float *)C + i * sC, ldc);
    return CUBLAS_STATUS_SUCCESS;
}
