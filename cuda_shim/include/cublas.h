/* cublas.h — SynthGPU cuBLAS API Declarations
 * =============================================
 * Implements the subset of cuBLAS used by PyTorch and TensorFlow.
 * Applications compile against this header unchanged.
 */

#pragma once
#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cublasContext *cublasHandle_t;

typedef enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS          = 0,
    CUBLAS_STATUS_NOT_INITIALIZED  = 1,
    CUBLAS_STATUS_ALLOC_FAILED     = 3,
    CUBLAS_STATUS_INVALID_VALUE    = 7,
    CUBLAS_STATUS_ARCH_MISMATCH    = 8,
    CUBLAS_STATUS_MAPPING_ERROR    = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR   = 14,
    CUBLAS_STATUS_NOT_SUPPORTED    = 15,
    CUBLAS_STATUS_LICENSE_ERROR    = 16,
} cublasStatus_t;

typedef enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_HERMITAN = 2,
    CUBLAS_OP_CONJG    = 3,
} cublasOperation_t;

typedef enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DEFAULT              = -1,
    CUBLAS_GEMM_ALGO0                = 0,
    CUBLAS_GEMM_ALGO1                = 1,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP    = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP      = 100,
} cublasGemmAlgo_t;

typedef enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_8I  = 3,
    CUDA_R_32I = 10,
} cudaDataType;

/* ── Handle management ─────────────────────────────────────────── */
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, int *mode);

/* ── GEMM ──────────────────────────────────────────────────────── */
cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc);

cublasStatus_t cublasDgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda,
    const double *B, int ldb,
    const double *beta,
    double *C, int ldc);

cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType Atype, int lda,
    const void *B, cudaDataType Btype, int ldb,
    const void *beta,
    void *C, cudaDataType Ctype, int ldc,
    cudaDataType computeType,
    cublasGemmAlgo_t algo);

/* Batched variants */
cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *const Aarray[], int lda,
    const float *const Barray[], int ldb,
    const float *beta,
    float *Carray[], int ldc,
    int batchCount);

cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, long long strideA,
    const float *B, int ldb, long long strideB,
    const float *beta,
    float *C, int ldc, long long strideC,
    int batchCount);

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType Atype, int lda, long long strideA,
    const void *B, cudaDataType Btype, int ldb, long long strideB,
    const void *beta,
    void *C, cudaDataType Ctype, int ldc, long long strideC,
    int batchCount,
    cudaDataType computeType,
    cublasGemmAlgo_t algo);

/* Aliases used by PyTorch internals */
#define cublasCreate        cublasCreate_v2
#define cublasDestroy       cublasDestroy_v2
#define cublasSetStream     cublasSetStream_v2
#define cublasGetStream     cublasGetStream_v2
#define cublasSgemm         cublasSgemm_v2
#define cublasDgemm         cublasDgemm_v2

#ifdef __cplusplus
}
#endif
