/*
 * SynthGPU CUDA Shim — Main Interception Layer (shim.c)
 * =======================================================
 * THE primary file. Defines every CUDA Runtime and cuBLAS function
 * that PyTorch / TensorFlow / HuggingFace call at runtime.
 *
 * When loaded via LD_PRELOAD (Linux) or DLL injection (Windows),
 * all CUDA API calls land here instead of NVIDIA's libcuda.so.
 *
 * Build:
 *   gcc -shared -fPIC -O3 -march=native \
 *       shim.c memory.c stream.c event.c bridge.c telemetry.c \
 *       -Iinclude -I../include \
 *       $(python3-config --includes --ldflags) \
 *       -lopenblas -ldl -lm -lpthread \
 *       -o libsynthgpu_cuda.so
 *
 * Dependencies:
 *   memory.c  — Virtual VRAM pool (synthgpu_alloc, synthgpu_free)
 *   bridge.c  — C→Python warp scheduler bridge
 *   stream.c  — CUDA stream emulation
 *   event.c   — CUDA event timing
 *   telemetry.c — Dashboard stats
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../include/cuda_runtime_api.h"
#include "../include/cublas.h"
#include "memory.h"
#include "bridge.h"

/* ── Global state ─────────────────────────────────────────────── */
static int _current_device = 0;
static int _last_error     = 0;   /* cudaSuccess */

/* ── Library constructor ──────────────────────────────────────── */
/* Runs automatically when the .so is loaded (LD_PRELOAD / dlopen) */

#ifdef _MSC_VER
/* Windows DLL entry point */
#include <windows.h>
BOOL WINAPI DllMain(HINSTANCE hInst, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        synthgpu_vram_init();
        synthgpu_bridge_init();
        fprintf(stderr, "\n[SynthGPU] CUDA Shim v0.3.0 loaded (Windows)\n\n");
    }
    (void)hInst; (void)reserved;
    return TRUE;
}
#else
__attribute__((constructor))
static void synthgpu_init(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  SynthGPU CUDA Compatibility Shim v0.3.0         ║\n");
    fprintf(stderr, "║  github.com/OpenVGPU/SynthGPU                    ║\n");
    fprintf(stderr, "║  NO PHYSICAL GPU — 100%% CPU Compute               ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════╝\n");

    synthgpu_vram_init();    /* Allocate virtual VRAM pool  */
    synthgpu_bridge_init();  /* Initialise Python bridge    */

    fprintf(stderr,
        "[SynthGPU] CUDA shim active — intercepting all CUDA calls\n"
        "[SynthGPU] Virtual VRAM: %zu MB   Compute units: %d\n\n",
        synthgpu_vram_total_bytes() / 1024 / 1024,
        synthgpu_compute_units());
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * SECTION 1 — Device functions
 * ═══════════════════════════════════════════════════════════════ */

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    *count = 1;  /* SynthGPU appears as one virtual GPU */
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    *device = _current_device;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    if (device != 0) { _last_error = cudaErrorInvalidDevice; return cudaErrorInvalidDevice; }
    _current_device = device;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    if (!prop || device != 0) { _last_error = cudaErrorInvalidDevice; return cudaErrorInvalidDevice; }

    memset(prop, 0, sizeof(*prop));

    /* Name shown by torch.cuda.get_device_name(0) */
    strncpy(prop->name, "SynthGPU Virtual Accelerator", 255);

    prop->totalGlobalMem      = synthgpu_vram_total_bytes();
    prop->sharedMemPerBlock   = 49152;         /* 48 KB */
    prop->regsPerBlock        = 65536;
    prop->warpSize            = 32;            /* matches our scheduler */
    prop->maxThreadsPerBlock  = 1024;
    prop->maxThreadsDim[0]    = 1024;
    prop->maxThreadsDim[1]    = 1024;
    prop->maxThreadsDim[2]    = 64;
    prop->maxGridSize[0]      = 2147483647;
    prop->maxGridSize[1]      = 65535;
    prop->maxGridSize[2]      = 65535;
    prop->clockRate           = 1700000;       /* 1.7 GHz */
    prop->totalConstMem       = 65536;
    prop->major               = 8;             /* Report as Ampere sm_80 */
    prop->minor               = 0;
    prop->multiProcessorCount = synthgpu_compute_units();
    prop->l2CacheSize         = 4194304;       /* 4 MB */
    prop->memoryClockRate     = 9001000;       /* ~9 GHz GDDR6x equivalent */
    prop->memoryBusWidth      = 256;
    prop->concurrentKernels   = 1;
    prop->computeMode         = cudaComputeModeDefault;

    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    if (!value || device != 0) return cudaErrorInvalidDevice;
    /* Return sensible defaults for the most commonly queried attributes */
    switch (attr) {
        case 1:  *value = 1024;  break;  /* maxThreadsPerBlock */
        case 4:  *value = 32;    break;  /* warpSize */
        case 16: *value = 8;     break;  /* major */
        case 17: *value = 0;     break;  /* minor */
        case 35: *value = synthgpu_compute_units(); break; /* multiProcessorCount */
        default: *value = 0;
    }
    return cudaSuccess;
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 2 — Memory functions
 * ═══════════════════════════════════════════════════════════════ */

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr || size == 0) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    void *ptr = synthgpu_alloc(size);
    if (!ptr) { _last_error = cudaErrorMemoryAllocation; return cudaErrorMemoryAllocation; }
    *devPtr = ptr;
    return cudaSuccess;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    /* Unified memory: same as cudaMalloc for SynthGPU (all is system RAM) */
    (void)flags;
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    if (!ptr) return cudaErrorInvalidValue;
    *ptr = malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) return cudaSuccess;
    synthgpu_free(devPtr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    if (!dst || !src) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    /* Device IS host RAM — every direction is just memcpy */
    memcpy(dst, src, count);
    (void)kind;
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                             cudaMemcpyKind kind, cudaStream_t stream) {
    /* Async = sync for CPU compute — no hardware queue */
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    if (!devPtr) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    size_t total = synthgpu_vram_total_bytes();
    size_t used  = synthgpu_vram_used_bytes();
    if (total_bytes) *total_bytes = total;
    if (free_bytes)  *free_bytes  = (total > used) ? (total - used) : 0;
    return cudaSuccess;
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 3 — Synchronisation
 * ═══════════════════════════════════════════════════════════════ */

cudaError_t cudaDeviceSynchronize(void) {
    /* Synchronous compute — nothing to wait for */
    return cudaSuccess;
}

cudaError_t cudaThreadSynchronize(void) { return cudaSuccess; }
cudaError_t cudaDeviceReset(void)       { return cudaSuccess; }

/* ═══════════════════════════════════════════════════════════════
 * SECTION 4 — Error handling
 * ═══════════════════════════════════════════════════════════════ */

cudaError_t cudaGetLastError(void) {
    int err = _last_error;
    _last_error = cudaSuccess;
    return (cudaError_t)err;
}

cudaError_t cudaPeekAtLastError(void) {
    return (cudaError_t)_last_error;
}

const char *cudaGetErrorString(cudaError_t error) {
    switch ((int)error) {
        case 0:   return "no error";
        case 1:   return "invalid argument";
        case 2:   return "out of memory";
        case 3:   return "initialization error";
        case 100: return "no CUDA-capable device is detected";
        case 101: return "invalid device ordinal";
        default:  return "unknown error";
    }
}

const char *cudaGetErrorName(cudaError_t error) {
    switch ((int)error) {
        case 0:   return "cudaSuccess";
        case 1:   return "cudaErrorInvalidValue";
        case 2:   return "cudaErrorMemoryAllocation";
        case 100: return "cudaErrorNoDevice";
        case 101: return "cudaErrorInvalidDevice";
        default:  return "cudaErrorUnknown";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 5 — Version queries
 * ═══════════════════════════════════════════════════════════════ */

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return cudaErrorInvalidValue;
    *driverVersion = 12020;   /* Report as CUDA 12.2 */
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) return cudaErrorInvalidValue;
    *runtimeVersion = 12020;
    return cudaSuccess;
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 6 — cuBLAS Handle management
 * ═══════════════════════════════════════════════════════════════ */

typedef struct { int id; } SynthGPUcuBLAS;
static SynthGPUcuBLAS _cublas_handle = {1};

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    *handle = (cublasHandle_t)&_cublas_handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    (void)handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
    (void)handle; (void)streamId;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode) {
    (void)handle; (void)mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    (void)handle;
    if (version) *version = 120200;   /* cuBLAS 12.2 */
    return CUBLAS_STATUS_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 7 — cuBLAS GEMM  (CRITICAL — 90% of transformer compute)
 * Routes to Python warp scheduler via bridge_sgemm().
 * ═══════════════════════════════════════════════════════════════ */

cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc)
{
    /* Resolve device pointers → host pointers (same address in SynthGPU) */
    float *hA = (float *)synthgpu_d2h_ptr(A);
    float *hB = (float *)synthgpu_d2h_ptr(B);
    float *hC = (float *)synthgpu_d2h_ptr(C);

    /* Route through bridge → Python warp scheduler for telemetry + compute */
    bridge_sgemm(hA, hB, hC, m, n, k,
                 *alpha, *beta,
                 transa == CUBLAS_OP_T,
                 transb == CUBLAS_OP_T);

    (void)handle; (void)lda; (void)ldb; (void)ldc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda,
    const double *B, int ldb,
    const double *beta,
    double *C, int ldc)
{
    double *hA = (double *)synthgpu_d2h_ptr(A);
    double *hB = (double *)synthgpu_d2h_ptr(B);
    double *hC = (double *)synthgpu_d2h_ptr(C);

    bridge_dgemm(hA, hB, hC, m, n, k, *alpha, *beta,
                 transa == CUBLAS_OP_T, transb == CUBLAS_OP_T);

    (void)handle; (void)lda; (void)ldb; (void)ldc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType Atype, int lda,
    const void *B, cudaDataType Btype, int ldb,
    const void *beta,
    void *C,       cudaDataType Ctype, int ldc,
    cudaDataType computeType,
    cublasGemmAlgo_t algo)
{
    /* Handle the common case: float32 */
    if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
        return cublasSgemm_v2(handle, transa, transb, m, n, k,
                               (const float *)alpha,
                               (const float *)A, lda,
                               (const float *)B, ldb,
                               (const float *)beta,
                               (float *)C, ldc);
    }
    (void)computeType; (void)algo;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *const Aarray[], int lda,
    const float *const Barray[], int ldb,
    const float *beta,
    float *Carray[], int ldc,
    int batchCount)
{
    for (int i = 0; i < batchCount; i++) {
        cublasSgemm_v2(handle, transa, transb, m, n, k,
                       alpha, Aarray[i], lda, Barray[i], ldb,
                       beta,  Carray[i], ldc);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, long long strideA,
    const float *B, int ldb, long long strideB,
    const float *beta,
    float *C,       int ldc, long long strideC,
    int batchCount)
{
    for (int i = 0; i < batchCount; i++) {
        cublasSgemm_v2(handle, transa, transb, m, n, k,
                       alpha, A + i * strideA, lda,
                              B + i * strideB, ldb,
                       beta,  C + i * strideC, ldc);
    }
    return CUBLAS_STATUS_SUCCESS;
}
