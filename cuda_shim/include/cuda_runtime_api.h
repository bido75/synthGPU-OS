/* cuda_runtime_api.h — SynthGPU CUDA Runtime API Declarations
 * ============================================================
 * Matches NVIDIA CUDA Runtime API exactly so any application
 * that includes this header can compile against SynthGPU unchanged.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes (match NVIDIA values exactly) ────────────────── */
typedef enum cudaError {
    cudaSuccess                    = 0,
    cudaErrorInvalidValue          = 1,
    cudaErrorMemoryAllocation      = 2,
    cudaErrorInitializationError   = 3,
    cudaErrorInvalidDevicePointer  = 17,
    cudaErrorInvalidDevice         = 101,
    cudaErrorNoDevice              = 100,
    cudaErrorNotSupported          = 906,
    cudaErrorUnknown               = 999,
} cudaError_t;

/* ── Memory copy kinds ──────────────────────────────────────────── */
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
} cudaMemcpyKind;

/* ── Compute mode ──────────────────────────────────────────────── */
typedef enum cudaComputeMode {
    cudaComputeModeDefault          = 0,
    cudaComputeModeExclusive        = 1,
    cudaComputeModeProhibited       = 2,
    cudaComputeModeExclusiveProcess = 3,
} cudaComputeMode;

/* ── Opaque handles ────────────────────────────────────────────── */
typedef struct CUstream_st  *cudaStream_t;
typedef struct CUevent_st   *cudaEvent_t;

/* ── Device properties — matches struct cudaDeviceProp layout ─── */
struct cudaDeviceProp {
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    size_t memPitch;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    clockRate;
    size_t totalConstMem;
    int    major;
    int    minor;
    size_t textureAlignment;
    int    deviceOverlap;
    int    multiProcessorCount;
    int    kernelExecTimeoutEnabled;
    int    integrated;
    int    canMapHostMemory;
    int    computeMode;
    int    maxTexture1D;
    int    maxTexture2D[2];
    int    maxTexture3D[3];
    int    concurrentKernels;
    int    ECCEnabled;
    int    pciBusID;
    int    pciDeviceID;
    int    pciDomainID;
    int    tccDriver;
    int    asyncEngineCount;
    int    unifiedAddressing;
    int    memoryClockRate;
    int    memoryBusWidth;
    size_t l2CacheSize;
    int    maxThreadsPerMultiProcessor;
    /* Padding — ensures struct is large enough for any CUDA version */
    char   _pad[512];
};

/* ── Runtime API declarations ──────────────────────────────────── */
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                             cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);

cudaError_t cudaDeviceSynchronize(void);

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                 unsigned int flags);
cudaError_t cudaStreamQuery(cudaStream_t stream);

cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char *cudaGetErrorString(cudaError_t error);
const char *cudaGetErrorName(cudaError_t error);

cudaError_t cudaDriverGetVersion(int *driverVersion);
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);

#ifdef __cplusplus
}
#endif
