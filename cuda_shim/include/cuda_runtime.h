/* cuda_runtime.h — SynthGPU CUDA Runtime (umbrella header)
 * ==========================================================
 * Included by application code that uses the CUDA Runtime API.
 * Pulls in cuda_runtime_api.h and adds common device macros.
 */

#pragma once

#include "cuda_runtime_api.h"

/* Standard CUDA event flags */
#define cudaEventDefault        0x00
#define cudaEventBlockingSync   0x01
#define cudaEventDisableTiming  0x02
#define cudaEventInterprocess   0x04

/* Stream flags */
#define cudaStreamDefault     0x00
#define cudaStreamNonBlocking 0x01

/* cudaMallocManaged flags */
#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost   0x02

/* Device attribute query */
typedef enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock              = 1,
    cudaDevAttrMaxBlockDimX                    = 2,
    cudaDevAttrMaxGridDimX                     = 5,
    cudaDevAttrWarpSize                        = 10,
    cudaDevAttrMultiProcessorCount             = 16,
    cudaDevAttrComputeCapabilityMajor          = 75,
    cudaDevAttrComputeCapabilityMinor          = 76,
    cudaDevAttrConcurrentKernels               = 31,
    cudaDevAttrUnifiedAddressing               = 41,
    cudaDevAttrTotalConstantMemory             = 57,
    cudaDevAttrSharedMemPerBlock               = 8,
} cudaDeviceAttr;

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceReset(void);

#ifdef __cplusplus
}
#endif
