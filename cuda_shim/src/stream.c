/*
 * SynthGPU CUDA Shim — CUDA Stream Emulation (stream.c)
 * =======================================================
 * CUDA streams allow asynchronous execution ordering.
 * On SynthGPU all execution is synchronous (CPU), so streams
 * are stub objects that always report immediate completion.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include "synthgpu_cuda.h"

/* Opaque stream handle */
struct CUstream_st {
    int id;
};

static int _stream_counter = 0;

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    if (!pStream) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    cudaStream_t s = (cudaStream_t)malloc(sizeof(*s));
    if (!s) { _last_error = cudaErrorMemoryAllocation; return cudaErrorMemoryAllocation; }
    s->id = ++_stream_counter;
    *pStream = s;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
    (void)flags;
    return cudaStreamCreate(pStream);
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority) {
    (void)flags; (void)priority;
    return cudaStreamCreate(pStream);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (stream) free(stream);
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    /* No-op: CPU execution is inherently synchronous */
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
    (void)stream; (void)event; (void)flags;
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    /* Always complete — CPU execution is synchronous */
    (void)stream;
    return cudaSuccess;  /* cudaSuccess doubles as cudaSuccess + complete */
}
