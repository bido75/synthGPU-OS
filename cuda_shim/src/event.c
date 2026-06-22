/*
 * SynthGPU CUDA Shim — CUDA Event Emulation (event.c)
 * =====================================================
 * CUDA events are used for timing and inter-stream synchronisation.
 * We use clock_gettime (Linux) / QueryPerformanceCounter (Windows)
 * to record real wall-clock timestamps so that cudaEventElapsedTime
 * returns correct values to profiling tools.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include "synthgpu_cuda.h"

#ifdef _WIN32
#  include <windows.h>
#else
#  include <time.h>
#endif

/* Event struct — stores a high-resolution timestamp */
struct CUevent_st {
    double time_ms;   /* wall-clock time in milliseconds */
    int    recorded;
};

static double _now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
#endif
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    if (!event) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    cudaEvent_t e = (cudaEvent_t)malloc(sizeof(*e));
    if (!e) { _last_error = cudaErrorMemoryAllocation; return cudaErrorMemoryAllocation; }
    e->time_ms = 0.0;
    e->recorded = 0;
    *event = e;
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    (void)flags;
    return cudaEventCreate(event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    cudaEvent_t e = event;
    e->time_ms  = _now_ms();
    e->recorded = 1;
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    /* No-op: CPU execution is synchronous */
    (void)event;
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    (void)event;
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms || !start || !end) {
        _last_error = cudaErrorInvalidValue;
        return cudaErrorInvalidValue;
    }
    cudaEvent_t s = start;
    cudaEvent_t e = end;
    *ms = (float)(e->time_ms - s->time_ms);
    if (*ms < 0.0f) *ms = 0.0f;
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (event) free(event);
    return cudaSuccess;
}
