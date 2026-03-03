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

#ifdef _WIN32
#  include <windows.h>
#else
#  include <time.h>
#endif

#define cudaSuccess           0
#define cudaErrorInvalidValue 11

/* Event struct — stores a high-resolution timestamp */
typedef struct {
    double time_ms;   /* wall-clock time in milliseconds */
    int    recorded;
} SynthGPUEvent;

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

int cudaEventCreate(void **event) {
    if (!event) return cudaErrorInvalidValue;
    SynthGPUEvent *e = (SynthGPUEvent *)malloc(sizeof(SynthGPUEvent));
    e->time_ms = 0.0;
    e->recorded = 0;
    *event = e;
    return cudaSuccess;
}

int cudaEventCreateWithFlags(void **event, unsigned int flags) {
    (void)flags;
    return cudaEventCreate(event);
}

int cudaEventRecord(void *event, void *stream) {
    if (!event) return cudaErrorInvalidValue;
    SynthGPUEvent *e = (SynthGPUEvent *)event;
    e->time_ms  = _now_ms();
    e->recorded = 1;
    (void)stream;
    return cudaSuccess;
}

int cudaEventSynchronize(void *event) {
    /* No-op: CPU execution is synchronous */
    (void)event;
    return cudaSuccess;
}

int cudaEventQuery(void *event) {
    (void)event;
    return cudaSuccess;
}

int cudaEventElapsedTime(float *ms, void *start, void *end) {
    if (!ms || !start || !end) return cudaErrorInvalidValue;
    SynthGPUEvent *s = (SynthGPUEvent *)start;
    SynthGPUEvent *e = (SynthGPUEvent *)end;
    *ms = (float)(e->time_ms - s->time_ms);
    if (*ms < 0.0f) *ms = 0.0f;
    return cudaSuccess;
}

int cudaEventDestroy(void *event) {
    if (event) free(event);
    return cudaSuccess;
}
