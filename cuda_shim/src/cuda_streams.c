/*
 * SynthGPU CUDA Shim — Stream & Event Emulation
 * ===============================================
 * CUDA streams provide ordering; CUDA events measure time.
 * CPU execution is synchronous, so streams are no-ops and
 * events just record wall-clock time.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <time.h>
#include "synthgpu_cuda.h"

#define cudaSuccess           0
#define cudaErrorInvalidValue 1

/* ── Streams ─────────────────────────────────────────────────────── */

typedef struct { int id; } SynthStream;
static int _stream_counter = 0;

int cudaStreamCreate(void **pStream) {
    if (!pStream) return cudaErrorInvalidValue;
    SynthStream *s = (SynthStream *)malloc(sizeof(SynthStream));
    if (!s) return 2; /* cudaErrorMemoryAllocation */
    s->id = ++_stream_counter;
    *pStream = s;
    return cudaSuccess;
}

int cudaStreamCreateWithFlags(void **pStream, unsigned int flags) {
    (void)flags;
    return cudaStreamCreate(pStream);
}

int cudaStreamDestroy(void *stream) {
    free(stream);
    return cudaSuccess;
}

int cudaStreamSynchronize(void *stream) {
    (void)stream;
    return cudaSuccess;
}

int cudaStreamWaitEvent(void *stream, void *event, unsigned int flags) {
    (void)stream; (void)event; (void)flags;
    return cudaSuccess;
}

/* ── Events ──────────────────────────────────────────────────────── */

typedef struct {
    double time_ms;   /* wall-clock at cudaEventRecord() */
} SynthEvent;

static double _wall_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

int cudaEventCreate(void **event) {
    if (!event) return cudaErrorInvalidValue;
    SynthEvent *e = (SynthEvent *)malloc(sizeof(SynthEvent));
    if (!e) return 2;
    e->time_ms = 0.0;
    *event = e;
    return cudaSuccess;
}

int cudaEventCreateWithFlags(void **event, unsigned int flags) {
    (void)flags;
    return cudaEventCreate(event);
}

int cudaEventRecord(void *event, void *stream) {
    (void)stream;
    if (!event) return cudaErrorInvalidValue;
    ((SynthEvent *)event)->time_ms = _wall_ms();
    return cudaSuccess;
}

int cudaEventSynchronize(void *event) {
    (void)event;
    return cudaSuccess;
}

int cudaEventElapsedTime(float *ms, void *start, void *end) {
    if (!ms || !start || !end) return cudaErrorInvalidValue;
    double diff = ((SynthEvent *)end)->time_ms - ((SynthEvent *)start)->time_ms;
    *ms = (float)(diff < 0 ? 0 : diff);
    return cudaSuccess;
}

int cudaEventDestroy(void *event) {
    free(event);
    return cudaSuccess;
}
