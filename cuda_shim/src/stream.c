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

#define cudaSuccess           0
#define cudaErrorInvalidValue 11

/* Opaque stream handle */
typedef struct {
    int id;
} SynthGPUStream;

static int _stream_counter = 0;

int cudaStreamCreate(void **pStream) {
    if (!pStream) return cudaErrorInvalidValue;
    SynthGPUStream *s = (SynthGPUStream *)malloc(sizeof(SynthGPUStream));
    s->id = ++_stream_counter;
    *pStream = s;
    return cudaSuccess;
}

int cudaStreamCreateWithFlags(void **pStream, unsigned int flags) {
    (void)flags;
    return cudaStreamCreate(pStream);
}

int cudaStreamCreateWithPriority(void **pStream, unsigned int flags, int priority) {
    (void)flags; (void)priority;
    return cudaStreamCreate(pStream);
}

int cudaStreamDestroy(void *stream) {
    if (stream) free(stream);
    return cudaSuccess;
}

int cudaStreamSynchronize(void *stream) {
    /* No-op: CPU execution is inherently synchronous */
    (void)stream;
    return cudaSuccess;
}

int cudaStreamWaitEvent(void *stream, void *event, unsigned int flags) {
    (void)stream; (void)event; (void)flags;
    return cudaSuccess;
}

int cudaStreamQuery(void *stream) {
    /* Always complete — CPU execution is synchronous */
    (void)stream;
    return cudaSuccess;  /* cudaSuccess doubles as cudaSuccess + complete */
}
