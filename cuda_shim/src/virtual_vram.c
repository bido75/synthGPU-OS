/*
 * SynthGPU CUDA Shim — Virtual VRAM Allocator
 * =============================================
 * cudaMalloc / cudaFree backed by system RAM.
 * All "device" pointers are regular system-RAM pointers —
 * cudaMemcpy is just memcpy() under the hood.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "synthgpu_cuda.h"

#ifdef _WIN32
  #include <windows.h>
  #define LOCK_T            CRITICAL_SECTION
  #define LOCK_INIT(l)      InitializeCriticalSection(&(l))
  #define LOCK_ACQUIRE(l)   EnterCriticalSection(&(l))
  #define LOCK_RELEASE(l)   LeaveCriticalSection(&(l))
#else
  #include <pthread.h>
  #define LOCK_T            pthread_mutex_t
  #define LOCK_INIT(l)      pthread_mutex_init(&(l), NULL)
  #define LOCK_ACQUIRE(l)   pthread_mutex_lock(&(l))
  #define LOCK_RELEASE(l)   pthread_mutex_unlock(&(l))
#endif

#define cudaSuccess               0
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidValue     1

/* ── Allocation registry ────────────────────────────────────────── */
#define MAX_ALLOCS 65536

typedef struct {
    void   *ptr;
    size_t  size;
    int     active;
} VRAMEntry;

static VRAMEntry  _alloc_table[MAX_ALLOCS];
static size_t     _total_allocated = 0;
static size_t     _vram_limit      = 0;
static LOCK_T     _alloc_lock;
static int        _lock_initialized = 0;

/* ── Determine pool size ────────────────────────────────────────── */
static void _ensure_limit(void) {
    if (_vram_limit) return;

    char *env = getenv("SYNTHGPU_VRAM_MB");
    if (env) {
        _vram_limit = (size_t)atoll(env) * 1024 * 1024;
        return;
    }

#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    _vram_limit = (size_t)(ms.ullAvailPhys * 40 / 100);
#else
    FILE *f = fopen("/proc/meminfo", "r");
    long long kb = 4LL * 1024 * 1024; /* 4GB fallback */
    if (f) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                sscanf(line, "MemAvailable: %lld kB", &kb);
                break;
            }
        }
        fclose(f);
    }
    _vram_limit = (size_t)(kb * 1024LL * 40 / 100);
#endif
    fprintf(stderr, "[SynthGPU] Virtual VRAM pool: %zu MB\n",
            _vram_limit / 1024 / 1024);
}

void synthgpu_vram_init(void) {
    if (!_lock_initialized) {
        LOCK_INIT(_alloc_lock);
        _lock_initialized = 1;
    }
    _ensure_limit();
}

/* ── Alloc / free ────────────────────────────────────────────────── */

void *synthgpu_alloc(size_t size) {
    synthgpu_vram_init();

    /* Align to 256 bytes — mirrors GPU alignment */
    size = (size + 255) & ~(size_t)255;

    LOCK_ACQUIRE(_alloc_lock);

    if (_total_allocated + size > _vram_limit) {
        LOCK_RELEASE(_alloc_lock);
        fprintf(stderr, "[SynthGPU] cudaMalloc OOM: need %zuMB, free %zuMB\n",
                size / 1024 / 1024,
                (_vram_limit - _total_allocated) / 1024 / 1024);
        return NULL;
    }

    void *ptr = malloc(size);
    if (!ptr) { LOCK_RELEASE(_alloc_lock); return NULL; }

    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (!_alloc_table[i].active) {
            _alloc_table[i].ptr    = ptr;
            _alloc_table[i].size   = size;
            _alloc_table[i].active = 1;
            _total_allocated      += size;
            LOCK_RELEASE(_alloc_lock);
            return ptr;
        }
    }
    free(ptr);
    LOCK_RELEASE(_alloc_lock);
    return NULL;
}

void synthgpu_free(void *ptr) {
    if (!ptr) return;
    LOCK_ACQUIRE(_alloc_lock);
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (_alloc_table[i].active && _alloc_table[i].ptr == ptr) {
            _total_allocated        -= _alloc_table[i].size;
            _alloc_table[i].active   = 0;
            _alloc_table[i].ptr      = NULL;
            free(ptr);
            LOCK_RELEASE(_alloc_lock);
            return;
        }
    }
    /* Not in registry — ignore silently (host pointer passed to cudaFree) */
    LOCK_RELEASE(_alloc_lock);
}

void *synthgpu_d2h_ptr(const void *device_ptr) {
    /* Device IS host — pointer is valid as-is */
    return (void *)device_ptr;
}

size_t synthgpu_vram_total_bytes(void) { _ensure_limit(); return _vram_limit; }
size_t synthgpu_vram_used_bytes(void)  { return _total_allocated; }

/* ── CUDA Memory API ─────────────────────────────────────────────── */

int cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr || !size) return cudaErrorInvalidValue;
    void *ptr = synthgpu_alloc(size);
    if (!ptr) return cudaErrorMemoryAllocation;
    *devPtr = ptr;
    return cudaSuccess;
}

int cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    (void)flags;
    return cudaMalloc(devPtr, size);
}

int cudaFree(void *devPtr) {
    synthgpu_free(devPtr);
    return cudaSuccess;
}

int cudaMemcpy(void *dst, const void *src, size_t count, int kind) {
    (void)kind;
    if (!dst || !src) return cudaErrorInvalidValue;
    memcpy(dst, src, count);
    return cudaSuccess;
}

int cudaMemcpyAsync(void *dst, const void *src, size_t count,
                    int kind, void *stream) {
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

int cudaMemset(void *devPtr, int value, size_t count) {
    if (!devPtr) return cudaErrorInvalidValue;
    memset(devPtr, value, count);
    return cudaSuccess;
}

int cudaMemsetAsync(void *devPtr, int value, size_t count, void *stream) {
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

int cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    _ensure_limit();
    if (total_bytes) *total_bytes = _vram_limit;
    if (free_bytes)  *free_bytes  = (_vram_limit > _total_allocated)
                                    ? _vram_limit - _total_allocated : 0;
    return cudaSuccess;
}
