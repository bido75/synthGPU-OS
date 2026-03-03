/*
 * SynthGPU CUDA Shim — Virtual VRAM Allocator (memory.c)
 * ========================================================
 * Manages a pool of system RAM that appears to CUDA applications
 * as GPU device memory. cudaMalloc/cudaFree/cudaMemcpy all
 * operate on regions within this pool.
 *
 * Key insight: "device" memory IS system RAM — there is no real
 * GPU, so there is nothing to copy to. We just track which regions
 * are allocated. cudaMemcpy (H→D or D→H) is just memcpy().
 *
 * Linux:   uses mmap with MAP_HUGETLB (2MB huge pages) for fast TLB.
 *          Falls back to standard 4KB pages if huge pages unavailable.
 * Windows: uses VirtualAlloc (mmap not available).
 */

#define _GNU_SOURCE
#include "memory.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <unistd.h>
#  ifndef MAP_HUGETLB
#    define MAP_HUGETLB 0  /* not all kernels define this */
#  endif
#endif

/* ── Configuration ───────────────────────────────────────────── */
#define VRAM_DEFAULT_MB  4096   /* 4 GB default virtual VRAM */
#define MAX_ALLOCATIONS  8192   /* Maximum concurrent allocations */
#define ALIGN_BYTES      256    /* GPU memory alignment requirement */

/* ── Internal state ──────────────────────────────────────────── */
static uint8_t *_vram_pool  = NULL;
static size_t   _vram_total = 0;
static size_t   _vram_used  = 0;
static int      _alloc_cnt  = 0;

typedef struct {
    void   *ptr;
    size_t  size;
    int     freed;
} Alloc;

static Alloc _alloc_table[MAX_ALLOCATIONS];

/* ── CPU core count ──────────────────────────────────────────── */
int synthgpu_compute_units(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 4;
#endif
}

/* ── Pool initialisation ─────────────────────────────────────── */
void synthgpu_vram_init(void) {
    if (_vram_pool) return;  /* already initialised */

    /* Respect SYNTHGPU_VRAM_MB env var */
    const char *env_mb = getenv("SYNTHGPU_VRAM_MB");
    size_t mb = env_mb ? (size_t)atol(env_mb) : 0;

    if (mb == 0) {
        /* Default: 40% of available system RAM, capped at 4 GB */
#ifdef _WIN32
        MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
        GlobalMemoryStatusEx(&ms);
        mb = (size_t)(ms.ullAvailPhys * 40 / 100 / (1024*1024));
#else
        FILE *f = fopen("/proc/meminfo", "r");
        long long avail_kb = 0;
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f))
                if (sscanf(line, "MemAvailable: %lld kB", &avail_kb) == 1) break;
            fclose(f);
        }
        mb = avail_kb > 0 ? (size_t)(avail_kb / 1024 * 40 / 100) : 4096;
#endif
        if (mb > 4096) mb = 4096;
        if (mb < 256)  mb = 256;
    }

    _vram_total = mb * 1024ULL * 1024ULL;

#ifdef _WIN32
    _vram_pool = (uint8_t *)VirtualAlloc(NULL, _vram_total,
                                          MEM_RESERVE | MEM_COMMIT,
                                          PAGE_READWRITE);
    if (!_vram_pool) {
        fprintf(stderr, "[SynthGPU] FATAL: VirtualAlloc failed for %zu MB\n", mb);
        exit(1);
    }
#else
    /* Try huge pages first (2 MB pages = faster TLB = faster matmul) */
    _vram_pool = (uint8_t *)mmap(NULL, _vram_total,
                                  PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                                  -1, 0);
    if (_vram_pool == MAP_FAILED) {
        /* Fallback: standard 4 KB pages */
        _vram_pool = (uint8_t *)mmap(NULL, _vram_total,
                                      PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS,
                                      -1, 0);
    }
    if (_vram_pool == MAP_FAILED) {
        fprintf(stderr, "[SynthGPU] FATAL: mmap failed for %zu MB virtual VRAM\n", mb);
        exit(1);
    }
#endif

    memset(_alloc_table, 0, sizeof(_alloc_table));
    fprintf(stderr, "[SynthGPU] Virtual VRAM ready: %zu MB\n", mb);
}

/* ── Allocator ───────────────────────────────────────────────── */
void *synthgpu_alloc(size_t size) {
    if (!_vram_pool) synthgpu_vram_init();

    /* Align to ALIGN_BYTES (GPU memory alignment requirement) */
    size = (size + (ALIGN_BYTES - 1)) & ~(size_t)(ALIGN_BYTES - 1);

    /* First-fit scan: sum used regions to find next offset */
    size_t offset = 0;
    for (int i = 0; i < _alloc_cnt; i++) {
        if (!_alloc_table[i].freed)
            offset += _alloc_table[i].size;
    }

    if (offset + size > _vram_total || _alloc_cnt >= MAX_ALLOCATIONS)
        return NULL;

    void *ptr = _vram_pool + offset;
    _alloc_table[_alloc_cnt].ptr   = ptr;
    _alloc_table[_alloc_cnt].size  = size;
    _alloc_table[_alloc_cnt].freed = 0;
    _alloc_cnt++;
    _vram_used += size;
    return ptr;
}

void synthgpu_free(void *ptr) {
    if (!ptr) return;
    for (int i = 0; i < _alloc_cnt; i++) {
        if (_alloc_table[i].ptr == ptr && !_alloc_table[i].freed) {
            _vram_used -= _alloc_table[i].size;
            _alloc_table[i].freed = 1;
            return;
        }
    }
}

/* ── Pointer conversion ──────────────────────────────────────── */
void *synthgpu_d2h_ptr(const void *device_ptr) {
    /* Device pointer IS a host pointer (both are in system RAM).
     * Validate it is within our pool; pass through either way. */
    if (_vram_pool) {
        const uint8_t *p = (const uint8_t *)device_ptr;
        if (p >= _vram_pool && p < _vram_pool + _vram_total)
            return (void *)device_ptr;
    }
    return (void *)device_ptr;
}

/* ── Pool queries ────────────────────────────────────────────── */
size_t synthgpu_vram_total_bytes(void) {
    if (!_vram_pool) synthgpu_vram_init();
    return _vram_total;
}
size_t synthgpu_vram_used_bytes(void) { return _vram_used; }
