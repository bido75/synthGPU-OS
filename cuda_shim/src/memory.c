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
#define MAX_ALLOCATIONS  8192   /* Maximum concurrent allocations */
#define ALIGN_BYTES      256    /* GPU memory alignment requirement */
#define CONSTRAINED_RAM_THRESHOLD_MB  (16 * 1024)
#define CONSTRAINED_POOL_MAX_MB       256
#define CONSTRAINED_POOL_SAFE_MB      128

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

/* ── Pool sizing ─────────────────────────────────────────────── */
size_t synthgpu_vram_budget_mb(size_t total_mb, size_t available_mb,
                               const char *override_mb) {
    size_t pool_mb;
    int constrained = total_mb < CONSTRAINED_RAM_THRESHOLD_MB;

    if (override_mb != NULL) {
        long long requested = atoll(override_mb);
        pool_mb = requested > 0 ? (size_t)requested : 0;
        if (constrained) {
            size_t safe_max = available_mb * 40 / 100;
            if (pool_mb > safe_max) pool_mb = safe_max;
            if (pool_mb > CONSTRAINED_POOL_MAX_MB)
                pool_mb = CONSTRAINED_POOL_MAX_MB;
        } else {
            size_t safe_max = available_mb * 80 / 100;
            if (pool_mb > safe_max) pool_mb = safe_max;
        }
        if (pool_mb < 64) pool_mb = 64;
    } else if (constrained) {
        pool_mb = available_mb * 25 / 100;
        if (pool_mb > CONSTRAINED_POOL_SAFE_MB)
            pool_mb = CONSTRAINED_POOL_SAFE_MB;
        if (pool_mb < 64) pool_mb = 64;
    } else {
        size_t avail_cap = available_mb * 75 / 100;
        pool_mb = total_mb * 40 / 100;
        if (pool_mb > avail_cap) pool_mb = avail_cap;
        if (pool_mb < 256) pool_mb = 256;
    }

    return (pool_mb / 64) * 64;
}

static void probe_system_memory_mb(size_t *total_mb, size_t *available_mb) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        *total_mb = (size_t)(ms.ullTotalPhys / (1024 * 1024));
        *available_mb = (size_t)(ms.ullAvailPhys / (1024 * 1024));
        return;
    }
#else
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        unsigned long long total_kb = 0;
        unsigned long long available_kb = 0;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %llu kB", &total_kb) == 1) continue;
            if (sscanf(line, "MemAvailable: %llu kB", &available_kb) == 1) continue;
        }
        fclose(f);
        if (total_kb > 0 && available_kb > 0) {
            *total_mb = (size_t)(total_kb / 1024);
            *available_mb = (size_t)(available_kb / 1024);
            return;
        }
    }
#endif
    *total_mb = 8192;
    *available_mb = 4096;
}

/* ── Pool initialisation ─────────────────────────────────────── */
void synthgpu_vram_init(void) {
    if (_vram_pool) return;  /* already initialised */

    size_t total_mb;
    size_t available_mb;
    probe_system_memory_mb(&total_mb, &available_mb);

    /* Read the override once, before selecting the hardware profile. */
    const char *env_mb = getenv("SYNTHGPU_VRAM_MB");
    size_t mb = synthgpu_vram_budget_mb(total_mb, available_mb, env_mb);

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
