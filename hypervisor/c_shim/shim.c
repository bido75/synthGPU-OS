#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <dlfcn.h>
#include <pthread.h>

/* ── Mock CUDA types ───────────────────────────────────────────── */
typedef enum {
    cudaSuccess                 = 0,
    cudaErrorMemoryAllocation   = 2,
    cudaErrorInvalidValue       = 11,
    cudaErrorInvalidDevice      = 101,
    cudaErrorNotSupported       = 801,
} cudaError_t;

typedef enum {
    cudaMemcpyHostToHost        = 0,
    cudaMemcpyHostToDevice      = 1,
    cudaMemcpyDeviceToHost      = 2,
    cudaMemcpyDeviceToDevice    = 3,
} cudaMemcpyKind;

/* ── Paths (must match vgpu_hypervisor.py) ─────────────────────── */
static const char *SHM_PATH = "/dev/shm/synth_vgpu_vram";
static const char *UDS_PATH = "/tmp/vgpu/control.sock";

/* ── Allocation tracking ───────────────────────────────────────── */
#define MAX_ALLOCS 1048576
static struct {
    void  *ptr;
    size_t size;
} _alloc_table[MAX_ALLOCS];
static int _alloc_count = 0;
static pthread_mutex_t _table_lock = PTHREAD_MUTEX_INITIALIZER;

/* ── Internal helpers ──────────────────────────────────────────── */

static int _uds_request(size_t requested_size, size_t *out_offset) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == -1) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, UDS_PATH, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        close(sock);
        return -1;
    }

    /* send requested size (8 bytes, little-endian) */
    uint64_t req = (uint64_t)requested_size;
    write(sock, &req, sizeof(req));

    /* read offset back (8 bytes) */
    uint64_t off;
    ssize_t n = read(sock, &off, sizeof(off));
    close(sock);

    if (n != sizeof(off)) return -1;
    *out_offset = (size_t)off;
    return 0;
}

static int _track_alloc(void *ptr, size_t size) {
    pthread_mutex_lock(&_table_lock);
    if (_alloc_count >= MAX_ALLOCS) {
        pthread_mutex_unlock(&_table_lock);
        return -1;
    }
    _alloc_table[_alloc_count].ptr  = ptr;
    _alloc_table[_alloc_count].size = size;
    _alloc_count++;
    pthread_mutex_unlock(&_table_lock);
    return 0;
}

static int _untrack_alloc(void *ptr, size_t *out_size) {
    pthread_mutex_lock(&_table_lock);
    for (int i = 0; i < _alloc_count; i++) {
        if (_alloc_table[i].ptr == ptr) {
            *out_size = _alloc_table[i].size;
            _alloc_table[i] = _alloc_table[_alloc_count - 1];
            _alloc_count--;
            pthread_mutex_unlock(&_table_lock);
            return 0;
        }
    }
    pthread_mutex_unlock(&_table_lock);
    return -1;
}

/* ── Intercepted CUDA API ─────────────────────────────────────── */

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) return cudaErrorInvalidValue;

    size_t offset;
    if (_uds_request(size, &offset) != 0) {
        fprintf(stderr, "[synthgpu-shim] cudaMalloc: hypervisor unreachable\n");
        return cudaErrorMemoryAllocation;
    }
    if (offset == (size_t)-1) {
        fprintf(stderr, "[synthgpu-shim] cudaMalloc: OOM (requested %zu bytes)\n", size);
        return cudaErrorMemoryAllocation;
    }

    int fd = open(SHM_PATH, O_RDWR);
    if (fd == -1) {
        perror("[synthgpu-shim] open /dev/shm");
        return cudaErrorMemoryAllocation;
    }

    void *mapped = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)offset);
    close(fd);

    if (mapped == MAP_FAILED) {
        perror("[synthgpu-shim] mmap");
        return cudaErrorMemoryAllocation;
    }

    _track_alloc(mapped, size);
    *devPtr = mapped;

    fprintf(stderr, "[synthgpu-shim] cudaMalloc(%zu) → %p (offset %zu)\n", size, mapped, offset);
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) return cudaSuccess;

    size_t size;
    if (_untrack_alloc(devPtr, &size) != 0) {
        fprintf(stderr, "[synthgpu-shim] cudaFree: unknown pointer %p\n", devPtr);
        return cudaErrorInvalidValue;
    }

    munmap(devPtr, size);
    fprintf(stderr, "[synthgpu-shim] cudaFree(%p) — %zu bytes released\n", devPtr, size);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    (void)kind;
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    fprintf(stderr, "[synthgpu-shim] cudaSetDevice(%d) → accepted (virtual)\n", device);
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (device) *device = 0;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int *count) {
    if (count) *count = 1;
    fprintf(stderr, "[synthgpu-shim] cudaGetDeviceCount → 1 device (virtual)\n");
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    /* report 6 GB free out of 8 GB total */
    if (free)  *free  = 6ULL * 1024 * 1024 * 1024;
    if (total) *total = 8ULL * 1024 * 1024 * 1024;
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    /* fall back to regular malloc for pinned memory */
    *ptr = malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height, size_t elemSize) {
    size_t row_bytes = width * elemSize;
    /* align pitch to 256 bytes (common GPU requirement) */
    *pitch = (row_bytes + 255) & ~255;
    size_t total = (*pitch) * height;
    return cudaMalloc(devPtr, total);
}
