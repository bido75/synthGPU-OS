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
#include <pthread.h>

/*
 * memory.c — Extended memory interceptors for libsynthgpu.so
 *
 * Provides cudaMemcpy2D, cudaMemcpyAsync, cudaMemsetAsync,
 * and a real /dev/shm page-cache warmer for zero-copy pre-fault.
 */

/* ── Forward declarations from shim.c ──────────────────────────── */
extern int _uds_request(size_t requested_size, size_t *out_offset);
extern int  _track_alloc(void *ptr, size_t size);

/* ── Paths (must match vgpu_hypervisor.py) ─────────────────────── */
static const char *SHM_PATH = "/dev/shm/synth_vgpu_vram";
static const char *UDS_PATH = "/tmp/vgpu/control.sock";

/* ── Internal: connect + send + recv pattern (no alloc tracking) ─ */
static int _uds_raw_request(size_t requested_size, size_t *out_offset) {
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

    uint64_t req = (uint64_t)requested_size;
    write(sock, &req, sizeof(req));

    uint64_t off;
    ssize_t n = read(sock, &off, sizeof(off));
    close(sock);

    if (n != sizeof(off)) return -1;
    *out_offset = (size_t)off;
    return 0;
}

/* ── Public API ────────────────────────────────────────────────── */

/*
 * cudaMemcpy2D — strided 2D copy (used by torch/cv ops).
 * Maps a temporary view for each row to handle pitch != width.
 */
typedef enum {
    cudaMemcpyHostToHost        = 0,
    cudaMemcpyHostToDevice      = 1,
    cudaMemcpyDeviceToHost      = 2,
    cudaMemcpyDeviceToDevice    = 3,
} cudaMemcpyKind;

typedef int cudaError_t;
#define cudaSuccess              0
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidValue     11

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch,
                         const void *src, size_t spitch,
                         size_t width, size_t height,
                         int kind) {
    (void)kind;
    if (!dst || !src) return cudaErrorInvalidValue;

    size_t row_bytes = width;
    for (size_t y = 0; y < height; y++) {
        memcpy((char *)dst + y * dpitch,
               (const char *)src + y * spitch,
               row_bytes);
    }
    fprintf(stderr, "[synthgpu-shim] cudaMemcpy2D %zux%zu (dpitch=%zu, spitch=%zu)\n",
            width, height, dpitch, spitch);
    return cudaSuccess;
}

/*
 * cudaMemcpyAsync / cudaMemsetAsync — synchronous shim;
 * real CUDA would queue on a stream, we just execute immediately.
 */
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            int kind, void *stream) {
    (void)kind; (void)stream;
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, void *stream) {
    (void)stream;
    memset(devPtr, value, count);
    return cudaSuccess;
}

/*
 * cudaMallocManaged — unified memory; same as cudaMalloc in our
 * virtual world (single address space).
 */
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    (void)flags;
    return cudaMalloc(devPtr, size);
}
/* forward-declare the real cudaMalloc from shim.c */
cudaError_t cudaMalloc(void **devPtr, size_t size);

/*
 * WarmVRAM — pre-fault every page in the shm file so first-touch
 * page-ins don't spike latency during inference.  Idempotent.
 */
void synthgpu_warm_vram(void) {
    int fd = open(SHM_PATH, O_RDWR);
    if (fd == -1) {
        perror("[synthgpu-shim] warm_vram: open");
        return;
    }

    off_t len = lseek(fd, 0, SEEK_END);
    if (len <= 0) {
        close(fd);
        return;
    }

    void *map = mmap(NULL, (size_t)len, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("[synthgpu-shim] warm_vram: mmap");
        close(fd);
        return;
    }

    /* touch every page (4KB stride) to force physical page allocation */
    volatile char *p = (volatile char *)map;
    for (off_t i = 0; i < len; i += 4096) {
        p[i] = 0;
    }

    munmap(map, (size_t)len);
    close(fd);
    fprintf(stderr, "[synthgpu-shim] VRAM pre-warmed: %ld bytes\n", (long)len);
}

/* cudaPointerGetAttributes — minimal stub */
typedef struct {
    int   type;          /* 1 = device pointer */
    int   device;
    void *devicePointer;
    void *hostPointer;
    int   isManaged;
} cudaPointerAttributes;

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attrs, const void *ptr) {
    if (!attrs) return cudaErrorInvalidValue;
    attrs->type          = 1;  /* device pointer */
    attrs->device        = 0;
    attrs->devicePointer = (void*)ptr;
    attrs->hostPointer   = NULL;
    attrs->isManaged     = 0;
    return cudaSuccess;
}
