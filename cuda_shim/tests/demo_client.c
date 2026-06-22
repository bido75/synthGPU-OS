#include <stdio.h>
#include <cuda_runtime.h>

/* Keep the client unlinked from any CUDA runtime. LD_PRELOAD must supply these. */
#pragma weak cudaGetDeviceCount
#pragma weak cudaMalloc
#pragma weak cudaMemGetInfo
#pragma weak cudaFree

#define VERIFY_MARKER "/tmp/synthgpu_cuda_verified"

int main(void) {
    int count = 0;
    cudaError_t err;
    void *dev_ptr = NULL;
    size_t free_b = 0;
    size_t total_b = 0;
    FILE *marker;

    if (!cudaGetDeviceCount || !cudaMalloc || !cudaMemGetInfo || !cudaFree) {
        fprintf(stderr,
                "CUDA runtime symbols unavailable; run with libsynthgpu_cuda.so in LD_PRELOAD\n");
        return 2;
    }

    err = cudaGetDeviceCount(&count);
    printf("cudaGetDeviceCount: err=%d count=%d\n", err, count);
    if (err != cudaSuccess || count < 1) {
        fprintf(stderr, "FAIL: no virtual device reported\n");
        return 1;
    }

    err = cudaMalloc(&dev_ptr, 1024 * 1024);
    printf("cudaMalloc(1MB): err=%d ptr=%p\n", err, dev_ptr);
    if (err != cudaSuccess || dev_ptr == NULL) {
        fprintf(stderr, "FAIL: allocation failed\n");
        return 1;
    }

    err = cudaMemGetInfo(&free_b, &total_b);
    printf("cudaMemGetInfo: err=%d free=%zu total=%zu\n", err, free_b, total_b);
    if (err != cudaSuccess || total_b == 0 || free_b > total_b) {
        fprintf(stderr, "FAIL: invalid memory information\n");
        cudaFree(dev_ptr);
        return 1;
    }

    err = cudaFree(dev_ptr);
    printf("cudaFree: err=%d\n", err);
    if (err != cudaSuccess) {
        fprintf(stderr, "FAIL: free failed\n");
        return 1;
    }

    marker = fopen(VERIFY_MARKER, "w");
    if (marker) {
        fputs("verified\n", marker);
        fclose(marker);
    } else {
        fprintf(stderr, "WARNING: could not write verification marker\n");
    }

    puts("ALL CHECKS PASSED - real LD_PRELOAD interception confirmed");
    return 0;
}
