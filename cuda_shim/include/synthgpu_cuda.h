/* synthgpu_cuda.h — SynthGPU CUDA Shim Internal Header
 * ======================================================
 * Shared declarations for all .c source files in cuda_shim/src/.
 * The C shim calls into Python via the bridge functions declared here.
 */

#ifndef SYNTHGPU_CUDA_H
#define SYNTHGPU_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA runtime error state, owned by device_info.c. */
extern cudaError_t _last_error;

/* ── Virtual VRAM ────────────────────────────────────────────────── */
void  synthgpu_vram_init(void);
size_t synthgpu_vram_budget_mb(size_t total_mb, size_t available_mb,
                               const char *override_mb);
void *synthgpu_alloc(size_t size);
void  synthgpu_free(void *ptr);
void *synthgpu_d2h_ptr(const void *device_ptr);
size_t synthgpu_vram_total_bytes(void);
size_t synthgpu_vram_used_bytes(void);

/* ── Device info ─────────────────────────────────────────────────── */
int         synthgpu_compute_units(void);
const char *synthgpu_device_name(void);

/* ── Python bridge (forward-declared — defined in python_bridge.c) ── */
int synthgpu_bridge_init(void);

void bridge_sgemm(
    const float *A, const float *B, float *C,
    int m, int n, int k,
    float alpha, float beta,
    int trans_a, int trans_b);

void bridge_dgemm(
    const double *A, const double *B, double *C,
    int m, int n, int k,
    double alpha, double beta,
    int trans_a, int trans_b);

void bridge_softmax(const float *input, float *output, int rows, int cols);
void bridge_relu(const float *input, float *output, int n);
void bridge_layer_norm(
    const float *input,
    const float *gamma, const float *beta,
    float *output,
    int rows, int cols,
    float eps);

/* ── Warp telemetry ──────────────────────────────────────────────── */
long  synthgpu_get_warps_executed(void);
float synthgpu_get_warp_throughput(void);

#ifdef __cplusplus
}
#endif

#endif /* SYNTHGPU_CUDA_H */
