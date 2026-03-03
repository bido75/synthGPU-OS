/*
 * SynthGPU CUDA Shim — C→Python Bridge Header (bridge.h)
 * =========================================================
 * Declares functions that route CUDA compute operations to the
 * Python warp scheduler (cuda_shim/kernels/bridge_api.py).
 */

#ifndef SYNTHGPU_BRIDGE_H
#define SYNTHGPU_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialise the Python interpreter and load bridge_api module.
 * Returns 0 on success, -1 if Python is unavailable (graceful fallback). */
int synthgpu_bridge_init(void);

/* Single-precision GEMM: C = alpha*op(A)*op(B) + beta*C
 * Routes through the Python warp scheduler for telemetry. */
void bridge_sgemm(const float  *A, const float  *B, float  *C,
                  int m, int n, int k,
                  float  alpha, float  beta,
                  int trans_a, int trans_b);

/* Double-precision GEMM */
void bridge_dgemm(const double *A, const double *B, double *C,
                  int m, int n, int k,
                  double alpha, double beta,
                  int trans_a, int trans_b);

/* Activation / norm helpers — warp telemetry only (no compute here) */
void bridge_softmax    (const float *input, float *output, int rows, int cols);
void bridge_relu       (const float *input, float *output, int n);
void bridge_layer_norm (const float *input, const float *gamma,
                        const float *beta, float *output,
                        int rows, int cols, float eps);

/* Warp counter accessors */
long  synthgpu_get_warps_executed(void);
float synthgpu_get_warp_throughput(void);

#ifdef __cplusplus
}
#endif

#endif /* SYNTHGPU_BRIDGE_H */
