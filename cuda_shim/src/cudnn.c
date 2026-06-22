/*
 * SynthGPU CUDA Shim — cuDNN Implementation
 * ===========================================
 * Activation functions, softmax, and tensor descriptors.
 * All compute is done directly in C (no Python bridge needed
 * for these simple element-wise operations).
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "synthgpu_cuda.h"

#define CUDNN_STATUS_SUCCESS         0
#define CUDNN_STATUS_NOT_INITIALIZED 1
#define CUDNN_STATUS_ALLOC_FAILED    2
#define CUDNN_STATUS_BAD_PARAM       6
#define CUDNN_STATUS_NOT_SUPPORTED   9

#define CUDNN_ACTIVATION_SIGMOID      0
#define CUDNN_ACTIVATION_RELU         1
#define CUDNN_ACTIVATION_TANH         2
#define CUDNN_ACTIVATION_CLIPPED_RELU 3
#define CUDNN_ACTIVATION_ELU          4
#define CUDNN_ACTIVATION_IDENTITY     5
#define CUDNN_ACTIVATION_SWISH        6

#define CUDNN_SOFTMAX_FAST     0
#define CUDNN_SOFTMAX_ACCURATE 1
#define CUDNN_SOFTMAX_LOG      2

#define CUDNN_SOFTMAX_MODE_INSTANCE 0
#define CUDNN_SOFTMAX_MODE_CHANNEL  1

/* ── Internal types ──────────────────────────────────────────────── */

typedef struct {
    int   valid;
    void *stream;
} CudnnCtx;

typedef struct {
    int    mode;
    double coef;
} CudnnActDesc;

typedef struct {
    int n, c, h, w;
    int data_type;   /* 0=fp32 */
    int format;      /* 0=NCHW */
} CudnnTensorDesc;

/* ── Handle lifecycle ────────────────────────────────────────────── */

int cudnnCreate(void **handle) {
    if (!handle) return CUDNN_STATUS_BAD_PARAM;
    CudnnCtx *ctx = (CudnnCtx *)calloc(1, sizeof(CudnnCtx));
    if (!ctx) return CUDNN_STATUS_ALLOC_FAILED;
    ctx->valid = 1;
    *handle = ctx;
    return CUDNN_STATUS_SUCCESS;
}

int cudnnDestroy(void *handle) { free(handle); return CUDNN_STATUS_SUCCESS; }

int cudnnSetStream(void *handle, void *stream) {
    if (!handle) return CUDNN_STATUS_NOT_INITIALIZED;
    ((CudnnCtx *)handle)->stream = stream;
    return CUDNN_STATUS_SUCCESS;
}

int cudnnGetStream(void *handle, void **stream) {
    if (!handle || !stream) return CUDNN_STATUS_BAD_PARAM;
    *stream = ((CudnnCtx *)handle)->stream;
    return CUDNN_STATUS_SUCCESS;
}

/* ── Tensor descriptor ───────────────────────────────────────────── */

int cudnnCreateTensorDescriptor(void **d) {
    if (!d) return CUDNN_STATUS_BAD_PARAM;
    *d = calloc(1, sizeof(CudnnTensorDesc));
    return *d ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

int cudnnDestroyTensorDescriptor(void *d) { free(d); return CUDNN_STATUS_SUCCESS; }

int cudnnSetTensor4dDescriptor(void *d, int format, int dtype,
                                int n, int c, int h, int w)
{
    if (!d) return CUDNN_STATUS_BAD_PARAM;
    CudnnTensorDesc *t = (CudnnTensorDesc *)d;
    t->n = n; t->c = c; t->h = h; t->w = w;
    t->data_type = dtype; t->format = format;
    return CUDNN_STATUS_SUCCESS;
}

/* ── Activation descriptor ───────────────────────────────────────── */

int cudnnCreateActivationDescriptor(void **d) {
    if (!d) return CUDNN_STATUS_BAD_PARAM;
    *d = calloc(1, sizeof(CudnnActDesc));
    return *d ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

int cudnnDestroyActivationDescriptor(void *d) { free(d); return CUDNN_STATUS_SUCCESS; }

int cudnnSetActivationDescriptor(void *d, int mode, int nan_prop, double coef) {
    (void)nan_prop;
    if (!d) return CUDNN_STATUS_BAD_PARAM;
    ((CudnnActDesc *)d)->mode = mode;
    ((CudnnActDesc *)d)->coef = coef;
    return CUDNN_STATUS_SUCCESS;
}

/* ── Activation forward ──────────────────────────────────────────── */

int cudnnActivationForward(void *handle, void *act_desc,
                            const void *alpha, void *x_desc, const void *x,
                            const void *beta,  void *y_desc, void *y)
{
    (void)y_desc;
    if (!handle || !act_desc || !x || !y) return CUDNN_STATUS_BAD_PARAM;

    CudnnTensorDesc *td = (CudnnTensorDesc *)x_desc;
    int n = td ? td->n * td->c * td->h * td->w : 0;
    if (n == 0) n = 1024; /* fallback for NULL descriptor */

    float fa = alpha ? *(const float *)alpha : 1.0f;
    float fb = beta  ? *(const float *)beta  : 0.0f;

    const float *src = (const float *)x;
    float *dst = (float *)y;
    int mode = ((CudnnActDesc *)act_desc)->mode;

    for (int i = 0; i < n; i++) {
        float v = src[i], r;
        switch (mode) {
            case CUDNN_ACTIVATION_RELU:
                r = v > 0.0f ? v : 0.0f; break;
            case CUDNN_ACTIVATION_TANH:
                r = tanhf(v); break;
            case CUDNN_ACTIVATION_SIGMOID:
                r = 1.0f / (1.0f + expf(-v)); break;
            case CUDNN_ACTIVATION_SWISH:
                r = v / (1.0f + expf(-v)); break;
            case CUDNN_ACTIVATION_ELU: {
                double coef = ((CudnnActDesc *)act_desc)->coef;
                r = v >= 0.0f ? v : (float)(coef * (expf(v) - 1.0f));
                break;
            }
            default:
                r = v; /* IDENTITY */
        }
        dst[i] = fa * r + fb * dst[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

/* ── Softmax forward ─────────────────────────────────────────────── */

int cudnnSoftmaxForward(void *handle, int algo, int mode,
                         const void *alpha, void *x_desc, const void *x,
                         const void *beta,  void *y_desc, void *y)
{
    (void)alpha; (void)beta; (void)y_desc;
    if (!handle || !x || !y) return CUDNN_STATUS_BAD_PARAM;

    CudnnTensorDesc *td = (CudnnTensorDesc *)x_desc;
    int batch   = td ? td->n : 1;
    int classes = td ? td->c : 1024;
    int spatial = td ? td->h * td->w : 1;
    if (spatial == 0) spatial = 1;

    const float *src = (const float *)x;
    float *dst = (float *)y;

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatial; s++) {
            float max_v = src[b * classes * spatial + s];
            for (int c = 1; c < classes; c++) {
                float v = src[b * classes * spatial + c * spatial + s];
                if (v > max_v) max_v = v;
            }
            float sum = 0.0f;
            for (int c = 0; c < classes; c++) {
                float e = expf(src[b * classes * spatial + c * spatial + s] - max_v);
                dst[b * classes * spatial + c * spatial + s] = e;
                sum += e;
            }
            for (int c = 0; c < classes; c++)
                dst[b * classes * spatial + c * spatial + s] /= sum;
        }
    }
    (void)algo; (void)mode;
    return CUDNN_STATUS_SUCCESS;
}

/* ── Version ─────────────────────────────────────────────────────── */
size_t cudnnGetVersion(void)     { return 8700; /* 8.7.0 */ }
size_t cudnnGetCudartVersion(void) { return 12020; }
