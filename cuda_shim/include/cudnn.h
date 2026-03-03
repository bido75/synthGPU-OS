/* cudnn.h — SynthGPU cuDNN API Declarations
 * ===========================================
 * Implements the deep-learning primitive subset used by PyTorch.
 */

#pragma once
#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudnnContext *cudnnHandle_t;
typedef struct cudnnTensorStruct *cudnnTensorDescriptor_t;
typedef struct cudnnActivationStruct *cudnnActivationDescriptor_t;
typedef struct cudnnDropoutStruct *cudnnDropoutDescriptor_t;
typedef struct cudnnFilterStruct *cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct *cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct *cudnnPoolingDescriptor_t;

typedef enum cudnnStatus_t {
    CUDNN_STATUS_SUCCESS          = 0,
    CUDNN_STATUS_NOT_INITIALIZED  = 1,
    CUDNN_STATUS_ALLOC_FAILED     = 2,
    CUDNN_STATUS_BAD_PARAM        = 6,
    CUDNN_STATUS_INTERNAL_ERROR   = 8,
    CUDNN_STATUS_NOT_SUPPORTED    = 9,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 17,
} cudnnStatus_t;

typedef enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
} cudnnTensorFormat_t;

typedef enum cudnnDataType_t {
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF   = 2,
    CUDNN_DATA_INT8   = 3,
    CUDNN_DATA_INT32  = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8  = 6,
    CUDNN_DATA_BFLOAT16 = 9,
} cudnnDataType_t;

typedef enum cudnnActivationMode_t {
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU          = 4,
    CUDNN_ACTIVATION_IDENTITY     = 5,
    CUDNN_ACTIVATION_SWISH        = 6,
} cudnnActivationMode_t;

typedef enum cudnnNanPropagation_t {
    CUDNN_NOT_PROPAGATE_NAN = 0,
    CUDNN_PROPAGATE_NAN     = 1,
} cudnnNanPropagation_t;

typedef enum cudnnSoftmaxAlgorithm_t {
    CUDNN_SOFTMAX_FAST     = 0,
    CUDNN_SOFTMAX_ACCURATE = 1,
    CUDNN_SOFTMAX_LOG      = 2,
} cudnnSoftmaxAlgorithm_t;

typedef enum cudnnSoftmaxMode_t {
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,
    CUDNN_SOFTMAX_MODE_CHANNEL  = 1,
} cudnnSoftmaxMode_t;

/* ── Handle ─────────────────────────────────────────────────────── */
cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream);
cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *stream);

/* ── Tensor descriptor ──────────────────────────────────────────── */
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);
cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format, cudnnDataType_t dataType,
    int n, int c, int h, int w);

/* ── Activation ─────────────────────────────────────────────────── */
cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc);
cudnnStatus_t cudnnSetActivationDescriptor(
    cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt,
    double coef);
cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

/* ── Softmax ────────────────────────────────────────────────────── */
cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algorithm,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

/* ── Version ────────────────────────────────────────────────────── */
size_t cudnnGetVersion(void);
size_t cudnnGetCudartVersion(void);

#ifdef __cplusplus
}
#endif
