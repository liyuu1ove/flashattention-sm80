// shared struct
#pragma once

#include <cuda_runtime.h>

#ifndef XYDNNWINAPI
#ifdef _WIN32
#define XYDNNWINAPI __stdcall
#else
#define XYDNNWINAPI
#endif
#endif

struct xydnnContext;
typedef struct xydnnContext *xydnnHandle_t;

struct xydnnTensorStruct;
typedef struct xydnnTensorStruct *xydnnTensorDescriptor_t;

typedef enum {
    XYDNN_DATA_FLOAT              = 0,
    XYDNN_DATA_DOUBLE             = 1,
    XYDNN_DATA_HALF               = 2,
    XYDNN_DATA_INT8               = 3,
    XYDNN_DATA_INT32              = 4,
    XYDNN_DATA_INT8x4             = 5,
    XYDNN_DATA_UINT8              = 6,
    XYDNN_DATA_UINT8x4            = 7,
    XYDNN_DATA_INT8x32            = 8,
    XYDNN_DATA_BFLOAT16           = 9,
    XYDNN_DATA_INT64              = 10,
    XYDNN_DATA_BOOLEAN            = 11,
    XYDNN_DATA_FP8_E4M3           = 12,
    XYDNN_DATA_FP8_E5M2           = 13,
    XYDNN_DATA_FAST_FLOAT_FOR_FP8 = 14,
} xydnnDataType_t;

typedef enum {
    XYDNN_DEFAULT_MATH                    = 0,
    XYDNN_TENSOR_OP_MATH                  = 1,
    XYDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    XYDNN_FMA_MATH                        = 3,
} xydnnMathType_t;


/*
 * XYDNN return codes
 */
typedef enum {
    XYDNN_STATUS_SUCCESS                      = 0,
    XYDNN_STATUS_NOT_INITIALIZED              = 1,
    XYDNN_STATUS_ALLOC_FAILED                 = 2,
    XYDNN_STATUS_BAD_PARAM                    = 3,
    XYDNN_STATUS_INTERNAL_ERROR               = 4,
    XYDNN_STATUS_INVALID_VALUE                = 5,
    XYDNN_STATUS_ARCH_MISMATCH                = 6,
    XYDNN_STATUS_MAPPING_ERROR                = 7,
    XYDNN_STATUS_EXECUTION_FAILED             = 8,
    XYDNN_STATUS_NOT_SUPPORTED                = 9,
    XYDNN_STATUS_LICENSE_ERROR                = 10,
    XYDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    XYDNN_STATUS_RUNTIME_IN_PROGRESS          = 12,
    XYDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13,
    XYDNN_STATUS_VERSION_MISMATCH             = 14,
} xydnnStatus_t;

typedef struct xydnnDropoutStruct *xydnnDropoutDescriptor_t;

extern "C" {

xydnnStatus_t XYDNNWINAPI
xydnnCreate(xydnnHandle_t *handle);

xydnnStatus_t XYDNNWINAPI
xydnnDestroy(xydnnHandle_t handle);

xydnnStatus_t XYDNNWINAPI
xydnnSetStream(xydnnHandle_t handle, cudaStream_t streamId);

xydnnStatus_t XYDNNWINAPI
xydnnGetStream(xydnnHandle_t handle, cudaStream_t *streamId);

xydnnStatus_t XYDNNWINAPI
xydnnCreateDropoutDescriptor(xydnnDropoutDescriptor_t *dropoutDesc);

xydnnStatus_t XYDNNWINAPI
xydnnDestroyDropoutDescriptor(xydnnDropoutDescriptor_t dropoutDesc);

xydnnStatus_t XYDNNWINAPI
xydnnDropoutGetStatesSize(xydnnHandle_t handle, size_t *sizeInBytes);

xydnnStatus_t XYDNNWINAPI
xydnnSetDropoutDescriptor(xydnnDropoutDescriptor_t dropoutDesc,
                          xydnnHandle_t handle,
                          float dropout,
                          void *states,
                          size_t stateSizeInBytes,
                          unsigned long long seed);

xydnnStatus_t XYDNNWINAPI
xydnnGetDropoutDescriptor(xydnnDropoutDescriptor_t dropoutDesc,
                          xydnnHandle_t handle,
                          float *dropout,
                          void **states,
                          unsigned long long *seed);

}
