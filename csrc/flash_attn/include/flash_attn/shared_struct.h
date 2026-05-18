// shared struct

typedef enum {
    CUDNN_DATA_FLOAT              = 0,
    CUDNN_DATA_DOUBLE             = 1,
    CUDNN_DATA_HALF               = 2,
    CUDNN_DATA_INT8               = 3,
    CUDNN_DATA_INT32              = 4,
    CUDNN_DATA_INT8x4             = 5,
    CUDNN_DATA_UINT8              = 6,
    CUDNN_DATA_UINT8x4            = 7,
    CUDNN_DATA_INT8x32            = 8,
    CUDNN_DATA_BFLOAT16           = 9,
    CUDNN_DATA_INT64              = 10,
    CUDNN_DATA_BOOLEAN            = 11,
    CUDNN_DATA_FP8_E4M3           = 12,
    CUDNN_DATA_FP8_E5M2           = 13,
    CUDNN_DATA_FAST_FLOAT_FOR_FP8 = 14,
} cudnnDataType_t;

typedef enum {
    CUDNN_DEFAULT_MATH                    = 0,
    CUDNN_TENSOR_OP_MATH                  = 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    CUDNN_FMA_MATH                        = 3,
} cudnnMathType_t;


/*
 * CUDNN return codes
 */
typedef enum {
    CUDNN_STATUS_SUCCESS                      = 0,
    CUDNN_STATUS_NOT_INITIALIZED              = 1,
    CUDNN_STATUS_ALLOC_FAILED                 = 2,
    CUDNN_STATUS_BAD_PARAM                    = 3,
    CUDNN_STATUS_INTERNAL_ERROR               = 4,
    CUDNN_STATUS_INVALID_VALUE                = 5,
    CUDNN_STATUS_ARCH_MISMATCH                = 6,
    CUDNN_STATUS_MAPPING_ERROR                = 7,
    CUDNN_STATUS_EXECUTION_FAILED             = 8,
    CUDNN_STATUS_NOT_SUPPORTED                = 9,
    CUDNN_STATUS_LICENSE_ERROR                = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS          = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13,
    CUDNN_STATUS_VERSION_MISMATCH             = 14,
} cudnnStatus_t;

typedef struct cudnnDropoutStruct *cudnnDropoutDescriptor_t;