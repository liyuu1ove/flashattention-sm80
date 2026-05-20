#include "flash_attn/xydnn_mha_fwd_api.h"

#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"

#ifndef FLASHATTENTION_DISABLE_DROPOUT
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cutlass/numeric_types.h>

struct xydnnContext
{
    cudaStream_t stream;
    cublasHandle_t cublas;
};

struct xydnnSeqDataStruct
{
    xydnnDataType_t dataType;
    int nbDims;
    int dimA[XYDNN_SEQDATA_DIM_COUNT];
    xydnnSeqDataAxis_t axes[XYDNN_SEQDATA_DIM_COUNT];
    size_t seqLengthArraySize;
    int *seqLengthArray;
    void *paddingFill;
};

struct xydnnTensorStruct
{
    xydnnDataType_t dataType;
    int nbDims;
    int dimA[4];
    int strideA[4];
};

struct xydnnDropoutStruct
{
    float dropout;
    void *states;
    size_t stateSizeInBytes;
    unsigned long long seed;
};

struct xydnnAttnStruct
{
    unsigned attnMode;
    int nHeads;
    double smScaler;
    xydnnDataType_t dataType;
    xydnnDataType_t computePrec;
    xydnnMathType_t mathType;
    xydnnDropoutDescriptor_t attnDropoutDesc;
    xydnnDropoutDescriptor_t postDropoutDesc;
    int qSize;
    int kSize;
    int vSize;
    int qProjSize;
    int kProjSize;
    int vProjSize;
    int oProjSize;
    int qoMaxSeqLength;
    int kvMaxSeqLength;
    int maxBatchSize;
    int maxBeamSize;
    int isCausal;
    int windowSizeLeft;
    int windowSizeRight;
    float softcap;
    const void *alibiSlopes;
    int alibiSlopesBatchStride;
    const int *cuSeqlensQO;
    const int *cuSeqlensKV;
    int totalQO;
    int totalKV;
};

namespace {

constexpr int kSeqDataDims = XYDNN_SEQDATA_DIM_COUNT;
constexpr size_t kDropoutRngStateBytes = 2 * sizeof(uint64_t);

int round_multiple(int value, int multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

size_t data_type_size(xydnnDataType_t dataType)
{
    switch (dataType) {
    case XYDNN_DATA_HALF:
    case XYDNN_DATA_BFLOAT16:
        return 2;
    case XYDNN_DATA_FLOAT:
        return 4;
    default:
        return 0;
    }
}

cublasComputeType_t compute_type_for(xydnnDataType_t dataType)
{
    switch (dataType) {
    case XYDNN_DATA_HALF:
        return CUBLAS_COMPUTE_32F_FAST_16F;
    case XYDNN_DATA_BFLOAT16:
        return CUBLAS_COMPUTE_32F_FAST_16BF;
    case XYDNN_DATA_FLOAT:
        return CUBLAS_COMPUTE_32F;
    default:
        return CUBLAS_COMPUTE_32F;
    }
}

xydnnStatus_t tensor_desc_for_weight(const xydnnAttnStruct *attnDesc,
                                    xydnnMultiHeadAttnWeightKind_t wKind,
                                    int *nbDims,
                                    int dimA[4],
                                    int strideA[4],
                                    size_t *elem_count)
{
    if (attnDesc == nullptr || nbDims == nullptr || dimA == nullptr || strideA == nullptr || elem_count == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    const bool enable_bias = (attnDesc->attnMode & XYDNN_ATTN_ENABLE_PROJ_BIASES) != 0;
    const int in_dim_q = attnDesc->qSize;
    const int in_dim_k = attnDesc->kSize;
    const int in_dim_v = attnDesc->vSize;
    const int proj_q = attnDesc->qProjSize > 0 ? attnDesc->qProjSize : attnDesc->qSize;
    const int proj_k = attnDesc->kProjSize > 0 ? attnDesc->kProjSize : attnDesc->kSize;
    const int proj_v = attnDesc->vProjSize > 0 ? attnDesc->vProjSize : attnDesc->vSize;
    const int proj_o = attnDesc->oProjSize > 0 ? attnDesc->oProjSize : attnDesc->qSize;
    int out_dim = 0;
    int in_dim = 0;
    bool is_bias = false;
    switch (wKind) {
    case XYDNN_MH_ATTN_Q_WEIGHTS:
        out_dim = proj_q;
        in_dim = in_dim_q;
        break;
    case XYDNN_MH_ATTN_K_WEIGHTS:
        out_dim = proj_k;
        in_dim = in_dim_k;
        break;
    case XYDNN_MH_ATTN_V_WEIGHTS:
        out_dim = proj_v;
        in_dim = in_dim_v;
        break;
    case XYDNN_MH_ATTN_O_WEIGHTS:
        out_dim = proj_o;
        in_dim = proj_q;
        break;
    case XYDNN_MH_ATTN_Q_BIASES:
        out_dim = proj_q;
        is_bias = true;
        break;
    case XYDNN_MH_ATTN_K_BIASES:
        out_dim = proj_k;
        is_bias = true;
        break;
    case XYDNN_MH_ATTN_V_BIASES:
        out_dim = proj_v;
        is_bias = true;
        break;
    case XYDNN_MH_ATTN_O_BIASES:
        out_dim = proj_o;
        is_bias = true;
        break;
    default:
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (is_bias) {
        if (!enable_bias) {
            return XYDNN_STATUS_NOT_SUPPORTED;
        }
        dimA[0] = out_dim;
        strideA[0] = 1;
        *nbDims = 1;
        *elem_count = static_cast<size_t>(out_dim);
        return XYDNN_STATUS_SUCCESS;
    }
    dimA[0] = out_dim;
    dimA[1] = in_dim;
    strideA[0] = 1;
    strideA[1] = out_dim;
    *nbDims = 2;
    *elem_count = static_cast<size_t>(out_dim) * static_cast<size_t>(in_dim);
    return XYDNN_STATUS_SUCCESS;
}

size_t count_weight_elems(const xydnnAttnStruct *attnDesc, xydnnMultiHeadAttnWeightKind_t wKind)
{
    int nbDims = 0;
    int dimA[4] = {};
    int strideA[4] = {};
    size_t elems = 0;
    if (tensor_desc_for_weight(attnDesc, wKind, &nbDims, dimA, strideA, &elems) != XYDNN_STATUS_SUCCESS) {
        return 0;
    }
    return elems;
}

size_t weight_buffer_bytes(const xydnnAttnStruct *attnDesc)
{
    if (attnDesc == nullptr) {
        return 0;
    }
    const size_t elem_size = data_type_size(attnDesc->dataType);
    size_t total_elems = 0;
    for (int kind = XYDNN_MH_ATTN_Q_WEIGHTS; kind <= XYDNN_MH_ATTN_O_WEIGHTS; ++kind) {
        total_elems += count_weight_elems(attnDesc, static_cast<xydnnMultiHeadAttnWeightKind_t>(kind));
    }
    if ((attnDesc->attnMode & XYDNN_ATTN_ENABLE_PROJ_BIASES) != 0) {
        for (int kind = XYDNN_MH_ATTN_Q_BIASES; kind <= XYDNN_MH_ATTN_O_BIASES; ++kind) {
            total_elems += count_weight_elems(attnDesc, static_cast<xydnnMultiHeadAttnWeightKind_t>(kind));
        }
    }
    return total_elems * elem_size;
}

size_t weight_offset_bytes(const xydnnAttnStruct *attnDesc, xydnnMultiHeadAttnWeightKind_t wKind)
{
    size_t offset_elems = 0;
    for (int kind = XYDNN_MH_ATTN_Q_WEIGHTS; kind < static_cast<int>(wKind); ++kind) {
        offset_elems += count_weight_elems(attnDesc, static_cast<xydnnMultiHeadAttnWeightKind_t>(kind));
    }
    return offset_elems * data_type_size(attnDesc->dataType);
}

size_t bias_offset_bytes(const xydnnAttnStruct *attnDesc, xydnnMultiHeadAttnWeightKind_t wKind)
{
    const size_t weight_bytes = weight_buffer_bytes(attnDesc);
    size_t offset_elems = 0;
    for (int kind = XYDNN_MH_ATTN_Q_BIASES; kind < static_cast<int>(wKind); ++kind) {
        offset_elems += count_weight_elems(attnDesc, static_cast<xydnnMultiHeadAttnWeightKind_t>(kind));
    }
    const size_t weight_elems = 0;
    return weight_bytes + offset_elems * data_type_size(attnDesc->dataType);
}

cublasStatus_t cublas_gemm_ex(cublasHandle_t handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int m,
                              int n,
                              int k,
                              const void *alpha,
                              const void *A,
                              cudaDataType_t Atype,
                              int lda,
                              const void *B,
                              cudaDataType_t Btype,
                              int ldb,
                              const void *beta,
                              void *C,
                              cudaDataType_t Ctype,
                              int ldc,
                              cublasComputeType_t computeType)
{
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        Atype,
                        lda,
                        B,
                        Btype,
                        ldb,
                        beta,
                        C,
                        Ctype,
                        ldc,
                        computeType,
                        CUBLAS_GEMM_DEFAULT);
}

size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

size_t seqlens_workspace_bytes(int batch_size)
{
    return 2 * static_cast<size_t>(batch_size + 1) * sizeof(int);
}

size_t projection_workspace_bytes(const xydnnAttnStruct *attnDesc, size_t q_rows, size_t k_rows)
{
    if (attnDesc == nullptr) {
        return 0;
    }
    const size_t elem_size = data_type_size(attnDesc->dataType);
    const int proj_q = attnDesc->qProjSize > 0 ? attnDesc->qProjSize : attnDesc->qSize;
    const int proj_k = attnDesc->kProjSize > 0 ? attnDesc->kProjSize : attnDesc->kSize;
    const int proj_v = attnDesc->vProjSize > 0 ? attnDesc->vProjSize : attnDesc->vSize;
    const int proj_o = attnDesc->oProjSize > 0 ? attnDesc->oProjSize : attnDesc->qSize;
    size_t bytes = 0;
    bytes = align_up(bytes, 256);
    bytes += q_rows * static_cast<size_t>(proj_q) * elem_size;
    bytes = align_up(bytes, 256);
    bytes += k_rows * static_cast<size_t>(proj_k) * elem_size;
    bytes = align_up(bytes, 256);
    bytes += k_rows * static_cast<size_t>(proj_v) * elem_size;
    bytes = align_up(bytes, 256);
    bytes += q_rows * static_cast<size_t>(proj_o) * elem_size;
    return bytes;
}

cudaDataType_t to_cuda_dtype(xydnnDataType_t dataType)
{
    switch (dataType) {
    case XYDNN_DATA_HALF:
        return CUDA_R_16F;
    case XYDNN_DATA_BFLOAT16:
        return CUDA_R_16BF;
    case XYDNN_DATA_FLOAT:
        return CUDA_R_32F;
    default:
        return CUDA_R_32F;
    }
}

template <typename T>
struct ScalarTraits;

template <>
struct ScalarTraits<float>
{
    static constexpr float one = 1.0f;
    static constexpr float zero = 0.0f;
};

template <>
struct ScalarTraits<double>
{
    static constexpr double one = 1.0;
    static constexpr double zero = 0.0;
};

int dim_for_axis(const xydnnSeqDataStruct *desc, xydnnSeqDataAxis_t axis)
{
    for (int i = 0; i < desc->nbDims; ++i) {
        if (desc->axes[i] == axis) {
            return desc->dimA[i];
        }
    }
    return 0;
}

xydnnStatus_t get_stream(xydnnHandle_t handle, cudaStream_t *stream)
{
    if (stream == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (handle == nullptr) {
        *stream = nullptr;
        return XYDNN_STATUS_SUCCESS;
    }
    *stream = handle->stream;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t build_cu_seqlens_host(const xydnnSeqDataStruct *desc,
                                    int batch_size,
                                    int max_seqlen,
                                    std::vector<int> &cu_seqlens,
                                    int *total)
{
    if (desc == nullptr || total == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (desc->seqLengthArray == nullptr || desc->seqLengthArraySize == 0) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
    cu_seqlens.assign(batch_size + 1, 0);
    if (desc->seqLengthArraySize == static_cast<size_t>(batch_size)) {
        int running = 0;
        for (int i = 0; i < batch_size; ++i) {
            const int len = desc->seqLengthArray[i];
            if (len <= 0 || len > max_seqlen) {
                return XYDNN_STATUS_BAD_PARAM;
            }
            cu_seqlens[i] = running;
            running += len;
        }
        cu_seqlens[batch_size] = running;
        *total = running;
        return XYDNN_STATUS_SUCCESS;
    }
    if (desc->seqLengthArraySize == static_cast<size_t>(batch_size + 1)) {
        if (desc->seqLengthArray[0] != 0) {
            return XYDNN_STATUS_BAD_PARAM;
        }
        for (int i = 0; i < batch_size; ++i) {
            const int begin = desc->seqLengthArray[i];
            const int end = desc->seqLengthArray[i + 1];
            if (end <= begin || end - begin > max_seqlen) {
                return XYDNN_STATUS_BAD_PARAM;
            }
            cu_seqlens[i] = begin;
        }
        cu_seqlens[batch_size] = desc->seqLengthArray[batch_size];
        *total = cu_seqlens[batch_size];
        return XYDNN_STATUS_SUCCESS;
    }
    return XYDNN_STATUS_BAD_PARAM;
}

void set_params_fprop_ptr(FLASH_NAMESPACE::Flash_fwd_params &params,
                          int batch_size,
                          int seqlen_q,
                          int seqlen_k,
                          int num_heads,
                          int num_heads_k,
                          int head_size,
                          const void *q,
                          const void *k,
                          const void *v,
                          void *out,
                          void *softmax_lse,
                          xydnnDataType_t dataType,
                          float softmax_scale,
                          bool is_causal,
                          int window_size_left,
                          int window_size_right,
                          float softcap,
                          const void *alibi_slopes,
                          int alibi_slopes_batch_stride,
                          const int *cu_seqlens_q,
                          const int *cu_seqlens_k,
                          int total_q,
                          float dropout)
{
    params = {};

    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    params.is_bf16 = dataType == XYDNN_DATA_BFLOAT16;

    params.q_ptr = const_cast<void *>(q);
    params.k_ptr = const_cast<void *>(k);
    params.v_ptr = const_cast<void *>(v);
    params.o_ptr = out;

    // XYDNN SeqData is T, N, B, V for regular input. For varlen input, pointers
    // are packed as [total_tokens, head, head_dim] and cu_seqlens carries batch offsets.
    params.q_row_stride = num_heads * head_size;
    params.k_row_stride = num_heads_k * head_size;
    params.v_row_stride = num_heads_k * head_size;
    params.o_row_stride = num_heads * head_size;
    params.q_head_stride = head_size;
    params.k_head_stride = head_size;
    params.v_head_stride = head_size;
    params.o_head_stride = head_size;
    if (cu_seqlens_q == nullptr) {
        params.q_batch_stride = seqlen_q * num_heads * head_size;
        params.k_batch_stride = seqlen_k * num_heads_k * head_size;
        params.v_batch_stride = seqlen_k * num_heads_k * head_size;
        params.o_batch_stride = seqlen_q * num_heads * head_size;
    }

    params.softmax_lse_ptr = softmax_lse;
    params.p_ptr = nullptr;

    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(seqlen_k, 128);
    params.d = head_size;
    params.d_rounded = head_size_rounded;
    params.total_q = total_q;

    if (softcap > 0.0f) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else {
        params.softcap = 0.0f;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.p_dropout = 1.0f - dropout;
    params.p_dropout_in_uint8_t = static_cast<uint8_t>(std::floor(params.p_dropout * 255.0f));
    params.rp_dropout = 1.0f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.is_causal = is_causal;
    params.cu_seqlens_q = const_cast<int *>(cu_seqlens_q);
    params.cu_seqlens_k = const_cast<int *>(cu_seqlens_k);
    params.seqused_k = nullptr;
    params.leftpad_k = nullptr;
    params.is_seqlens_k_cumulative = true;
    params.alibi_slopes_ptr = const_cast<void *>(alibi_slopes);
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;
    params.unpadded_lse = cu_seqlens_q != nullptr;
    params.seqlenq_ngroups_swapped = false;
}

void run_mha_fwd_ptr(FLASH_NAMESPACE::Flash_fwd_params &params, cudaStream_t stream)
{
    FP16_SWITCH(!params.is_bf16, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            switch (params.d) {
            case 32:
                FLASH_NAMESPACE::run_mha_fwd_<elem_type, 32, Is_causal>(params, stream);
                break;
            case 64:
                FLASH_NAMESPACE::run_mha_fwd_<elem_type, 64, Is_causal>(params, stream);
                break;
            case 96:
                FLASH_NAMESPACE::run_mha_fwd_<elem_type, 96, Is_causal>(params, stream);
                break;
            case 128:
                FLASH_NAMESPACE::run_mha_fwd_<elem_type, 128, Is_causal>(params, stream);
                break;
            case 192:
                FLASH_NAMESPACE::run_mha_fwd_<elem_type, 192, Is_causal>(params, stream);
                break;
            case 256:
                FLASH_NAMESPACE::run_mha_fwd_<elem_type, 256, Is_causal>(params, stream);
                break;
            default:
                break;
            }
        });
    });
}

template <typename Elem>
xydnnStatus_t launch_linear_projection(cublasHandle_t handle,
                                       const Elem *x,
                                       const Elem *w,
                                       Elem *y,
                                       int rows,
                                       int in_dim,
                                       int out_dim,
                                       xydnnDataType_t dataType)
{
    if (rows <= 0 || in_dim <= 0 || out_dim <= 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    const auto alpha = ScalarTraits<float>::one;
    const auto beta = ScalarTraits<float>::zero;
    const cudaDataType_t cuda_dtype = to_cuda_dtype(dataType);
    const cublasComputeType_t compute_type = compute_type_for(dataType);
    cublasStatus_t status = cublas_gemm_ex(handle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_T,
                                           out_dim,
                                           rows,
                                           in_dim,
                                           &alpha,
                                           w,
                                           cuda_dtype,
                                           in_dim,
                                           x,
                                           cuda_dtype,
                                           in_dim,
                                           &beta,
                                           y,
                                           cuda_dtype,
                                           out_dim,
                                           compute_type);
    return status == CUBLAS_STATUS_SUCCESS ? XYDNN_STATUS_SUCCESS : XYDNN_STATUS_EXECUTION_FAILED;
}

xydnnStatus_t launch_linear_projection(cublasHandle_t handle,
                                       const void *x,
                                       const void *w,
                                       void *y,
                                       int rows,
                                       int in_dim,
                                       int out_dim,
                                       xydnnDataType_t dataType)
{
    if (rows <= 0 || in_dim <= 0 || out_dim <= 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudaDataType_t cuda_dtype = to_cuda_dtype(dataType);
    const cublasComputeType_t compute_type = compute_type_for(dataType);
    const cublasStatus_t status = cublas_gemm_ex(handle,
                                                 CUBLAS_OP_T,
                                                 CUBLAS_OP_T,
                                                 out_dim,
                                                 rows,
                                                 in_dim,
                                                 &alpha,
                                                 w,
                                                 cuda_dtype,
                                                 in_dim,
                                                 x,
                                                 cuda_dtype,
                                                 in_dim,
                                                 &beta,
                                                 y,
                                                 cuda_dtype,
                                                 out_dim,
                                                 compute_type);
    return status == CUBLAS_STATUS_SUCCESS ? XYDNN_STATUS_SUCCESS : XYDNN_STATUS_EXECUTION_FAILED;
}

}  // namespace

extern "C" {

xydnnStatus_t
xydnnCreate(xydnnHandle_t *handle)
{
    if (handle == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    auto *ptr = static_cast<xydnnContext *>(malloc(sizeof(xydnnContext)));
    if (ptr == nullptr) {
        return XYDNN_STATUS_ALLOC_FAILED;
    }
    ptr->stream = nullptr;
    ptr->cublas = nullptr;
    if (cublasCreate(&ptr->cublas) != CUBLAS_STATUS_SUCCESS) {
        free(ptr);
        return XYDNN_STATUS_INTERNAL_ERROR;
    }
    *handle = ptr;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnDestroy(xydnnHandle_t handle)
{
    if (handle != nullptr) {
        if (handle->cublas != nullptr) {
            cublasDestroy(handle->cublas);
        }
        free(handle);
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnSetStream(xydnnHandle_t handle, cudaStream_t streamId)
{
    if (handle == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    handle->stream = streamId;
    if (handle->cublas != nullptr) {
        cublasStatus_t st = cublasSetStream(handle->cublas, streamId);
        if (st != CUBLAS_STATUS_SUCCESS) {
            return XYDNN_STATUS_INTERNAL_ERROR;
        }
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnGetStream(xydnnHandle_t handle, cudaStream_t *streamId)
{
    if (handle == nullptr || streamId == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    *streamId = handle->stream;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnCreateDropoutDescriptor(xydnnDropoutDescriptor_t *dropoutDesc)
{
    if (dropoutDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    auto *ptr = static_cast<xydnnDropoutStruct *>(std::calloc(1, sizeof(xydnnDropoutStruct)));
    if (ptr == nullptr) {
        return XYDNN_STATUS_ALLOC_FAILED;
    }
    *dropoutDesc = ptr;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnDestroyDropoutDescriptor(xydnnDropoutDescriptor_t dropoutDesc)
{
    if (dropoutDesc != nullptr) {
        std::free(dropoutDesc);
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnDropoutGetStatesSize(xydnnHandle_t handle, size_t *sizeInBytes)
{
    (void)handle;
    if (sizeInBytes == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    *sizeInBytes = 0;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnSetDropoutDescriptor(xydnnDropoutDescriptor_t dropoutDesc,
                          xydnnHandle_t handle,
                          float dropout,
                          void *states,
                          size_t stateSizeInBytes,
                          unsigned long long seed)
{
    (void)handle;
    if (dropoutDesc == nullptr || dropout < 0.0f || dropout >= 1.0f) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    dropoutDesc->dropout = dropout;
    dropoutDesc->states = states;
    dropoutDesc->stateSizeInBytes = stateSizeInBytes;
    dropoutDesc->seed = seed;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnGetDropoutDescriptor(xydnnDropoutDescriptor_t dropoutDesc,
                          xydnnHandle_t handle,
                          float *dropout,
                          void **states,
                          unsigned long long *seed)
{
    (void)handle;
    if (dropoutDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (dropout) {
        *dropout = dropoutDesc->dropout;
    }
    if (states) {
        *states = dropoutDesc->states;
    }
    if (seed) {
        *seed = dropoutDesc->seed;
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t XYDNNWINAPI
xydnnCreateTensorDescriptor(xydnnTensorDescriptor_t *tensorDesc)
{
    if (tensorDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    auto *ptr = static_cast<xydnnTensorStruct *>(malloc(sizeof(xydnnTensorStruct)));
    if (ptr == nullptr) {
        return XYDNN_STATUS_ALLOC_FAILED;
    }
    memset(ptr, 0, sizeof(xydnnTensorStruct));
    *tensorDesc = ptr;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t XYDNNWINAPI
xydnnDestroyTensorDescriptor(xydnnTensorDescriptor_t tensorDesc)
{
    if (tensorDesc != nullptr) {
        free(tensorDesc);
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t XYDNNWINAPI
xydnnSetTensorNdDescriptor(xydnnTensorDescriptor_t tensorDesc,
                           xydnnDataType_t dataType,
                           int nbDims,
                           const int dimA[],
                           const int strideA[])
{
    if (tensorDesc == nullptr || dimA == nullptr || strideA == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (nbDims <= 0 || nbDims > 4 || data_type_size(dataType) == 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    tensorDesc->dataType = dataType;
    tensorDesc->nbDims = nbDims;
    for (int i = 0; i < nbDims; ++i) {
        if (dimA[i] <= 0 || strideA[i] <= 0) {
            return XYDNN_STATUS_BAD_PARAM;
        }
        tensorDesc->dimA[i] = dimA[i];
        tensorDesc->strideA[i] = strideA[i];
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t XYDNNWINAPI
xydnnGetTensorNdDescriptor(const xydnnTensorDescriptor_t tensorDesc,
                           int nbDimsRequested,
                           xydnnDataType_t *dataType,
                           int *nbDims,
                           int dimA[],
                           int strideA[])
{
    if (tensorDesc == nullptr || nbDimsRequested < 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (dataType) {
        *dataType = tensorDesc->dataType;
    }
    if (nbDims) {
        *nbDims = tensorDesc->nbDims;
    }
    const int copy_dims = std::min(nbDimsRequested, tensorDesc->nbDims);
    if (dimA) {
        memcpy(dimA, tensorDesc->dimA, sizeof(int) * copy_dims);
    }
    if (strideA) {
        memcpy(strideA, tensorDesc->strideA, sizeof(int) * copy_dims);
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t XYDNNWINAPI
xydnnCreateSeqDataDescriptor(xydnnSeqDataDescriptor_t *seqDataDesc)
{
    if (seqDataDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    auto *ptr = static_cast<xydnnSeqDataStruct *>(malloc(sizeof(xydnnSeqDataStruct)));
    if (ptr == nullptr) {
        return XYDNN_STATUS_ALLOC_FAILED;
    }
    memset(ptr, 0, sizeof(xydnnSeqDataStruct));
    ptr->nbDims = kSeqDataDims;
    ptr->axes[0] = XYDNN_SEQDATA_TIME_DIM;
    ptr->axes[1] = XYDNN_SEQDATA_BATCH_DIM;
    ptr->axes[2] = XYDNN_SEQDATA_BEAM_DIM;
    ptr->axes[3] = XYDNN_SEQDATA_VECT_DIM;
    ptr->seqLengthArray = nullptr;
    *seqDataDesc = ptr;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t XYDNNWINAPI
xydnnDestroySeqDataDescriptor(xydnnSeqDataDescriptor_t seqDataDesc)
{
    if (seqDataDesc != nullptr) {
        free(seqDataDesc->seqLengthArray);
        free(seqDataDesc);
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t XYDNNWINAPI
xydnnSetSeqDataDescriptor(xydnnSeqDataDescriptor_t seqDataDesc,
                          xydnnDataType_t dataType,
                          int nbDims,
                          const int dimA[],
                          const xydnnSeqDataAxis_t axes[],
                          size_t seqLengthArraySize,
                          const int seqLengthArray[],
                          void *paddingFill)
{
    if (seqDataDesc == nullptr || dimA == nullptr || axes == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (nbDims != kSeqDataDims) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
    if (data_type_size(dataType) == 0) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }

    bool seen[kSeqDataDims] = {};
    for (int i = 0; i < nbDims; ++i) {
        if (dimA[i] <= 0 || axes[i] < XYDNN_SEQDATA_TIME_DIM || axes[i] > XYDNN_SEQDATA_VECT_DIM) {
            return XYDNN_STATUS_BAD_PARAM;
        }
        if (seen[axes[i]]) {
            return XYDNN_STATUS_BAD_PARAM;
        }
        seen[axes[i]] = true;
    }

    seqDataDesc->dataType = dataType;
    seqDataDesc->nbDims = nbDims;
    memcpy(seqDataDesc->dimA, dimA, sizeof(int) * nbDims);
    memcpy(seqDataDesc->axes, axes, sizeof(xydnnSeqDataAxis_t) * nbDims);
    seqDataDesc->seqLengthArraySize = seqLengthArraySize;
    free(seqDataDesc->seqLengthArray);
    seqDataDesc->seqLengthArray = nullptr;
    if (seqLengthArraySize > 0) {
        if (seqLengthArray == nullptr) {
            return XYDNN_STATUS_BAD_PARAM;
        }
        seqDataDesc->seqLengthArray = static_cast<int *>(malloc(sizeof(int) * seqLengthArraySize));
        if (seqDataDesc->seqLengthArray == nullptr) {
            return XYDNN_STATUS_ALLOC_FAILED;
        }
        memcpy(seqDataDesc->seqLengthArray, seqLengthArray, sizeof(int) * seqLengthArraySize);
    }
    seqDataDesc->paddingFill = paddingFill;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t XYDNNWINAPI
xydnnGetSeqDataDescriptor(const xydnnSeqDataDescriptor_t seqDataDesc,
                          xydnnDataType_t *dataType,
                          int *nbDims,
                          int nbDimsRequested,
                          int dimA[],
                          xydnnSeqDataAxis_t axes[],
                          size_t *seqLengthArraySize,
                          size_t seqLengthSizeRequested,
                          int seqLengthArray[],
                          void *paddingFill)
{
    if (seqDataDesc == nullptr || nbDimsRequested < 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (dataType) {
        *dataType = seqDataDesc->dataType;
    }
    if (nbDims) {
        *nbDims = seqDataDesc->nbDims;
    }
    if (dimA) {
        memcpy(dimA, seqDataDesc->dimA, sizeof(int) * std::min(nbDimsRequested, seqDataDesc->nbDims));
    }
    if (axes) {
        memcpy(axes, seqDataDesc->axes, sizeof(xydnnSeqDataAxis_t) * std::min(nbDimsRequested, seqDataDesc->nbDims));
    }
    if (seqLengthArraySize) {
        *seqLengthArraySize = seqDataDesc->seqLengthArraySize;
    }
    const size_t seq_length_copy_count = std::min(seqLengthSizeRequested, seqDataDesc->seqLengthArraySize);
    if (seqLengthArray && seq_length_copy_count > 0) {
        memcpy(seqLengthArray,
               seqDataDesc->seqLengthArray,
               sizeof(int) * seq_length_copy_count);
    }
    if (paddingFill) {
        memcpy(paddingFill, &seqDataDesc->paddingFill, sizeof(void *));
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnCreateAttnDescriptor(xydnnAttnDescriptor_t *attnDesc)
{
    if (attnDesc == NULL) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    struct xydnnAttnStruct *ptr = (struct xydnnAttnStruct *)malloc(sizeof(struct xydnnAttnStruct));
    
    if (ptr == NULL) {
        return XYDNN_STATUS_ALLOC_FAILED;
    }

    // Initialize all fields to 0 / NULL. 
    memset(ptr, 0, sizeof(struct xydnnAttnStruct));
    ptr->smScaler = 1.0;
    ptr->isCausal = 0;
    ptr->windowSizeLeft = -1;
    ptr->windowSizeRight = -1;
    ptr->softcap = 0.0f;
    ptr->alibiSlopes = nullptr;
    ptr->alibiSlopesBatchStride = 0;
    ptr->cuSeqlensQO = nullptr;
    ptr->cuSeqlensKV = nullptr;
    ptr->totalQO = 0;
    ptr->totalKV = 0;
    *attnDesc = ptr;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnDestroyAttnDescriptor(xydnnAttnDescriptor_t attnDesc){
    if (attnDesc == NULL) {
        return XYDNN_STATUS_SUCCESS;
    }
    free(attnDesc);
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnSetAttnDescriptor(xydnnAttnDescriptor_t attnDesc,
                       unsigned attnMode,
                       int nHeads,
                       double smScaler,
                       xydnnDataType_t dataType,
                       xydnnDataType_t computePrec,
                       xydnnMathType_t mathType,
                       xydnnDropoutDescriptor_t attnDropoutDesc,
                       xydnnDropoutDescriptor_t postDropoutDesc,
                       int qSize,
                       int kSize,
                       int vSize,
                       int qProjSize,
                       int kProjSize,
                       int vProjSize,
                       int oProjSize,
                       int qoMaxSeqLength,
                       int kvMaxSeqLength,
                       int maxBatchSize,
                       int maxBeamSize)
{
    if (attnDesc == NULL) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (nHeads <= 0 || maxBatchSize <= 0 || maxBeamSize <= 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }

    if (qSize <= 0 || kSize <= 0 || vSize <= 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (qProjSize < 0 || kProjSize < 0 || vProjSize < 0 || oProjSize < 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if ((qProjSize != 0 && qProjSize != qSize) ||
        (kProjSize != 0 && kProjSize != kSize) ||
        (vProjSize != 0 && vProjSize != vSize) ||
        (oProjSize != 0 && oProjSize != qSize)) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }

    if (qoMaxSeqLength <= 0 || kvMaxSeqLength <= 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }

    // Check attnMode logic
    unsigned mapMask = XYDNN_ATTN_QUERYMAP_ONE_TO_ONE; // (1U << 0)
    unsigned biasMask = XYDNN_ATTN_ENABLE_PROJ_BIASES;  // (1U << 1)
    if (attnMode & ~(mapMask | biasMask)) {
        return XYDNN_STATUS_BAD_PARAM; 
    }


    attnDesc->attnMode        = attnMode;
    attnDesc->nHeads          = nHeads;
    attnDesc->smScaler        = smScaler;
    attnDesc->dataType        = dataType;
    attnDesc->computePrec     = computePrec;
    attnDesc->mathType        = mathType;
    attnDesc->attnDropoutDesc = attnDropoutDesc;
    attnDesc->postDropoutDesc = postDropoutDesc;
    attnDesc->qSize           = qSize;
    attnDesc->kSize           = kSize;
    attnDesc->vSize           = vSize;
    attnDesc->qProjSize       = qProjSize;
    attnDesc->kProjSize       = kProjSize;
    attnDesc->vProjSize       = vProjSize;
    attnDesc->oProjSize       = oProjSize;
    attnDesc->qoMaxSeqLength  = qoMaxSeqLength;
    attnDesc->kvMaxSeqLength  = kvMaxSeqLength;
    attnDesc->maxBatchSize    = maxBatchSize;
    attnDesc->maxBeamSize     = maxBeamSize;

    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnGetAttnDescriptor(xydnnAttnDescriptor_t attnDesc,
                       unsigned *attnMode,
                       int *nHeads,
                       double *smScaler,
                       xydnnDataType_t *dataType,
                       xydnnDataType_t *computePrec,
                       xydnnMathType_t *mathType,
                       xydnnDropoutDescriptor_t *attnDropoutDesc,
                       xydnnDropoutDescriptor_t *postDropoutDesc,
                       int *qSize,
                       int *kSize,
                       int *vSize,
                       int *qProjSize,
                       int *kProjSize,
                       int *vProjSize,
                       int *oProjSize,
                       int *qoMaxSeqLength,
                       int *kvMaxSeqLength,
                       int *maxBatchSize,
                       int *maxBeamSize)
{
    if (attnDesc == NULL) {
        return XYDNN_STATUS_BAD_PARAM;
    }

    if (attnMode)          *attnMode          = attnDesc->attnMode;
    if (nHeads)            *nHeads            = attnDesc->nHeads;
    if (smScaler)          *smScaler          = attnDesc->smScaler;
    if (dataType)          *dataType          = attnDesc->dataType;
    if (computePrec)       *computePrec       = attnDesc->computePrec;
    if (mathType)          *mathType          = attnDesc->mathType;
    if (attnDropoutDesc)   *attnDropoutDesc   = attnDesc->attnDropoutDesc;
    if (postDropoutDesc)   *postDropoutDesc   = attnDesc->postDropoutDesc;
    if (qSize)             *qSize             = attnDesc->qSize;
    if (kSize)             *kSize             = attnDesc->kSize;
    if (vSize)             *vSize             = attnDesc->vSize;
    if (qProjSize)         *qProjSize         = attnDesc->qProjSize;
    if (kProjSize)         *kProjSize         = attnDesc->kProjSize;
    if (vProjSize)         *vProjSize         = attnDesc->vProjSize;
    if (oProjSize)         *oProjSize         = attnDesc->oProjSize;
    if (qoMaxSeqLength)    *qoMaxSeqLength    = attnDesc->qoMaxSeqLength;
    if (kvMaxSeqLength)    *kvMaxSeqLength    = attnDesc->kvMaxSeqLength;
    if (maxBatchSize)      *maxBatchSize      = attnDesc->maxBatchSize;
    if (maxBeamSize)       *maxBeamSize       = attnDesc->maxBeamSize;

    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnSetAttnDescriptorOptions(xydnnAttnDescriptor_t attnDesc,
                              int isCausal,
                              int windowSizeLeft,
                              int windowSizeRight,
                              float softcap)
{
    if (attnDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (isCausal != 0 && isCausal != 1) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (softcap < 0.0f) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    attnDesc->isCausal = isCausal;
    attnDesc->windowSizeLeft = windowSizeLeft;
    attnDesc->windowSizeRight = windowSizeRight;
    attnDesc->softcap = softcap;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnGetAttnDescriptorOptions(xydnnAttnDescriptor_t attnDesc,
                              int *isCausal,
                              int *windowSizeLeft,
                              int *windowSizeRight,
                              float *softcap)
{
    if (attnDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (isCausal) {
        *isCausal = attnDesc->isCausal;
    }
    if (windowSizeLeft) {
        *windowSizeLeft = attnDesc->windowSizeLeft;
    }
    if (windowSizeRight) {
        *windowSizeRight = attnDesc->windowSizeRight;
    }
    if (softcap) {
        *softcap = attnDesc->softcap;
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnSetAttnDescriptorAlibiSlopes(xydnnAttnDescriptor_t attnDesc,
                                  const void *alibiSlopes,
                                  int batchStride)
{
    if (attnDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (batchStride < 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    attnDesc->alibiSlopes = alibiSlopes;
    attnDesc->alibiSlopesBatchStride = batchStride;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnGetAttnDescriptorAlibiSlopes(xydnnAttnDescriptor_t attnDesc,
                                  const void **alibiSlopes,
                                  int *batchStride)
{
    if (attnDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (alibiSlopes) {
        *alibiSlopes = attnDesc->alibiSlopes;
    }
    if (batchStride) {
        *batchStride = attnDesc->alibiSlopesBatchStride;
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnSetAttnDescriptorVarlen(xydnnAttnDescriptor_t attnDesc,
                             const int *cuSeqlensQO,
                             const int *cuSeqlensKV,
                             int totalQO,
                             int totalKV)
{
    if (attnDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if ((cuSeqlensQO == nullptr) != (cuSeqlensKV == nullptr)) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (cuSeqlensQO != nullptr && (totalQO <= 0 || totalKV <= 0)) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (cuSeqlensQO == nullptr && (totalQO != 0 || totalKV != 0)) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    attnDesc->cuSeqlensQO = cuSeqlensQO;
    attnDesc->cuSeqlensKV = cuSeqlensKV;
    attnDesc->totalQO = totalQO;
    attnDesc->totalKV = totalKV;
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnGetAttnDescriptorVarlen(xydnnAttnDescriptor_t attnDesc,
                             const int **cuSeqlensQO,
                             const int **cuSeqlensKV,
                             int *totalQO,
                             int *totalKV)
{
    if (attnDesc == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (cuSeqlensQO) {
        *cuSeqlensQO = attnDesc->cuSeqlensQO;
    }
    if (cuSeqlensKV) {
        *cuSeqlensKV = attnDesc->cuSeqlensKV;
    }
    if (totalQO) {
        *totalQO = attnDesc->totalQO;
    }
    if (totalKV) {
        *totalKV = attnDesc->totalKV;
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnGetMultiHeadAttnBuffers(xydnnHandle_t handle,
                             const xydnnAttnDescriptor_t attnDesc,
                             size_t *weightSizeInBytes,
                             size_t *workSpaceSizeInBytes,
                             size_t *reserveSpaceSizeInBytes)
{
    (void)handle;
    if (attnDesc == nullptr || weightSizeInBytes == nullptr ||
        workSpaceSizeInBytes == nullptr || reserveSpaceSizeInBytes == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    *weightSizeInBytes = weight_buffer_bytes(attnDesc);
    const size_t max_q_rows = static_cast<size_t>(attnDesc->maxBatchSize) *
                              static_cast<size_t>(attnDesc->qoMaxSeqLength);
    const size_t max_k_rows = static_cast<size_t>(attnDesc->maxBatchSize) *
                              static_cast<size_t>(attnDesc->kvMaxSeqLength);
    *workSpaceSizeInBytes = seqlens_workspace_bytes(attnDesc->maxBatchSize) +
                            projection_workspace_bytes(attnDesc, max_q_rows, max_k_rows);
    *reserveSpaceSizeInBytes = static_cast<size_t>(attnDesc->maxBatchSize) *
                               static_cast<size_t>(attnDesc->nHeads) *
                               static_cast<size_t>(attnDesc->qoMaxSeqLength) *
                               sizeof(float);
    if (attnDesc->attnDropoutDesc != nullptr && attnDesc->attnDropoutDesc->dropout != 0.0f) {
        *reserveSpaceSizeInBytes += kDropoutRngStateBytes;
    }
    return XYDNN_STATUS_SUCCESS;
}

xydnnStatus_t
xydnnGetMultiHeadAttnWeights(xydnnHandle_t handle,
                             const xydnnAttnDescriptor_t attnDesc,
                             xydnnMultiHeadAttnWeightKind_t wKind,
                             size_t weightSizeInBytes,
                             const void *weights,
                             xydnnTensorDescriptor_t wDesc,
                             void **wAddr)
{
    (void)handle;
    (void)wKind;
    (void)weightSizeInBytes;
    (void)weights;
    (void)wDesc;
    if (attnDesc == nullptr || wAddr == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    const size_t elem_size = data_type_size(attnDesc->dataType);
    if (elem_size == 0) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }

    int nbDims = 0;
    int dimA[4] = {};
    int strideA[4] = {};
    size_t elem_count = 0;
    xydnnStatus_t status = tensor_desc_for_weight(attnDesc, wKind, &nbDims, dimA, strideA, &elem_count);
    if (status != XYDNN_STATUS_SUCCESS) {
        return status;
    }
    const size_t offset_bytes = weight_offset_bytes(attnDesc, wKind);
    if (weights == nullptr || weightSizeInBytes < offset_bytes + elem_count * elem_size) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    *wAddr = const_cast<char *>(static_cast<const char *>(weights) + offset_bytes);
    if (wDesc != nullptr) {
        status = xydnnSetTensorNdDescriptor(wDesc, attnDesc->dataType, nbDims, dimA, strideA);
        if (status != XYDNN_STATUS_SUCCESS) {
            return status;
        }
    }
    return XYDNN_STATUS_SUCCESS;
}


xydnnStatus_t
xydnnMultiHeadAttnForward(xydnnHandle_t handle,
                          const xydnnAttnDescriptor_t attnDesc,
                          int currIdx,
                          const int loWinIdx[],
                          const int hiWinIdx[],
                          const int devSeqLengthsQO[],
                          const int devSeqLengthsKV[],
                          const xydnnSeqDataDescriptor_t qDesc,
                          const void *queries,
                          const void *residuals,
                          const xydnnSeqDataDescriptor_t kDesc,
                          const void *keys,
                          const xydnnSeqDataDescriptor_t vDesc,
                          const void *values,
                          const xydnnSeqDataDescriptor_t oDesc,
                          void *out,
                          size_t weightSizeInBytes,
                          const void *weights,
                          size_t workSpaceSizeInBytes,
                          void *workSpace,
                          size_t reserveSpaceSizeInBytes,
                          void *reserveSpace)
{
    (void)currIdx;
    (void)loWinIdx;
    (void)hiWinIdx;

    if (attnDesc == nullptr || qDesc == nullptr || kDesc == nullptr || vDesc == nullptr || oDesc == nullptr ||
        queries == nullptr || keys == nullptr || values == nullptr || out == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (residuals != nullptr) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
    if (weights == nullptr && weightSizeInBytes != 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    const float attn_dropout = attnDesc->attnDropoutDesc != nullptr ? attnDesc->attnDropoutDesc->dropout : 0.0f;
    const float post_dropout = attnDesc->postDropoutDesc != nullptr ? attnDesc->postDropoutDesc->dropout : 0.0f;
    if (attn_dropout < 0.0f || attn_dropout >= 1.0f || post_dropout < 0.0f || post_dropout >= 1.0f) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (post_dropout != 0.0f) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
    if (attn_dropout != 0.0f && attnDesc->softcap > 0.0f) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
#ifdef FLASHATTENTION_DISABLE_DROPOUT
    if (attn_dropout != 0.0f) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
#endif
    if (attnDesc->dataType != XYDNN_DATA_HALF && attnDesc->dataType != XYDNN_DATA_BFLOAT16) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
    if (qDesc->dataType != attnDesc->dataType || kDesc->dataType != attnDesc->dataType ||
        vDesc->dataType != attnDesc->dataType || oDesc->dataType != attnDesc->dataType) {
        return XYDNN_STATUS_BAD_PARAM;
    }

    const int batch_size = dim_for_axis(qDesc, XYDNN_SEQDATA_BATCH_DIM);
    const int q_beam = dim_for_axis(qDesc, XYDNN_SEQDATA_BEAM_DIM);
    const int k_beam = dim_for_axis(kDesc, XYDNN_SEQDATA_BEAM_DIM);
    const int v_beam = dim_for_axis(vDesc, XYDNN_SEQDATA_BEAM_DIM);
    const int o_beam = dim_for_axis(oDesc, XYDNN_SEQDATA_BEAM_DIM);
    const int seqlen_q = dim_for_axis(qDesc, XYDNN_SEQDATA_TIME_DIM);
    const int seqlen_k = dim_for_axis(kDesc, XYDNN_SEQDATA_TIME_DIM);
    const int seqlen_v = dim_for_axis(vDesc, XYDNN_SEQDATA_TIME_DIM);
    const int seqlen_o = dim_for_axis(oDesc, XYDNN_SEQDATA_TIME_DIM);
    const int q_vect = dim_for_axis(qDesc, XYDNN_SEQDATA_VECT_DIM);
    const int k_vect = dim_for_axis(kDesc, XYDNN_SEQDATA_VECT_DIM);
    const int v_vect = dim_for_axis(vDesc, XYDNN_SEQDATA_VECT_DIM);
    const int o_vect = dim_for_axis(oDesc, XYDNN_SEQDATA_VECT_DIM);

    if (batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || seqlen_v != seqlen_k || seqlen_o != seqlen_q) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (q_beam != 1 || k_beam != 1 || v_beam != 1 || o_beam != 1) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
    if (dim_for_axis(kDesc, XYDNN_SEQDATA_BATCH_DIM) != batch_size ||
        dim_for_axis(vDesc, XYDNN_SEQDATA_BATCH_DIM) != batch_size ||
        dim_for_axis(oDesc, XYDNN_SEQDATA_BATCH_DIM) != batch_size) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (q_vect != attnDesc->qSize || k_vect != attnDesc->kSize ||
        v_vect != attnDesc->vSize || o_vect != attnDesc->qSize) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (q_vect % attnDesc->nHeads != 0 || o_vect % attnDesc->nHeads != 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }

    const int head_size = q_vect / attnDesc->nHeads;
    if (o_vect / attnDesc->nHeads != head_size) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (k_vect % head_size != 0 || v_vect % head_size != 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    const int num_heads_k = k_vect / head_size;
    const int num_heads_v = v_vect / head_size;
    if (num_heads_k <= 0 || num_heads_v != num_heads_k || attnDesc->nHeads % num_heads_k != 0) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (head_size != 32 && head_size != 64 && head_size != 96 &&
        head_size != 128 && head_size != 192 && head_size != 256) {
        return XYDNN_STATUS_NOT_SUPPORTED;
    }
    if (batch_size > attnDesc->maxBatchSize || seqlen_q > attnDesc->qoMaxSeqLength ||
        seqlen_k > attnDesc->kvMaxSeqLength) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (reserveSpace == nullptr) {
        return XYDNN_STATUS_BAD_PARAM;
    }

    const size_t softmax_lse_bytes = static_cast<size_t>(batch_size) *
                                    static_cast<size_t>(attnDesc->nHeads) *
                                    static_cast<size_t>(seqlen_q) *
                                    sizeof(float);
    const size_t dropout_rng_bytes = attn_dropout != 0.0f ? kDropoutRngStateBytes : 0;
    const size_t required_reserve = dropout_rng_bytes + softmax_lse_bytes;
    if (reserveSpaceSizeInBytes < required_reserve) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (attnDesc->alibiSlopes != nullptr && attnDesc->alibiSlopesBatchStride < attnDesc->nHeads) {
        return XYDNN_STATUS_BAD_PARAM;
    }

    const bool use_projection = weights != nullptr && weightSizeInBytes != 0;
    const bool enable_bias = (attnDesc->attnMode & XYDNN_ATTN_ENABLE_PROJ_BIASES) != 0;
    const int proj_q = attnDesc->qProjSize > 0 ? attnDesc->qProjSize : attnDesc->qSize;
    const int proj_k = attnDesc->kProjSize > 0 ? attnDesc->kProjSize : attnDesc->kSize;
    const int proj_v = attnDesc->vProjSize > 0 ? attnDesc->vProjSize : attnDesc->vSize;
    const int proj_o = attnDesc->oProjSize > 0 ? attnDesc->oProjSize : attnDesc->qSize;

    const int *cu_seqlens_q = attnDesc->cuSeqlensQO;
    const int *cu_seqlens_k = attnDesc->cuSeqlensKV;
    int total_q = attnDesc->totalQO;
    int total_k = attnDesc->totalKV;
    std::vector<int> cu_seqlens_q_host;
    std::vector<int> cu_seqlens_k_host;
    bool use_packed_varlen = false;
    bool copy_host_seqlens_to_workspace = false;
    if (cu_seqlens_q == nullptr && qDesc->seqLengthArray != nullptr && kDesc->seqLengthArray != nullptr) {
        xydnnStatus_t q_seq_status = build_cu_seqlens_host(qDesc, batch_size, seqlen_q, cu_seqlens_q_host, &total_q);
        if (q_seq_status != XYDNN_STATUS_SUCCESS) {
            return q_seq_status;
        }
        xydnnStatus_t k_seq_status = build_cu_seqlens_host(kDesc, batch_size, seqlen_k, cu_seqlens_k_host, &total_k);
        if (k_seq_status != XYDNN_STATUS_SUCCESS) {
            return k_seq_status;
        }
        cu_seqlens_q = cu_seqlens_q_host.data();
        cu_seqlens_k = cu_seqlens_k_host.data();
        use_packed_varlen = true;
        copy_host_seqlens_to_workspace = true;
    } else if (cu_seqlens_q == nullptr && devSeqLengthsQO != nullptr && devSeqLengthsKV != nullptr) {
        cu_seqlens_q = devSeqLengthsQO;
        cu_seqlens_k = devSeqLengthsKV;
        use_packed_varlen = true;
        if (total_q <= 0 || total_k <= 0) {
            return XYDNN_STATUS_BAD_PARAM;
        }
    } else if (cu_seqlens_q != nullptr) {
        use_packed_varlen = true;
    }
    if (use_packed_varlen && (cu_seqlens_k == nullptr || total_q <= 0 || total_k <= 0)) {
        return XYDNN_STATUS_BAD_PARAM;
    }
    if (use_packed_varlen) {
        const size_t required_workspace = seqlens_workspace_bytes(batch_size);
        if (copy_host_seqlens_to_workspace && (workSpace == nullptr || workSpaceSizeInBytes < required_workspace)) {
            return XYDNN_STATUS_BAD_PARAM;
        }
    }

    cudaStream_t stream = nullptr;
    xydnnStatus_t stream_status = get_stream(handle, &stream);
    if (stream_status != XYDNN_STATUS_SUCCESS) {
        return stream_status;
    }

    if (copy_host_seqlens_to_workspace) {
        const size_t seqlens_bytes = static_cast<size_t>(batch_size + 1) * sizeof(int);
        int *workspace_int = static_cast<int *>(workSpace);
        int *device_cu_seqlens_q = workspace_int;
        int *device_cu_seqlens_k = workspace_int + batch_size + 1;
        cudaError_t q_copy_status = cudaMemcpyAsync(device_cu_seqlens_q,
                                                    cu_seqlens_q_host.data(),
                                                    seqlens_bytes,
                                                    cudaMemcpyHostToDevice,
                                                    stream);
        if (q_copy_status != cudaSuccess) {
            return XYDNN_STATUS_EXECUTION_FAILED;
        }
        cudaError_t k_copy_status = cudaMemcpyAsync(device_cu_seqlens_k,
                                                    cu_seqlens_k_host.data(),
                                                    seqlens_bytes,
                                                    cudaMemcpyHostToDevice,
                                                    stream);
        if (k_copy_status != cudaSuccess) {
            return XYDNN_STATUS_EXECUTION_FAILED;
        }
        cu_seqlens_q = device_cu_seqlens_q;
        cu_seqlens_k = device_cu_seqlens_k;
    }

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    (void)cc_minor;
    if (cc_major < 8) {
        return XYDNN_STATUS_ARCH_MISMATCH;
    }

    const void *q_input = queries;
    const void *k_input = keys;
    const void *v_input = values;
    void *o_output = out;
    void *projected_q = nullptr;
    void *projected_k = nullptr;
    void *projected_v = nullptr;
    void *projected_o = nullptr;
    const void *q_w = nullptr;
    const void *k_w = nullptr;
    const void *v_w = nullptr;
    const void *o_w = nullptr;
    const void *q_b = nullptr;
    const void *k_b = nullptr;
    const void *v_b = nullptr;
    const void *o_b = nullptr;
    if (use_projection) {
        q_w = static_cast<const char *>(weights) + weight_offset_bytes(attnDesc, XYDNN_MH_ATTN_Q_WEIGHTS);
        k_w = static_cast<const char *>(weights) + weight_offset_bytes(attnDesc, XYDNN_MH_ATTN_K_WEIGHTS);
        v_w = static_cast<const char *>(weights) + weight_offset_bytes(attnDesc, XYDNN_MH_ATTN_V_WEIGHTS);
        o_w = static_cast<const char *>(weights) + weight_offset_bytes(attnDesc, XYDNN_MH_ATTN_O_WEIGHTS);
        if (enable_bias) {
            q_b = static_cast<const char *>(weights) + bias_offset_bytes(attnDesc, XYDNN_MH_ATTN_Q_BIASES);
            k_b = static_cast<const char *>(weights) + bias_offset_bytes(attnDesc, XYDNN_MH_ATTN_K_BIASES);
            v_b = static_cast<const char *>(weights) + bias_offset_bytes(attnDesc, XYDNN_MH_ATTN_V_BIASES);
            o_b = static_cast<const char *>(weights) + bias_offset_bytes(attnDesc, XYDNN_MH_ATTN_O_BIASES);
        }
        const size_t q_rows = static_cast<size_t>(total_q > 0 ? total_q : batch_size * seqlen_q);
        const size_t k_rows = static_cast<size_t>(total_k > 0 ? total_k : batch_size * seqlen_k);
        const size_t seqlens_bytes = seqlens_workspace_bytes(batch_size);
        const size_t projection_bytes = projection_workspace_bytes(attnDesc, q_rows, k_rows);
        if (workSpace == nullptr || workSpaceSizeInBytes < seqlens_bytes + projection_bytes) {
            return XYDNN_STATUS_BAD_PARAM;
        }
        unsigned char *projection_base = static_cast<unsigned char *>(workSpace) + seqlens_bytes;
        size_t offset = 0;
        const size_t elem_size = data_type_size(attnDesc->dataType);
        offset = align_up(offset, 256);
        projected_q = projection_base + offset;
        offset += q_rows * static_cast<size_t>(proj_q) * elem_size;
        offset = align_up(offset, 256);
        projected_k = projection_base + offset;
        offset += k_rows * static_cast<size_t>(proj_k) * elem_size;
        offset = align_up(offset, 256);
        projected_v = projection_base + offset;
        offset += k_rows * static_cast<size_t>(proj_v) * elem_size;
        offset = align_up(offset, 256);
        projected_o = projection_base + offset;
        q_input = projected_q;
        k_input = projected_k;
        v_input = projected_v;
        o_output = projected_o;

        cublasHandle_t cublas = handle->cublas;
        if (cublas == nullptr) {
            return XYDNN_STATUS_NOT_INITIALIZED;
        }
        if (cublasSetStream(cublas, stream) != CUBLAS_STATUS_SUCCESS) {
            return XYDNN_STATUS_INTERNAL_ERROR;
        }
        if (launch_linear_projection(cublas,
                                     queries,
                                     q_w,
                                     projected_q,
                                     static_cast<int>(q_rows),
                                     attnDesc->qSize,
                                     proj_q,
                                     attnDesc->dataType) != XYDNN_STATUS_SUCCESS) {
            return XYDNN_STATUS_EXECUTION_FAILED;
        }
        if (launch_linear_projection(cublas,
                                     keys,
                                     k_w,
                                     projected_k,
                                     static_cast<int>(k_rows),
                                     attnDesc->kSize,
                                     proj_k,
                                     attnDesc->dataType) != XYDNN_STATUS_SUCCESS) {
            return XYDNN_STATUS_EXECUTION_FAILED;
        }
        if (launch_linear_projection(cublas,
                                     values,
                                     v_w,
                                     projected_v,
                                     static_cast<int>(k_rows),
                                     attnDesc->vSize,
                                     proj_v,
                                     attnDesc->dataType) != XYDNN_STATUS_SUCCESS) {
            return XYDNN_STATUS_EXECUTION_FAILED;
        }
    }

    FLASH_NAMESPACE::Flash_fwd_params params;
    int window_size_left = attnDesc->windowSizeLeft;
    int window_size_right = attnDesc->windowSizeRight;
    bool is_causal = attnDesc->isCausal != 0;

    if (window_size_left >= seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k) {
        window_size_right = -1;
    }
    if (seqlen_q == 1) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_left = -1;
        window_size_right = 0;
    }
    if (window_size_left < 0 && window_size_right >= 0) {
        window_size_left = seqlen_k;
    }
    if (window_size_left >= 0 && window_size_right < 0) {
        window_size_right = seqlen_k;
    }

    set_params_fprop_ptr(params,
                         batch_size,
                         seqlen_q,
                         seqlen_k,
                         attnDesc->nHeads,
                         num_heads_k,
                         head_size,
                         q_input,
                         k_input,
                         v_input,
                         o_output,
                         static_cast<unsigned char *>(reserveSpace) + dropout_rng_bytes,
                         attnDesc->dataType,
                         static_cast<float>(attnDesc->smScaler),
                         is_causal,
                         window_size_left,
                         window_size_right,
                         attnDesc->softcap,
                         attnDesc->alibiSlopes,
                         attnDesc->alibiSlopesBatchStride,
                         cu_seqlens_q,
                         cu_seqlens_k,
                         total_q,
                         attn_dropout);
    if (use_packed_varlen) {
        params.total_q = total_q;
    }
    params.rng_state = reinterpret_cast<uint64_t *>(reserveSpace);
    if (attn_dropout != 0.0f) {
        uint64_t rng_state_host[2] = {attnDesc->attnDropoutDesc->seed, 0ULL};
        cudaError_t rng_copy_status = cudaMemcpyAsync(params.rng_state,
                                                      rng_state_host,
                                                      sizeof(rng_state_host),
                                                      cudaMemcpyHostToDevice,
                                                      stream);
        if (rng_copy_status != cudaSuccess) {
            return XYDNN_STATUS_EXECUTION_FAILED;
        }
#ifndef FLASHATTENTION_DISABLE_DROPOUT
        params.philox_args = at::PhiloxCudaState(attnDesc->attnDropoutDesc->seed, 0ULL);
#endif
    }

    try {
        run_mha_fwd_ptr(params, stream);
    } catch (...) {
        return XYDNN_STATUS_EXECUTION_FAILED;
    }

    if (use_projection) {
        cublasHandle_t cublas = handle->cublas;
        if (cublas == nullptr) {
            return XYDNN_STATUS_NOT_INITIALIZED;
        }
        if (cublasSetStream(cublas, stream) != CUBLAS_STATUS_SUCCESS) {
            return XYDNN_STATUS_INTERNAL_ERROR;
        }
        const size_t q_rows = static_cast<size_t>(total_q > 0 ? total_q : batch_size * seqlen_q);
        if (launch_linear_projection(cublas,
                                     projected_o,
                                     o_w,
                                     out,
                                     static_cast<int>(q_rows),
                                     proj_o,
                                     attnDesc->qSize,
                                     attnDesc->dataType) != XYDNN_STATUS_SUCCESS) {
            return XYDNN_STATUS_EXECUTION_FAILED;
        }
    }

    return XYDNN_STATUS_SUCCESS;
}

}  // extern "C"
