#include <cuda_runtime.h>
#include <cudnn.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess) {
        std::cerr << what << " failed: " << cudaGetErrorString(status) << "\n";
        std::exit(1);
    }
}

void check_cudnn(cudnnStatus_t status, const char *what)
{
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << what << " failed: " << cudnnGetErrorString(status)
                  << " (" << static_cast<int>(status) << ")\n";
        std::exit(1);
    }
}

struct DeviceBuffer {
    void *ptr = nullptr;

    explicit DeviceBuffer(size_t bytes = 0)
    {
        if (bytes != 0) {
            check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
            check_cuda(cudaMemset(ptr, 0, bytes), "cudaMemset");
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

struct DropoutDesc {
    cudnnDropoutDescriptor_t desc = nullptr;
    DeviceBuffer states;

    DropoutDesc(cudnnHandle_t handle, unsigned long long seed)
    {
        size_t state_bytes = 0;
        check_cudnn(cudnnDropoutGetStatesSize(handle, &state_bytes), "cudnnDropoutGetStatesSize");
        states = DeviceBuffer(state_bytes);
        check_cudnn(cudnnCreateDropoutDescriptor(&desc), "cudnnCreateDropoutDescriptor");
        check_cudnn(cudnnSetDropoutDescriptor(desc, handle, 0.0f, states.ptr, state_bytes, seed),
                    "cudnnSetDropoutDescriptor");
    }

    DropoutDesc(const DropoutDesc &) = delete;
    DropoutDesc &operator=(const DropoutDesc &) = delete;

    ~DropoutDesc()
    {
        if (desc != nullptr) {
            cudnnDestroyDropoutDescriptor(desc);
        }
    }
};

struct SeqDesc {
    cudnnSeqDataDescriptor_t desc = nullptr;

    SeqDesc(cudnnDataType_t dtype,
            const std::vector<int> &dims,
            const std::vector<cudnnSeqDataAxis_t> &axes,
            const std::vector<int> &seq_lengths)
    {
        check_cudnn(cudnnCreateSeqDataDescriptor(&desc), "cudnnCreateSeqDataDescriptor");
        check_cudnn(cudnnSetSeqDataDescriptor(desc,
                                              dtype,
                                              static_cast<int>(dims.size()),
                                              dims.data(),
                                              axes.data(),
                                              seq_lengths.size(),
                                              seq_lengths.data(),
                                              nullptr),
                    "cudnnSetSeqDataDescriptor");
    }

    SeqDesc(const SeqDesc &) = delete;
    SeqDesc &operator=(const SeqDesc &) = delete;

    ~SeqDesc()
    {
        if (desc != nullptr) {
            cudnnDestroySeqDataDescriptor(desc);
        }
    }
};

}  // namespace

int main()
{
    constexpr int batch = 1;
    constexpr int beam = 1;
    constexpr int seqlen = 4;
    constexpr int heads = 1;
    constexpr int head_dim = 32;
    constexpr int embed = heads * head_dim;
    constexpr cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    constexpr cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::cerr << "No CUDA device is available\n";
        return 1;
    }

    cudnnHandle_t handle = nullptr;
    cudnnAttnDescriptor_t attn = nullptr;
    check_cudnn(cudnnCreate(&handle), "cudnnCreate");
    check_cudnn(cudnnCreateAttnDescriptor(&attn), "cudnnCreateAttnDescriptor");

    DropoutDesc attn_dropout(handle, 1234ULL);
    DropoutDesc post_dropout(handle, 5678ULL);

    check_cudnn(cudnnSetAttnDescriptor(attn,
                                       CUDNN_ATTN_QUERYMAP_ONE_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES,
                                       heads,
                                       1.0 / std::sqrt(static_cast<double>(head_dim)),
                                       dtype,
                                       compute_type,
                                       CUDNN_DEFAULT_MATH,
                                       attn_dropout.desc,
                                       post_dropout.desc,
                                       embed,
                                       embed,
                                       embed,
                                       embed,
                                       embed,
                                       embed,
                                       embed,
                                       seqlen,
                                       seqlen,
                                       batch,
                                       beam),
                "cudnnSetAttnDescriptor");

    const std::vector<int> seq_lengths(batch * beam, seqlen);
    const std::vector<int> dims = {seqlen, batch, beam, embed};
    const std::vector<cudnnSeqDataAxis_t> axes = {
        CUDNN_SEQDATA_TIME_DIM,
        CUDNN_SEQDATA_BATCH_DIM,
        CUDNN_SEQDATA_BEAM_DIM,
        CUDNN_SEQDATA_VECT_DIM,
    };

    SeqDesc q_desc(dtype, dims, axes, seq_lengths);
    SeqDesc k_desc(dtype, dims, axes, seq_lengths);
    SeqDesc v_desc(dtype, dims, axes, seq_lengths);
    SeqDesc o_desc(dtype, dims, axes, seq_lengths);

    size_t weight_bytes = 0;
    size_t workspace_bytes = 0;
    size_t reserve_bytes = 0;
    check_cudnn(cudnnGetMultiHeadAttnBuffers(handle, attn, &weight_bytes, &workspace_bytes, &reserve_bytes),
                "cudnnGetMultiHeadAttnBuffers");

    const size_t tensor_bytes = static_cast<size_t>(seqlen) * batch * beam * embed * sizeof(float);
    DeviceBuffer queries(tensor_bytes);
    DeviceBuffer keys(tensor_bytes);
    DeviceBuffer values(tensor_bytes);
    DeviceBuffer out(tensor_bytes);
    DeviceBuffer weights(weight_bytes);
    DeviceBuffer workspace(workspace_bytes);
    DeviceBuffer reserve(reserve_bytes);
    DeviceBuffer dev_q_lens(sizeof(int) * seq_lengths.size());
    DeviceBuffer dev_k_lens(sizeof(int) * seq_lengths.size());
    check_cuda(cudaMemcpy(dev_q_lens.ptr,
                          seq_lengths.data(),
                          sizeof(int) * seq_lengths.size(),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(dev_q_lens)");
    check_cuda(cudaMemcpy(dev_k_lens.ptr,
                          seq_lengths.data(),
                          sizeof(int) * seq_lengths.size(),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(dev_k_lens)");

    std::vector<int> lo_win(seqlen, 0);
    std::vector<int> hi_win(seqlen, seqlen);

    check_cudnn(cudnnMultiHeadAttnForward(handle,
                                          attn,
                                          -1,
                                          lo_win.data(),
                                          hi_win.data(),
                                          static_cast<const int *>(dev_q_lens.ptr),
                                          static_cast<const int *>(dev_k_lens.ptr),
                                          q_desc.desc,
                                          queries.ptr,
                                          nullptr,
                                          k_desc.desc,
                                          keys.ptr,
                                          v_desc.desc,
                                          values.ptr,
                                          o_desc.desc,
                                          out.ptr,
                                          weight_bytes,
                                          weights.ptr,
                                          workspace_bytes,
                                          workspace.ptr,
                                          reserve_bytes,
                                          reserve.ptr),
                "cudnnMultiHeadAttnForward");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cudnn(cudnnDestroyAttnDescriptor(attn), "cudnnDestroyAttnDescriptor");
    check_cudnn(cudnnDestroy(handle), "cudnnDestroy");

    std::cout << "cuDNN version: " << cudnnGetVersion() << "\n";
    std::cout << "cudnnMultiHeadAttnForward ok\n";
    std::cout << "weights=" << weight_bytes
              << " workspace=" << workspace_bytes
              << " reserve=" << reserve_bytes << "\n";
    return 0;
}
