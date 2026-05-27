#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cudnn.h>
#include <torch/torch.h>

#include "flash_attn/flash_attn_api.h"
#include "flash_attn/xydnn_mha_fwd_api.h"

namespace {

int read_env_or(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return value == nullptr ? fallback : std::atoi(value);
}

double read_env_or(const char *name, double fallback)
{
    const char *value = std::getenv(name);
    return value == nullptr ? fallback : std::atof(value);
}

std::string read_env_or(const char *name, const char *fallback)
{
    const char *value = std::getenv(name);
    return value == nullptr ? std::string(fallback) : std::string(value);
}

struct PrecisionConfig {
    std::string name;
    xydnnDataType_t xydnn_dtype;
    torch::Dtype torch_dtype;
};

PrecisionConfig read_precision()
{
    const std::string dtype = read_env_or("DTYPE", "fp16");
    if (dtype == "bf16") {
        return {"bf16", XYDNN_DATA_BFLOAT16, torch::kBFloat16};
    }
    if (dtype == "fp16" || dtype == "half") {
        return {"fp16", XYDNN_DATA_HALF, torch::kFloat16};
    }
    std::cerr << "DTYPE must be fp16 or bf16, got '" << dtype << "'\n";
    std::exit(1);
}

bool is_supported_head_dim(int head_dim)
{
    return head_dim == 32 || head_dim == 64 || head_dim == 96 ||
           head_dim == 128 || head_dim == 192 || head_dim == 256;
}

void check_xydnn(xydnnStatus_t status, const char *what)
{
    if (status != XYDNN_STATUS_SUCCESS) {
        std::cerr << what << " failed with xyDNN status " << static_cast<int>(status) << "\n";
        std::exit(1);
    }
}

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
        std::cerr << what << " failed with cuDNN status " << static_cast<int>(status)
                  << " (" << cudnnGetErrorString(status) << ")\n";
        std::exit(1);
    }
}

struct Problem {
    int batch;
    int seqlen;
    int heads;
    int kv_heads;
    int head_dim;
    bool causal;
    bool varlen;
    float softmax_scale;
    float softcap;
    float dropout;
    unsigned long long dropout_seed;
    int window_left;
    int window_right;
    int runtime_window;
    PrecisionConfig precision;
};

struct XyDnnState {
    xydnnHandle_t handle = nullptr;
    xydnnDropoutDescriptor_t attn_dropout = nullptr;
    xydnnDropoutDescriptor_t post_dropout = nullptr;
    xydnnAttnDescriptor_t attn = nullptr;
    xydnnSeqDataDescriptor_t q_desc = nullptr;
    xydnnSeqDataDescriptor_t k_desc = nullptr;
    xydnnSeqDataDescriptor_t v_desc = nullptr;
    xydnnSeqDataDescriptor_t o_desc = nullptr;
    at::Tensor workspace;
    at::Tensor reserve;
    std::vector<int> lo_win;
    std::vector<int> hi_win;
};

struct CudnnState {
    cudnnHandle_t handle = nullptr;
    cudnnAttnDescriptor_t attn = nullptr;
    cudnnDropoutDescriptor_t attn_dropout = nullptr;
    cudnnDropoutDescriptor_t post_dropout = nullptr;
    cudnnSeqDataDescriptor_t q_desc = nullptr;
    cudnnSeqDataDescriptor_t k_desc = nullptr;
    cudnnSeqDataDescriptor_t v_desc = nullptr;
    cudnnSeqDataDescriptor_t o_desc = nullptr;
    at::Tensor workspace;
    at::Tensor reserve;
    at::Tensor weights;
    at::Tensor attn_dropout_states;
    at::Tensor post_dropout_states;
    at::Tensor seq_lengths_q;
    at::Tensor seq_lengths_k;
    std::vector<int> lo_win;
    std::vector<int> hi_win;
    size_t weight_bytes = 0;
    size_t workspace_bytes = 0;
    size_t reserve_bytes = 0;
};

std::vector<int> make_cu_seqlens(int batch, int max_seqlen)
{
    std::vector<int> cu(batch + 1, 0);
    for (int i = 0; i < batch; ++i) {
        const int trim = (i * 17) % std::max(1, max_seqlen / 2);
        const int len = std::max(1, max_seqlen - trim);
        cu[i + 1] = cu[i] + len;
    }
    return cu;
}

void create_seq_desc(xydnnSeqDataDescriptor_t *desc,
                     xydnnDataType_t dtype,
                     int time_dim,
                     int batch,
                     int vect,
                     const int *seq_array,
                     size_t seq_array_size)
{
    const int dim_a[] = {time_dim, batch, 1, vect};
    const xydnnSeqDataAxis_t axes[] = {
        XYDNN_SEQDATA_TIME_DIM,
        XYDNN_SEQDATA_BATCH_DIM,
        XYDNN_SEQDATA_BEAM_DIM,
        XYDNN_SEQDATA_VECT_DIM,
    };
    check_xydnn(xydnnCreateSeqDataDescriptor(desc), "xydnnCreateSeqDataDescriptor");
    check_xydnn(xydnnSetSeqDataDescriptor(*desc,
                                          dtype,
                                          4,
                                          dim_a,
                                          axes,
                                          seq_array_size,
                                          seq_array,
                                          nullptr),
                "xydnnSetSeqDataDescriptor");
}

struct CudnnTryResult {
    bool ok = false;
    std::string reason;
};

bool try_cudnn(cudnnStatus_t status, const char *what, std::string *reason)
{
    if (status == CUDNN_STATUS_SUCCESS) {
        return true;
    }
    std::ostringstream oss;
    oss << what << " failed with cuDNN status " << static_cast<int>(status)
        << " (" << cudnnGetErrorString(status) << ")";
    if (reason != nullptr) {
        *reason = oss.str();
    }
    return false;
}

CudnnTryResult create_cudnn_seq_desc(cudnnSeqDataDescriptor_t *desc,
                                     cudnnDataType_t dtype,
                                     int time_dim,
                                     int batch,
                                     int vect,
                                     const int *seq_array,
                                     size_t seq_array_size)
{
    std::string reason;
    const int dim_a[] = {time_dim, batch, 1, vect};
    const cudnnSeqDataAxis_t axes[] = {
        CUDNN_SEQDATA_TIME_DIM,
        CUDNN_SEQDATA_BATCH_DIM,
        CUDNN_SEQDATA_BEAM_DIM,
        CUDNN_SEQDATA_VECT_DIM,
    };
    if (!try_cudnn(cudnnCreateSeqDataDescriptor(desc), "cudnnCreateSeqDataDescriptor", &reason) ||
        !try_cudnn(cudnnSetSeqDataDescriptor(*desc,
                                             dtype,
                                             4,
                                             dim_a,
                                             axes,
                                             seq_array_size,
                                             seq_array,
                                             nullptr),
                   "cudnnSetSeqDataDescriptor",
                   &reason)) {
        return {false, reason};
    }
    return {true, ""};
}

CudnnTryResult initialize_cudnn_identity_weights(CudnnState &state, int heads, int head_dim)
{
    const int vect = heads * head_dim;
    const size_t float_count = (state.weight_bytes + sizeof(float) - 1) / sizeof(float);
    std::vector<float> host_weights(float_count, 0.0f);
    auto *base = static_cast<char *>(state.weights.data_ptr());

    const cudnnMultiHeadAttnWeightKind_t kinds[] = {
        CUDNN_MH_ATTN_Q_WEIGHTS,
        CUDNN_MH_ATTN_K_WEIGHTS,
        CUDNN_MH_ATTN_V_WEIGHTS,
        CUDNN_MH_ATTN_O_WEIGHTS,
    };

    for (cudnnMultiHeadAttnWeightKind_t kind : kinds) {
        cudnnTensorDescriptor_t w_desc = nullptr;
        cudnnStatus_t status = cudnnCreateTensorDescriptor(&w_desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            std::string reason;
            try_cudnn(status, "cudnnCreateTensorDescriptor", &reason);
            return {false, reason};
        }

        void *w_addr = nullptr;
        status = cudnnGetMultiHeadAttnWeights(state.handle,
                                              state.attn,
                                              kind,
                                              state.weight_bytes,
                                              state.weights.data_ptr(),
                                              w_desc,
                                              &w_addr);
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyTensorDescriptor(w_desc);
            std::string reason;
            try_cudnn(status, "cudnnGetMultiHeadAttnWeights", &reason);
            return {false, reason};
        }

        cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
        int nb_dims = 0;
        int dim_a[8] = {};
        int stride_a[8] = {};
        status = cudnnGetTensorNdDescriptor(w_desc, 8, &dtype, &nb_dims, dim_a, stride_a);
        cudnnDestroyTensorDescriptor(w_desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            std::string reason;
            try_cudnn(status, "cudnnGetTensorNdDescriptor", &reason);
            return {false, reason};
        }
        if (dtype != CUDNN_DATA_FLOAT) {
            return {false, "cuDNN projection weights are not float32"};
        }

        const auto byte_offset = static_cast<ptrdiff_t>(static_cast<char *>(w_addr) - base);
        if (byte_offset < 0 || static_cast<size_t>(byte_offset) >= state.weight_bytes ||
            byte_offset % static_cast<ptrdiff_t>(sizeof(float)) != 0) {
            return {false, "cuDNN returned an invalid projection weight pointer"};
        }
        const size_t base_index = static_cast<size_t>(byte_offset) / sizeof(float);
        auto set = [&](int i0, int i1, int i2, int i3) {
            const int idx[4] = {i0, i1, i2, i3};
            size_t rel = 0;
            for (int d = 0; d < nb_dims; ++d) {
                rel += static_cast<size_t>(idx[d]) * static_cast<size_t>(stride_a[d]);
            }
            const size_t pos = base_index + rel;
            if (pos < host_weights.size()) {
                host_weights[pos] = 1.0f;
            }
        };

        if (nb_dims == 2 && dim_a[0] == vect && dim_a[1] == vect) {
            for (int i = 0; i < vect; ++i) {
                set(i, i, 0, 0);
            }
        } else if (nb_dims == 3 && dim_a[0] == heads && dim_a[1] == head_dim && dim_a[2] == vect) {
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    set(h, d, h * head_dim + d, 0);
                }
            }
        } else if (nb_dims == 3 && dim_a[0] == heads && dim_a[1] == vect && dim_a[2] == head_dim) {
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    set(h, h * head_dim + d, d, 0);
                }
            }
        } else if (nb_dims == 3 && dim_a[0] == vect && dim_a[1] == heads && dim_a[2] == head_dim) {
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    set(h * head_dim + d, h, d, 0);
                }
            }
        } else if (nb_dims == 3 && dim_a[0] == heads && dim_a[1] == vect && dim_a[2] == vect) {
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    const int idx = h * head_dim + d;
                    set(h, idx, idx, 0);
                }
            }
        } else {
            std::ostringstream oss;
            oss << "unsupported cuDNN projection weight descriptor: nbDims=" << nb_dims;
            for (int d = 0; d < nb_dims; ++d) {
                oss << " dim" << d << "=" << dim_a[d] << " stride" << d << "=" << stride_a[d];
            }
            return {false, oss.str()};
        }
    }

    check_cuda(cudaMemcpy(state.weights.data_ptr(),
                          host_weights.data(),
                          state.weight_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(cuDNN identity weights)");
    return {true, ""};
}

XyDnnState create_xydnn_state(const Problem &p,
                              int q_time_dim,
                              int k_time_dim,
                              int total_q,
                              const int *q_seq_array,
                              const int *k_seq_array,
                              size_t seq_array_size)
{
    XyDnnState state;
    const int q_vect = p.heads * p.head_dim;
    const int kv_vect = p.kv_heads * p.head_dim;
    check_xydnn(xydnnCreate(&state.handle), "xydnnCreate");
    check_xydnn(xydnnSetStream(state.handle, at::cuda::getCurrentCUDAStream().stream()), "xydnnSetStream");
    if (p.dropout > 0.0f) {
        check_xydnn(xydnnCreateDropoutDescriptor(&state.attn_dropout), "xydnnCreateDropoutDescriptor(attn)");
        check_xydnn(xydnnSetDropoutDescriptor(state.attn_dropout,
                                              state.handle,
                                              p.dropout,
                                              nullptr,
                                              0,
                                              p.dropout_seed),
                    "xydnnSetDropoutDescriptor(attn)");
    }
    check_xydnn(xydnnCreateAttnDescriptor(&state.attn), "xydnnCreateAttnDescriptor");
    check_xydnn(xydnnSetAttnDescriptor(state.attn,
                                       XYDNN_ATTN_QUERYMAP_ONE_TO_ONE,
                                       p.heads,
                                       p.softmax_scale,
                                       p.precision.xydnn_dtype,
                                       XYDNN_DATA_FLOAT,
                                       XYDNN_DEFAULT_MATH,
                                       state.attn_dropout,
                                       nullptr,
                                       q_vect,
                                       kv_vect,
                                       kv_vect,
                                       0,
                                       0,
                                       0,
                                       0,
                                       q_time_dim,
                                       k_time_dim,
                                       p.batch,
                                       1),
                "xydnnSetAttnDescriptor");
    check_xydnn(xydnnSetAttnDescriptorOptions(state.attn,
                                              p.causal ? 1 : 0,
                                              p.window_left,
                                              p.window_right,
                                              p.softcap),
                "xydnnSetAttnDescriptorOptions");
    check_xydnn(xydnnSetAttnDescriptorAlibiSlopes(state.attn, nullptr, 0),
                "xydnnSetAttnDescriptorAlibiSlopes");

    create_seq_desc(&state.q_desc, p.precision.xydnn_dtype, q_time_dim, p.batch, q_vect, q_seq_array, seq_array_size);
    create_seq_desc(&state.k_desc, p.precision.xydnn_dtype, k_time_dim, p.batch, kv_vect, k_seq_array, seq_array_size);
    create_seq_desc(&state.v_desc, p.precision.xydnn_dtype, k_time_dim, p.batch, kv_vect, k_seq_array, seq_array_size);
    create_seq_desc(&state.o_desc, p.precision.xydnn_dtype, q_time_dim, p.batch, q_vect, q_seq_array, seq_array_size);

    size_t weight_bytes = 0;
    size_t workspace_bytes = 0;
    size_t reserve_bytes = 0;
    check_xydnn(xydnnGetMultiHeadAttnBuffers(state.handle,
                                             state.attn,
                                             &weight_bytes,
                                             &workspace_bytes,
                                             &reserve_bytes),
                "xydnnGetMultiHeadAttnBuffers");
    (void)weight_bytes;
    state.workspace = torch::empty({static_cast<long>((workspace_bytes + sizeof(int) - 1) / sizeof(int))},
                                   torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    state.reserve = torch::empty({static_cast<long>((reserve_bytes + sizeof(float) - 1) / sizeof(float))},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    if (p.varlen) {
        check_xydnn(xydnnSetAttnDescriptorVarlen(state.attn, nullptr, nullptr, 0, 0),
                    "xydnnSetAttnDescriptorVarlen(clear)");
        (void)total_q;
    }
    if (p.runtime_window != 0) {
        state.lo_win.resize(q_time_dim);
        state.hi_win.resize(q_time_dim);
        for (int row = 0; row < q_time_dim; ++row) {
            const int row_base = row + k_time_dim - q_time_dim;
            int lo = 0;
            int hi = k_time_dim;
            if (p.runtime_window == 1) {
                lo = 0;
                hi = p.causal ? std::min(k_time_dim, row_base + 1) : k_time_dim;
            } else if (p.runtime_window == 2) {
                lo = std::max(0, row_base - std::max(0, p.window_left));
                hi = std::min(k_time_dim, row_base + 1 + std::max(0, p.window_right));
            } else if (p.runtime_window == 3) {
                lo = row % 2;
                hi = k_time_dim;
            }
            state.lo_win[row] = lo;
            state.hi_win[row] = hi;
        }
    }
    return state;
}

void destroy_xydnn_state(XyDnnState &state)
{
    check_xydnn(xydnnDestroySeqDataDescriptor(state.q_desc), "xydnnDestroySeqDataDescriptor(q)");
    check_xydnn(xydnnDestroySeqDataDescriptor(state.k_desc), "xydnnDestroySeqDataDescriptor(k)");
    check_xydnn(xydnnDestroySeqDataDescriptor(state.v_desc), "xydnnDestroySeqDataDescriptor(v)");
    check_xydnn(xydnnDestroySeqDataDescriptor(state.o_desc), "xydnnDestroySeqDataDescriptor(o)");
    check_xydnn(xydnnDestroyAttnDescriptor(state.attn), "xydnnDestroyAttnDescriptor");
    if (state.post_dropout != nullptr) {
        check_xydnn(xydnnDestroyDropoutDescriptor(state.post_dropout), "xydnnDestroyDropoutDescriptor(post)");
    }
    if (state.attn_dropout != nullptr) {
        check_xydnn(xydnnDestroyDropoutDescriptor(state.attn_dropout), "xydnnDestroyDropoutDescriptor(attn)");
    }
    check_xydnn(xydnnDestroy(state.handle), "xydnnDestroy");
}

CudnnTryResult create_cudnn_state(const Problem &p, CudnnState &state)
{
    std::string reason;
    const int vect = p.heads * p.head_dim;
    std::vector<int> full_lengths(p.batch, p.seqlen);
    const auto cuda_opts = torch::TensorOptions().device(torch::kCUDA);

    if (!try_cudnn(cudnnCreate(&state.handle), "cudnnCreate", &reason) ||
        !try_cudnn(cudnnSetStream(state.handle, at::cuda::getCurrentCUDAStream().stream()), "cudnnSetStream", &reason) ||
        !try_cudnn(cudnnCreateDropoutDescriptor(&state.attn_dropout), "cudnnCreateDropoutDescriptor(attn)", &reason) ||
        !try_cudnn(cudnnCreateDropoutDescriptor(&state.post_dropout), "cudnnCreateDropoutDescriptor(post)", &reason)) {
        return {false, reason};
    }

    size_t dropout_state_bytes = 0;
    if (!try_cudnn(cudnnDropoutGetStatesSize(state.handle, &dropout_state_bytes), "cudnnDropoutGetStatesSize", &reason)) {
        return {false, reason};
    }
    state.attn_dropout_states = torch::empty({static_cast<long>((dropout_state_bytes + 3) / 4)},
                                             cuda_opts.dtype(torch::kInt32));
    state.post_dropout_states = torch::empty({static_cast<long>((dropout_state_bytes + 3) / 4)},
                                             cuda_opts.dtype(torch::kInt32));
    if (!try_cudnn(cudnnSetDropoutDescriptor(state.attn_dropout,
                                             state.handle,
                                             0.0f,
                                             state.attn_dropout_states.data_ptr(),
                                             dropout_state_bytes,
                                             1234ULL),
                   "cudnnSetDropoutDescriptor(attn)",
                   &reason) ||
        !try_cudnn(cudnnSetDropoutDescriptor(state.post_dropout,
                                             state.handle,
                                             0.0f,
                                             state.post_dropout_states.data_ptr(),
                                             dropout_state_bytes,
                                             5678ULL),
                   "cudnnSetDropoutDescriptor(post)",
                   &reason) ||
        !try_cudnn(cudnnCreateAttnDescriptor(&state.attn), "cudnnCreateAttnDescriptor", &reason)) {
        return {false, reason};
    }

    if (!try_cudnn(cudnnSetAttnDescriptor(state.attn,
                                          CUDNN_ATTN_QUERYMAP_ONE_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES,
                                          p.heads,
                                          p.softmax_scale,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_DEFAULT_MATH,
                                          state.attn_dropout,
                                          state.post_dropout,
                                          vect,
                                          vect,
                                          vect,
                                          vect,
                                          vect,
                                          vect,
                                          vect,
                                          p.seqlen,
                                          p.seqlen,
                                          p.batch,
                                          1),
                   "cudnnSetAttnDescriptor",
                   &reason)) {
        return {false, reason};
    }

    CudnnTryResult seq_try;
    seq_try = create_cudnn_seq_desc(&state.q_desc, CUDNN_DATA_FLOAT, p.seqlen, p.batch, vect, full_lengths.data(), full_lengths.size());
    if (!seq_try.ok) {
        return seq_try;
    }
    seq_try = create_cudnn_seq_desc(&state.k_desc, CUDNN_DATA_FLOAT, p.seqlen, p.batch, vect, full_lengths.data(), full_lengths.size());
    if (!seq_try.ok) {
        return seq_try;
    }
    seq_try = create_cudnn_seq_desc(&state.v_desc, CUDNN_DATA_FLOAT, p.seqlen, p.batch, vect, full_lengths.data(), full_lengths.size());
    if (!seq_try.ok) {
        return seq_try;
    }
    seq_try = create_cudnn_seq_desc(&state.o_desc, CUDNN_DATA_FLOAT, p.seqlen, p.batch, vect, full_lengths.data(), full_lengths.size());
    if (!seq_try.ok) {
        return seq_try;
    }

    if (!try_cudnn(cudnnGetMultiHeadAttnBuffers(state.handle,
                                                state.attn,
                                                &state.weight_bytes,
                                                &state.workspace_bytes,
                                                &state.reserve_bytes),
                   "cudnnGetMultiHeadAttnBuffers",
                   &reason)) {
        return {false, reason};
    }
    state.weights = torch::zeros({static_cast<long>((state.weight_bytes + sizeof(float) - 1) / sizeof(float))},
                                 cuda_opts.dtype(torch::kFloat32));
    state.workspace = torch::empty({static_cast<long>((state.workspace_bytes + sizeof(int) - 1) / sizeof(int))},
                                   cuda_opts.dtype(torch::kInt32));
    state.reserve = torch::empty({static_cast<long>((state.reserve_bytes + sizeof(int) - 1) / sizeof(int))},
                                 cuda_opts.dtype(torch::kInt32));
    state.seq_lengths_q = torch::full({p.batch}, p.seqlen, cuda_opts.dtype(torch::kInt32));
    state.seq_lengths_k = torch::full({p.batch}, p.seqlen, cuda_opts.dtype(torch::kInt32));
    state.lo_win.resize(p.seqlen, 0);
    state.hi_win.resize(p.seqlen, p.seqlen);
    if (p.causal) {
        for (int i = 0; i < p.seqlen; ++i) {
            state.hi_win[i] = i + 1;
        }
    }
    return {true, ""};
}

void destroy_cudnn_state(CudnnState &state)
{
    if (state.post_dropout != nullptr) {
        check_cudnn(cudnnDestroyDropoutDescriptor(state.post_dropout), "cudnnDestroyDropoutDescriptor(post)");
    }
    if (state.attn_dropout != nullptr) {
        check_cudnn(cudnnDestroyDropoutDescriptor(state.attn_dropout), "cudnnDestroyDropoutDescriptor(attn)");
    }
    if (state.q_desc != nullptr) {
        check_cudnn(cudnnDestroySeqDataDescriptor(state.q_desc), "cudnnDestroySeqDataDescriptor(q)");
    }
    if (state.k_desc != nullptr) {
        check_cudnn(cudnnDestroySeqDataDescriptor(state.k_desc), "cudnnDestroySeqDataDescriptor(k)");
    }
    if (state.v_desc != nullptr) {
        check_cudnn(cudnnDestroySeqDataDescriptor(state.v_desc), "cudnnDestroySeqDataDescriptor(v)");
    }
    if (state.o_desc != nullptr) {
        check_cudnn(cudnnDestroySeqDataDescriptor(state.o_desc), "cudnnDestroySeqDataDescriptor(o)");
    }
    if (state.attn != nullptr) {
        check_cudnn(cudnnDestroyAttnDescriptor(state.attn), "cudnnDestroyAttnDescriptor");
    }
    if (state.handle != nullptr) {
        check_cudnn(cudnnDestroy(state.handle), "cudnnDestroy");
    }
}

void run_xydnn(const XyDnnState &state,
               const at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               at::Tensor &out)
{
    check_xydnn(xydnnMultiHeadAttnForward(state.handle,
                                          state.attn,
                                          -1,
                                          state.lo_win.empty() ? nullptr : state.lo_win.data(),
                                          state.hi_win.empty() ? nullptr : state.hi_win.data(),
                                          nullptr,
                                          nullptr,
                                          state.q_desc,
                                          q.data_ptr(),
                                          nullptr,
                                          state.k_desc,
                                          k.data_ptr(),
                                          state.v_desc,
                                          v.data_ptr(),
                                          state.o_desc,
                                          out.data_ptr(),
                                          0,
                                          nullptr,
                                          state.workspace.nbytes(),
                                          state.workspace.data_ptr(),
                                          state.reserve.nbytes(),
                                          state.reserve.data_ptr()),
                "xydnnMultiHeadAttnForward");
}

void run_cudnn_backend(const CudnnState &state,
                       const at::Tensor &q_tnbv,
                       const at::Tensor &k_tnbv,
                       const at::Tensor &v_tnbv,
                       at::Tensor &out_tnbv)
{
    check_cudnn(cudnnMultiHeadAttnForward(state.handle,
                                          state.attn,
                                          -1,
                                          state.lo_win.data(),
                                          state.hi_win.data(),
                                          state.seq_lengths_q.data_ptr<int>(),
                                          state.seq_lengths_k.data_ptr<int>(),
                                          state.q_desc,
                                          q_tnbv.data_ptr(),
                                          nullptr,
                                          state.k_desc,
                                          k_tnbv.data_ptr(),
                                          state.v_desc,
                                          v_tnbv.data_ptr(),
                                          state.o_desc,
                                          out_tnbv.data_ptr(),
                                          state.weight_bytes,
                                          state.weights.data_ptr(),
                                          state.workspace_bytes,
                                          state.workspace.data_ptr(),
                                          state.reserve_bytes,
                                          state.reserve.data_ptr()),
                "cudnnMultiHeadAttnForward");
}

CudnnTryResult try_run_cudnn_backend(const CudnnState &state,
                                     const at::Tensor &q_tnbv,
                                     const at::Tensor &k_tnbv,
                                     const at::Tensor &v_tnbv,
                                     at::Tensor &out_tnbv)
{
    const cudnnStatus_t status = cudnnMultiHeadAttnForward(state.handle,
                                                           state.attn,
                                                           -1,
                                                           state.lo_win.data(),
                                                           state.hi_win.data(),
                                                           state.seq_lengths_q.data_ptr<int>(),
                                                           state.seq_lengths_k.data_ptr<int>(),
                                                           state.q_desc,
                                                           q_tnbv.data_ptr(),
                                                           nullptr,
                                                           state.k_desc,
                                                           k_tnbv.data_ptr(),
                                                           state.v_desc,
                                                           v_tnbv.data_ptr(),
                                                           state.o_desc,
                                                           out_tnbv.data_ptr(),
                                                           state.weight_bytes,
                                                           state.weights.data_ptr(),
                                                           state.workspace_bytes,
                                                           state.workspace.data_ptr(),
                                                           state.reserve_bytes,
                                                           state.reserve.data_ptr());
    std::string reason;
    if (!try_cudnn(status, "cudnnMultiHeadAttnForward", &reason)) {
        return {false, reason};
    }
    const cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        std::ostringstream oss;
        oss << "cuDNN forward cuda sync failed: " << cudaGetErrorString(cuda_status);
        return {false, oss.str()};
    }
    return {true, ""};
}

float benchmark_ms(const std::string &name, int warmup, int iters, const std::function<void()> &fn)
{
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    check_cuda(cudaDeviceSynchronize(), (name + " warmup sync").c_str());

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start), (name + " start").c_str());
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    check_cuda(cudaEventRecord(stop), (name + " stop").c_str());
    check_cuda(cudaEventSynchronize(stop), (name + " sync").c_str());
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return elapsed_ms / static_cast<float>(iters);
}

}  // namespace

int main()
{
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available\n";
        return 1;
    }

    Problem p;
    p.batch = read_env_or("BATCH", 4);
    p.seqlen = read_env_or("SEQLEN", 1024);
    p.heads = read_env_or("HEADS", 8);
    p.kv_heads = read_env_or("KV_HEADS", p.heads);
    p.head_dim = read_env_or("HEAD_DIM", 64);
    p.causal = read_env_or("CAUSAL", 0) != 0;
    p.varlen = read_env_or("VARLEN", 0) != 0;
    p.softmax_scale = 1.0f / std::sqrt(static_cast<float>(p.head_dim));
    p.softcap = static_cast<float>(read_env_or("SOFTCAP", 0.0));
    p.dropout = static_cast<float>(read_env_or("DROPOUT", 0.0));
    p.dropout_seed = static_cast<unsigned long long>(read_env_or("DROPOUT_SEED", 1234));
    p.window_left = read_env_or("WINDOW_LEFT", -1);
    p.window_right = read_env_or("WINDOW_RIGHT", -1);
    p.runtime_window = read_env_or("RUNTIME_WINDOW", 0);
    p.precision = read_precision();
    const int warmup = read_env_or("WARMUP", 20);
    const int iters = read_env_or("ITERS", 100);

    if (!is_supported_head_dim(p.head_dim)) {
        std::cerr << "This xyDNN fast testbench supports HEAD_DIM in {32,64,96,128,192,256}\n";
        return 1;
    }
    if (p.heads % p.kv_heads != 0) {
        std::cerr << "KV_HEADS must divide HEADS\n";
        return 1;
    }
    if (p.dropout < 0.0f || p.dropout >= 1.0f) {
        std::cerr << "DROPOUT must be in [0, 1)\n";
        return 1;
    }
    if (p.dropout > 0.0f && p.softcap > 0.0f) {
        std::cerr << "DROPOUT with SOFTCAP is not supported by the current FA forward path\n";
        return 1;
    }

    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(p.precision.torch_dtype);
    std::vector<int> q_cu;
    std::vector<int> k_cu;
    int total_q = p.batch * p.seqlen;
    int total_k = p.batch * p.seqlen;
    int q_time_dim = p.seqlen;
    int k_time_dim = p.seqlen;
    size_t seq_array_size = 0;
    const int *q_seq_array = nullptr;
    const int *k_seq_array = nullptr;

    if (p.varlen) {
        q_cu = make_cu_seqlens(p.batch, p.seqlen);
        k_cu = make_cu_seqlens(p.batch, p.seqlen);
        total_q = q_cu.back();
        total_k = k_cu.back();
        q_time_dim = p.seqlen;
        k_time_dim = p.seqlen;
        seq_array_size = q_cu.size();
        q_seq_array = q_cu.data();
        k_seq_array = k_cu.data();
    }

    at::Tensor q;
    at::Tensor k;
    at::Tensor v;
    at::Tensor xydnn_out;
    at::Tensor q_cu_tensor;
    at::Tensor k_cu_tensor;
    at::Tensor fa_out_tensor;
    at::Tensor real_cudnn_out_tensor;
    at::Tensor xydnn_out_float;
    at::Tensor q_cudnn_tnbv;
    at::Tensor k_cudnn_tnbv;
    at::Tensor v_cudnn_tnbv;
    if (p.varlen) {
        q = torch::randn({total_q, p.heads, p.head_dim}, opts);
        k = torch::randn({total_k, p.kv_heads, p.head_dim}, opts);
        v = torch::randn({total_k, p.kv_heads, p.head_dim}, opts);
        xydnn_out = torch::empty_like(q);
        fa_out_tensor = torch::empty_like(q);
        q_cu_tensor = torch::from_blob(q_cu.data(),
                                       {static_cast<long>(q_cu.size())},
                                       torch::TensorOptions().dtype(torch::kInt32)).clone().to(torch::kCUDA);
        k_cu_tensor = torch::from_blob(k_cu.data(),
                                       {static_cast<long>(k_cu.size())},
                                       torch::TensorOptions().dtype(torch::kInt32)).clone().to(torch::kCUDA);
    } else {
        q = torch::randn({p.batch, p.seqlen, p.heads, p.head_dim}, opts);
        k = torch::randn({p.batch, p.seqlen, p.kv_heads, p.head_dim}, opts);
        v = torch::randn({p.batch, p.seqlen, p.kv_heads, p.head_dim}, opts);
        xydnn_out = torch::empty_like(q);
        xydnn_out_float = torch::empty({p.batch, p.seqlen, p.heads, p.head_dim}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        fa_out_tensor = torch::empty_like(q);
        auto qf = q.to(torch::kFloat);
        auto kf = k.to(torch::kFloat);
        auto vf = v.to(torch::kFloat);
        q_cudnn_tnbv = qf.reshape({p.batch, p.seqlen, p.heads * p.head_dim})
                              .permute({1, 0, 2})
                              .contiguous()
                              .reshape({p.seqlen, p.batch, 1, p.heads * p.head_dim});
        k_cudnn_tnbv = kf.reshape({p.batch, p.seqlen, p.kv_heads * p.head_dim})
                              .permute({1, 0, 2})
                              .contiguous()
                              .reshape({p.seqlen, p.batch, 1, p.kv_heads * p.head_dim});
        v_cudnn_tnbv = vf.reshape({p.batch, p.seqlen, p.kv_heads * p.head_dim})
                              .permute({1, 0, 2})
                              .contiguous()
                              .reshape({p.seqlen, p.batch, 1, p.kv_heads * p.head_dim});
        real_cudnn_out_tensor = torch::empty({p.seqlen, p.batch, 1, p.heads * p.head_dim},
                                             torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }

    XyDnnState xydnn = create_xydnn_state(p, q_time_dim, k_time_dim, total_q, q_seq_array, k_seq_array, seq_array_size);
    const bool has_local_window = p.window_left >= 0 || p.window_right >= 0;
    const bool should_try_real_cudnn = !p.varlen && p.kv_heads == p.heads && p.softcap == 0.0f &&
                                       p.dropout == 0.0f && !has_local_window;
    bool enable_real_cudnn = false;
    CudnnState cudnn;
    std::string cudnn_skip_reason;
    if (should_try_real_cudnn) {
        auto cudnn_try = create_cudnn_state(p, cudnn);
        if (!cudnn_try.ok) {
            cudnn_skip_reason = cudnn_try.reason;
            destroy_cudnn_state(cudnn);
        } else {
            auto weight_try = initialize_cudnn_identity_weights(cudnn, p.heads, p.head_dim);
            if (!weight_try.ok) {
                cudnn_skip_reason = weight_try.reason;
                destroy_cudnn_state(cudnn);
                enable_real_cudnn = false;
            } else {
                auto forward_try = try_run_cudnn_backend(cudnn, q_cudnn_tnbv, k_cudnn_tnbv, v_cudnn_tnbv, real_cudnn_out_tensor);
                if (forward_try.ok) {
                    enable_real_cudnn = true;
                } else {
                    cudnn_skip_reason = forward_try.reason;
                    destroy_cudnn_state(cudnn);
                    enable_real_cudnn = false;
                }
            }
        }
    } else {
        cudnn_skip_reason = "requires VARLEN=0, KV_HEADS=HEADS, SOFTCAP=0, DROPOUT=0, WINDOW_LEFT=-1, WINDOW_RIGHT=-1";
    }

    std::optional<at::Tensor> fa_out = fa_out_tensor;
    std::optional<at::Tensor> alibi = std::nullopt;
    auto make_fa_generator = [&]() -> std::optional<at::Generator> {
        if (p.dropout == 0.0f) {
            return std::nullopt;
        }
        at::Generator gen = at::cuda::detail::createCUDAGenerator();
        auto gen_impl = at::check_generator<at::CUDAGeneratorImpl>(gen);
        gen_impl->set_current_seed(p.dropout_seed);
        gen_impl->set_philox_offset_per_thread(0);
        return gen;
    };

    auto run_fa_reference = [&]() {
        std::optional<at::Generator> gen = make_fa_generator();
        if (p.varlen) {
            std::optional<at::Tensor> seqused_k = std::nullopt;
            std::optional<const at::Tensor> leftpad_k = std::nullopt;
            std::optional<at::Tensor> block_table = std::nullopt;
            auto result = flash_attn_sm80_varlen_fwd(q,
                                                     k,
                                                     v,
                                                     fa_out,
                                                     q_cu_tensor,
                                                     k_cu_tensor,
                                                     seqused_k,
                                                     leftpad_k,
                                                     block_table,
                                                     alibi,
                                                     p.seqlen,
                                                     p.seqlen,
                                                     p.dropout,
                                                     p.softmax_scale,
                                                     false,
                                                     p.causal,
                                                     p.window_left,
                                                     p.window_right,
                                                     p.softcap,
                                                     false,
                                                     gen,
                                                     0);
            (void)result;
        } else {
            auto result = flash_attn_sm80_fwd(q,
                                              k,
                                              v,
                                              fa_out,
                                              alibi,
                                              p.dropout,
                                              p.softmax_scale,
                                              p.causal,
                                              p.window_left,
                                              p.window_right,
                                              p.softcap,
                                              false,
                                              gen);
            (void)result;
        }
    };

    auto run_xydnn_backend = [&]() {
        run_xydnn(xydnn, q, k, v, xydnn_out);
    };

    auto run_real_cudnn_backend = [&]() {
        run_cudnn_backend(cudnn, q_cudnn_tnbv, k_cudnn_tnbv, v_cudnn_tnbv, real_cudnn_out_tensor);
    };

    run_fa_reference();
    run_xydnn_backend();
    if (enable_real_cudnn) {
        run_real_cudnn_backend();
    }
    check_cuda(cudaDeviceSynchronize(), "initial sync");

    const auto fa_diff = (xydnn_out - fa_out.value()).abs();
    const float fa_max_abs = fa_diff.max().item<float>();
    const float fa_mean_abs = fa_diff.mean().item<float>();
    const float fa_ref_max = fa_out.value().abs().max().item<float>();
    const float fa_rel_max = fa_max_abs / std::max(fa_ref_max, 1.0e-6f);
    float cudnn_max_abs = 0.0f;
    float cudnn_mean_abs = 0.0f;
    float cudnn_rel_max = 0.0f;
    if (enable_real_cudnn) {
        xydnn_out_float = xydnn_out.to(torch::kFloat);
        auto real_cudnn_bthd = real_cudnn_out_tensor.reshape({p.seqlen, p.batch, p.heads * p.head_dim})
                                                   .permute({1, 0, 2})
                                                   .contiguous()
                                                   .reshape({p.batch, p.seqlen, p.heads, p.head_dim});
        const auto cudnn_diff = (xydnn_out_float - real_cudnn_bthd).abs();
        cudnn_max_abs = cudnn_diff.max().item<float>();
        cudnn_mean_abs = cudnn_diff.mean().item<float>();
        const float cudnn_ref_max = real_cudnn_bthd.abs().max().item<float>();
        cudnn_rel_max = cudnn_max_abs / std::max(cudnn_ref_max, 1.0e-6f);
    }

    const float fa_ms = benchmark_ms("FA reference", warmup, iters, run_fa_reference);
    const float xydnn_ms = benchmark_ms("xyDNN", warmup, iters, run_xydnn_backend);
    float real_cudnn_ms = 0.0f;
    if (enable_real_cudnn) {
        real_cudnn_ms = benchmark_ms("cuDNN", warmup, iters, run_real_cudnn_backend);
    }

    const double tokens_q = static_cast<double>(total_q);
    const double max_k = p.varlen ? static_cast<double>(total_k) / p.batch : static_cast<double>(p.seqlen);
    const double flops = (p.causal && !p.varlen ? 2.0 : 4.0) * tokens_q * max_k * p.heads * p.head_dim;
    const double fa_tflops = flops / (fa_ms * 1.0e-3) / 1.0e12;
    const double xydnn_tflops = flops / (xydnn_ms * 1.0e-3) / 1.0e12;
    const double real_cudnn_tflops = enable_real_cudnn ? flops / (real_cudnn_ms * 1.0e-3) / 1.0e12 : 0.0;

    std::cout << "xyDNN vs cuDNN forward testbench\n";
    std::cout << "reference: repository flash_attn_sm80_* path; real cuDNN: system cudnnMultiHeadAttnForward when supported.\n";
    std::cout << "B=" << p.batch
              << " S=" << p.seqlen
              << " H=" << p.heads
              << " KV_H=" << p.kv_heads
              << " D=" << p.head_dim
              << " causal=" << p.causal
              << " varlen=" << p.varlen
              << " window_left=" << p.window_left
              << " window_right=" << p.window_right
              << " dropout=" << p.dropout
              << " dtype=" << p.precision.name
              << " warmup=" << warmup
              << " iters=" << iters << "\n";
    if (p.varlen) {
        std::cout << "total_q=" << total_q << " total_k=" << total_k << "\n";
    }
    std::cout << "precision xyDNN-vs-FA max_abs=" << fa_max_abs
              << " mean_abs=" << fa_mean_abs
              << " rel_max=" << fa_rel_max << "\n";
    if (enable_real_cudnn) {
        std::cout << "precision xyDNN-vs-cuDNN max_abs=" << cudnn_max_abs
                  << " mean_abs=" << cudnn_mean_abs
                  << " rel_max=" << cudnn_rel_max << "\n";
    } else {
        std::cout << "real cuDNN backend skipped: " << cudnn_skip_reason << "\n";
    }
    std::cout << "FA ref  avg latency: " << fa_ms << " ms, approx throughput: " << fa_tflops << " TFLOP/s\n";
    if (enable_real_cudnn) {
        std::cout << "cuDNN   avg latency: " << real_cudnn_ms << " ms, approx throughput: " << real_cudnn_tflops << " TFLOP/s\n";
    }
    std::cout << "xyDNN    avg latency: " << xydnn_ms << " ms, approx throughput: " << xydnn_tflops << " TFLOP/s\n";
    std::cout << "speedup xyDNN / FA-ref: " << (fa_ms / xydnn_ms) << "x\n";
    if (enable_real_cudnn) {
        std::cout << "speedup xyDNN / cuDNN: " << (real_cudnn_ms / xydnn_ms) << "x\n";
    }

    destroy_xydnn_state(xydnn);
    if (enable_real_cudnn) {
        destroy_cudnn_state(cudnn);
    }
    return fa_max_abs < 5.0e-2f ? 0 : 2;
}
