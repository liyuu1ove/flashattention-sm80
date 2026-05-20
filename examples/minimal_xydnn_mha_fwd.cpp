#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "flash_attn/xydnn_mha_fwd_api.h"

namespace {

at::Tensor reference_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, float softmax_scale)
{
    auto qf = q.to(torch::kFloat);
    const int q_heads = q.size(2);
    const int kv_heads = k.size(2);
    auto kf = k.repeat_interleave(q_heads / kv_heads, 2).to(torch::kFloat);
    auto vf = v.repeat_interleave(q_heads / kv_heads, 2).to(torch::kFloat);
    auto scores = torch::einsum("bthd,bshd->bhts", {qf, kf}) * softmax_scale;
    auto probs = torch::softmax(scores, -1);
    return torch::einsum("bhts,bshd->bthd", {probs, vf}).to(q.scalar_type());
}

at::Tensor reference_varlen_attention(const at::Tensor& q,
                                      const at::Tensor& k,
                                      const at::Tensor& v,
                                      const std::vector<int>& q_lens,
                                      const std::vector<int>& k_lens,
                                      float softmax_scale)
{
    auto qf = q.to(torch::kFloat);
    auto kf = k.to(torch::kFloat);
    auto vf = v.to(torch::kFloat);
    std::vector<at::Tensor> outs;
    int q_off = 0;
    int k_off = 0;
    for (size_t i = 0; i < q_lens.size(); ++i) {
        auto qi = qf.narrow(0, q_off, q_lens[i]);
        auto ki = kf.narrow(0, k_off, k_lens[i]).repeat_interleave(q.size(1) / k.size(1), 1);
        auto vi = vf.narrow(0, k_off, k_lens[i]).repeat_interleave(q.size(1) / k.size(1), 1);
        auto scores = torch::einsum("thd,shd->hts", {qi, ki}) * softmax_scale;
        auto probs = torch::softmax(scores, -1);
        outs.push_back(torch::einsum("hts,shd->thd", {probs, vi}));
        q_off += q_lens[i];
        k_off += k_lens[i];
    }
    return torch::cat(outs, 0).to(q.scalar_type());
}

void check(xydnnStatus_t status, const char *what)
{
    if (status != XYDNN_STATUS_SUCCESS) {
        std::cerr << what << " failed with status " << static_cast<int>(status) << "\n";
        std::exit(1);
    }
}

bool use_bf16()
{
    const char *dtype = std::getenv("DTYPE");
    return dtype != nullptr && std::string(dtype) == "bf16";
}

}

int main()
{
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available\n";
        return 1;
    }

    const int batch = 2;
    const int seqlen = 128;
    const int heads = 4;
    const int kv_heads = 2;
    const int head_dim = 64;
    const int vect = heads * head_dim;
    const int kv_vect = kv_heads * head_dim;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const bool bf16 = use_bf16();
    const auto torch_dtype = bf16 ? torch::kBFloat16 : torch::kFloat16;
    const auto xydnn_dtype = bf16 ? XYDNN_DATA_BFLOAT16 : XYDNN_DATA_HALF;
    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch_dtype);
    const std::vector<int> q_lens = {80, 48};
    const std::vector<int> k_lens = {64, 32};
    const int total_q = q_lens[0] + q_lens[1];
    const int total_k = k_lens[0] + k_lens[1];
    const int max_q = 80;
    const int max_k = 64;
    const int q_cu_seqlens[] = {0, q_lens[0], total_q};
    const int k_cu_seqlens[] = {0, k_lens[0], total_k};
    auto q = torch::randn({total_q, heads, head_dim}, opts);
    auto k = torch::randn({total_k, kv_heads, head_dim}, opts);
    auto v = torch::randn({total_k, kv_heads, head_dim}, opts);
    auto out = torch::empty_like(q);
    auto reserve = torch::empty({batch, heads, seqlen}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    auto workspace = torch::empty({2 * (batch + 1)}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    xydnnHandle_t handle = nullptr;
    check(xydnnCreate(&handle), "xydnnCreate");
    check(xydnnSetStream(handle, at::cuda::getCurrentCUDAStream().stream()), "xydnnSetStream");

    xydnnAttnDescriptor_t attn = nullptr;
    check(xydnnCreateAttnDescriptor(&attn), "xydnnCreateAttnDescriptor");
    check(xydnnSetAttnDescriptor(attn,
                                 XYDNN_ATTN_QUERYMAP_ONE_TO_ONE,
                                 heads,
                                 softmax_scale,
                                 xydnn_dtype,
                                 XYDNN_DATA_FLOAT,
                                 XYDNN_DEFAULT_MATH,
                                 nullptr,
                                 nullptr,
                                 vect,
                                 kv_vect,
                                 kv_vect,
                                 0,
                                 0,
                                 0,
                                 0,
                                 seqlen,
                                 seqlen,
                                 batch,
                                 1),
          "xydnnSetAttnDescriptor");
    check(xydnnSetAttnDescriptorOptions(attn, 0, -1, -1, 0.0f),
          "xydnnSetAttnDescriptorOptions");
    check(xydnnSetAttnDescriptorAlibiSlopes(attn, nullptr, 0),
          "xydnnSetAttnDescriptorAlibiSlopes");

    const int dimQO[] = {max_q, batch, 1, vect};
    const int dimKV[] = {max_k, batch, 1, kv_vect};
    const xydnnSeqDataAxis_t axes[] = {
        XYDNN_SEQDATA_TIME_DIM,
        XYDNN_SEQDATA_BATCH_DIM,
        XYDNN_SEQDATA_BEAM_DIM,
        XYDNN_SEQDATA_VECT_DIM,
    };

    xydnnSeqDataDescriptor_t qDesc = nullptr;
    xydnnSeqDataDescriptor_t kDesc = nullptr;
    xydnnSeqDataDescriptor_t vDesc = nullptr;
    xydnnSeqDataDescriptor_t oDesc = nullptr;
    check(xydnnCreateSeqDataDescriptor(&qDesc), "xydnnCreateSeqDataDescriptor(q)");
    check(xydnnCreateSeqDataDescriptor(&kDesc), "xydnnCreateSeqDataDescriptor(k)");
    check(xydnnCreateSeqDataDescriptor(&vDesc), "xydnnCreateSeqDataDescriptor(v)");
    check(xydnnCreateSeqDataDescriptor(&oDesc), "xydnnCreateSeqDataDescriptor(o)");
    check(xydnnSetSeqDataDescriptor(qDesc, xydnn_dtype, 4, dimQO, axes, 3, q_cu_seqlens, nullptr), "xydnnSetSeqDataDescriptor(q)");
    check(xydnnSetSeqDataDescriptor(kDesc, xydnn_dtype, 4, dimKV, axes, 3, k_cu_seqlens, nullptr), "xydnnSetSeqDataDescriptor(k)");
    check(xydnnSetSeqDataDescriptor(vDesc, xydnn_dtype, 4, dimKV, axes, 3, k_cu_seqlens, nullptr), "xydnnSetSeqDataDescriptor(v)");
    check(xydnnSetSeqDataDescriptor(oDesc, xydnn_dtype, 4, dimQO, axes, 3, q_cu_seqlens, nullptr), "xydnnSetSeqDataDescriptor(o)");

    check(xydnnMultiHeadAttnForward(handle,
                                    attn,
                                    -1,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    qDesc,
                                    q.data_ptr(),
                                    nullptr,
                                    kDesc,
                                    k.data_ptr(),
                                    vDesc,
                                    v.data_ptr(),
                                    oDesc,
                                    out.data_ptr(),
                                    0,
                                    nullptr,
                                    workspace.nbytes(),
                                    workspace.data_ptr(),
                                    reserve.nbytes(),
                                    reserve.data_ptr()),
          "xydnnMultiHeadAttnForward");
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    auto out_ref = reference_varlen_attention(q, k, v, q_lens, k_lens, softmax_scale);
    auto max_diff = (out - out_ref).abs().max().item<float>();
    auto mean_diff = (out - out_ref).abs().mean().item<float>();

    check(xydnnDestroySeqDataDescriptor(qDesc), "xydnnDestroySeqDataDescriptor(q)");
    check(xydnnDestroySeqDataDescriptor(kDesc), "xydnnDestroySeqDataDescriptor(k)");
    check(xydnnDestroySeqDataDescriptor(vDesc), "xydnnDestroySeqDataDescriptor(v)");
    check(xydnnDestroySeqDataDescriptor(oDesc), "xydnnDestroySeqDataDescriptor(o)");
    check(xydnnDestroyAttnDescriptor(attn), "xydnnDestroyAttnDescriptor");
    check(xydnnDestroy(handle), "xydnnDestroy");

    std::cout << "xydnnMultiHeadAttnForward ok (" << (bf16 ? "bf16" : "fp16") << ")\n";
    std::cout << "max abs diff: " << max_diff << "\n";
    std::cout << "mean abs diff: " << mean_diff << "\n";

    return max_diff < 5e-2f ? 0 : 2;
}
