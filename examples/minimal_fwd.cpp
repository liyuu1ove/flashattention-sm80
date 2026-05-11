#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "flash_attn/flash_attn_api.h"

namespace {

at::Tensor reference_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    float softmax_scale
) {
    auto qf = q.to(torch::kFloat);
    auto kf = k.to(torch::kFloat);
    auto vf = v.to(torch::kFloat);

    auto scores = torch::einsum("bthd,bshd->bhts", {qf, kf}) * softmax_scale;
    if (causal) {
        const auto seqlen_q = q.size(1);
        const auto seqlen_k = k.size(1);
        auto mask = torch::ones({seqlen_q, seqlen_k}, torch::TensorOptions().device(q.device()).dtype(torch::kBool)).tril();
        scores = scores.masked_fill(mask.logical_not().unsqueeze(0).unsqueeze(0), -INFINITY);
    }
    auto probs = torch::softmax(scores, -1);
    return torch::einsum("bhts,bshd->bthd", {probs, vf}).to(q.scalar_type());
}

}

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available\n";
        return 1;
    }

    const int batch = 2;
    const int seqlen = 128;
    const int heads = 4;
    const int head_dim = 64;
    const bool causal = true;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
    auto q = torch::randn({batch, seqlen, heads, head_dim}, opts);
    auto k = torch::randn({batch, seqlen, heads, head_dim}, opts);
    auto v = torch::randn({batch, seqlen, heads, head_dim}, opts);

    std::optional<at::Tensor> out = std::nullopt;
    std::optional<at::Tensor> alibi = std::nullopt;
    std::optional<at::Generator> gen = std::nullopt;

    auto result = flash_attn_sm80_fwd(
        q,
        k,
        v,
        out,
        alibi,
        0.0f,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0f,
        false,
        gen
    );

    auto out_fa = result[0];
    auto out_ref = reference_attention(q, k, v, causal, softmax_scale);
    auto max_diff = (out_fa - out_ref).abs().max().item<float>();
    auto mean_diff = (out_fa - out_ref).abs().mean().item<float>();

    std::cout << "flash_attn_sm80_fwd ok\n";
    std::cout << "output shape: " << out_fa.sizes() << "\n";
    std::cout << "max abs diff: " << max_diff << "\n";
    std::cout << "mean abs diff: " << mean_diff << "\n";

    return max_diff < 5e-2f ? 0 : 2;
}
