#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "flash_attn/flash_attn_api.h"

namespace {

int read_env_or(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::atoi(value);
}

}

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available\n";
        return 1;
    }

    const int batch = read_env_or("BATCH", 8);
    const int seqlen = read_env_or("SEQLEN", 2048);
    const int heads = read_env_or("HEADS", 16);
    const int head_dim = read_env_or("HEAD_DIM", 64);
    const int warmup = read_env_or("WARMUP", 20);
    const int iters = read_env_or("ITERS", 100);
    const bool causal = read_env_or("CAUSAL", 1) != 0;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
    auto q = torch::randn({batch, seqlen, heads, head_dim}, opts);
    auto k = torch::randn({batch, seqlen, heads, head_dim}, opts);
    auto v = torch::randn({batch, seqlen, heads, head_dim}, opts);

    std::optional<at::Tensor> out = std::nullopt;
    std::optional<at::Tensor> alibi = std::nullopt;
    std::optional<at::Generator> gen = std::nullopt;

    for (int i = 0; i < warmup; ++i) {
        auto result = flash_attn_sm80_fwd(
            q, k, v, out, alibi, 0.0f, softmax_scale, causal, -1, -1, 0.0f, false, gen
        );
        (void)result;
    }
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    C10_CUDA_CHECK(cudaEventCreate(&start));
    C10_CUDA_CHECK(cudaEventCreate(&stop));

    C10_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        auto result = flash_attn_sm80_fwd(
            q, k, v, out, alibi, 0.0f, softmax_scale, causal, -1, -1, 0.0f, false, gen
        );
        (void)result;
    }
    C10_CUDA_CHECK(cudaEventRecord(stop));
    C10_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    C10_CUDA_CHECK(cudaEventDestroy(start));
    C10_CUDA_CHECK(cudaEventDestroy(stop));

    const double avg_ms = elapsed_ms / static_cast<double>(iters);
    const double flops = causal
        ? 2.0 * batch * heads * head_dim * seqlen * seqlen
        : 4.0 * batch * heads * head_dim * seqlen * seqlen;
    const double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

    std::cout << "flash_attn_sm80_fwd benchmark\n";
    std::cout << "B=" << batch
              << " S=" << seqlen
              << " H=" << heads
              << " D=" << head_dim
              << " causal=" << causal
              << " warmup=" << warmup
              << " iters=" << iters << "\n";
    std::cout << "avg latency: " << avg_ms << " ms\n";
    std::cout << "approx throughput: " << tflops << " TFLOP/s\n";

    return 0;
}
