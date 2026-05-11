#pragma once

#include <optional>
#include <vector>

#include <torch/torch.h>

extern "C" {

std::vector<at::Tensor> flash_attn_sm80_fwd(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    std::optional<at::Tensor> &out,
    std::optional<at::Tensor> &alibi_slopes,
    float p_dropout,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool return_softmax,
    std::optional<at::Generator> gen
);

std::vector<at::Tensor> flash_attn_sm80_varlen_fwd(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    std::optional<at::Tensor> &out,
    const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k,
    std::optional<at::Tensor> &seqused_k,
    std::optional<const at::Tensor> &leftpad_k,
    std::optional<at::Tensor> &block_table,
    std::optional<at::Tensor> &alibi_slopes,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool return_softmax,
    std::optional<at::Generator> gen,
    int num_splits
);

std::vector<at::Tensor> flash_attn_sm80_bwd(
    const at::Tensor &dout,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &softmax_lse,
    std::optional<at::Tensor> &dq,
    std::optional<at::Tensor> &dk,
    std::optional<at::Tensor> &dv,
    std::optional<at::Tensor> &alibi_slopes,
    float p_dropout,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool deterministic,
    std::optional<at::Generator> gen,
    std::optional<at::Tensor> &rng_state
);

std::vector<at::Tensor> flash_attn_sm80_varlen_bwd(
    const at::Tensor &dout,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &softmax_lse,
    std::optional<at::Tensor> &dq,
    std::optional<at::Tensor> &dk,
    std::optional<at::Tensor> &dv,
    const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k,
    std::optional<at::Tensor> &alibi_slopes,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool deterministic,
    std::optional<at::Generator> gen,
    std::optional<at::Tensor> &rng_state
);

std::vector<at::Tensor> flash_attn_sm80_fwd_kvcache(
    at::Tensor &q,
    const at::Tensor &kcache,
    const at::Tensor &vcache,
    std::optional<const at::Tensor> &k,
    std::optional<const at::Tensor> &v,
    std::optional<const at::Tensor> &seqlens_k,
    std::optional<const at::Tensor> &rotary_cos,
    std::optional<const at::Tensor> &rotary_sin,
    std::optional<const at::Tensor> &cache_batch_idx,
    std::optional<const at::Tensor> &leftpad_k,
    std::optional<at::Tensor> &block_table,
    std::optional<at::Tensor> &alibi_slopes,
    std::optional<at::Tensor> &out,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool is_rotary_interleaved,
    int num_splits
);

}
