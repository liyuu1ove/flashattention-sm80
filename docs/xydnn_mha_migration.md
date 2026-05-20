# xyDNN MHA Migration Table

This document tracks the current FlashAttention-to-xyDNN MHA migration status.
The xyDNN API intentionally follows the cuDNN MHA shape: descriptors, raw device
pointers, caller-owned workspace/reserve buffers, and explicit handle/stream state.

## Status Legend

- Done: implemented in the xyDNN forward path
- Partial: API or core wiring exists, but behavior is limited
- Planned: suitable for a later migration step
- Not planned: intentionally out of the current minimal-forward scope

## Current Scope

- Basic forward MHA only
- Raw pointer xyDNN execution path; no `at::Tensor` dependency inside the xyDNN API implementation
- SM80 fast kernels for `HEAD_DIM` in `{32, 64, 96, 128, 192, 256}`
- `fp16` and `bf16` inputs/outputs
- Causal and non-causal forward
- GQA/MQA through fewer K/V heads than Q heads
- Packed varlen forward through cumulative sequence offsets or seq-length arrays
- Attention dropout forward through xyDNN dropout descriptors
- Optional local window, softcap, and ALiBi descriptor options
- Optional Q/K/V/O projection weights for same-size projections
- No post-dropout, residuals, backward, paged KV cache, rotary embedding, or split-KV path yet

## Migration Table

| Feature | cuDNN behavior | FA capability | xyDNN status | Notes |
|---|---|---|---|---|
| Handle / stream | Library handle binds a CUDA stream | FA launch path accepts a stream | Done | `xydnnCreate`, `xydnnDestroy`, `xydnnSetStream`, `xydnnGetStream`; cuBLAS projection handle follows the same stream |
| Descriptor-driven API | Attention, tensor, and seq descriptors drive execution | FA uses internal params structs | Done | `xydnnAttnDescriptor_t`, `xydnnSeqDataDescriptor_t`, `xydnnTensorDescriptor_t` |
| Bare pointers | Inputs, outputs, weights, workspace, reserve are raw pointers | FA kernels already use device pointers internally | Done | xyDNN implementation does not require torch tensors |
| Build targets | cuDNN links as a system library | FA kernels are compiled into local shared objects | Done | `make xydnn-fwd`, `make minimal-xydnn`, `make minimal-cudnn`, `make bench-xydnn-vs-cudnn` |
| Basic forward MHA | `cudnnMultiHeadAttnForward` | FA forward kernels | Done | `xydnnMultiHeadAttnForward` maps descriptors into `Flash_fwd_params` |
| Causal / non-causal | Both supported | FA supports both | Done | Descriptor option controls causal mode |
| fp16 | Supported input/output dtype | FA supports fp16 kernels | Done | Built and verified through `DTYPE=fp16` |
| bf16 | Supported input/output dtype | FA supports bf16 kernels on SM80+ | Done | Built and verified through `DTYPE=bf16` |
| Attention scale | User-provided scale | FA uses `softmax_scale` | Done | Stored in attention descriptor |
| Max batch / max seq | Descriptor carries upper bounds | FA validates runtime shapes | Done | Used for validation and workspace/reserve sizing |
| GQA / MQA | Q heads may map to fewer K/V heads | FA supports `h` and `h_k` | Done | K/V head count is inferred from K/V vector size and Q head dimension |
| Varlen / sequence length arrays | SeqData supports per-batch sequence lengths | FA supports packed varlen offsets | Done | Supports descriptor cumulative offsets and host seq-length arrays copied through workspace |
| Device cumulative seqlens | Caller can provide device-side arrays | FA varlen uses device cumulative offsets | Done | `xydnnSetAttnDescriptorVarlen` accepts device `cu_seqlens_q/k` plus totals |
| Padded regular tensors | cuDNN seqdata can describe padded TNBV tensors | FA regular path uses dense packed batch tensors | Done | Dense non-varlen path works for full-length batches |
| Auto-pack padded varlen tensors | cuDNN can use seq lengths with padded descriptors | FA varlen expects packed tokens | Partial | xyDNN handles packed varlen; it does not compact padded TNBV input automatically |
| Local window | Left/right windowed attention | FA supports local attention | Done | Descriptor options map to FA window params; cuDNN per-token window arrays are ignored |
| Softcap | Logit softcapping | FA supports softcap | Done | Descriptor option maps to FA softcap params |
| ALiBi | Per-head attention slope bias | FA supports ALiBi slopes | Done | Descriptor stores raw slope pointer and batch stride |
| Q/K/V/O projection weights | cuDNN owns optional projection matrices | Projections can be run outside FA core | Partial | Same-size Q/K/V/O projection GEMMs are wired through cuBLAS and device workspace |
| Projection weight descriptors | cuDNN exposes packed weight sub-tensors | xyDNN can expose tensor views | Done | `xydnnGetMultiHeadAttnWeights` fills descriptor metadata and returns raw sub-buffer address |
| Projection biases | cuDNN supports optional Q/K/V/O biases | FA core does not consume projection bias | Partial | Bias descriptors exist, but forward does not apply bias yet |
| Dimension-changing projections | cuDNN can project to configured sizes | FA core requires head-aligned Q/K/V | Partial | Current xyDNN only accepts zero or same-size projection dims |
| Workspace / reserve | Caller provides buffers | FA uses auxiliary buffers for LSE and varlen metadata | Done | `xydnnGetMultiHeadAttnBuffers` includes seq-length scratch and projection scratch |
| cuDNN comparison bench | Compare against system cuDNN | FA reference exists | Done | `examples/bench_xydnn_vs_cudnn.cpp` compares xyDNN vs FA and tries real cuDNN when supported |
| Minimal cuDNN forward | Standalone cuDNN smoke test | N/A | Done | `examples/minimal_cudnn_mha_forward.cpp` verifies local cuDNN availability |
| Dropout | Attention dropout and reserve states | FA has dropout-capable paths | Partial | Attention dropout forward is wired through `xydnnDropoutDescriptor_t`; post-dropout and softcap+dropout are still rejected |
| Residuals | Optional residual input | Could be fused after attention | Not planned for minimal path | xyDNN forward rejects non-null residual pointer |
| Return softmax/probabilities | cuDNN reserve can expose internals indirectly | FA can write probabilities in dropout-oriented paths | Planned | Need a clear xyDNN policy before exposing |
| Paged KV cache | Block table / cache paging | FA supports KV cache paths | Planned | Requires additional API surface |
| Rotary embedding | Rotary Q/K mixing | FA supports rotary in KV-cache path | Planned | Separate from basic MHA |
| Backward / gradients | dQ/dK/dV and weight gradients | FA supports backward kernels | Not planned for minimal path | Add after forward parity is stable |
| Split-KV | Parallel split reduction for performance | FA has split-KV dispatch | Planned | Current Makefile fast target intentionally builds only direct fwd kernels |

## Verified Commands

The current state has been compiled and smoke-tested with:

```bash
make xydnn-fwd minimal-xydnn bench-xydnn-vs-cudnn -j1
./build/minimal_xydnn_mha_fwd
DTYPE=bf16 ./build/minimal_xydnn_mha_fwd
BATCH=2 SEQLEN=128 HEADS=4 KV_HEADS=4 HEAD_DIM=64 WARMUP=2 ITERS=5 DTYPE=fp16 ./build/bench_xydnn_vs_cudnn
BATCH=2 SEQLEN=128 HEADS=4 KV_HEADS=4 HEAD_DIM=64 WARMUP=2 ITERS=5 DTYPE=bf16 ./build/bench_xydnn_vs_cudnn
BATCH=2 SEQLEN=128 HEADS=4 KV_HEADS=4 HEAD_DIM=64 WARMUP=1 ITERS=2 DTYPE=fp16 DROPOUT=0.1 DROPOUT_SEED=1234 ./build_dropout_check/bench_xydnn_vs_cudnn
BATCH=2 SEQLEN=128 HEADS=4 KV_HEADS=4 HEAD_DIM=64 WARMUP=1 ITERS=2 DTYPE=bf16 DROPOUT=0.1 DROPOUT_SEED=1234 ./build_dropout_check/bench_xydnn_vs_cudnn
```

Observed smoke-test status:

| Test | Status | Notes |
|---|---|---|
| Minimal xyDNN fp16 | Pass | Varlen packed forward against torch reference |
| Minimal xyDNN bf16 | Pass | Varlen packed forward against torch reference |
| Bench xyDNN vs FA fp16 | Pass | `max_abs=0` for the tested shape |
| Bench xyDNN vs FA bf16 | Pass | `max_abs=0` for the tested shape |
| Bench xyDNN vs FA dropout fp16 | Pass | `max_abs=0` with fixed `DROPOUT_SEED` |
| Bench xyDNN vs FA dropout bf16 | Pass | `max_abs=0` with fixed `DROPOUT_SEED` |
| Bench xyDNN vs cuDNN fp16 input path | Pass | cuDNN side currently runs float descriptors/weights for the supported comparison case |
| Bench xyDNN vs cuDNN bf16 input path | Pass | xyDNN/FA run bf16; cuDNN comparison remains the float cuDNN MHA path |

## Remaining Near-Term Work

1. Apply projection biases in the forward path.
2. Add tests for projection weights, including `fp16` and `bf16`.
3. Decide whether xyDNN should auto-pack padded varlen inputs or require packed varlen pointers.
4. Add projection-weight tests across the supported head dimensions.
5. Decide the public behavior for return-softmax / reserve contents.
