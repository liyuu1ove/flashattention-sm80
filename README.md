# FlashAttention SM80 Only

This repository is reduced to the SM80 CUDA implementation and organized around direct C++/CUDA builds.
It also contains an xyDNN MHA forward API that mirrors the cuDNN descriptor/raw-pointer style while using
FlashAttention kernels internally.

## Layout

- `csrc/flash_attn/include/flash_attn/`
  Public exported C++ API
- `csrc/flash_attn/src/core/`
  Handwritten core headers and kernel support code
- `csrc/flash_attn/src/templates/`
  Shared launch templates included by generated `.cu` files
- `csrc/flash_attn/src/generated/`
  Auto-generated kernel instantiation files
- `csrc/flash_attn/tools/`
  Code generation utilities
- `examples/`
  Minimal correctness test and benchmark
- `docs/`
  xyDNN migration notes and benchmark documentation
- `scripts/`
  xyDNN test-matrix runner

## Build

Forward-only FlashAttention library for the torch-facing examples:

```bash
make fwd
```

Full torch-facing FlashAttention library with all generated SM80 forward/backward objects:

```bash
make full
```

xyDNN MHA forward shared library:

```bash
make xydnn-fwd
```

xyDNN minimal example and xyDNN-vs-FA/cuDNN benchmark:

```bash
make minimal-xydnn
make bench-xydnn-vs-cudnn
```

Standalone cuDNN smoke test, using the system cuDNN installation:

```bash
make minimal-cudnn
```

Run the xyDNN implemented-path matrix:

```bash
make test-xydnn-mha
```

Run a wider matrix:

```bash
XYDNN_TEST_SUITE=full make test-xydnn-mha
```

## Outputs

- `build/libflash_attn_sm80_fwd.so`
- `build/libflash_attn_sm80.so`
- `build/libxydnn_mha_fwd.so`
- `build/minimal_xydnn_mha_fwd`
- `build/bench_xydnn_vs_cudnn`
- `build/minimal_cudnn_mha_forward`

Public API declaration:

- `csrc/flash_attn/include/flash_attn/flash_attn_api.h`
- `csrc/flash_attn/include/flash_attn/xydnn_mha_fwd_api.h`

## Entry Points

- `flash_attn_sm80_fwd`
- `flash_attn_sm80_varlen_fwd`
- `flash_attn_sm80_bwd`
- `flash_attn_sm80_varlen_bwd`
- `flash_attn_sm80_fwd_kvcache`

Implementations live in `csrc/flash_attn/flash_api.cpp`. These APIs use `at::Tensor`, so runtime still depends on PyTorch C++/CUDA libraries.

## xyDNN MHA API

The xyDNN MHA API is implemented in `csrc/flash_attn/xydnn_mha_fwd_api.cpp`.
It follows the cuDNN-style API shape:

- `xydnnHandle_t` owns stream state and an internal cuBLAS handle for projection GEMMs
- `xydnnAttnDescriptor_t` stores MHA options
- `xydnnSeqDataDescriptor_t` describes Q/K/V/O seqdata layout
- `xydnnTensorDescriptor_t` describes packed projection weight views
- `xydnnMultiHeadAttnForward` accepts raw device pointers for Q/K/V/O, weights, workspace, and reserve

Currently implemented xyDNN features:

- Basic MHA forward
- Raw-pointer execution path with no `at::Tensor` usage inside the xyDNN API implementation
- `fp16` and `bf16` input/output paths
- SM80 fast forward kernels
- `HEAD_DIM` in `{32, 64, 96, 128, 192, 256}`
- Dense full-length forward
- Packed varlen forward through cumulative sequence offsets or seq-length arrays
- Causal and non-causal forward
- GQA/MQA by using fewer K/V heads than Q heads
- Attention dropout through `xydnnDropoutDescriptor_t`
- Local window options through `xydnnSetAttnDescriptorOptions`
- Softcap through `xydnnSetAttnDescriptorOptions`
- ALiBi slopes through `xydnnSetAttnDescriptorAlibiSlopes`
- Same-size Q/K/V/O projection weights through cuBLAS GEMM and caller workspace
- Projection weight metadata through `xydnnGetMultiHeadAttnWeights`
- Buffer sizing through `xydnnGetMultiHeadAttnBuffers`

Projection bias descriptors are present, but projection bias application is not wired into forward yet.
Dimension-changing projections are not supported yet; projection sizes must be either zero or equal to the corresponding input/output vector size.

## xyDNN Unsupported/Error Cases

The forward path returns `XYDNN_STATUS_NOT_SUPPORTED` for these intentionally unsupported paths:

- Non-zero post-dropout descriptors
- Attention dropout combined with softcap
- Non-null `residuals`
- Data types other than `XYDNN_DATA_HALF` and `XYDNN_DATA_BFLOAT16`
- Beam dimensions other than `1`
- `HEAD_DIM` outside `{32, 64, 96, 128, 192, 256}`
- Projection sizes that are non-zero but not equal to the corresponding Q/K/V/O vector size
- SeqData descriptors with a dimension count other than `XYDNN_SEQDATA_DIM_COUNT`
- Bias weight queries when projection biases are disabled
- Devices below SM80 return `XYDNN_STATUS_ARCH_MISMATCH`

Common `XYDNN_STATUS_BAD_PARAM` cases:

- Null required descriptors or Q/K/V/O pointers
- Mismatched descriptor data types
- Invalid or inconsistent batch, sequence, beam, or vector dimensions
- K/V sequence lengths disagree
- K/V head size does not match Q head size
- K/V heads do not divide Q heads
- Runtime batch/sequence exceeds descriptor maximums
- Missing or undersized reserve buffer
- Missing or undersized workspace when varlen host seq arrays or projection scratch are needed
- Invalid cumulative sequence lengths or totals
- Invalid ALiBi batch stride
- `weights == nullptr` with a non-zero weight byte size, or undersized weight buffers

Not implemented yet:

- Post/residual dropout execution
- Residual add
- Projection bias application
- Dimension-changing projections
- Automatic packing from padded varlen tensors
- Return-softmax/probability output policy
- Backward through xyDNN
- Paged KV cache
- Rotary embedding
- Split-KV dispatch through xyDNN

## Examples

```bash
make minimal
./build/minimal_fwd
```

```bash
make bench
./build/bench_fwd
```

Benchmark parameters can be overridden with env vars:

```bash
BATCH=8 SEQLEN=2048 HEADS=16 HEAD_DIM=64 WARMUP=20 ITERS=100 CAUSAL=1 ./build/bench_fwd
```

xyDNN minimal forward:

```bash
make minimal-xydnn
./build/minimal_xydnn_mha_fwd
DTYPE=bf16 ./build/minimal_xydnn_mha_fwd
```

xyDNN vs FA/cuDNN benchmark:

```bash
make bench-xydnn-vs-cudnn
BATCH=2 SEQLEN=128 HEADS=4 KV_HEADS=4 HEAD_DIM=64 DTYPE=fp16 ./build/bench_xydnn_vs_cudnn
BATCH=2 SEQLEN=128 HEADS=4 KV_HEADS=2 VARLEN=1 DTYPE=bf16 ./build/bench_xydnn_vs_cudnn
WINDOW_LEFT=32 WINDOW_RIGHT=16 DTYPE=fp16 ./build/bench_xydnn_vs_cudnn
```

Benchmark parameters:

- `BATCH`, `SEQLEN`, `HEADS`, `KV_HEADS`, `HEAD_DIM`
- `DTYPE=fp16|bf16`
- `CAUSAL=0|1`
- `VARLEN=0|1`
- `SOFTCAP`
- `DROPOUT`, `DROPOUT_SEED`
- `WINDOW_LEFT`, `WINDOW_RIGHT`
- `WARMUP`, `ITERS`

The benchmark always compares xyDNN against the repository FlashAttention reference path.
It also tries system cuDNN when supported by the current cuDNN comparison shim:
`VARLEN=0`, `KV_HEADS=HEADS`, `SOFTCAP=0`, `DROPOUT=0`, and no local window. Other combinations report cuDNN as skipped while still validating xyDNN against FA.
