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

Currently implemented xyDNN MHA forward features:

- cuDNN-style handle, descriptor, seqdata descriptor, tensor descriptor, dropout descriptor, workspace, reserve, and raw-pointer forward entry points.
- Raw-pointer Q/K/V/O execution path inside the xyDNN API implementation. The xyDNN forward wrapper builds `Flash_fwd_params` directly instead of using `at::Tensor`.
- Dense packed forward for full-length batches.
- Packed varlen forward through device cumulative sequence offsets from `xydnnSetAttnDescriptorVarlen`.
- Packed varlen forward through host seq-length or cumulative arrays stored in Q/K SeqData descriptors; xyDNN copies these arrays into caller workspace.
- `fp16` and `bf16` input/output data paths on SM80+.
- Head dimensions `{32, 64, 96, 128, 192, 256}`.
- Non-causal, causal, and fixed local-window attention.
- Runtime `loWinIdx[]` / `hiWinIdx[]` translation when the arrays describe one fixed full, causal, or local-window pattern.
- GQA/MQA by using fewer K/V heads than Q heads, as long as K/V heads divide Q heads.
- Attention dropout through `xydnnDropoutDescriptor_t` and reserve-space RNG state.
- Softcap through `xydnnSetAttnDescriptorOptions`.
- ALiBi slopes through `xydnnSetAttnDescriptorAlibiSlopes`.
- Same-size Q/K/V/O projection weights through cuBLAS GEMM and caller-provided workspace.
- Projection weight and bias metadata queries through `xydnnGetMultiHeadAttnWeights`.
- Weight, workspace, and reserve sizing through `xydnnGetMultiHeadAttnBuffers`.
- xyDNN-vs-FA benchmark, with best-effort system cuDNN comparison when the shape is compatible with the standalone cuDNN path.

Projection bias descriptors are present, but projection bias application is not wired into forward yet.
Dimension-changing projections are not supported yet; projection sizes must be either zero or equal to the corresponding input/output vector size.

## xyDNN MHA Function Signatures

Header:

```cpp
#include "flash_attn/xydnn_mha_fwd_api.h"
```

Handle and stream:

```cpp
xydnnStatus_t xydnnCreate(xydnnHandle_t *handle);
xydnnStatus_t xydnnDestroy(xydnnHandle_t handle);
xydnnStatus_t xydnnSetStream(xydnnHandle_t handle, cudaStream_t streamId);
xydnnStatus_t xydnnGetStream(xydnnHandle_t handle, cudaStream_t *streamId);
```

Attention descriptor:

```cpp
xydnnStatus_t xydnnCreateAttnDescriptor(xydnnAttnDescriptor_t *attnDesc);
xydnnStatus_t xydnnDestroyAttnDescriptor(xydnnAttnDescriptor_t attnDesc);

xydnnStatus_t xydnnSetAttnDescriptor(
    xydnnAttnDescriptor_t attnDesc,
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
    int maxBeamSize);
```

Optional descriptor controls:

```cpp
xydnnStatus_t xydnnSetAttnDescriptorOptions(
    xydnnAttnDescriptor_t attnDesc,
    int isCausal,
    int windowSizeLeft,
    int windowSizeRight,
    float softcap);

xydnnStatus_t xydnnSetAttnDescriptorAlibiSlopes(
    xydnnAttnDescriptor_t attnDesc,
    const void *alibiSlopes,
    int batchStride);

xydnnStatus_t xydnnSetAttnDescriptorVarlen(
    xydnnAttnDescriptor_t attnDesc,
    const int *cuSeqlensQO,
    const int *cuSeqlensKV,
    int totalQO,
    int totalKV);
```

SeqData descriptor:

```cpp
xydnnStatus_t xydnnCreateSeqDataDescriptor(xydnnSeqDataDescriptor_t *seqDataDesc);
xydnnStatus_t xydnnDestroySeqDataDescriptor(xydnnSeqDataDescriptor_t seqDataDesc);

xydnnStatus_t xydnnSetSeqDataDescriptor(
    xydnnSeqDataDescriptor_t seqDataDesc,
    xydnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const xydnnSeqDataAxis_t axes[],
    size_t seqLengthArraySize,
    const int seqLengthArray[],
    void *paddingFill);
```

Dropout descriptor:

```cpp
xydnnStatus_t xydnnCreateDropoutDescriptor(xydnnDropoutDescriptor_t *dropoutDesc);
xydnnStatus_t xydnnDestroyDropoutDescriptor(xydnnDropoutDescriptor_t dropoutDesc);
xydnnStatus_t xydnnDropoutGetStatesSize(xydnnHandle_t handle, size_t *sizeInBytes);

xydnnStatus_t xydnnSetDropoutDescriptor(
    xydnnDropoutDescriptor_t dropoutDesc,
    xydnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed);
```

Buffer and weight helpers:

```cpp
xydnnStatus_t xydnnGetMultiHeadAttnBuffers(
    xydnnHandle_t handle,
    const xydnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes,
    size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes);

xydnnStatus_t xydnnGetMultiHeadAttnWeights(
    xydnnHandle_t handle,
    const xydnnAttnDescriptor_t attnDesc,
    xydnnMultiHeadAttnWeightKind_t wKind,
    size_t weightSizeInBytes,
    const void *weights,
    xydnnTensorDescriptor_t wDesc,
    void **wAddr);
```

Forward:

```cpp
xydnnStatus_t xydnnMultiHeadAttnForward(
    xydnnHandle_t handle,
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
    void *reserveSpace);
```

## xyDNN MHA Forward Behavior

Typical call order:

1. Create a handle with `xydnnCreate`, optionally bind a CUDA stream with `xydnnSetStream`.
2. Create and fill `xydnnSeqDataDescriptor_t` for Q/K/V/O.
3. Create and fill `xydnnAttnDescriptor_t`.
4. Optionally configure causal/window/softcap, ALiBi, varlen, dropout, and projection weights.
5. Query sizes with `xydnnGetMultiHeadAttnBuffers`.
6. Allocate caller-owned `workSpace` and `reserveSpace`.
7. Call `xydnnMultiHeadAttnForward`.

Dense inputs are interpreted as packed batch-major device memory:

```text
Q/O: [batch, seqlen_q, nHeads, headDim]
K/V: [batch, seqlen_k, kvHeads, headDim]
```

Packed varlen inputs are interpreted as:

```text
Q/O: [total_q, nHeads, headDim]
K/V: [total_k, kvHeads, headDim]
```

Varlen can be provided in two ways:

- `xydnnSetAttnDescriptorVarlen` with device cumulative arrays `cuSeqlensQO` and `cuSeqlensKV`.
- Host seq-length or cumulative arrays in Q/K SeqData descriptors. In this mode xyDNN copies them into `workSpace` before launching FA.

`currIdx` is accepted for API shape compatibility and currently ignored. `loWinIdx` and `hiWinIdx` may be null; when both are provided, xyDNN translates them to the fixed FlashAttention window form if every query row matches one common full/causal/local pattern. Irregular per-token windows return `XYDNN_STATUS_NOT_SUPPORTED`. Non-null `residuals` is rejected.

When `weights == nullptr` and `weightSizeInBytes == 0`, Q/K/V/O are passed directly to FlashAttention. When `weights` is provided, xyDNN runs same-size Q/K/V projections into `workSpace`, runs attention, then runs the output projection back to `out`. Projection bias addresses can be queried, but bias application is not implemented yet.

`reserveSpace` is required. It stores FA softmax LSE data as `float`; when attention dropout is enabled, xyDNN also reserves 16 bytes at the start for the seed/offset pair. Use `xydnnGetMultiHeadAttnBuffers` to compute the required size.

## xyDNN Unsupported/Error Cases

The forward path returns `XYDNN_STATUS_NOT_SUPPORTED` for these intentionally unsupported paths:

- Non-zero post-dropout descriptors
- Attention dropout combined with softcap
- Non-null `residuals`
- Irregular per-token `loWinIdx[]` / `hiWinIdx[]` windows that cannot be represented as one fixed full, causal, or local FA window
- Data types other than `XYDNN_DATA_HALF` and `XYDNN_DATA_BFLOAT16`
- Beam dimensions other than `1`
- `HEAD_DIM` outside `{32, 64, 96, 128, 192, 256}`
- Projection sizes that are non-zero but not equal to the corresponding Q/K/V/O vector size
- SeqData descriptors with a dimension count other than `XYDNN_SEQDATA_DIM_COUNT`
- Bias weight queries when projection biases are disabled
- Devices below SM80 return `XYDNN_STATUS_ARCH_MISMATCH`

Common `XYDNN_STATUS_BAD_PARAM` cases:

- Null required output pointers in create/get/set helpers, such as `handle`, `streamId`, descriptor pointers, buffer-size pointers, or `wAddr`
- Null required forward descriptors or data pointers: `attnDesc`, Q/K/V/O descriptors, `queries`, `keys`, `values`, or `out`
- `weights == nullptr` with a non-zero `weightSizeInBytes`
- Dropout probability outside `[0, 1)`
- Invalid attention descriptor values: `nHeads <= 0`, `maxBatchSize <= 0`, `maxBeamSize <= 0`, non-positive Q/K/V sizes, non-positive max sequence lengths, negative projection sizes, or unknown `attnMode` bits
- Invalid options: `isCausal` not equal to `0` or `1`, negative `softcap`, or negative ALiBi `batchStride`
- Invalid tensor descriptor values: null dims/strides, `nbDims <= 0`, `nbDims > 4`, non-positive dims/strides, or unsupported data type in tensor descriptor setup
- Invalid SeqData values: null dims/axes, non-positive dimensions, duplicated axes, axes outside the `xydnnSeqDataAxis_t` enum, or non-zero `seqLengthArraySize` with null `seqLengthArray`
- Mismatched Q/K/V/O descriptor data types relative to `attnDesc->dataType`
- Invalid or inconsistent runtime shape: non-positive batch/sequence dimensions, K/V sequence lengths disagree, O sequence length differs from Q sequence length, descriptor batch sizes disagree, or vector sizes do not match `qSize`, `kSize`, `vSize`
- Invalid head layout: Q/O vector size not divisible by `nHeads`, O head size differs from Q head size, K/V vector size not divisible by Q head size, K/V head counts disagree, or K/V heads do not divide Q heads
- Runtime batch or sequence length exceeds `maxBatchSize`, `qoMaxSeqLength`, or `kvMaxSeqLength`
- Missing `reserveSpace` or `reserveSpaceSizeInBytes` smaller than the LSE storage plus dropout RNG state when dropout is enabled
- Invalid varlen configuration: only one of Q/K cumulative arrays is set, totals are missing or non-positive with device seqlens, totals are non-zero while cumulative pointers are null, or host seq arrays contain invalid lengths/cumulative offsets
- Missing or undersized `workSpace` when xyDNN must copy host seq arrays to device memory or when projection scratch is needed
- Undersized `weights` buffer in `xydnnGetMultiHeadAttnWeights`
- Invalid runtime window arrays: only one of `loWinIdx` / `hiWinIdx` is provided, a window index is negative, `hiWinIdx[i] < loWinIdx[i]`, or `hiWinIdx[i] > seqlen_k`

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
