# FlashAttention SM80 Only

This repository is reduced to the SM80 CUDA implementation and organized around direct C++/CUDA builds.

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

## Build

Fast forward-only library for examples:

```bash
make fwd
```

Full library with forward + backward:

```bash
make full
```

## Outputs

- `build/libflash_attn_sm80_fwd.so`
- `build/libflash_attn_sm80.so`

Public API declaration:

- `csrc/flash_attn/include/flash_attn/flash_attn_api.h`

## Entry Points

- `flash_attn_sm80_fwd`
- `flash_attn_sm80_varlen_fwd`
- `flash_attn_sm80_bwd`
- `flash_attn_sm80_varlen_bwd`
- `flash_attn_sm80_fwd_kvcache`

Implementations live in `csrc/flash_attn/flash_api.cpp`. These APIs use `at::Tensor`, so runtime still depends on PyTorch C++/CUDA libraries.

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
