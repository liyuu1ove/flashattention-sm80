# xyDNN vs cuDNN/FA Testbench

Build:

```bash
make bench-xydnn-vs-cudnn
```

Run the implemented xyDNN MHA path matrix:

```bash
make test-xydnn-mha
```

The default matrix is a short smoke suite covering dense, causal, packed varlen,
GQA/MQA-style K/V heads, softcap, local window, `fp16`, and `bf16`. For a wider
shape sweep:

```bash
XYDNN_TEST_SUITE=full make test-xydnn-mha
```

Run padded/full attention:

```bash
BATCH=2 SEQLEN=128 HEADS=4 KV_HEADS=4 ITERS=20 WARMUP=5 ./build/bench_xydnn_vs_cudnn
```

Run packed varlen attention through xyDNN seq length arrays:

```bash
BATCH=2 SEQLEN=128 HEADS=4 KV_HEADS=4 VARLEN=1 ITERS=20 WARMUP=5 ./build/bench_xydnn_vs_cudnn
```

Environment knobs:

- `BATCH`, `SEQLEN`, `HEADS`, `KV_HEADS`, `HEAD_DIM`
- `DTYPE=fp16|bf16`
- `CAUSAL=0|1`
- `VARLEN=0|1`
- `SOFTCAP`
- `WINDOW_LEFT`, `WINDOW_RIGHT`
- `WARMUP`, `ITERS`
- `XYDNN_TEST_SUITE=smoke|full` for `scripts/run_xydnn_mha_matrix.py`
- `XYDNN_SKIP_BUILD=1` to skip the script's internal build step

Current note: the bench always compares xyDNN against the repository's torch-facing
`flash_attn_sm80_fwd` / `flash_attn_sm80_varlen_fwd` path. It also tries the system
NVIDIA cuDNN MHA backend when that combination is supported by the current cuDNN shim
(`VARLEN=0`, `KV_HEADS=HEADS`, no softcap, no local window); otherwise it reports the
cuDNN backend as skipped and still validates xyDNN against FA.
