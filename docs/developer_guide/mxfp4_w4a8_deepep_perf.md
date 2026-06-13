# MXFP4 W4A8 DeepEP Performance Log

Date: 2026-06-13

This note records the current MiMoV2 MXFP4 W4A8 DeepEP optimization state and the
fixed-shape serving benchmark results used during development.

## Test Setup

Model and tokenizer: `/preset-models`

Server shape:

```bash
python3 -m sglang.launch_server \
  --model-path /preset-models \
  --served-model-name mimo-v2-flash \
  --pp-size 1 \
  --tp-size 8 \
  --page-size 1 \
  --host 127.0.0.1 \
  --port 31084 \
  --trust-remote-code \
  --watchdog-timeout 1000000 \
  --mem-fraction-static 0.80 \
  --max-total-tokens 32768 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser mimo \
  --context-length 8192 \
  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
  --load-balance-method round_robin \
  --attention-backend fa3 \
  --allow-auto-truncate \
  --quantization fp8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend mxfp4_w4a8 \
  --moe-dense-tp-size 1 \
  --cuda-graph-bs 1 2 4 8 \
  --cuda-graph-max-bs 8
```

Benchmark shape:

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai \
  --host 127.0.0.1 \
  --port 31084 \
  --model /preset-models \
  --served-model-name mimo-v2-flash \
  --tokenizer /preset-models \
  --dataset-name random-ids \
  --random-input-len 512 \
  --random-output-len 64 \
  --random-range-ratio 1.0 \
  --num-prompts 8 \
  --max-concurrency 8 \
  --warmup-requests 1 \
  --disable-tqdm
```

Use `--max-concurrency 1` for C1.

Runtime note: with this model and `tp-size=8`, CUDA graph capture logs
`Capture cuda graph bs [8]`. Requested `1 2 4` are filtered by the attention TP
gather constraint, so decode graph replay pads smaller concurrency to the bs=8
graph. Decode logs confirm `cuda graph: True`.

## Implemented Changes

Kept changes:

- Added MXFP4 W4A8 support for DeepEP normal dispatch. This enables
  `--deepep-mode auto`: prefill/extend uses DeepEP normal and decode uses DeepEP
  low-latency.
- Added a contiguous grouped Triton GEMM path for the DeepEP normal output
  layout, keyed by `expert_start` and `num_tokens_per_expert`.
- Reused `ep_scatter` cumsum output through an optional `expert_start_out`,
  avoiding a separate `torch.cumsum` on the normal path.
- Copied `num_recv_tokens_per_expert` to GPU via pinned CPU memory with
  non-blocking transfer when copy engines are allowed.
- Changed normal-path `output_index` from `topk_ids` dtype to `int32`.
- Passed actual routed token count into the low-latency MXFP4 W4A8 Triton path
  and used it to cap GEMM grid M, avoiding launches over the full padded
  DeepEP low-latency M dimension.
- Tuned MXFP4 W4A8 normal and low-latency Triton tile choices for the observed
  MiMoV2 dimensions.

Tried and reverted:

- Low-latency `BLOCK_M=2/4` for small routed-token counts. End-to-end C1/C8 did
  not improve versus `BLOCK_M=8`.
- Skipping `m_indices` writes in `ep_scatter`. Smoke test passed, but synthetic
  FP8 scatter timing was unchanged: `with_m_indices=0.0389 ms`,
  `without_m_indices=0.0391 ms` for `512 x 6144`, `topk=8`, `all_tokens=4096`.

## Benchmarks

All rows use fixed 512 input tokens and 64 output tokens.

| Path | Concurrency | Mean TTFT (ms) | Mean TPOT (ms) | Output tok/s | Notes |
|---|---:|---:|---:|---:|---|
| No DeepEP, W4A16-like baseline | 1 | 202.20 | 86.13 | - | Previous reference run |
| No DeepEP, W4A16-like baseline | 8 | 399.20 | 86.95 | 87.05 | Previous reference run |
| DeepEP auto before these optimizations | 1 | 2636.54 | 81.27 | - | After earlier fused activation/max work |
| DeepEP auto before these optimizations | 8 | 7400.12 | 141.80 | - | After earlier fused activation/max work |
| Final MXFP4 W4A8 DeepEP auto | 1 | 332.50 | 68.58 | 13.73 | Warm-server run |
| Final MXFP4 W4A8 DeepEP auto | 8 | 750.84 | 106.57 | 67.67 | Average of two warm-server runs |

Final deltas versus the previous DeepEP auto path:

| Concurrency | TTFT Delta | TPOT Delta |
|---|---:|---:|
| 1 | -87.39% | -15.61% |
| 8 | -89.85% | -24.84% |

Final deltas versus the no-DeepEP W4A16-like reference:

| Concurrency | TTFT Delta | TPOT Delta |
|---|---:|---:|
| 1 | +64.44% | -20.38% |
| 8 | +88.08% | +22.56% |

## Current Bottleneck

The current Triton MXFP4 path decodes E2M1 nibbles and applies scales inside the
GEMM loop. This keeps the implementation checkpoint-layout-compatible, but it is
still much slower than a native/specialized MXFP4 W4A8 GEMM would be. The next
large performance step is likely one of:

- a specialized CUTLASS/DeepGEMM-style MXFP4 E2M1 x FP8 kernel for these shapes;
- a load-time weight conversion into an existing W4A8-compatible layout, if the
  accuracy and layout semantics are acceptable;
- a deeper DeepEP low-latency path that can avoid padding smaller decode batches
  to the bs=8 CUDA graph work shape under `tp-size=8`.
