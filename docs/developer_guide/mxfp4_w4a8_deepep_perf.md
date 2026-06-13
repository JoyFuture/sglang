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
- Fixed the DeepEP normal-path activation quantization. The normal path now uses
  the same EP `silu_and_mul_masked_post_quant_fwd` gate/up convention as the
  low-latency path instead of the generic fused FP8 quant helper.
- Added a Hopper/SM90 `tl.dot_scaled` MXFP4 W4A8 grouped GEMM path for both
  DeepEP low-latency and normal layouts. The kernel uses raw
  `e4m3 x e2m1` dot-scaled tiles and applies the existing float32 activation and
  weight scales outside the dot. This is enabled by default and can be disabled
  with `SGLANG_MXFP4_W4A8_DOT_SCALED=0` to fall back to the previous hand-decoded
  E2M1 Triton kernel.
- Added an experimental low-latency decode path that converts MXFP4 weight
  scales to e8m0/uint8 at load time and passes them directly to `tl.dot_scaled`.
  It is disabled by default after end-to-end serving benchmarks regressed. Set
  `SGLANG_MXFP4_W4A8_E8M0_LL=1` to enable it for targeted experiments.

Correctness note:

- A post-optimization serving smoke test returned repetitive unrelated Chinese
  tokens for a fixed prompt. Synthetic isolation showed the low-latency Triton
  path matched the reference within quantization error, but the normal contig
  path diverged heavily after fused SiLU+mul+FP8 quantization.
- After the fix, synthetic `mxfp4_w4a8_deepep_normal_triton` output matches the
  equivalent low-latency padded Triton path exactly for the tested routed-token
  layout: `max_abs_diff=0.0`, `mean_abs_diff=0.0`.
- The full DeepEP normal path including `ep_scatter` and `ep_gather` also matches
  manual aggregation over the low-latency Triton output exactly for the tested
  `16 tokens x topk 8` synthetic case: `max_abs_diff=0.0`,
  `mean_abs_diff=0.0`.

Tried and reverted:

- Low-latency `BLOCK_M=2/4` for small routed-token counts. End-to-end C1/C8 did
  not improve versus `BLOCK_M=8`.
- Skipping `m_indices` writes in `ep_scatter`. Smoke test passed, but synthetic
  FP8 scatter timing was unchanged: `with_m_indices=0.0389 ms`,
  `without_m_indices=0.0391 ms` for `512 x 6144`, `topk=8`, `all_tokens=4096`.
- A local 2D contiguous SiLU+mul+FP8 activation-quant candidate for the DeepEP
  normal path. Synthetic full-path comparison still matched the low-latency
  equivalent (`max_abs_diff=0.0`, `mean_abs_diff=0.0`), and the isolated kernel
  was neutral at `M=512` and about `1.31x` faster at `M=4096`. It was not
  committed or kept after a serving bad-text report; the running service was
  restored to the known-good 3D `silu_and_mul_masked_post_quant_fwd` wrapper
  from the correctness fix before further tuning.
- Passing the existing float32 scales directly into `tl.dot_scaled`. Triton 3.6.0
  failed in the SM90 MLIR lowering pipeline for that shape, so the kept kernel
  leaves `tl.dot_scaled` scales as `None` and applies float32 scales to each
  32-wide K tile after the raw dot.
- Enabling e8m0 MXFP4 weight scales directly in the low-latency decode
  `tl.dot_scaled` kernels. Synthetic low-latency GEMM timings improved, but
  serving TPOT regressed and the extra scale buffers reduced available rank-0
  GPU memory after CUDA graph capture from about `49.64 GB` to about `46.01 GB`.
  The code remains behind `SGLANG_MXFP4_W4A8_E8M0_LL=1`, but the default path
  does not allocate those buffers and keeps the external float-scale kernel.

## Benchmarks

All rows use fixed 512 input tokens and 64 output tokens.

| Path | Concurrency | Mean TTFT (ms) | Mean TPOT (ms) | Output tok/s | Notes |
|---|---:|---:|---:|---:|---|
| No DeepEP, W4A16-like baseline | 1 | 202.20 | 86.13 | - | Previous reference run |
| No DeepEP, W4A16-like baseline | 8 | 399.20 | 86.95 | 87.05 | Previous reference run |
| DeepEP auto before these optimizations | 1 | 2636.54 | 81.27 | - | After earlier fused activation/max work |
| DeepEP auto before these optimizations | 8 | 7400.12 | 141.80 | - | After earlier fused activation/max work |
| MXFP4 W4A8 DeepEP auto before correctness fix | 1 | 332.50 | 68.58 | 13.73 | Warm-server run; later found to produce bad text |
| MXFP4 W4A8 DeepEP auto before correctness fix | 8 | 750.84 | 106.57 | 67.67 | Average of two warm-server runs; later found to produce bad text |
| Correctness-fixed MXFP4 W4A8 DeepEP auto | 1 | 1559.08 | 69.13 | 10.80 | Warm-server run after normal-path activation quantization fix |
| Correctness-fixed MXFP4 W4A8 DeepEP auto | 8 | 695.21 | 107.75 | 67.46 | Warm-server run after normal-path activation quantization fix |
| Hopper dot_scaled MXFP4 W4A8 DeepEP auto | 1 | 280.19 | 28.33 | 30.93 | Warm-server run with decode CUDA graph enabled |
| Hopper dot_scaled MXFP4 W4A8 DeepEP auto | 8 | 318.81 | 34.12 | 204.61 | Warm-server run with decode CUDA graph enabled |
| Experimental e8m0 low-latency scale path | 1 | 284.93 | 31.92 | 27.82 | `SGLANG_MXFP4_W4A8_E8M0_LL=1`; warm-server run |
| Experimental e8m0 low-latency scale path | 8 | 320.79 | 39.68 | 179.02 | Hot rerun; first C8 run was `335.35 ms / 39.71 ms / 178.01 tok/s` |
| Current default after e8m0 default-off | 1 | 280.30 | 28.28 | 30.98 | `SGLANG_MXFP4_W4A8_E8M0_LL` unset; decode CUDA graph enabled |
| Current default after e8m0 default-off | 8 | 328.04 | 34.08 | 204.10 | `SGLANG_MXFP4_W4A8_E8M0_LL` unset; decode CUDA graph enabled |

Correctness-fixed deltas versus the previous DeepEP auto path:

| Concurrency | TTFT Delta | TPOT Delta |
|---|---:|---:|
| 1 | -40.87% | -14.94% |
| 8 | -90.61% | -24.01% |

Correctness-fixed deltas versus the no-DeepEP W4A16-like reference:

| Concurrency | TTFT Delta | TPOT Delta |
|---|---:|---:|
| 1 | +671.06% | -19.74% |
| 8 | +74.15% | +23.92% |

Correctness-fixed deltas versus the bad-text pre-correctness MXFP4 W4A8 DeepEP
path:

| Concurrency | TTFT Delta | TPOT Delta |
|---|---:|---:|
| 1 | +368.90% | +0.80% |
| 8 | -7.41% | +1.11% |

Hopper dot_scaled deltas versus the correctness-fixed MXFP4 W4A8 DeepEP path:

| Concurrency | TTFT Delta | TPOT Delta |
|---|---:|---:|
| 1 | -82.03% | -59.02% |
| 8 | -54.14% | -68.33% |

Hopper dot_scaled deltas versus the no-DeepEP W4A16-like reference:

| Concurrency | TTFT Delta | TPOT Delta |
|---|---:|---:|
| 1 | +38.57% | -67.11% |
| 8 | -20.14% | -60.76% |

Kernel microbenchmarks for the dot_scaled path:

| Kernel Shape | Previous Decode Kernel (ms) | dot_scaled Kernel (ms) | Speedup | Mean Diff | Mean Abs Ref |
|---|---:|---:|---:|---:|---:|
| Low-latency GEMM1, `N=32768,K=6144`, 16 routed tokens | 4.9073 | 1.0476 | 4.68x | 0.1487 | 185.7580 |
| Low-latency GEMM2, `N=6144,K=16384`, 16 routed tokens | 3.5192 | 1.0582 | 3.33x | 0.2535 | 311.1642 |
| Normal GEMM1, `N=32768,K=6144`, 4096 routed tokens | 98.1322 | 11.7592 | 8.35x | 0.1499 | 188.5663 |
| Normal GEMM2, `N=6144,K=16384`, 4096 routed tokens | 57.0733 | 6.8607 | 8.32x | 0.2511 | 308.2648 |

Synthetic low-latency e8m0 scale microbenchmarks against the current external
float-scale dot_scaled kernel:

| Kernel Shape | External Float Scale (ms) | e8m0 Scale (ms) | Speedup | Max Diff | Mean Diff | Mean Abs Ref |
|---|---:|---:|---:|---:|---:|---:|
| Low-latency GEMM1, `N=32768,K=6144`, 16 routed tokens | 1.0475 | 0.7609 | 1.38x | 32.0 | 0.000263 | 2011.4004 |
| Low-latency GEMM2, `N=6144,K=16384`, 16 routed tokens | 1.0566 | 0.8596 | 1.23x | 32.0 | 0.000791 | 3307.1658 |
| Normal GEMM1, `N=32768,K=6144`, 4096 routed tokens | 11.7406 | 12.7812 | 0.92x | 64.0 | 0.000191 | 2010.3130 |
| Normal GEMM2, `N=6144,K=16384`, 4096 routed tokens | 6.8358 | 7.3526 | 0.93x | 64.0 | 0.000449 | 3295.7183 |

A smaller direct-launch smoke with exactly e8m0-derived random scales also showed
low-latency kernel wins (`1.16x` for GEMM1 and `1.47x` for GEMM2, max diff
`0.5`), but the end-to-end serving regression above makes this path unsuitable
as the default.

Synthetic full-path comparison against the previous hand-decoded Triton path:

- Low-latency path, small `H=128,I=256` shape: active-row
  `max_diff=3072.0`, `mean_diff=224.7762`, `mean_abs_ref=31104.3242`.
- Normal path, small `H=128,I=256` shape:
  `max_diff=2048.0`, `mean_diff=212.0821`, `mean_abs_ref=26055.5254`.
- The full-path differences are expected from different dot accumulation order
  and the second FP8 activation quantization; serving smoke did not show the
  previous bad-token failure.

Serving smoke after the correctness fix:

- Fixed prompt `请只回答：北京是中国的首都。` no longer produced the prior repeated
  unrelated Chinese tokens. With the configured reasoning parser, the response
  emitted semantic reasoning text in `reasoning_content`.
- Fixed prompt `用一句话说明水的化学式是什么。` returned a normal Chinese answer:
  water is `H2O`.
- Server logs confirmed decode CUDA graph replay during smoke and benchmark
  requests with `cuda graph: True`.

Serving smoke after restoring the known-good activation quant path:

- Short prompt `用一句中文回答：1+1等于几？` returned normal content:
  `1+1等于2。`.
- Translation prompt `把下面这句话翻译成英文：今天我们继续优化推理性能。` returned
  normal content: `Today we continue to optimize inference performance.`
- A repeated Chinese long-prefill prompt with `2364` prompt tokens returned
  normal content: `这是一段重复的测试文本。`
- Server logs confirmed decode CUDA graph replay during the smoke requests with
  `cuda graph: True`.

Serving smoke after enabling the Hopper dot_scaled kernel:

- Short prompt `用一句中文回答：1+1等于几？` returned normal content:
  `1+1等于2。`.
- Translation prompt `把下面这句话翻译成英文：今天我们继续优化推理性能。` returned
  normal content: `Today we continue to optimize inference performance.`
- A repeated Chinese long-prefill prompt with `2365` prompt tokens returned
  normal summary content: `这是一段用于测试推理服务的重复性填充文本`.
- Server logs confirmed CUDA graph capture `Capture cuda graph bs [8]` and
  decode graph replay during smoke and benchmark requests with
  `cuda graph: True`.

Serving smoke after the e8m0 experiment was made default-off:

- Short prompt `1+1等于几？只回答数字。` returned normal content: `2`.
- Translation prompt `Translate to English: 今天我们继续优化推理性能。` returned
  normal content: `Today we continue to optimize inference performance.`
- A repeated Chinese long-prefill prompt with `1820` prompt tokens returned
  normal content: `这段文本是用于测试推理服务的重复性填充文本。`
- A prior `max_tokens=96` long-prefill smoke returned empty `content` with
  `finish_reason=length` because all generated tokens were used by
  `reasoning_content`. Increasing `max_tokens` to `240` returned normal content,
  so this symptom is request-budget related rather than a kernel bad-text
  regression.
- Server logs confirmed CUDA graph capture `Capture cuda graph bs [8]` and
  decode graph replay with `cuda graph: True`.

## Current Bottleneck

The current Hopper dot_scaled path keeps the checkpoint layout and avoids manual
E2M1 nibble decode in the GEMM loop. It still applies float32 activation and
weight scales outside `tl.dot_scaled` by default. A direct e8m0 weight-scale
variant exists for low-latency decode experiments, but it is default-off because
end-to-end serving regressed and the load-time buffers consumed several GB of
additional GPU memory. The next large performance step is likely one of:

- load-time conversion of MXFP4 scales to a layout that can be consumed directly
  by `tl.dot_scaled` or a CUTLASS/DeepGEMM-style kernel;
- a load-time weight conversion into an existing W4A8-compatible layout, if the
  accuracy and layout semantics are acceptable;
- a deeper DeepEP low-latency path that can avoid padding smaller decode batches
  to the bs=8 CUDA graph work shape under `tp-size=8`.
