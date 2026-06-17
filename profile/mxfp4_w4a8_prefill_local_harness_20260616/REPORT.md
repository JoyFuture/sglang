# MXFP4 W4A8 Prefill Contig Grouped GEMM 局部验证报告

## 范围

本目录用于局部验证 prefill / extend normal path 的 `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`。测试不覆盖 decode / low-latency path。

局部形状：

- w13-like：`N=4096, K=6144`
- w2-like：`N=6144, K=2048`
- `total_m=16384`
- `num_experts=48`
- skew token 分布，`max_tokens_per_expert=872`

正确性 reference 使用当前 `_launch_grouped_gemm_contig` 输出。候选输出与 reference 对比，当前保留候选的 `max_abs_err=0`，`allclose(atol=1e-2, rtol=1e-2)=true`。

## 候选 1：normal contig dot-scaled launch 增加 maxnreg=168

改动：

- 仅影响 `_launch_grouped_gemm_contig` 中 `_USE_DOT_SCALED` 的 normal contig path。
- 默认 `SGLANG_MXFP4_W4A8_CONTIG_MAXNREG=168`。
- 设置 `SGLANG_MXFP4_W4A8_CONTIG_MAXNREG=0` 可关闭。
- decode / low-latency `_launch_grouped_gemm` 未改动。

microbenchmark：

| shape | 关闭 maxnreg | maxnreg=168 | speedup |
| --- | ---: | ---: | ---: |
| w13-like | 6.523 ms | 6.069 ms | 1.075x |
| w2-like | 3.718 ms | 3.077 ms | 1.208x |

NCU full 对照：

| shape | 配置 | registers/thread | theoretical occupancy | achieved occupancy | NCU duration |
| --- | --- | ---: | ---: | ---: | ---: |
| w13-like | baseline | 207 | 12.50% | 12.42% | 8.51 ms |
| w13-like | maxnreg=168 | 168 | 18.75% | 18.56% | 7.85 ms |
| w2-like | baseline | 205 | 12.50% | 12.40% | 4.75 ms |
| w2-like | maxnreg=168 | 168 | 18.75% | 18.57% | 3.98 ms |

结论：

- 当前 kernel 明确受 register pressure 限制，`maxnreg=168` 可以把 resident blocks 从 2 提高到 3，achieved occupancy 从约 `12.4%` 提高到约 `18.6%`。
- w2-like 收益高于 w13-like，符合 NCU 中 eligible warp 和 issue slot 改善的方向。
- 该候选没有改变数学语义，风险主要是 forced register cap 可能在其他 shape 上引入 spill 退化；因此保留环境变量开关。
- 该候选不足以解决主瓶颈。优化后 L1/TEX throughput 仍约 `80%`，DRAM throughput 低于 `13%`，NCU 仍报告 uncoalesced global access excessive sectors 约 `54%-55%`，shared excessive wavefronts 约 `28%`。

## 已拒绝方向

- Activation scale 复用 prototype：正确但退化。w13-like `6.076 ms -> 6.431 ms`，w2-like `3.072 ms -> 3.455 ms`。原因判断是减少 `a_scale` load 的收益小于更长生命周期带来的寄存器/调度代价。
- 缩小 `BLOCK_M/BLOCK_N` 或使用 `BLOCK_N=256`：正确但明显变慢。
- `num_warps=8`：明显变慢。
- `num_stages=4`：只有约 `0.1%-0.2%`，视为噪声。
- B transpose-load prototype：正确但几乎无收益。
- E8M0 rhs scale prototype：w2 约 `1.5%` 小收益，但 w13 约 `7.7%` 退化，不接入 normal path。
- `maxnreg=144`：过度 spill，明显变慢。
- `maxnreg=160`：有效但弱于 `168`。
- `maxnreg=170/176/192`：退化或收益弱于 `168`。

## 候选 3：aligned N/K mask 简化

改动：

- 新增 `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_aligned_nk_kernel`。
- 仅在 `n % BLOCK_N == 0` 且 `k % DOT_K == 0` 时使用。
- 去掉 N/K 边界 mask 和冗余 store `tl.where`，保留 M tail mask。
- 默认开启，可用 `SGLANG_MXFP4_W4A8_CONTIG_ALIGNED_NK=0` 关闭。

50-iter microbenchmark：

| shape | maxnreg168 原 kernel | aligned kernel | speedup |
| --- | ---: | ---: | ---: |
| w13-like | 6.046 ms | 5.932 ms | 1.019x |
| w2-like | 3.074 ms | 3.038 ms | 1.012x |

接入后默认 `_launch_grouped_gemm_contig`：

| shape | latency |
| --- | ---: |
| w13-like | 5.958 ms |
| w2-like | 3.041 ms |

NCU w13-like：

- registers/thread：`168`
- theoretical occupancy：`18.75%`
- achieved occupancy：`18.57%`
- NCU duration：`7.71 ms`
- uncoalesced global access 仍然存在，excessive sectors 约 `55%`

结论：

- 保留。收益小但稳定，且条件分支不会影响非 aligned shape。
- 它不改变当前主要瓶颈判断：下一步仍需处理 register pressure 之外的 L1/TEX、uncoalesced access 和可能的 grouped GEMM tail effect。

## 候选 4：Humming/CUDA WGMMA normal path prototype

用户明确要求先不跑端到端 profile，因此本阶段改为评估 CUDA/CUTLASS/Humming 结构性替代方案。

新增 harness：

```text
profile/mxfp4_w4a8_prefill_local_harness_20260616/harness/bench_humming_bridge.py
```

该 harness 直接把当前 SGLang layout 桥接到 Humming：

- `b_packed: [E, N, K/2]` 通过 `.contiguous().view(torch.int32)` 交给 Humming repack。
- `b_scale.float32` 先转 `bf16`。
- `expert_start/counts` 转成 Humming `expert_layout: [E+1]`。
- reference 仍是当前 Triton `_launch_grouped_gemm_contig`。

bridge 结果：

| shape | Triton optimized | Humming bridge | speedup | correctness |
| --- | ---: | ---: | ---: | --- |
| w13-like | `5.961 ms` | `3.093 ms` | `1.93x` | `allclose_1e-2=true`, `max_abs_err=0.0078125` |
| w2-like | `3.061 ms` | `1.686 ms` | `1.82x` | `allclose_1e-2=true`, `max_abs_err=0.00937` |

已接入一个默认关闭的 SGLang 实验开关：

```bash
SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL=1
SGLANG_HUMMING_ROOT=/sgl-workspace/humming-main
```

范围：

- 只替换 `_launch_grouped_gemm_contig`。
- 只影响 prefill / extend normal path。
- 不影响 decode / low-latency path。
- 首次使用会做 Humming weight repack 和 NVRTC/JIT，后续按 weight data pointer 缓存。

开关路径局部结果：

| shape | Humming switch | Triton direct reference correctness |
| --- | ---: | --- |
| w13-like | `3.157 ms` | `allclose_1e-2=true`, `max_abs_err=0.00861` |
| w2-like | `1.770 ms` | `allclose_1e-2=true`, `max_abs_err=0.00827` |

E8M0 weight scale 也做了试验：

| shape | Humming e8m0 latency | correctness |
| --- | ---: | --- |
| w13-like | `1.338 ms` | fail，`max_abs_err=3.93`, `mean_abs_err=0.522` |
| w2-like | `0.665 ms` | fail，`max_abs_err=2.28`, `mean_abs_err=0.301` |

结论：

- Humming 对当前 kernel 有明确参考价值，而且已经能在局部同形状上跑通 SGLang packed weight 语义。
- bf16 weight scale 是当前可行的 prototype；直接 E8M0 cast 不符合当前 float32 scale 语义，不能接入。
- Humming switch 比 bridge 稍慢，主要是每次 GEMM 仍要更新 `expert_layout`。生产化时应让 DeepEP normal scatter 直接输出 `E+1` layout，避免 Python 侧临时处理。
- 按局部数据估算，Humming 路线能把当前 Triton grouped GEMM 再降约 `1.8-1.9x`。这仍未必达到 `<=200 ms/step` 理想目标，但比继续 Triton meta tuning 更有希望。

## CUTLASS 路线判断

仓库已有 `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`，但它不是当前 MXFP4 W4A8 DeepEP normal path 的 drop-in：

- 它面向 int4 W4A8 CUTLASS grouped GEMM。
- weight scale layout 与当前 `[E, N, K/32]` MXFP4 scale 不同。
- activation quant 更偏 per-tensor / tensorwise 流程，而当前 normal path 是 per-token-group FP8 scale。
- 要支持当前语义，需要重写 iterator / scale load / grouped-contiguous scheduler / weight repack，工程量接近自研 CUDA kernel。

因此短期建议优先走 Humming prototype，长期如果不想引入 Humming runtime，再按 Humming 的 WGMMA + offline repack + 有效 tile scheduler 思路自研 CUDA/CUTLASS-style kernel。

## 后续验证

端到端暂不跑。下一步建议先做真实模型权重局部验证：

- 用真实 `w13_weight/w2_weight` 和真实 `w13_weight_scale/w2_weight_scale` 构造 Humming layer。
- 对 normal path 的 w13/w2 输出分别和 Triton reference 比较误差。
- 统计首次 repack/JIT 成本与后续 steady-state latency。
- 如果局部真实权重也通过，再考虑服务内 A/B。

之后再做真实 16k prefill 端到端 profile，对比：

- step GPU wall
- kernel sum
- `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel` / `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_aligned_nk_kernel` sum
- `cached_notify_combine`
- dispatch/combine actual movement

如果端到端确认候选 1+3 稳定收益，再进入结构性候选：定位 uncoalesced global/shared access 的源行、评估正确处理每 32K scale 语义的 `DOT_K`/scale 复用方案，或进一步对 grouped GEMM tail effect 做调度优化。
