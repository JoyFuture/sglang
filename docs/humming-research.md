# Humming W4A8 / MXFP4 MoE GEMM 调研记录

## 结论

Humming 对当前 MXFP4 W4A8 prefill grouped GEMM 有明确参考价值，而且比继续只调 Triton meta 参数更有希望。

本地在 H20Z / sm90 上用 Humming 的 `grouped_contiguous` FP8 activation + FP4 E2M1 weight 路径做同形状 microbenchmark：

| shape | 当前 SGLang Triton 优化后 | Humming grouped_contiguous | 粗略 speedup |
| --- | ---: | ---: | ---: |
| w13-like `N=4096,K=6144,total_m=16384,E=48` | `5.958 ms` | `3.031 ms` | `1.97x` |
| w2-like `N=6144,K=2048,total_m=16384,E=48` | `3.041 ms` | `1.593 ms` | `1.91x` |

这里的 Humming 测试使用了与当前 harness 相同的 synthetic skew token 分布：`min=1, max=872, mean=341.33`。

这说明 CUDA/WGMMA 路线确实可能把当前 Triton kernel 再砍掉接近一半。按现有 step 中 grouped GEMM `~757 ms/step` 粗略估算，如果同等收益能落到真实模型，GEMM 可能降到 `~380 ms/step` 量级；这仍不到 `<=200 ms/step` 理想目标，但已经是比当前 Triton 微调更大的结构性收益。

## Humming 的关键设计点

Humming 是 NVRTC JIT CUDA kernel，不是 Triton kernel。它有以下与当前问题相关的能力：

- 支持 Dense GEMM 和 MoE GEMM。
- 支持 `grouped_contiguous` 和 `grouped_masked`。
- 支持 FP8 E4M3 activation + FP4 E2M1 weight。
- 支持 H20/Hopper 的 sm90 heuristic，`humming/tune/sm90_h20.py` 里有专门配置。
- 支持 WGMMA 路径、stream-K、persistent/grid-size heuristic、TMA 相关路径。

对当前 kernel 最有启发的点：

1. **有效 tile 调度**
   - 当前 SGLang Triton contig path 的 grid 是：
     - `num_experts * ceil(max_m / BLOCK_M) * ceil(N / BLOCK_N)`
   - 小专家的多余 M-block 进 kernel 后 early return。
   - Humming grouped_contiguous 会根据 `expert_layout` 计算每个专家真实 token 数，再累加真实 `ceil(M_e / BlockM)`，只调度有效 M-block。
   - 这能减少空 CTA，也可能缓解专家不均衡下的 tail effect。

2. **WGMMA + CUDA pipeline**
   - 当前 Triton `tl.dot_scaled` kernel 的 NCU 显示 register pressure 高、occupancy 低，且 uncoalesced global/shared access 明显。
   - Humming 把 packed weight 预处理成自己的 WGMMA 友好布局，kernel 内从 shared 到 register 再做 FP4 到 FP8 的转换，并使用 WGMMA。
   - 这绕开了当前 Triton `tl.dot_scaled` 生成代码的一部分限制。

3. **shape-aware heuristic**
   - H20 sm90 heuristic 对 MoE shape 会选小 M tile：
     - 例如本地 w13/w2 skew 选择了 `block_shape_m=8/16/32/48/64` 的分段配置。
   - 这比当前 SGLang 只用 `max_m` 粗略决定 `BLOCK_M` 更细。

## 不能直接替换的点

Humming 目前不是 drop-in replacement，主要差异如下：

1. **weight layout 不同**
   - 当前 SGLang kernel 直接消费 `[E, N, K/2]` packed MXFP4 weight。
   - Humming 需要 `prepare_humming_weight()` 做 repack，生成它自己的 WGMMA/MMA 布局。
   - 接入生产路径时应在模型加载或第一次使用时预处理，不能每次 forward 做。

2. **scale dtype 不同**
   - 当前 SGLang normal path 的 weight scale 是 `float32`，shape `[E, N, K/32]`。
   - Humming 支持 `float16` / `bfloat16` / FP8 scale / UE8M0 scale，不直接用 float32 scale。
   - 本地 benchmark 使用 `bfloat16` weight scale。
   - 若接入，需要验证 `float32 -> bf16` 或 `float32 -> e8m0` 后的误差。

3. **scale 语义需要核对**
   - 当前 SGLang 做：
     - `raw_acc(e4m3 x e2m1) * a_scale_fp32 * b_scale_fp32`
   - Humming 在 kernel 内把 E2M1 weight 转成 activation dtype 对应格式，再在 C accumulator 上应用 input/weight scale。
   - 数学目标相近，但 accumulator 分段、scale 应用时机、scale dtype 都可能造成差异。

4. **launcher / dependency 风险**
   - Humming 依赖自己的 NVRTC 编译、cubin 注册和 `torch.ops.humming.launch_kernel`。
   - 直接引入到 SGLang 需要处理打包、缓存、编译耗时、server 多进程/多 rank 编译风暴、CUDA 版本兼容。
   - 本地跑 benchmark 时，Humming 的 `tops_bench` 在 H20Z 上触发过 `wgmma.mma_async.m256n64k32` ptxas 报错；主 GEMM 可跑，但需要避开或修正该辅助 benchmark 路径。

## 建议路线

### 路线 A：先做 Humming 原型后端

新增一个实验性 backend，例如：

```text
--moe-runner-backend mxfp4_w4a8_humming
```

或者先在当前 backend 下加环境变量：

```text
SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL=1
```

第一阶段只接 prefill / extend normal path：

- `mxfp4_w4a8_deepep_normal_triton`
- w13 grouped_contiguous GEMM
- w2 grouped_contiguous GEMM

暂不接 decode / low-latency。

### 路线 B：只借鉴 Humming，重写一个轻量 CUDA kernel

如果不想引入 Humming runtime，可以借鉴这些设计重写 SGLang 内部 CUDA/CUTLASS-style kernel：

- contiguous grouped scheduler：只调度真实有效 M-block。
- weight offline repack 到 WGMMA 友好布局。
- FP4 weight 在 CUDA kernel 内转换到 FP8/BF16 register。
- 对每 32K weight scale、每 128K activation scale 分段累加。
- 保持 SGLang 自己的 build/runtime，不引入外部 JIT launcher。

这条路线工程量更大，但长期更可控。

### 路线 C：CUTLASS 直接做

纯 CUTLASS 直接替换当前 kernel 的可行性不如 Humming 明确：

- Hopper 对 FP8 WGMMA 支持成熟。
- 但当前问题是 `FP8 activation x packed MXFP4 weight + per-token/per-channel scale + grouped_contiguous MoE`。
- CUTLASS 可能需要自定义 iterator / epilogue / grouped scheduler / offline repack，最终工程量接近自研 CUDA kernel。

因此短期更建议先基于 Humming 做 prototype，确认真实模型 correctness 和端到端收益，再决定是否 vendor / 改写。

## 下一步验证清单

1. 用真实 SGLang weight tensor 做一次 Humming layout 转换：
   - 输入：`w13_weight / w2_weight` 和 float32 `weight_scale`。
   - 转换：weight repack，scale 先试 `bf16`。
   - 校验：Humming output vs 当前 Triton output。

2. 正确性统计：
   - `max_abs_err`
   - `mean_abs_err`
   - `max_rel_err`
   - `mean_rel_err`
   - `allclose(atol=1e-2, rtol=1e-2)` 先作为宽松门槛。
   - 如果 bf16 scale 误差大，再试 E8M0 或保留 float32 scale 的自研 kernel。

3. 性能统计：
   - 局部 harness w13/w2。
   - 真实 16k prefill 中 w13/w2 两类 kernel sum。
   - `cached_notify_combine` 是否随 GEMM 下降。

4. 工程风险评估：
   - NVRTC 编译缓存。
   - 多 rank 并发编译。
   - H20/H200 兼容。
   - SGLang wheel/容器打包。

## 2026-06-16 追加：SGLang layout 到 Humming layout 桥接验证

已新增局部 harness：

```text
profile/mxfp4_w4a8_prefill_local_harness_20260616/harness/bench_humming_bridge.py
```

该 harness 直接使用当前 SGLang synthetic tensor：

- `a: [total_m, K]`，`torch.float8_e4m3fn`
- `a_scale: [total_m, K / 128]`，`float32`
- `b_packed: [E, N, K / 2]`，packed MXFP4
- `b_scale: [E, N, K / 32]`，`float32`
- `expert_start / num_tokens_per_expert`

桥接方式：

- `b_packed.contiguous().view(torch.int32)` 作为 Humming packed weight 输入。
- `b_scale.float32 -> bf16`，再走 Humming `weight_scale_group_size=32`。
- `expert_start/counts` 转成 Humming grouped-contiguous `expert_layout`，长度为 `E + 1`。
- `input_scale_group_size=128`，对应当前 activation scale layout。

验证结论：

| shape | 当前 Triton optimized | Humming bridge bf16 scale | speedup | correctness |
| --- | ---: | ---: | ---: | --- |
| w13-like `N=4096,K=6144` | `5.961 ms` | `3.093 ms` | `1.93x` | `allclose_1e-2=true`, `max_abs_err=0.0078125` |
| w2-like `N=6144,K=2048` | `3.061 ms` | `1.686 ms` | `1.82x` | `allclose_1e-2=true`, `max_abs_err=0.00937` |

这说明 Humming 的 weight repack 与当前 SGLang packed nibble 顺序基本兼容；bf16 weight scale 造成的误差在本地 synthetic case 下可接受。

同时测试了 `float32 -> e8m0` weight scale：

| shape | Humming e8m0 latency | correctness |
| --- | ---: | --- |
| w13-like | `1.338 ms` | fail，`max_abs_err=3.93`, `mean_abs_err=0.522` |
| w2-like | `0.665 ms` | fail，`max_abs_err=2.28`, `mean_abs_err=0.301` |

E8M0 路径很快，但直接把当前 float32 scale cast 成 E8M0 会严重改变数学语义，不能作为 drop-in。除非模型加载时本来就提供 UE8M0/MXFP4 scale，或重新设计 scale/global-scale 约定，否则不能接入生产路径。

## 2026-06-16 追加：SGLang 实验性 Humming normal path 开关

注：本节记录的是 side-cache 原型。后续“方案 3”已把默认 Humming 行为改成加载期主权重替换，并补了 low-latency decode 路径；以本文后面的“方案 3 实现记录”为准。

已在 `mxfp4_w4a8_deepep_triton.py` 增加默认关闭的实验开关：

```bash
SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL=1
SGLANG_HUMMING_ROOT=/sgl-workspace/humming-main
```

范围：

- 只影响 `_launch_grouped_gemm_contig`，也就是 prefill / extend normal path。
- 不影响 decode / low-latency `_launch_grouped_gemm`。
- 只在 `a_scale_group_size=128`、`b_scale.float32`、`N/K` 满足 Humming 约束时使用。
- 首次使用会对 weight 做 Humming repack，并触发 NVRTC/JIT 编译；后续按 weight data pointer 缓存。

通过 `bench_contig.py` 开启该开关后的局部结果：

| shape | Humming switch latency | Triton direct reference correctness |
| --- | ---: | --- |
| w13-like | `3.157 ms` | `allclose_1e-2=true`, `max_abs_err=0.00861` |
| w2-like | `1.770 ms` | `allclose_1e-2=true`, `max_abs_err=0.00827` |

开关路径比纯 bridge 稍慢，主要原因是每次 GEMM 仍要把当前 `expert_start` 更新成 Humming `expert_layout`。这部分如果要生产化，最好让 DeepEP normal scatter 直接产出 `E+1` 的 expert layout，避免 Python 侧临时更新。

## CUDA / CUTLASS 路线判断

SGLang 仓库已有 `cutlass_w4a8_moe.py`，但它不是当前 MXFP4 W4A8 normal path 的直接替换：

- 当前文件面向 int4 W4A8 CUTLASS grouped GEMM。
- weight scale layout 是类似 `[E, K // 512, N * 8]` / `[E, N // 512, K * 4]` 的 CUTLASS 友好格式。
- activation scale 更偏 per-tensor / 静态 tensorwise quant 流程。
- 当前目标 kernel 是 FP8 activation + packed MXFP4 E2M1 weight + per-token-group activation scale + per-32K per-channel weight scale。

因此，直接套现有 CUTLASS W4A8 MoE 工程风险较高；它需要重新处理 weight repack、scale iterator、grouped-contiguous scheduler 和 epilogue 语义。短期最可行的是：

1. 继续用 Humming 原型验证真实模型局部 correctness 和性能。
2. 若收益稳定，再决定是 vendor Humming runtime，还是按 Humming 的 WGMMA scheduler / repack 思路重写 SGLang 内部 CUDA kernel。
3. CUTLASS 更适合作为长期自研 CUDA kernel 的构件参考，而不是本轮直接接入。

## 2026-06-16 追加：方案 3 实现记录，加载期替换主权重

已把 Humming 接入从 side-cache 原型改成默认的加载期权重替换路径：

- 当 `SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL=1` 时，默认等价于 `SGLANG_MXFP4_W4A8_HUMMING_REPLACE_WEIGHTS=1`。
- `process_weights_after_loading()` 中先把 SGLang 原始 MXFP4 packed weight/float32 scale 转成 Humming repacked weight/bf16 scale。
- 转换完成后删除 `w13_weight`、`w2_weight`、`w13_weight_scale_inv`、`w2_weight_scale_inv`，不再在运行期保留一份原始 MXFP4 主权重和一份 Humming cache。
- 若需要回到旧 side-cache 实验行为，可以显式设置 `SGLANG_MXFP4_W4A8_HUMMING_REPLACE_WEIGHTS=0`；这仍会额外占显存，不建议用于满载服务。

为了释放原始权重后仍能跑完整服务，本次同时补了 Humming low-latency decode 路径：

- prefill/extend normal 使用 Humming `GROUPED_CONTIGUOUS`，输入仍是 DeepEP normal scatter 后的 contiguous expert token layout。
- decode/low-latency 使用 Humming `GROUPED_MASKED`，把 SGLang `[E, expected_m, K]` flatten 成 `[E * expected_m, K]`，`masked_m` 作为每个 expert 的有效 token 数。
- low-latency 的无效 padding 行 Humming 不保证清零；DeepEP combine 只消费有效路由 token，局部 correctness 按有效行比较。

局部验证：

- `py_compile` 通过：
  - `mxfp4_w4a8_deepep_triton.py`
  - `mxfp4_w4a8.py`
  - `mxfp4_w4a8_moe.py`
- 小尺寸 normal 两段 GEMM：替换后的 Humming 权重 vs Triton reference，`max_abs_err=0.0`，`allclose_2e-2=true`。
- 小尺寸 low-latency 两段 GEMM：有效 token 行 vs Triton reference，`valid_max_abs_err=1.62e-05`，`allclose_valid_2e-2=true`。

这个实现解决的是用户服务中看到的根因：之前 Humming 在模型加载后或首次 forward 时还要额外构建 repacked weight cache，显存会从“原始 MXFP4 权重”变成“原始 MXFP4 权重 + Humming repacked 权重”。现在 Humming 模式下原始权重会被替换和释放，峰值额外内存只出现在单层转换过程中，常驻权重以内核实际使用的 Humming layout 为主。

## 2026-06-16 追加：Humming grouped-contiguous 单 kernel NCU

按用户要求单独跑了 Humming steady-state grouped-contiguous GEMM 的 NCU。采集方式：

- 只 profile Humming GEMM 本体。
- Humming layer 构建、weight repack、NVRTC/JIT、warmup 都在 capture 外。
- 形状仍是本地 synthetic skew：
  - w13-like：`N=4096,K=6144,total_m=16384,E=48`
  - w2-like：`N=6144,K=2048,total_m=16384,E=48`

报告位置：

- `profile/mxfp4_w4a8_prefill_local_harness_20260616/reports/humming_w13_grouped_contig_full.ncu-rep`
- `profile/mxfp4_w4a8_prefill_local_harness_20260616/reports/humming_w2_grouped_contig_full.ncu-rep`
- `profile/mxfp4_w4a8_prefill_local_harness_20260616/analysis/details_humming_w13_grouped_contig_full.txt`
- `profile/mxfp4_w4a8_prefill_local_harness_20260616/analysis/details_humming_w2_grouped_contig_full.txt`

关键指标：

| shape | NCU duration | SM throughput | Memory throughput | DRAM throughput | L1/TEX throughput | achieved occupancy | registers/thread | grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w13-like | `3.96 ms` | `58.85%` | `66.56%` | `9.47%` | `67.46%` | `21.72%` | `128` | `264` CTAs |
| w2-like | `2.11 ms` | `58.12%` | `65.21%` | `8.89%` | `66.07%` | `21.89%` | `128` | `264` CTAs |

与当前 Triton optimized NCU 对比：

- Triton w13-like：`7.85 ms`，Humming w13-like：`3.96 ms`，NCU 口径约 `1.98x`。
- Triton w2-like：`3.98 ms`，Humming w2-like：`2.11 ms`，NCU 口径约 `1.89x`。
- Triton 的 uncoalesced global excessive sectors 约 `54%-55%`，Humming 降到 `7%-8%`。
- Triton 的 uncoalesced shared excessive wavefronts 约 `28%`，Humming 降到约 `9%`。
- Triton `registers/thread=168` 后 occupancy 约 `18.6%`，Humming `registers/thread=128`，occupancy 约 `21.8%`。
- Triton grid 是 `num_experts * ceil(max_m / 64) * ceil(N / 128)`，本地 skew case 下会发 `21504/32256` 个 CTA；Humming 只发 `264` 个 CTA，说明它按真实 expert token layout 调度有效 M tile，避免大量 early-return 空 CTA。

瓶颈判断：

- Humming 仍不是 DRAM bandwidth-bound：DRAM throughput 只有约 `9%`。
- Humming 的 L1/TEX 和 memory throughput 仍高于 DRAM，说明主要压力仍在片上访存、scale/weight layout、pipeline 和调度，而不是 HBM 带宽。
- 相比 Triton，Humming 已明显缓解 uncoalesced global/shared access 和 grouped GEMM 空 CTA，因此性能提升主要来自 weight repack layout、有效 tile scheduler、CUDA/WGMMA pipeline，而不是单纯更高 HBM 带宽。
- Humming 当前剩余瓶颈更像 compute/memory mixed：SM throughput 约 `58%`，memory throughput 约 `65%-67%`，NCU 也判断 compute 和 memory 较均衡。继续优化需要同时减少 scale/FP4 handling 指令、片上访存 transaction，以及提高 eligible warp/issue slot。

## 2026-06-16 追加：Humming grouped-contiguous tuning 覆盖

继续优化时没有直接修改 Humming CUDA 源码，而是先扫描 Humming 已暴露的 tuning knobs：

- `block_shape`
- `warp_shape`
- `num_sms`
- `num_stages`
- `num_ctas_per_sm`
- `use_stream_k`

原因是这些配置可以在 SGLang 集成层覆盖，验证和回滚成本明显低于直接改 Humming `.cuh`。本轮 synthetic skew 条件：

- `total_m=16384`
- `num_experts=48`
- `seed=0`
- w13-like：`N=4096,K=6144`
- w2-like：`N=6144,K=2048`

扫描结果：

| shape | Humming heuristic | best tuned | tuned config | speedup |
| --- | ---: | ---: | --- | ---: |
| w13-like | `3.068 ms` | `2.615 ms` | `BM64/BN128/BK128, WM64/WN32/WK128, stages4, ctas2, num_sms132` | `1.17x` |
| w2-like | `1.780 ms` | `1.397 ms` | `BM48/BN128/BK128, WM48/WN32/WK128, stages4, ctas2, num_sms132` | `1.27x` |

短迭代 8k 检查显示：

| shape | Humming heuristic | best tuned | tuned config |
| --- | ---: | ---: | --- |
| w13-like 8k | `1.727 ms` | `1.425 ms` | `BM48/BN128/BK128, WN32` |
| w2-like 8k | `1.051 ms` | `0.777 ms` | `BM48/BN128/BK128, WN32` |

因此当前 SGLang 覆盖策略：

- 仅影响 Humming normal grouped-contiguous path。
- 仅覆盖 `shape_m > 4096` 的 prefill/extend 大 M 区间，小 batch 和 decode 保持 Humming heuristic。
- w13-like `N=4096,K=6144`：
  - `4096 < M <= 12288`：`BM48/WN32`
  - `M > 12288`：`BM64/WN32`
- w2-like `N=6144,K=2048`：
  - `M > 4096`：`BM48/WN32`
- 可用 `SGLANG_MXFP4_W4A8_HUMMING_PREFILL_TUNING=0` 关闭该覆盖。

SGLang 集成层验证结果：

| shape | Triton harness | Humming tuned via SGLang entry | speedup vs Triton | correctness |
| --- | ---: | ---: | ---: | --- |
| w13-like 16k | `5.954 ms` | `2.695 ms` | `2.21x` | `allclose_1e-2=true`, `max_abs_err=2.44e-4` |
| w2-like 16k | `3.037 ms` | `1.465 ms` | `2.07x` | `allclose_1e-2=true`, `max_abs_err=1.22e-4` |

这轮结果说明：Humming 默认 heuristic 对 H20/Hopper 上的 MXFP4 W4A8 grouped-contiguous 大 M MoE shape 仍有明显 tuning 空间，主要收益来自把 `warp_shape_n` 从 `16` 调到 `32`，并在 w2/down-proj 上使用 `block_m=48`。`num_sms`、`num_ctas_per_sm`、`block_n=256`、`block_k=256` 等方向大多无收益或明显退化。

随后对 tuned config 重新采了单 kernel NCU：

| shape | old Humming NCU | tuned Humming NCU | SM throughput | Memory throughput | DRAM throughput | L1/TEX throughput | achieved occupancy | registers/thread | excessive global sectors | excessive shared wavefronts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w13-like | `3.96 ms` | `3.37 ms` | `58.48%` | `40.65%` | `7.62%` | `41.40%` | `12.46%` | `255` | `9%` | `4%` |
| w2-like | `2.11 ms` | `1.77 ms` | `61.14%` | `41.57%` | `7.90%` | `42.59%` | `12.31%` | `254` | `7%` | `4%` |

NCU 解读：

- tuned config 的 duration 比旧 Humming NCU 继续下降约 `15%-16%`。
- `warp_shape_n=32` 后 block size 从旧配置的 `256` threads 降到 `128` threads，但每线程寄存器升到约 `254-255`，理论/实际 occupancy 降到约 `12.5%/12.3%`。
- 尽管 occupancy 更低，SM throughput 仍在 `58%-61%`，memory throughput 从旧配置约 `65%-67%` 降到约 `41%-42%`，说明这轮收益主要来自减少片上/内存 transaction 和改善 tile work distribution，而不是提升 resident warp 数。
- DRAM throughput 仍只有约 `8%`，因此 tuned Humming 依然不是 HBM 带宽受限。
- excessive shared wavefronts 从旧配置约 `9%` 降到约 `4%`；global excessive sectors 维持在 `7%-9%`。下一步源码级优化更应该盯 shared/local/register pressure 和 scale/FP4 handling，而不是单纯调 HBM 访问。

下一步如果继续向下优化，再考虑 Humming 源码级改动：

- 针对 FP8 activation + FP4 E2M1 weight + bf16 scale + grouped-contiguous 写更专门的 scale load/apply 路径。
- 检查 NCU 中 local memory / shared wavefront 的剩余浪费。
- 评估 grouped scheduler 是否能减少每 CTA 的控制开销。
- 评估让 DeepEP normal scatter 直接产出 Humming `expert_layout`，避免每次 GEMM 前更新布局缓存。

## 2026-06-16 追加：2.2w tokens/s 后的下一轮优化记录

用户端到端实测 Humming + tuned config 后，16k prefill 吞吐已从约 `1.2w tokens/s` 提升到约 `2.2w tokens/s`。这说明此前接入 Humming repack layout 与 `warp_shape_n=32` tuning 对真实服务有效，但距离 W8A8 双机 `~3w tokens/s` 仍有差距。

本轮继续尝试了更细的 Humming tuning 扫描，重点覆盖：

- `block_m=32/40/48/56/64/72/80`
- `warp_n=16/32/64`
- `num_ctas_per_sm=1/2`
- `num_stages=3/4/5`
- `use_stream_k=true/false`
- `num_sms=112/132/160/192/224/264/320`

但当前 8 卡都在运行服务，每卡剩余显存约 `0.1-3.2 GiB`，Humming JIT 新配置时需要额外 output / workspace / cubin 加载空间，继续扫会频繁 OOM，甚至可能触发非法指令后的 CUDA context 错误。因此没有把这轮不完整扫描结果固化到生产路径。

已确认的下一层瓶颈：

- tuned Humming 不是 HBM bandwidth-bound，DRAM throughput 仍只有约 `8%`。
- `warp_shape_n=32` 版本每线程寄存器约 `254-255`，occupancy 约 `12.3%`；继续优化重点应是 register pressure、shared/local memory transaction 和 scale handling。
- grouped-contiguous scheduler 每个 CTA 都会读取/处理 `expert_layout`，但当前只有 `264` CTA，调度开销不是主瓶颈。
- `expert_layout` 只需要保存每个 expert 的 token 起点，16k/百万级 context 下 int32 足够，没必要用 int64。

本轮落地了一个低风险小优化：

- Humming normal grouped-contiguous path 的 `expert_layout` cache 固定使用 `torch.int32`。
- 这样 Humming scheduler 走 `use_int64_expert_layout=false` 分支，layout load 和 shared copy 宽度减半。
- 正确性用 4096-token synthetic w13/w2 验证，均为 `allclose_1e-2=true`。

这个 int32 layout 优化预期收益较小，但它在每层 MoE、每个 grouped GEMM launch 上都会发生，且风险低；大收益的下一步仍然需要空闲 GPU 上做完整 micro tuning 或进入 Humming 源码级优化。

## 2026-06-16 追加：空闲 GPU 后的 focused tuning

释放 GPU 后，继续围绕当前最佳配置做 focused tuning，而不是盲扫全空间。重点测试：

- `num_stages=3/4/5`
- `use_stream_k=true/false`
- `num_ctas_per_sm=1/2`
- `num_sms=132/192/264`
- `num_write_splits=2`
- w13 的 `BM48` 与 `BM64` 对比

关键 microbenchmark 结论：

| shape | 上一版 tuned config | 上一版 latency | 新 tuned config | 新 latency | 变化 |
| --- | --- | ---: | --- | ---: | ---: |
| w13-like 16k | `BM64/BN128/BK128/WN32/stages4` | `~2.64 ms` | `BM48/BN128/BK128/WN32/stages3` | `~2.49 ms` | `~1.06x` |
| w2-like 16k | `BM48/BN128/BK128/WN32/stages4` | `~1.40 ms` | `BM48/BN128/BK128/WN32/stages3` | `~1.35 ms` | `~1.04x` |

8k 复核也显示 `stages3` 更快：

| shape | stages3 | stages4 | stages5 |
| --- | ---: | ---: | ---: |
| w13-like 8k | `1.361 ms` | `1.424 ms` | `1.434 ms` |
| w2-like 8k | `0.755 ms` | `0.778 ms` | `0.794 ms` |

因此 SGLang 覆盖策略更新为：

- w13-like `N=4096,K=6144`：`M > 4096` 全部使用 `BM48/BN128/BK128/WN32/stages3/ctas2/num_sms132/stream_k=true`。
- w2-like `N=6144,K=2048`：`M > 4096` 使用同一套 `BM48/BN128/BK128/WN32/stages3/ctas2/num_sms132/stream_k=true`。
- `num_write_splits=2` 对 `BM48` 不合法，Humming epilogue 有 `BlockShape::M % 32 == 0` 的 static assertion。
- `num_sms>132` 在这两个 shape 上明显退化。
- `num_ctas_per_sm=1` 明显退化。
- `use_stream_k=false` 对 w13 的部分随机测量有波动收益，但在 w2 和综合稳定性上不如 `stream_k=true`，暂不固化。

SGLang entry 级验证：

| shape | SGLang Humming entry latency | tuning tail | correctness |
| --- | ---: | --- | --- |
| w13-like 16k | `2.525 ms` | `BM48/WN32/stages3` | `allclose_1e-2=true` |
| w2-like 16k | `~1.40 ms p50` | `BM48/WN32/stages3` | 已在 focused tuning 中验证 `allclose_1e-2=true` |

stage3 后重新采集单 kernel NCU：

| shape | stage4 NCU duration | stage3 NCU duration | SM throughput | Memory throughput | DRAM throughput | achieved occupancy | registers/thread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w13-like | `3.37 ms` | `3.17 ms` | `65.74%` | `43.52%` | `8.89%` | `12.25%` | `252` |
| w2-like | `1.77 ms` | `1.69 ms` | `63.90%` | `42.80%` | `8.14%` | `12.21%` | `255` |

解读：

- `num_stages=3` 降低 pipeline stage 开销后，SM throughput 从上一版约 `58%-61%` 提升到 `64%-66%`。
- 寄存器仍在 `252-255/thread`，occupancy 仍约 `12%`，说明下一层瓶颈还是 register pressure / local+shared transaction / scale apply，而不是 HBM。
- 这轮是小幅但稳定的 kernel-level tuning 收益；端到端是否能继续从 `2.2w tokens/s` 上涨，需要再跑真实 prefill profile。

## 2026-06-16 追加：Humming 源码级候选试验

在不继续回写 SGLang 的前提下，单独对 Humming kernel 做了两类试验：

1. 隔离进程扫描 `BM/WN/stages3` 候选，避免非法 kernel 污染 CUDA context。
2. 临时修改 Humming epilogue，放宽 `num_write_splits=2` 对 `BM48` 的限制，验证是否能降低写回压力。

隔离扫描结果显示，当前选择仍是最优：

| shape | best isolated config | mean latency | 次优配置 | 次优 latency |
| --- | --- | ---: | --- | ---: |
| w13-like 16k | `BM48/BN128/BK128/WN32/stages3` | `2.453 ms` | `BM40/WN32/stages3` | `2.599 ms` |
| w2-like 16k | `BM48/BN128/BK128/WN32/stages3` | `1.347 ms` | `BM56/WN32/stages3` | `1.405 ms` |

部分非标准 tile 会触发非法指令，所以后续扫配置必须继续使用单候选独立进程；不要把所有候选放在一个 Python 进程里顺序跑。

`num_write_splits=2` 源码试验：

- 修改点：临时把 Humming epilogue 里 `BlockShape::M % 32 == 0` 放宽到 `BlockShape::M % 16 == 0`，使 `BM48` 可以编译 split2。
- 正确性：w13/w2 都能 `allclose_1e-2=true`。
- 性能：
  - w13 `split1=2.456 ms`，`split2=2.459 ms`，基本持平略慢。
  - w2 `split1=1.346 ms`，`split2=1.339 ms`，约 `0.5%` 小收益。
- 结论：收益太小，不值得保留 Humming 源码改动；试验后已回滚 Humming 源码。

当前 Humming 源码级结论：

- 继续单纯调整 `BM/WN/stages/ctas/num_sms/split` 的收益已经接近尾声。
- 主要瓶颈仍是 `252-255 regs/thread` 带来的低 occupancy，以及 scale apply / epilogue / shared-local transaction。
- 如果还要继续从 kernel 内部拿明显收益，需要深入改 `mainloop_arith.cuh` 的 scale 处理或 `epilogue` 的 register footprint，而不是只改 tuning config。

## 2026-06-16 追加：DeepGEMM PR 332 评估

用户提到 DeepGEMM PR 332：`https://github.com/deepseek-ai/DeepGEMM/pull/332`。本地已拉取到：

```text
/sgl-workspace/DeepGEMM-pr332
```

PR 里和当前问题相关的文件：

- `csrc/jit_kernels/impls/sm90_fp8_fp4_gemm_1d2d.hpp`
- `csrc/jit_kernels/impls/sm90_fp8_fp4_gemm_1d2d_rs.hpp`
- `tests/test_sm90_fp8_fp4.py`
- `csrc/apis/gemm.hpp`

本地构建情况：

- PR head 可拉取并编译 Python 扩展。
- 因网络问题，CUTLASS submodule 没有完整拉下来；临时使用系统已有 `/usr/local/lib/python3.12/dist-packages/deep_gemm/include` 作为 CUTLASS/CuTe include。
- `deep_gemm` import 成功，且暴露：
  - `m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma`
  - `m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma`

自带测试结果：

- `test_sm90_fp8_fp4_contiguous` 可通过。
- `test_sm90_fp8_fp4_masked` 可通过。
- 但测试表里 W4 路径在这些 synthetic case 下比 DeepGEMM 自己的 FP8 对照慢，例如 contiguous `groups=8,m/group=128,n=4096,k=7168`：
  - W4：`190 us`
  - FP8：`86 us`
  - speedup：`0.45x`
- masked case 也类似，W4 通常是 FP8 的 `0.4x-0.8x`。

和当前 SGLang/Humming 场景的关键差异：

1. DeepGEMM contiguous API 使用 `grouped_layout`，它是每个 token 的 group id，形状接近 `[M]`。
   - 当前 Humming/SGLang normal path 使用 expert prefix layout，形状是 `[E + 1]`。
   - 如果直接接 DeepGEMM，需要额外构造 `[M]` layout，16k prefill 下是可接受的，但每层/每 launch 都会多一份 layout 读写。

2. DeepGEMM PR 的 FP4 scale 语义和当前 SGLang MXFP4 W4A8 不一致。
   - PR 测试里 `per_token_cast_to_fp4(..., use_ue8m0=True)`，并支持 E8M0/direct scale fast path。
   - 当前 SGLang 权重 scale 是 float32 `[E, N, K/32]`，Humming 接入时转成 bf16 scale。
   - 之前直接把 SGLang float32 scale cast 到 E8M0 已验证会严重错，不能作为 drop-in。

3. 在 SGLang 目标形状上直接跑 DeepGEMM PR 的 grouped contiguous synthetic case，数值偏差明显大于 Humming 路径：
   - w13-like `groups=48,m/group=341,N=4096,K=6144`：`w4_diff≈0.273`。
   - w2-like `groups=48,m/group=341,N=6144,K=2048`：`w4_diff≈0.273`。
   - 这说明 PR 的量化/scale 语义不能直接对齐当前 MXFP4 W4A8 reference。

可参考点：

- PR 的 SM90 1d2d grouped FP8xFP4 kernel 已经把 contiguous/masked FP4 WGMMA 路径串起来，可作为 scheduler 和 TMA/SFA/SFB layout 参考。
- 它允许 `block_m_override/block_n_override` 做 block shape sweep，这一点可借鉴到 Humming/SGLang 的 tuning harness。
- PR 的 1d2d psum scheduler 对 grouped contiguous 的处理值得阅读，但不能直接替换 Humming，因为输入 layout 和 scale 约定不同。

结论：

- DeepGEMM PR 332 可以参考，但当前不适合作为 SGLang MXFP4 W4A8 prefill normal path 的直接替换。
- 若要采用，需要至少解决两件事：
  1. 把 SGLang `expert_start/counts` 转换或改造成 DeepGEMM 所需的 per-token `grouped_layout`。
  2. 重新对齐 MXFP4 scale 语义，支持当前 float32/bf16 per-32K per-N scale，而不是直接走 UE8M0 fast path。
- 短期继续沿 Humming 路径优化更现实；DeepGEMM PR 主要作为源码设计参考。

建议下一步优先级：

1. 在空闲 GPU 上完整复测 focused tuning，尤其是 `use_stream_k=false`、`num_write_splits=2`、`BM48 vs BM64`、`stages=3/4/5` 的组合。
2. 若 tuning 不能继续提升，再改 Humming 源码，优先看 `mainloop_arith.cuh` 的 scale dequant/apply、`loader_as.cuh/loader_bs.cuh` 的 shared/global transaction，以及 epilogue 的 register footprint。
3. 端到端再 profile 一次，统计 Humming GEMM 降低后 `cached_notify_combine` 是否同步下降；如果 notify/wait 仍大，需要转向 DeepEP overlap / expert imbalance。
