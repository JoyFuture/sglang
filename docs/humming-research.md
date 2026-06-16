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
