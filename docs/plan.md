# MXFP4 W4A8 DeepEP Prefill Grouped GEMM 优化计划

## 目标

优化 prefill / extend normal path 的 `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`，优先降低 16k prefill 中该 kernel 的总耗时。

当前 baseline：

- W4A8 单机 16k prefill 中该 kernel 约 `754-757 ms/step`。
- 约 `137-138 launches/step`。
- trace 显示 registers/thread 约 `205-207`，estimated occupancy 约 `13%`。

第一阶段目标：

- 先建立可靠的局部 correctness 和 latency harness。
- 先做最小风险的 shape-aware meta tuning，不改变数学语义。
- 每个候选只改一个方向，并记录正确性、性能和是否保留。

## 不做的事

- 第一轮不改 decode / low-latency path。
- 第一轮不改 DeepEP dispatch/combine 逻辑。
- 第一轮不改权重加载格式或全局 quant_info 格式。
- 第一轮不引入 Blackwell / sm100 专用假设。

## 正确性基线

局部 kernel 正确性以当前实现作为 baseline：

- 输入相同的 `a`、`a_scale`、`b_packed`、`b_scale`、`expert_start`、`num_tokens_per_expert`。
- baseline 输出来自当前 `_launch_grouped_gemm_contig`。
- 候选输出与 baseline 输出比较。

覆盖形状：

- w13-like：`N=4096, K=6144`。
- w2-like：`N=6144, K=2048`。
- 专家数使用实际 trace 中出现的 `num_experts=48`。
- token 分布覆盖均匀和偏斜两类。

误差记录：

- `max_abs_err`
- `mean_abs_err`
- `max_rel_err`
- `mean_rel_err`
- `allclose(atol=1e-2, rtol=1e-2)`

候选如果只是 meta tuning，理论上数值路径不变，应尽量接近 baseline；若误差异常，直接拒绝候选。

## 性能测量基线

先用局部 harness 测单次 grouped GEMM latency：

- warmup 后用 CUDA event 计时。
- 分 w13-like 和 w2-like 形状分别测。
- 每个候选记录 P50 / P90 / P99 或平均值。
- 记录 meta 参数：
  - `BLOCK_M`
  - `BLOCK_N`
  - `DOT_K`
  - `num_warps`
  - `num_stages`
  - grid 形状

局部 benchmark 只用于快速筛选。候选通过后，再回到 16k prefill 端到端 profile 验证：

- step GPU wall。
- kernel sum。
- `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel` sum。
- `cached_notify_combine`。
- dispatch/combine actual movement。

## NCU 验证

当 microbenchmark 出现有希望的候选后，创建独立 profile run 目录：

```text
profile/<run_name>/
  harness/
  reports/
  analysis/
  REPORT.md
```

对 w13-like 和 w2-like 分别采集：

```bash
ncu --set full \
  --section PmSampling \
  --section PmSampling_WarpStates \
  -k "regex:_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel" \
  -c 1 \
  -o profile/<run_name>/reports/full_<tag> \
  <harness command>
```

```bash
ncu --set source \
  --section SourceCounters \
  -k "regex:_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel" \
  -c 1 \
  -o profile/<run_name>/reports/source_<tag> \
  <harness command>
```

重点读：

- achieved occupancy / theoretical occupancy。
- registers per thread / shared memory。
- tensor core utilization。
- global load sectors/request。
- L1/L2 hit rate。
- long_scoreboard / short_scoreboard / wait / lg_throttle。
- PM sampling timeline 是否有 tail effect。

## 候选顺序

### 候选 0：验证 harness

只新增或使用 profile 目录下的 benchmark/harness，不改 kernel。

通过条件：

- baseline w13-like 和 w2-like 都能运行。
- 能输出 correctness 统计和 latency。
- 可重复运行，结果波动在可接受范围内。

### 候选 1：w13/w2 shape-aware meta tuning

最小代码改动：

- 保持 kernel 数学逻辑不变。
- 在 `_launch_grouped_gemm_contig` 中根据 `N/K/max_m` 区分 w13-like 和 w2-like。
- 第一轮只调整 launch meta：
  - `BLOCK_M`
  - `BLOCK_N`
  - `DOT_K`
  - `num_warps`
  - `num_stages`

初始搜索重点：

- 当前 baseline：`BLOCK_M=64, BLOCK_N=128, DOT_K=32, num_warps=4, num_stages=3`。
- 尝试降低 register pressure：
  - `BLOCK_M=32, BLOCK_N=128, DOT_K=32`
  - `BLOCK_M=64, BLOCK_N=64, DOT_K=32`
  - `BLOCK_M=32, BLOCK_N=64, DOT_K=32`
- 尝试 w2 的不同 N tile：
  - `BLOCK_N=128` vs `BLOCK_N=256`，仅在编译和 correctness 通过时保留。
- 暂不把 `DOT_K=64` 与候选 1 混在一起；它属于候选 2。

通过条件：

- correctness 通过。
- w13-like 或 w2-like 至少一个显著变快，另一个不明显变慢。
- 若只优化 w13，w2 不应退化超过 3%。
- 若只优化 w2，w13 不应退化超过 3%。

### 候选 2：DOT_K / activation scale 复用

只有候选 1 完成后再做。

内容：

- 评估 `DOT_K=64` 是否可正确处理每 32K 一个 b_scale 的语义。
- 如果不能直接改 `DOT_K`，再考虑 128K 外层 group 内部复用 a_scale。

通过条件：

- correctness 通过。
- ncu 确认 scale load 或相关 stall 降低。
- latency 有稳定收益。

### 候选 3：E8M0 weight scale normal path prototype

只有候选 1/2 收益不足时再做。

内容：

- 参考 low-latency E8M0 kernel，做 normal path prototype。
- 不立即改模型加载主路径，先用局部转换测试。

通过条件：

- 输出误差可接受。
- weight scale load bytes 或 register pressure 明显下降。
- microbenchmark 收益足以覆盖后续接入风险。

### 候选 4：tile compaction / tail-effect 调度

只有 ncu 或真实 token 分布证明 tail effect 是主因时再做。

内容：

- 记录 `num_tokens_per_expert` 分布。
- 评估 compact tile map 或 persistent work queue。

通过条件：

- tail effect 在 PM sampling 中明确存在。
- tile map 生成成本小于 kernel 节省。
- 不破坏 normal path 的 DeepEP layout。

## 记录方式

每个候选都在本文件追加一段结果：

```text
候选 X：
- 改动：
- 正确性：
- latency：
- ncu：
- 结论：保留 / 回滚 / 继续改
```

如果候选被拒绝，保留原因，避免重复尝试。

## 当前状态

- `docs/draft.md` 已完成。
- `docs/plan.md` 已生成。
- 候选 0 已建立局部 correctness/latency harness：
  - `profile/mxfp4_w4a8_prefill_local_harness_20260616/harness/bench_contig.py`
  - baseline w13-like：`N=4096, K=6144, total_m=16384, E=48, skew`，direct kernel 与 `_launch_grouped_gemm_contig` 输出完全一致，`max_abs_err=0`，mean latency `6.58 ms`。
  - baseline w2-like：`N=6144, K=2048, total_m=16384, E=48, skew`，direct kernel 与 `_launch_grouped_gemm_contig` 输出完全一致，`max_abs_err=0`，mean latency `3.73 ms`。
  - 当前 synthetic skew 的 `max_tokens_per_expert=872`，grid 分别为 `[48, 14, 32]` 和 `[48, 14, 48]`。
- 候选 1 已实现并验证：对 normal contig dot-scaled launch 增加 `maxnreg=168`，可用 `SGLANG_MXFP4_W4A8_CONTIG_MAXNREG=0` 关闭或用该环境变量改成其他寄存器上限。
  - 改动文件：`python/sglang/srt/layers/moe/moe_runner/mxfp4_w4a8_deepep_triton.py`。
  - 正确性：w13-like 和 w2-like 输出都与 baseline 完全一致，`max_abs_err=0`。
  - 16k skew microbenchmark：
    - w13-like：关闭 maxnreg `6.523 ms`，默认 maxnreg168 `6.069 ms`，约 `1.075x`。
    - w2-like：关闭 maxnreg `3.718 ms`，默认 maxnreg168 `3.077 ms`，约 `1.208x`。
  - 1k uniform normal microbenchmark：
    - w13-like：`1.20 ms -> 1.10 ms`，约 `1.09x`。
    - w2-like：`0.61 ms -> 0.57 ms`，约 `1.07x`。
  - ncu w13-like：
    - baseline registers/thread `207`，theoretical occupancy `12.5%`，achieved occupancy `12.42%`。
    - maxnreg168 registers/thread `168`，theoretical occupancy `18.75%`，achieved occupancy `18.56%`。
    - NCU duration：baseline `8.51 ms`，maxnreg168 `7.85 ms`。
  - ncu w2-like：
    - baseline registers/thread `205`，theoretical occupancy `12.5%`，achieved occupancy `12.40%`。
    - maxnreg168 registers/thread `168`，theoretical occupancy `18.75%`，achieved occupancy `18.57%`。
    - NCU duration：baseline `4.75 ms`，maxnreg168 `3.98 ms`。
  - 代价：强制寄存器上限会引入或放大 local load/store spill 风险；当前 w13/w2 microbenchmark latency 仍下降，但需要端到端验证是否稳定。
  - 仍未解决的问题：
    - w13/w2 的 L1/TEX throughput 仍在 `80%` 左右，DRAM throughput 低于 `13%`，说明不是单纯 HBM 带宽受限。
    - NCU 仍报告 uncoalesced global access，excessive sectors 约 `54%-55%`；shared excessive wavefronts 约 `28%`。
    - 因此候选 1 是低风险局部收益，不是通往 `<=200 ms/step` 的充分优化。
  - 同一方向已拒绝的寄存器上限：
    - `maxnreg=144`：过度 spill，明显变慢。
    - `maxnreg=160`：有效，但弱于 `168`，w13 约 `6.31 ms`，w2 约 `3.14 ms`。
    - `maxnreg=170`：明显退化。
    - `maxnreg=176/192`：收益较弱或不稳定，弱于 `168`。
  - 已拒绝的同轮方向：
    - `BLOCK_M/BLOCK_N` 缩小或 `BLOCK_N=256` 均明显变慢。
    - `num_warps=8` 明显变慢。
    - `num_stages=4` 只有约 `0.1%-0.2%`，视为噪声，不单独接入。
    - B coalesced transpose-load prototype 正确但几乎无收益。
    - E8M0 rhs scale prototype：w2 约 `1.5%` 小收益，但 w13 约 `7.7%` 退化，不接入 normal path。
- 端到端 16k prefill profile 暂按用户要求不跑。当前优先评估 Humming/CUDA/CUTLASS 结构性替代方案。

候选 2：
- 改动：仅在局部 harness 中实现 activation scale 复用 prototype，保持 `DOT_K=32`，在每 128K activation scale group 外层加载一次 `a_scale`，内部仍按每 32K 读取对应 `b_scale`。
- 正确性：w13-like 和 w2-like 输出都与当前默认 baseline 完全一致，`max_abs_err=0`。
- latency：
  - w13-like：当前默认 baseline `6.076 ms`，prototype `6.431 ms`，约 `0.945x`。
  - w2-like：当前默认 baseline `3.072 ms`，prototype `3.455 ms`，约 `0.889x`。
- 结论：拒绝。虽然减少了重复 `a_scale` load，但延长了 `a_scale` 生命周期并改变了编译器调度，寄存器/调度代价超过收益。

候选 3：
- 改动：新增 aligned N/K 专用 normal contig dot-scaled kernel，只在 `n % BLOCK_N == 0` 且 `k % DOT_K == 0` 时使用；去掉 N/K 边界 mask 和冗余 store `tl.where`，仅保留 M tail mask。
- 生产开关：默认开启，可用 `SGLANG_MXFP4_W4A8_CONTIG_ALIGNED_NK=0` 关闭。
- 正确性：w13-like 和 w2-like 输出都与原始 direct kernel 完全一致，`max_abs_err=0`。
- 50-iter microbenchmark：
  - w13-like：当前默认 maxnreg168 原 kernel `6.046 ms`，aligned kernel `5.932 ms`，约 `1.019x`。
  - w2-like：当前默认 maxnreg168 原 kernel `3.074 ms`，aligned kernel `3.038 ms`，约 `1.012x`。
- 接入后默认 `_launch_grouped_gemm_contig`：
  - w13-like：`5.958 ms`。
  - w2-like：`3.041 ms`。
- ncu w13-like aligned：
  - registers/thread `168`，theoretical occupancy `18.75%`，achieved occupancy `18.57%`。
  - NCU duration `7.71 ms`，相对 maxnreg168 原 kernel `7.85 ms` 小幅下降。
  - uncoalesced global/shared access 仍未改善，excessive sectors 仍约 `55%`。
- 结论：保留。收益小但稳定，且有条件保护；它不是主瓶颈优化，主要减少 aligned shape 下不必要的边界判断和 store value select。

候选 4：
- 改动：新增 Humming bridge harness，并在 `_launch_grouped_gemm_contig` 增加默认关闭的实验性 Humming normal path 开关。
- 新增 harness：
  - `profile/mxfp4_w4a8_prefill_local_harness_20260616/harness/bench_humming_bridge.py`
- 生产开关：
  - `SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL=1`
  - `SGLANG_HUMMING_ROOT=/sgl-workspace/humming-main`
- 影响范围：
  - 只影响 prefill / extend normal path 的 `_launch_grouped_gemm_contig`。
  - decode / low-latency path 不变。
  - 只在 `a_scale_group_size=128`、`b_scale.float32`、`N/K` 满足 Humming 约束时使用。
  - 首次使用会做 Humming weight repack 和 NVRTC/JIT，后续按 weight data pointer 缓存。
- bridge correctness：
  - w13-like：`max_abs_err=0.0078125`，`mean_abs_err=1.26e-5`，`allclose_1e-2=true`。
  - w2-like：`max_abs_err=0.00937`，`mean_abs_err=7.22e-6`，`allclose_1e-2=true`。
- bridge latency：
  - w13-like：Triton `5.961 ms`，Humming `3.093 ms`，约 `1.93x`。
  - w2-like：Triton `3.061 ms`，Humming `1.686 ms`，约 `1.82x`。
- SGLang 开关路径 correctness：
  - w13-like：Triton direct reference 对比 `allclose_1e-2=true`，`max_abs_err=0.00861`。
  - w2-like：Triton direct reference 对比 `allclose_1e-2=true`，`max_abs_err=0.00827`。
- SGLang 开关路径 latency：
  - w13-like：`3.157 ms`。
  - w2-like：`1.770 ms`。
- 已拒绝子方向：
  - 直接 `float32 -> e8m0` weight scale：速度很快，但误差不可接受。w13-like `max_abs_err=3.93`，w2-like `max_abs_err=2.28`。
- CUTLASS 判断：
  - 现有 `cutlass_w4a8_moe.py` 是 int4 W4A8 CUTLASS grouped GEMM，不是当前 MXFP4 E2M1 + per-32K scale 的 drop-in。
  - 直接接入需要重做 weight repack、scale iterator、grouped-contiguous scheduler 和 epilogue，工程量接近自研 CUDA kernel。
- 结论：Humming 是目前最有参考价值的 CUDA/WGMMA 路线。短期建议继续用 Humming prototype 做真实模型权重局部验证；长期再决定 vendor Humming runtime，或按 Humming 的 WGMMA scheduler / offline repack 思路自研 CUDA/CUTLASS-style kernel。

下一步：
- 不跑端到端时，优先做真实模型权重的局部 Humming 验证，确认 synthetic 误差结论能迁移到实际 `w13_weight/w2_weight`。
- 记录首次 repack/JIT 成本和 steady-state latency。
- 如果真实权重局部验证通过，再考虑服务内 opt-in A/B；最终仍需回到 16k prefill 端到端 profile，观察 GEMM sum 和 `cached_notify_combine` 是否同步下降。
