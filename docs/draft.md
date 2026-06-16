# MXFP4 W4A8 DeepEP Prefill Grouped GEMM 优化草稿

## 任务边界

目标是优化 SGLang 中 DeepEP normal / prefill / extend 路径的 MXFP4 W4A8 MoE grouped GEMM，重点 kernel 为：

- `python/sglang/srt/layers/moe/moe_runner/mxfp4_w4a8_deepep_triton.py`
- `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`

本阶段只做分析和计划草稿，不修改 kernel。decode / low-latency 路径暂不作为第一轮优化目标。

当前环境与约束：

- 业务目标 GPU：H200/Hopper。
- 实测 GPU：8 x NVIDIA H20Z，compute capability 9.0，约 143771 MiB/卡。
- CUDA runtime 12.9，torch 2.11.0+cu129，triton 3.6.0，ncu 2025.2.1.0。
- 优化判断按 Hopper / sm90，不直接套用 Blackwell / sm100 专用假设。

## 已阅读材料

KDA 流程：

- `/sgl-workspace/kernel-design-agents/README.md`
- `/sgl-workspace/kernel-design-agents/CLAUDE.md`
- `/sgl-workspace/kernel-design-agents/docs/agent-flow.md`
- `/sgl-workspace/kernel-design-agents/prompts/basic-flow.md`

本地技能说明与参考：

- `/sgl-workspace/kernel-design-agents/skills/KernelWiki/SKILL.md`
- `/sgl-workspace/kernel-design-agents/skills/ncu-report-skill/SKILL.md`
- `ncu-report-skill/reference/00-directory-layout.md`
- `ncu-report-skill/reference/03-collection.md`
- `ncu-report-skill/reference/05-analysis-dimensions.md`
- `ncu-report-skill/reference/06-diagnosis-playbook.md`

目标代码与相关路径：

- `python/sglang/srt/layers/moe/moe_runner/mxfp4_w4a8.py`
- `python/sglang/srt/layers/moe/moe_runner/mxfp4_w4a8_deepep_triton.py`
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
- `python/sglang/srt/layers/moe/utils.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`

KernelWiki 本地查询过的相关条目：

- `wiki/kernels/grouped-gemm.md`
- `sources/prs/sglang/PR-16014.md`
- `sources/prs/flashinfer/PR-2193.md`
- `sources/prs/flashinfer/PR-1396.md`

其中 `grouped-gemm.md` 对当前任务最有直接帮助：MoE prefill 更适合 contiguous layout，decode 更常用 masked layout；专家 token 数不均衡和 tail effect 是 grouped GEMM 的常见瓶颈。

## 当前执行路径

DeepEP mode 选择：

- `DeepEPMode.AUTO.resolve(is_extend_in_batch)` 中，extend / prefill 返回 `NORMAL`，decode 返回 `LOW_LATENCY`。
- DeepEP dispatcher 在 `token_dispatcher/deepep.py` 的 `_get_impl()` 中根据 `get_is_extend_in_batch()` 选择 normal 或 low-latency dispatcher。

Prefill / extend normal 路径：

1. `fused_experts_deepep_to_mxfp4_w4a8(...)`
2. 如果 dispatch output 是 DeepEP normal 格式，进入 `_mxfp4_w4a8_deepep_normal(...)`。
3. `_mxfp4_w4a8_deepep_normal(...)` 使用 `ep_scatter` 把 dispatch 后的 tokens 按专家连续排列，生成：
   - `input_tensor: [all_tokens, hidden_size]`，dtype 为 `torch.float8_e4m3fn`。
   - `input_tensor_scale: [all_tokens, hidden_size / 128]`，dtype 为 `float32`。
   - `expert_start: [num_experts]`，每个专家在 contiguous token buffer 中的起始 offset。
   - `num_tokens_per_expert_gpu: [num_experts]`。
4. 调用 `mxfp4_w4a8_deepep_normal_triton(...)`。
5. normal Triton 函数内先做 w13 gate/up grouped GEMM：
   - `_launch_grouped_gemm_contig(..., n=gateup_size, k=hidden_size)`。
6. 对 w13 输出执行 `silu_and_mul_masked_post_quant_fwd`，得到 down projection 的 FP8 输入和 per-token-group scale。
7. 再做 w2 down grouped GEMM：
   - `_launch_grouped_gemm_contig(..., n=hidden_size, k=intermediate_size)`。
8. 返回 contiguous down output，外层 `_mxfp4_w4a8_deepep_normal(...)` 再用 `ep_gather` 按 topk 权重聚合回原 token 顺序。

当前 `_launch_grouped_gemm_contig` 的调度：

- `max_m = max(num_tokens_per_expert)`，来自 Python 侧 list。
- 根据 `max_m` 粗略选择 `BLOCK_M`：
  - `<=8 -> 8`
  - `<=16 -> 16`
  - `<=32 -> 32`
  - 否则 `64`
- `BLOCK_N = 128`。
- `BLOCK_K = 64` 被传入，但 dot-scaled kernel 实际循环使用 `DOT_K`。
- 默认 `_USE_DOT_SCALED = True`，`DOT_K = 32`。
- grid 为 `(num_experts, ceil(max_m / BLOCK_M), ceil(N / BLOCK_N))`。
- 每个 program 对应一个 `(expert_id, m_block, n_block)` tile，遇到 `token_count <= m_block * BLOCK_M` 直接 return。

## 当前 kernel 的功能和数据流

`_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel` 做的是按专家分组的矩阵乘：

```text
C[expert_start[e] : expert_start[e] + M_e, 0:N]
  = A[expert_start[e] : expert_start[e] + M_e, 0:K]
    @ dequant_mxfp4(B[e, 0:N, 0:K]).T
```

其中：

- `A` 是 DeepEP scatter 或中间激活 quant 后的 FP8 E4M3 数据。
- `a_scale` 是 activation per-token-group float32 scale，当前 normal path group size 为 128。
- `B` 是 per-expert packed MXFP4 weight，两个 E2M1 元素打包在一个 byte。
- `b_scale` 是 weight per-output-channel / per-32-K group 的 float32 scale。
- 输出 `C` 是 bf16。

kernel 内每个 K tile：

1. 从 contiguous `A` 读 `BLOCK_M x DOT_K` 的 FP8 E4M3。
2. 从 `B` 读 `(DOT_K / 2) x BLOCK_N` 的 packed E2M1 bytes。
3. 用 `tl.dot_scaled(a_raw, None, "e4m3", b_raw, None, "e2m1")` 计算未乘外部 scale 的 dot。
4. 分别加载：
   - `a_scale[global_m, k_start / A_SCALE_GROUP_SIZE]`，shape 为 `BLOCK_M`。
   - `b_scale[expert, offs_n, k_start / DOT_K]`，shape 为 `BLOCK_N`。
5. 执行 `acc += raw_acc * a_scale[:, None] * b_scale[None, :]`。
6. K 循环结束后将 FP32 accumulator 转成 bf16 store。

相对非 dot-scaled fallback kernel，dot-scaled 版本不手写 E2M1 nibble decode，而是把 FP8/E2M1 原始值交给 Triton `tl.dot_scaled`。但当前 scale 仍然在 dot 后逐 tile 外乘，且 scale 是 float32 载入。

## Prefill normal 与 decode low-latency 的区别

Prefill / extend normal：

- 输入是 `[all_tokens, K]` 的 contiguous token buffer。
- 每个专家通过 `expert_start` 和 `num_tokens_per_expert` 定位自己的 token 段。
- 使用 `_launch_grouped_gemm_contig` 和 `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`。
- grid 的 M 维使用全局 `max_m`，小专家的多余 M block 会进 kernel 后 early return。
- 输出也是 `[all_tokens, N]` contiguous，后续 `ep_gather` 组合。

Decode / low-latency：

- 输入是 `[num_experts, expected_m, K]` 的 masked layout。
- 每个专家有固定 `expected_m`，实际有效 token 数由 `masked_m[expert]` 控制。
- 使用 `_launch_grouped_gemm` 和 `_mxfp4_w4a8_grouped_gemm_dot_scaled_kernel`。
- 小 M 场景有额外 heuristic，例如 `m <= 8`、`num_routed_tokens <= 32` 时使用更小 `BLOCK_M`。
- low-latency 路径还存在可选的 E8M0 weight scale kernel：`_mxfp4_w4a8_grouped_gemm_dot_scaled_e8m0_kernel`，但 normal path 当前只接受 float32 weight scale。

本次优先 normal path，因为 16k prefill profile 中主要耗时来自 `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`。

## Profile 证据复核

用户给出的主口径：

- W4A8 单机 16k prefill：
  - GPU wall 约 `1322.18 ms/step`
  - kernel sum 约 `1280.37 ms/step`
  - prefill 吞吐约 `16384 / 1.322s = 1.24w tokens/s`
- W8A8 双机 16k prefill：
  - GPU wall 约 `525.44 ms/step`
  - kernel sum 约 `495.95 ms/step`
  - prefill 吞吐约 `16384 / 0.525s = 3.12w tokens/s`
- W4A8 比 W8A8 慢约 `2.52x`。

本地用 trace 中 `step[EXTEND bs=2 toks=16384]` 且 CPU annotation duration > 100ms 的长 step 做复核，按 kernel start 落在 CPU step 区间聚合，得到近似值：

- W4A8：
  - 长 step 数：7。
  - 平均 CPU step：`1308.85 ms`。
  - 平均 GPU kernel wall：`1312.71 ms`。
  - 平均 kernel sum：`1270.93 ms`。
  - `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`：`753.72 ms/step`，约 `137 launches/step`。
- W8A8：
  - 长 step 数：15。
  - 平均 CPU step：`523.59 ms`。
  - 平均 GPU kernel wall：`523.97 ms`。
  - 平均 kernel sum：`494.51 ms`。
  - `deep_gemm::sm90_fp8_gemm_1d2d_impl` 主 MoE GEMM 合计约 `109.41 ms/step`。

这与用户给出的 `757.22 ms/step` 和 `109.42 ms/step` 在量级和排序上吻合。不同过滤口径会造成 1% 左右差异，后续优化比较应固定同一脚本和同一 trace/benchmark 口径。

W4A8 prefill 长 step 中主要 kernel 聚合：

- `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`：约 `754-757 ms/step`，占 W4A8 kernel sum 约 `59%`。
- `cached_notify_combine`：本地复核约 `215 ms/step`，用户口径约 `219.48 ms/step`。
- `deep_ep::intranode::combine`：约 `37 ms/step`。
- `deep_ep::intranode::dispatch`：约 `20 ms/step`。

W8A8 对照：

- DeepGEMM FP8 MoE GEMM 主耗时约 `109 ms/step`。
- internode combine actual movement 约 `82 ms/step`，cached notify 约 `48 ms/step`。
- 虽然 W4A8 单机实际通信比 W8A8 双机少约 `60 ms/step`，但 MXFP4 W4A8 GEMM 多出约 `648 ms/step`，通信节省无法抵消 GEMM 差距。

trace 暴露出的当前 Triton kernel launch 参数：

- kernel name：`_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`。
- 每个 launch 的 block 为 `[128, 1, 1]`，即 4 warps。
- shared memory：`16384` bytes。
- registers per thread：
  - 一组为 `207`。
  - 另一组为 `205`。
- trace 中估算 achieved occupancy 为 `13%`。
- grid 第三维有两类：
  - `ceil(N / 128) = 32`，约 `69 launches/step`，合计约 `482.6 ms/step`，单 launch p50 约 `6.98 ms`。
  - `ceil(N / 128) = 48`，约 `68 launches/step`，合计约 `271.1 ms/step`，单 launch p50 约 `3.98 ms`。
- 结合 DeepGEMM FP8 kernel 名称中的形状，当前模型很可能是两类投影：
  - w13：`N ~= 4096, K ~= 6144`，更重，对应 `grid_nblocks = 32`。
  - w2：`N ~= 6144, K ~= 2048`，较轻，对应 `grid_nblocks = 48`。

这说明 w13 和 w2 的 shape 差异明显，当前一个 `_launch_grouped_gemm_contig` heuristic 同时服务两者，可能不是最优。

## 初步瓶颈判断

当前 kernel 很可能不是单纯 DRAM 带宽受限，而是多个因素叠加：

1. Tensor Core 路径存在，但利用率可能不高。
   - `tl.dot_scaled` 应使用矩阵乘硬件路径，而不是 fallback 的标量 decode + dot。
   - 但 trace 中 estimated occupancy 只有 `13%`，registers/thread 达到 `205-207`，每个 CTA 的 resident blocks/warps 可能受寄存器限制。
   - 需要 ncu 确认 `sm__pipe_tensor_cycles_active`、`sm__throughput`、`sm__warps_active` 和 stall reason。

2. scale load 和 scale 外乘开销可能很重。
   - 每个 `DOT_K=32` tile 都加载 `BLOCK_M` 个 activation scale 和 `BLOCK_N` 个 weight scale。
   - `a_scale` group size 为 128，但当前循环每 32 K 一次；同一个 `a_scale` 会在连续 4 个 `DOT_K` tile 中重复加载。
   - `b_scale` group size 正好 32，不能简单跨 DOT_K 复用，但可评估 E8M0 scale 或布局/载入方式。
   - `acc += raw_acc * a_scale[:, None] * b_scale[None, :]` 对 `BLOCK_M x BLOCK_N` 做 FP32 外乘，增加寄存器和指令压力。

3. register pressure / occupancy 是高风险瓶颈。
   - `BLOCK_M=64, BLOCK_N=128` 的 accumulator 是 `64 x 128` 逻辑 tile，虽然 Triton 会分配到 warps/threads，但 trace 显示寄存器已经非常高。
   - 低 occupancy 会放大全局 load latency、scale load latency和 grouped GEMM tail effect。

4. grouped GEMM tail effect 和专家负载不均衡需要验证。
   - grid M 维由 `max_m` 决定，很多专家如果 token_count 较小会产生 early-return block 或短 block。
   - 如果 token 分布长尾，少数高 token expert 的 tiles 会拖住每层 GEMM。
   - 当前每层两个 GEMM launch，约 137-138 launches/step；每个 launch 独立调度，tail effect 会重复出现。

5. packed MXFP4 weight 访问模式可能影响 L2/L1 效率。
   - `b_packed` 地址形状是 `(DOT_K/2, BLOCK_N)`，通过 `offs_n[None, :] * stride_bn + offs_k2[:, None] * stride_bk2` 加载。
   - 当前 weight tensor 要求 `stride(-1) == 1`，K/2 维连续；对一个 program 来说，K2 维连续、N 维跨 stride。
   - `tl.dot_scaled` 对 packed E2M1 的期望布局和当前 `[N, K/2]` 布局是否达到 Hopper 上的最佳 coalescing，需要用 ncu 的 sectors/request、L1/L2 hit rate 和源行 stall 验证。

6. 当前 `BLOCK_K` 参数对 dot-scaled kernel 不生效。
   - `_launch_grouped_gemm_contig` 会设置 `block_k=64`，但 dot-scaled kernel 的 K 循环使用 `DOT_K=32`。
   - 调 `BLOCK_K` 对当前默认路径可能没有效果，除非改 kernel 逻辑或禁用 dot-scaled fallback。

## 与 W8A8 DeepGEMM 的差距

W8A8 走 DeepGEMM `grouped_gemm_nt_f8f8bf16_contig`，核心差异：

- DeepGEMM 是专门的 sm90 FP8 grouped GEMM，使用 TMA / WGMMA / tuned tile 调度，且有 contiguous layout。
- W8A8 的两类 MoE GEMM 合计约 `109 ms/step`；W4A8 Triton dot-scaled 约 `754-757 ms/step`，主差距约 `6.9x`。
- W4A8 理论权重 bytes 更少，但需要：
  - packed FP4 解码或 `dot_scaled` 处理。
  - per-token activation scale。
  - per-output-channel weight scale。
  - scale 外乘。
  - Triton 编译出的寄存器和调度开销。
- 因此 W4A8 没有自动转化成吞吐优势；当前更像是算子实现效率不足，而不是通信瓶颈。

需要注意：DeepGEMM 的调度思想可以借鉴，例如 contiguous grouped layout、shape-specific tile、tail 处理、TMA/WGMMA pipeline，但不能直接假设 MXFP4 W4A8 在 Triton 上能达到 FP8 DeepGEMM 的同等效率。

## Correctness reference

后续实现任何候选优化前，必须先固定 correctness reference：

1. 局部 kernel reference：
   - 使用当前 `_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel` 作为 baseline。
   - 对同一随机输入和同一 `expert_start / num_tokens_per_expert`，比较候选 kernel 输出与 baseline。

## 2026-06-16 补充：结构性替代路线

在 Triton meta tuning 收益有限后，追加评估了 Humming/CUDA/CUTLASS 路线。

Humming 结论：

- Humming 是 NVRTC CUDA / WGMMA 路径，支持 FP8 activation + FP4 E2M1 weight 和 grouped-contiguous MoE。
- 当前 SGLang packed weight 可通过 `b_packed.contiguous().view(torch.int32)` 交给 Humming repack，nibble 顺序在本地验证中兼容。
- `b_scale.float32 -> bf16` 后，局部 synthetic correctness 可通过 `allclose(atol=1e-2, rtol=1e-2)`。
- 同形状 latency：
  - w13-like：Triton `5.961 ms`，Humming bridge `3.093 ms`，约 `1.93x`。
  - w2-like：Triton `3.061 ms`，Humming bridge `1.686 ms`，约 `1.82x`。
- 已加入默认关闭的实验开关 `SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL=1`，只替换 normal contig path，不影响 decode。

E8M0 scale 结论：

- 直接把当前 float32 weight scale cast 到 E8M0 虽然很快，但误差不可接受。
- w13-like `max_abs_err=3.93`，w2-like `max_abs_err=2.28`。
- 除非模型加载时本来就提供符合语义的 UE8M0/MXFP4 scale，否则不能作为 drop-in。

CUTLASS 结论：

- 仓库现有 `cutlass_w4a8_moe.py` 面向 int4 W4A8，不是当前 MXFP4 E2M1 + per-32K scale 的直接替换。
- 要用 CUTLASS 生产化当前语义，需要重写 weight layout、scale iterator、grouped scheduler 和 epilogue，工程量接近自研 CUDA kernel。
- 短期更建议用 Humming prototype 验证真实权重和服务内 A/B；长期再决定是否 vendor Humming runtime 或按其设计自研 CUDA/CUTLASS-style kernel。
   - 重点覆盖 w13 shape 和 w2 shape 两类。

2. PyTorch reference：
   - 参考 `mxfp4_w4a8.py` 中 `_dequant_mxfp4_matrix`、`_dequant_deepep_activation` 和 `_mxfp4_w4a8_deepep_ll_reference` 的逻辑。
   - 对 normal contiguous layout 可按专家切片：
     - dequant FP8 activation 到 bf16/float32。
     - dequant packed MXFP4 weight 到 bf16/float32。
     - 执行 `A @ W.T`。
     - w13 后接 SiLU/mul 和 FP8 quant-dequant，再做 w2。
   - 由于 FP8/FP4/scale 路径存在量化误差，候选与当前 Triton baseline 的 bitwise 或近似比较优先级高于完整 PyTorch reference。

建议误差统计：

- 输出 dtype：bf16。
- 指标：
  - `max_abs_err`
  - `mean_abs_err`
  - `max_rel_err`
  - `mean_rel_err`
  - `allclose` 通过率。
- 候选对 baseline 初始阈值：
  - `atol=1e-2, rtol=1e-2` 作为起点。
  - 如果仅改变 tiling / scheduling，不改变数值顺序太多，应收紧并记录实际分布。
  - 如果改变 K 分块或 scale 合并导致累加顺序变化，应以误差统计和端到端 accuracy smoke test 共同判断。

## 候选优化方向

### 方向 1：w13 / w2 分 shape tuning

内容：

- 将 `_launch_grouped_gemm_contig` 的 heuristic 从只看 `max_m`，扩展为同时看 `N` 和 `K`。
- 对 w13 的 `N ~= 4096, K ~= 6144` 和 w2 的 `N ~= 6144, K ~= 2048` 分别选择 `BLOCK_M / BLOCK_N / DOT_K / num_warps / num_stages`。
- 首先评估不改变 kernel 数学逻辑的 meta 参数组合。

可能收益：

- 当前 w13 占约 `483 ms/step`，w2 占约 `271 ms/step`；w13 是第一优先级。
- w2 K 较小，可能更适合更大 N tile 或不同 num_warps。
- w13 K 较大，可能更受 scale 重复加载、register pressure 和 pipeline 影响。

风险：

- Triton `tl.dot_scaled` 对 shape 有隐藏约束；某些 `DOT_K` 或 tile 组合可能编译失败或退化。
- 更大 tile 可能进一步提高 registers/thread，降低 occupancy。
- 更小 tile 可能增加 program 数和 launch 内调度 overhead。

验证：

- 用局部 harness 跑 w13/w2 两类 representative shape，比较 correctness 和 latency。
- 从 trace 或 microbenchmark 记录每类 grid 的 p50/p95 latency。
- ncu 确认 registers/thread、achieved occupancy、tensor pipe utilization 和 global load efficiency。

### 方向 2：评估 `DOT_K=64` 或跨 128 group 的 scale 复用

内容：

- 当前 activation scale group size 是 128，但 `DOT_K=32`，同一 `a_scale` 在 4 个 K tile 中重复加载。
- 尝试 `DOT_K=64`，理论上把 a_scale 重复 load 从每 128K 4 次降为 2 次。
- 更激进地，可在 kernel 内按 128 K group 外层循环，复用 `a_scale`，内部做 4 个 dot_scaled。

可能收益：

- 降低 activation scale load 次数和外乘次数调度开销。
- 对 w13 K 较大更可能有效。

风险：

- `tl.dot_scaled` 对 e2m1 packed B 的 K 维 tile 可能偏好 32；`DOT_K=64` 可能增加 registers 或生成更差代码。
- b_scale group size 为 32，`DOT_K=64` 需要两个 b_scale group，不能简单用单个 `b_scale`。
- 如果一次 raw_acc 覆盖 64K，而 scale 在 32K 内变化，数值语义会错；必须保证 b_scale 分段正确。

验证：

- 先只做 compile / correctness smoke，确认数值不变。
- ncu 看 source-level stalls 是否从 scale load / FP32 multiply 相关 PC 下降。
- latency 按 w13/w2 分开统计。

### 方向 3：减少 float32 weight scale load / decode 开销

内容：

- normal path 当前要求 `w13_weight_scale` 和 `w2_weight_scale` 为 float32。
- low-latency path 已存在可选 `_dot_scaled_e8m0_kernel`，但 normal path 未用。
- 评估能否在 normal path 使用 E8M0 scale，或在权重准备阶段生成更适合 dot_scaled 的 scale layout。

可能收益：

- weight scale bytes 从 float32 降低到 uint8/E8M0。
- `tl.dot_scaled(..., b_scale, "e2m1")` 可能把 scale 处理更靠近 dot_scaled 内部路径，减少外乘和寄存器压力。

风险：

- 上游 quant_info 当前 normal path 明确要求 float32 scale，改动会牵涉模型加载、权重准备和 dtype 兼容。
- E8M0 rounding 与现有 float32 scale 语义可能不完全一致，可能影响 accuracy。
- 不是第一轮最小改动，建议在 meta tuning 后再做。

验证：

- 先做离线转换 float32 scale -> e8m0 -> float32 的误差统计。
- 候选 kernel 对当前 baseline 比较输出误差。
- 端到端小模型 accuracy smoke，再跑 16k prefill performance。

### 方向 4：优化 mask / early return / tile schedule，降低 tail effect

内容：

- 当前 grid 使用 `ceil(max_m / BLOCK_M)` 覆盖所有专家，短专家会产生大量 early return 或短 tile。
- 可评估生成 compact tile schedule，让 grid 只包含有效 `(expert, m_block, n_block)`。
- 或对大 prefill 下 `max_m` 较高的场景，做 persistent / work-queue 式 tile 调度，减少尾波。

可能收益：

- 如果专家 token 分布非常不均，tail effect 可能显著。
- 减少无效 blocks，也减少 kernel 内 `token_count` early return 开销。

风险：

- 需要额外生成 tile map，增加前处理 kernel 或 CPU/GPU 同步。
- 每层 MoE 都会调用，tile map 生成成本必须小于节省。
- persistent/work-queue 在 Triton 中实现和验证复杂。

验证：

- 先从实际 trace/harness 记录每层 `num_tokens_per_expert` 分布：
  - max/mean
  - p50/p90/p99
  - empty expert 比例
  - tile 数浪费比例
- 用 ncu PM sampling 看是否存在明显 long tail。
- 若 tail 明显，再实现最小 tile compaction 候选。

### 方向 5：weight 访问布局和 coalescing 检查

内容：

- 当前 packed weight 是 `[expert, N, K/2]`，K/2 维连续。
- dot_scaled kernel 中 B load 形状为 `(DOT_K/2, BLOCK_N)`，对每个 N 读取连续 K2，但在 tile 内 N 维跨 stride。
- 评估是否需要转置/重排 weight，使 B tile 更符合 `tl.dot_scaled` 或 Hopper WGMMA 的访问偏好。

可能收益：

- 如果 ncu 显示 global load sectors/request 很差或 long_scoreboard 集中在 B load，layout 改动可能有较大收益。

风险：

- 权重重排会影响模型加载和内存布局，改动面比 meta tuning 大。
- 需要确保 w13/w2 权重准备和所有调用路径同步更新。
- 可能与其他 backend 共用 weight tensor，不能破坏现有逻辑。

验证：

- 先用 ncu 确认 B load 是热点。
- 单独写 prototype layout，不接入主路径，比较 microbenchmark。
- 确认内存占用和加载时间不会抵消 kernel 收益。

### 方向 6：尝试降低 register pressure

内容：

- 通过减小 `BLOCK_M`、`BLOCK_N` 或改写 scale 外乘，降低 accumulator 和临时 tensor 的 live range。
- 评估 `num_warps=4` 之外的组合，如 w2 使用 8 warps 或 w13 使用更小 tile。
- 如果 Triton 支持相关 hint，可尝试减少 num_stages 或调整 `tl.load` / `tl.dot_scaled` 结构，避免过多临时。

可能收益：

- trace 已显示 `205-207 registers/thread` 和 `13%` estimated occupancy。
- 如果 register 是主限制，降低寄存器可能明显提高 latency hiding。

风险：

- 寄存器下降不一定代表更快；tile 变小后 tensor core 利用率和 arithmetic intensity 可能下降。
- `tl.dot_scaled` 生成代码受 compiler 影响，某些写法收益不可预测。

验证：

- 每个候选记录 Triton 编译信息 / trace args / ncu launch stats。
- 重点看 `launch__registers_per_thread`、`sm__warps_active`、`smsp` stall reason 和 kernel latency。

## 第一轮建议排序

1. 建立局部 correctness + latency harness，锁定 w13/w2 representative shape。
2. 固定 baseline profile 脚本，能按 w13/w2 分组统计 kernel latency、launch args、grid。
3. 做 meta tuning：按 w13/w2 分别扫 `BLOCK_M / BLOCK_N / DOT_K / num_warps / num_stages`，只改 launch meta，不改语义。
4. 对 baseline 和最好的 meta 候选跑 ncu：
   - full + PM sampling。
   - source counters。
   - 重点确认 register、occupancy、tensor core utilization、memory coalescing、stall reason、tail effect。
5. 若 meta tuning 收益有限，再进入 scale 复用 / E8M0 scale / weight layout / tile schedule 等结构性改动。

## 计划中的验证命令

后续 `docs/plan.md` 需要把这些命令固化为可执行步骤。当前草稿阶段先列出需要的验证类别：

- 本地单 kernel correctness harness：
  - 构造 contiguous layout 的随机 FP8 activation、float32 activation scale、packed MXFP4 weight、float32 weight scale。
  - 覆盖 w13 和 w2 两组 N/K。
  - 覆盖均匀和偏斜 `num_tokens_per_expert`。
  - 候选输出与 baseline 输出比较误差统计。

- 单 kernel latency benchmark：
  - warmup 后多次调用 `_launch_grouped_gemm_contig`。
  - 分 w13/w2 分别记录 P50/P90/P99。
  - 记录每个候选的 meta 参数、Triton 编译参数、registers/thread。

- ncu：
  - profile 输出目录按 `profile/<new_run_name>/{harness,reports,analysis}`。
  - kernel regex：`regex:_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel`。
  - full pass：`ncu --set full --section PmSampling --section PmSampling_WarpStates ...`。
  - source pass：`ncu --set source --section SourceCounters ...`。
  - 分别 profile w13 和 w2，避免把两类 shape 混在一起。

- 端到端：
  - 候选通过局部 correctness 和 microbenchmark 后，再跑 16k prefill profile。
  - 统计：
    - step GPU wall。
    - kernel sum。
    - MXFP4 grouped GEMM sum。
    - `cached_notify_combine`。
    - dispatch / combine actual movement。
  - 观察 GEMM 下降后 notify/wait 是否同步下降。

## 当前未知数

- 实际 16k prefill 每层 `num_tokens_per_expert` 分布尚未从运行时直接导出；trace 只能间接看到 grid M 维分布。
- 尚未有 ncu 指标确认是 tensor core 利用不足、scale load 受限、global memory latency、register spill，还是 tail effect 为主。
- `tl.dot_scaled` 在 Triton 3.6.0 + sm90 上对 E4M3 x E2M1 的最优 tile 约束需要通过编译和 ncu 验证，不能只靠直觉。
- normal path 若引入 E8M0 scale，可能牵涉 quant_info 和权重准备流程，改动边界需要进一步确认。

## 初步结论

当前 W4A8 prefill 端到端差距主要来自 MXFP4 W4A8 grouped GEMM，而不是实际 DeepEP 通信。`_mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel` 在 16k prefill 中约 `754-757 ms/step`，相比 W8A8 DeepGEMM 的约 `109 ms/step` 慢约 `6.9x`，占 W4A8 kernel sum 约 `59%`。

最可能的第一批瓶颈是：低 occupancy / 高 register pressure、scale load 与 scale 外乘开销、w13/w2 shape 未区分 tuning，以及专家 token 分布导致的 grouped GEMM tail effect。下一步不应直接做大重构，而应先建立 correctness 和 microbenchmark，然后按 w13/w2 分开做 meta tuning，并用 ncu 确认主瓶颈后再决定是否进入 scale/layout/scheduler 级改动。
