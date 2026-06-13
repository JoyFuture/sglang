# DeepEP + MXFP4 在线反量化 W4A8 专家计算方案

## 1. 背景

当前工作目录：

```text
/sgl-workspace/sglang
```

模型目录：

```text
/preset-models
```

模型是 MiMoV2。当前 checkpoint 的量化配置大致是：

```json
{
  "quant_method": "fp8",
  "store_dtype": "mxfp4",
  "weight_block_size": [128, 128],
  "mxfp4_block_size": 32
}
```

这表示 routed experts 的权重存储格式是 MXFP4：

- expert weight 是 packed FP4 / E2M1；
- expert weight scale 是 E8M0；
- MXFP4 block size 是 32；
- checkpoint 本身应该继续保持 MXFP4 格式。

本需求的关键点是：**checkpoint 格式和运行时 GEMM 计算精度不是一回事**。

最终目标不是把 checkpoint 离线改成一个所谓 `w4afp8` checkpoint，而是：

```text
保持 MXFP4 checkpoint
开启 DeepEP
专家 GEMM 使用 FP8 activation
专家权重保持 MXFP4/W4，并在 GEMM 路径中在线反量化
在 Hopper / SM90 上执行 W4A8 expert GEMM
```

也就是：

```text
W = MXFP4 / W4 expert weight
A = FP8 activation
GEMM = W4A8 expert GEMM
输出 = BF16
通信/dispatch = DeepEP
目标硬件 = Hopper / SM90
```

本机已确认是 8 张 NVIDIA L20X，PyTorch 返回 compute capability 为：

```text
(9, 0)
```

所以 Hopper / SM90 路线是成立的。

## 2. 原始问题

最初执行上级目录 `run.sh` 时，在模型加载阶段报错：

```text
KeyError: 'model.layers.1.mlp.experts.w2_weight_scale'
```

根因是 MiMoV2 MXFP4 checkpoint 里的 expert scale 命名和 SGLang runtime 参数命名不一致：

checkpoint 里是：

```text
*_weight_scale
```

runtime 参数里是：

```text
*_weight_scale_inv
```

另外，checkpoint 里的 scale 是 `uint8`，语义上是 E8M0 scale；runtime 某些路径需要把它按 `torch.float8_e8m0fnu` 解释或转成对应参数 dtype。

## 3. 当前已完成的修改

已经提交了一个修复加载问题的 commit：

```text
f1aa784994f604a1b207d3e58e3060fb2454ad5b
Fix MiMoV2 MXFP4 expert loading
```

这个 commit 修改了 3 个文件：

```text
python/sglang/srt/configs/model_config.py
python/sglang/srt/models/mimo_v2.py
python/sglang/srt/server_args.py
```

### 3.1 model_config.py

新增了 MiMoV2 MXFP4 expert 检测逻辑：

```text
is_mimo_v2_mxfp4_experts(hf_config)
```

检测条件：

```text
architectures 包含 MiMoV2
quantization_config.quant_method == "fp8"
quantization_config.store_dtype == "mxfp4"
```

检测到后会设置：

```text
ModelConfig.is_fp4_experts = True
```

### 3.2 mimo_v2.py

修复了 MiMoV2 权重加载逻辑：

```text
checkpoint: *_weight_scale
runtime:    *_weight_scale_inv
```

现在加载时会把 checkpoint 里的 expert scale 名称映射到 runtime 参数名。

同时增加了 MXFP4 scale 处理：

```text
uint8 scale bytes -> torch.float8_e8m0fnu view/conversion
```

这解决了最初的 `KeyError`。

### 3.3 server_args.py

对 MiMoV2 MXFP4 experts 增加了一个保守 fallback：

如果：

```text
--moe-runner-backend auto
GPU 是 SM90 或 SM120
模型是 MiMoV2 MXFP4 experts
```

则自动选择：

```text
--moe-runner-backend marlin
```

这保证当前 checkpoint 能加载并完成推理。

## 4. 当前可运行脚本

创建了一个仓库外脚本：

```text
/sgl-workspace/run_mxfp4.sh
```

这个脚本可以用当前 checkpoint 正常启动并完成简单推理。

当前脚本大致配置：

```text
tp-size = 8
quantization = fp8
moe-runner-backend = marlin
DeepEP = 未开启
CUDA graph = disabled
```

这个路径的性质是：

```text
MXFP4 checkpoint + Marlin expert GEMM
```

它是当前可用的 fallback，但不是最终目标。

原因：

- 没有开启 DeepEP；
- expert activation 不是 FP8；
- 更接近 W4A16 / BF16 activation 路径；
- 不满足“DeepEP + MXFP4 在线反量化 + W4A8 expert GEMM”。

## 5. 已做过的实验和结论

### 5.1 DP Attention + DeepEP

尝试过：

```text
dp-size = 2
enable-dp-attention
deepep
```

失败原因：

```text
MiMoV2 fused QKV 要求 effective attention TP size = 8
dp-size = 2 时 effective attention TP size 变成 4
```

结论：

```text
当前模型/配置下应优先使用 TP8。
```

### 5.2 DeepEP + Marlin

尝试过：

```text
tp-size 8
deepep
moe-runner-backend marlin
```

失败报错核心：

```text
Runner backend MoeRunnerBackend.MARLIN requires a fused func for a2a backend deepep,
but none is registered.
```

结论：

```text
当前仓库没有 deepep -> marlin fused path。
```

即使补了这个路径，Marlin 当前也不是目标 W4A8 expert GEMM，因为它不是 FP8 activation 路径。

### 5.3 DeepEP + DeepGEMM + 当前 MXFP4

尝试过：

```text
tp-size 8
deepep
moe-runner-backend deep_gemm
```

模型能加载完 shards，并进入 forward，但在 kernel 路径失败：

```text
tvm.error.InternalError:
Assertion ... ab.scalar_type() == kPackedFP4 and arch_major == 10
```

结论：

```text
这个 DeepGEMM FP4 路径不是目标 Hopper W4A8 路线，
并且表现出 SM100 / Blackwell 相关限制。
```

## 6. 现有路径分析

### 6.1 Marlin MXFP4

相关文件：

```text
python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py
python/sglang/srt/layers/quantization/marlin_utils_fp4.py
python/sglang/srt/layers/moe/moe_runner/marlin.py
```

当前状态：

```text
能跑当前 checkpoint
能消费 MXFP4 experts
不开 DeepEP
不是 FP8 activation W4A8
```

结论：

```text
可以作为 fallback 和正确性参考，但不是最终方案。
```

### 6.2 现有 W4AFp8 / CUTLASS W4A8

相关文件：

```text
python/sglang/srt/layers/quantization/w4afp8.py
python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
sgl-kernel/csrc/moe/cutlass_moe/w4a8
```

这条路径确实有 Hopper 可用的 W4A8 kernel。

它的语义更接近：

```text
A = FP8 e4m3 activation
B = packed INT4 weight
B scale = 对应 INT4 量化 scale layout
```

但当前 checkpoint 是：

```text
B = packed MXFP4 / E2M1 weight
B scale = E8M0 per 32
```

所以不能简单把当前 MXFP4 checkpoint 直接塞进现有 `cutlass_w4a8_moe_mm`。

要复用这条 kernel，需要把 MXFP4 转成它期待的 INT4 权重格式。但这不符合最终目标，因为最终目标是：

```text
MXFP4 权重在线反量化 + W4A8 计算
```

而不是离线或加载后改成另一种 INT4 表示。

### 6.3 FlashInfer MXFP4

相关文件：

```text
python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py
python/sglang/srt/layers/moe/moe_runner/flashinfer_mxfp4.py
python/sglang/srt/layers/quantization/mxfp4.py
```

这条路径更接近当前 checkpoint，因为它处理的是：

```text
MXFP4 weight
E8M0 scale
SM90 / Hopper layout
```

但当前实现注释和逻辑描述的是：

```text
MXFP4 x BF16
W4A16
```

不是目标：

```text
MXFP4 x FP8
W4A8
```

结论：

```text
它不是最终方案，但非常适合作为 MXFP4 权重 layout、scale interleave、SM90 约束的参考。
```

### 6.4 FlashInfer CuteDSL FP4

相关文件：

```text
python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py
python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py
```

这条路径更偏 FP4/NVFP4 activation dispatch，不是当前需求的：

```text
DeepEP FP8 activation + MXFP4 expert weight + Hopper W4A8 GEMM
```

## 7. 最终需求的准确表述

最终需求应该表述为：

```text
在 MiMoV2 MXFP4 checkpoint 上开启 DeepEP，
并让 routed experts 的 GEMM 在 Hopper / SM90 上使用：

  MXFP4 expert weight 在线反量化
  FP8 e4m3 activation
  W4A8 expert GEMM
  BF16 output

checkpoint 仍保持 MXFP4 格式。
```

换句话说：

```text
不是要 w4afp8 checkpoint
而是要 MXFP4 checkpoint 的 W4A8 runtime expert GEMM
```

## 8. 推荐实现路线

推荐新增一个专门的 MoE runner backend，例如：

```text
mxfp4_w4a8
```

并新增 fused path：

```python
@register_fused_func("deepep", "mxfp4_w4a8")
```

优先支持的启动形态：

```bash
--tp-size 8 \
--moe-a2a-backend deepep \
--deepep-mode low_latency \
--deepep-dispatcher-output-dtype fp8 \
--moe-runner-backend mxfp4_w4a8
```

优先选择 `low_latency` 的原因：

- DeepEP low-latency dispatch 可以在 `use_fp8=True` 时返回量化后的 hidden states 和 scales；
- 更贴近“dispatch 后 FP8 activation 直接进入 expert GEMM”的目标；
- 可以避免 normal path 里额外再做一轮不必要的 activation quant。

## 9. 具体实现拆分

### 9.1 保持当前 checkpoint 加载逻辑

继续使用当前 MXFP4 checkpoint：

```text
w13_weight: packed MXFP4 / E2M1
w2_weight: packed MXFP4 / E2M1
w13_weight_scale: E8M0 per 32
w2_weight_scale: E8M0 per 32
```

不把 checkpoint 离线转换成 `w4afp8`。

不把 MXFP4 权重永久改写成现有 CUTLASS W4A8 kernel 使用的 INT4 格式。

### 9.2 新增 MoE runner backend

新增 enum/backend，例如：

```text
MoeRunnerBackend.MXFP4_W4A8
```

新增 quant info，例如：

```python
@dataclass
class Mxfp4W4A8QuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    ...
```

新增 fused function：

```python
@register_fused_func("deepep", "mxfp4_w4a8")
def fused_experts_deepep_to_mxfp4_w4a8(...):
    ...
```

输入：

```text
DeepEPLLDispatchOutput
```

输出：

```text
DeepEPLLCombineInput
```

### 9.3 新增或改造 Hopper kernel

kernel 的逻辑 contract：

```text
A:
  FP8 e4m3 activation from DeepEP

A scale:
  FP8 activation scale from DeepEP

B:
  packed MXFP4 / E2M1 expert weight

B scale:
  E8M0 block scale, block size 32

Output:
  BF16
```

expert 计算流程：

```text
GEMM1:
  FP8 activation x MXFP4 weight
  kernel 内在线 decode/dequant MXFP4 weight
  输出 BF16

SwiGLU:
  BF16 intermediate
  再量化成 FP8

GEMM2:
  FP8 intermediate x MXFP4 weight
  kernel 内在线 decode/dequant MXFP4 weight
  输出 BF16
```

### 9.4 Hopper 限制

第一阶段只支持：

```text
SM90 / Hopper
```

不要混入 SM100 / Blackwell DeepGEMM FP4 假设。

## 10. 分阶段计划

### Stage 1：最小功能跑通

目标：

```text
TP8 + DeepEP low_latency + mxfp4_w4a8 启动成功，并完成一次简单推理。
```

任务：

- 添加 `mxfp4_w4a8` runner backend；
- 添加启动参数 choices；
- 注册 `deepep -> mxfp4_w4a8` fused function；
- 构造 `Mxfp4W4A8QuantInfo`；
- 让 MiMoV2 MXFP4 experts 在显式指定 backend 时走这条路径；
- 实现第一版 Hopper kernel 或 wrapper；
- 验证 server 启动和一次 chat/completions 请求。

预期命令形态：

```bash
python3 -m sglang.launch_server \
  --model-path /preset-models \
  --tp-size 8 \
  --trust-remote-code \
  --quantization fp8 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend mxfp4_w4a8
```

### Stage 2：正确性验证

和当前能跑的 Marlin fallback 做对照：

```text
MXFP4 + Marlin
```

需要验证：

- 单 token decode；
- 小 batch；
- prefill；
- top-k routing；
- expert id mapping；
- scale layout；
- TP8 分片；
- DeepEP dispatch/combine 前后 token 顺序；
- 输出是否稳定，没有 NaN/Inf。

注意：Marlin 路径不是 FP8 activation，所以数值不一定 bitwise 对齐，但可以作为 sanity baseline。

### Stage 3：性能优化

优化方向：

- MXFP4 scale layout；
- FP8 activation scale 处理；
- 避免 DeepEP dispatch 后重复 quant；
- GEMM1/SwiGLU/GEMM2 fusion；
- DeepEP low-latency combine overlap；
- prefill/decode 不同 batch size 下的 kernel tuning。

### Stage 4：扩展能力

低优先级扩展：

- 支持 `DeepEP normal`；
- 支持更复杂的 EP/DP 组合；
- 根据性能决定是否保留 Marlin fallback 自动选择；
- 补测试和 benchmark。

## 11. 非目标

以下不是最终方案：

- 把 checkpoint 改成 `w4afp8` checkpoint；
- 继续使用 Marlin 作为最终 expert runner；
- 使用 BF16 activation 的 W4A16 路径，然后称为 W4A8；
- 依赖 Blackwell / SM100 的 DeepGEMM FP4 路径；
- 只改启动参数，不补缺失的 runtime/kernel 路径。

## 12. 当前状态总结

已经完成：

```text
MiMoV2 MXFP4 checkpoint 可以加载
原始 KeyError 已修复
Marlin fallback 可以启动并完成推理
相关修复已提交
```

仍缺失：

```text
DeepEP + MXFP4 在线反量化 + FP8 activation W4A8 expert GEMM
```

下一步推荐工程任务：

```text
新增 Hopper-only 的 mxfp4_w4a8 MoE runner，
实现 deepep -> mxfp4_w4a8 fused path，
并接入一个能直接消费 MXFP4/E2M1 weight + E8M0 scale + FP8 activation 的 expert GEMM kernel。
```

## 13. 2026-06-13 进展记录

已完成 Stage 1 prototype：

- 新增 `mxfp4_w4a8` MoE runner backend，并支持通过 `--moe-runner-backend mxfp4_w4a8` 显式选择；
- 对 MiMoV2 `fp8 + store_dtype=mxfp4` expert 权重接入 `Mxfp4W4A8MoEMethod`；
- DeepEP low_latency 路径注册 `deepep -> mxfp4_w4a8` fused runner；
- 当前实现直接消费 checkpoint 中的 MXFP4/E2M1 packed weight 和 E8M0 scale，不转换成 `w4afp8` checkpoint；
- 当前 prototype 输出 BF16，并要求 DeepEP dispatcher 输出 FP8 activation。

已通过的实机验证命令：

```bash
python3 -m sglang.launch_server \
  --model-path /preset-models \
  --served-model-name mimo-v2-flash \
  --host 127.0.0.1 \
  --port 31082 \
  --tp-size 8 \
  --trust-remote-code \
  --quantization fp8 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend mxfp4_w4a8 \
  --moe-dense-tp-size 1 \
  --mem-fraction-static 0.80 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

验证结果：

- server 成功加载模型；
- 日志确认 MoE 层使用 `DeepEP MXFP4 W4A8 prototype runner`；
- `/generate` 返回 HTTP 200；
- 返回 tensor 路径形状、dtype、finite 状态符合预期；
- 初次未设置 `--mem-fraction-static 0.80` 时 DeepEP/NVSHMEM 余量不足，设置后通过。

当前限制：

- 只支持 Hopper/SM90；
- 只支持 DeepEP low_latency；
- 只支持 `--deepep-dispatcher-output-dtype fp8`；
- 只支持 `silu` MoE activation；
- 当前 expert 执行仍是 Python/Torch reference path，不是最终 Hopper kernel；
- 当前 decode 路径不是 CUDA graph safe，不能打开 decode CUDA graph。

下一阶段目标：

```text
实现 graph-safe 的 Hopper/SM90 MXFP4/W4A8 decode expert kernel：
- device 侧使用 masked_m，不能有 masked_m.item()；
- 固定 shape / 固定 workspace，避免 forward 内动态分配；
- kernel 内在线 decode MXFP4/E2M1 weight + E8M0 scale；
- 直接消费 DeepEP FP8 activation；
- 输出 BF16；
- 支持 decode CUDA graph capture/replay。
```
