from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
)


@dataclass
class Mxfp4W4A8QuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    swiglu_limit: Optional[float] = None


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is not None:
            return scale.contiguous().view(e8m0_dtype).to(torch.float32)
        return torch.exp2(scale.to(torch.float32) - 127)
    if scale.dtype == torch.int8:
        return _scale_to_float(scale.contiguous().view(torch.uint8))
    return scale.to(torch.float32)


def _dequant_mxfp4_matrix(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    packed = weight_packed.contiguous().view(torch.uint8)
    unpacked = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = (packed >> 4) & 0x0F

    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    magnitude = values[(unpacked & 0x07).to(torch.long)]
    sign = torch.where((unpacked & 0x08) != 0, -1.0, 1.0)
    dequant = magnitude * sign

    scale = _scale_to_float(weight_scale).repeat_interleave(32, dim=-1)
    return (dequant * scale).to(out_dtype)


def _dequant_deepep_activation(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if hidden_states.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return hidden_states.to(out_dtype)

    output = hidden_states.to(torch.float32)
    if hidden_states_scale is not None:
        scale = _scale_to_float(hidden_states_scale)
        if scale.ndim == output.ndim and scale.shape[-1] != output.shape[-1]:
            repeat = (output.shape[-1] + scale.shape[-1] - 1) // scale.shape[-1]
            scale = scale.repeat_interleave(repeat, dim=-1)[..., : output.shape[-1]]
        elif scale.ndim == output.ndim - 1:
            scale = scale.unsqueeze(-1)
        output = output * scale
    return output.to(out_dtype)


def _quant_dequant_fp8_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    scale = (x.abs().amax().to(torch.float32) / fp8_max).clamp(min=1e-12)
    x_fp8 = (
        (x.to(torch.float32) / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    )
    return (x_fp8.to(torch.float32) * scale).to(x.dtype)


def _apply_swiglu(gate_up: torch.Tensor, swiglu_limit: Optional[float]) -> torch.Tensor:
    inner = gate_up.shape[-1] // 2
    gate = gate_up[..., :inner]
    up = gate_up[..., inner:]
    if swiglu_limit is not None and swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


@torch.no_grad()
def _mxfp4_w4a8_deepep_ll_reference(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    masked_m: torch.Tensor,
    quant_info: Mxfp4W4A8QuantInfo,
) -> torch.Tensor:
    compute_dtype = torch.bfloat16
    output = torch.zeros_like(hidden_states, dtype=compute_dtype)
    hidden_states_bf16 = _dequant_deepep_activation(
        hidden_states, hidden_states_scale, compute_dtype
    )

    num_experts = hidden_states.shape[0]
    for expert_id in range(num_experts):
        num_tokens = int(masked_m[expert_id].item())
        if num_tokens == 0:
            continue

        expert_input = hidden_states_bf16[expert_id, :num_tokens, :]
        w13 = _dequant_mxfp4_matrix(
            quant_info.w13_weight[expert_id],
            quant_info.w13_weight_scale[expert_id],
            compute_dtype,
        )
        gate_up = torch.matmul(expert_input, w13.transpose(0, 1))
        intermediate = _apply_swiglu(gate_up, quant_info.swiglu_limit)
        intermediate = _quant_dequant_fp8_tensor(intermediate)

        w2 = _dequant_mxfp4_matrix(
            quant_info.w2_weight[expert_id],
            quant_info.w2_weight_scale[expert_id],
            compute_dtype,
        )
        output[expert_id, :num_tokens, :] = torch.matmul(
            intermediate, w2.transpose(0, 1)
        )

    return output


@register_fused_func("deepep", "mxfp4_w4a8")
def fused_experts_deepep_to_mxfp4_w4a8(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> DeepEPLLCombineInput:
    if not DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
        raise NotImplementedError(
            "mxfp4_w4a8 currently supports DeepEP low_latency dispatch only. "
            "Start the server with --deepep-mode low_latency."
        )
    if runner_config.activation != "silu":
        raise NotImplementedError("mxfp4_w4a8 currently supports silu MoE only.")
    if runner_config.apply_router_weight_on_input:
        raise NotImplementedError(
            "mxfp4_w4a8 does not support apply_router_weight_on_input."
        )
    if not isinstance(quant_info, Mxfp4W4A8QuantInfo):
        raise TypeError(f"Unexpected quant_info type: {type(quant_info)}")

    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, _ = (
        dispatch_output
    )
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "mxfp4_w4a8 requires DeepEP FP8 dispatch output. "
            "Start the server with --deepep-dispatcher-output-dtype fp8."
        )
    output = _mxfp4_w4a8_deepep_ll_reference(
        hidden_states,
        hidden_states_scale,
        masked_m,
        quant_info,
    )
    return DeepEPLLCombineInput(
        hidden_states=output,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
