from __future__ import annotations

import gc
import os
import sys
import logging
from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd

logger = logging.getLogger(__name__)

_USE_DOT_SCALED = os.environ.get("SGLANG_MXFP4_W4A8_DOT_SCALED", "1") != "0"
_USE_E8M0_WEIGHT_SCALE_LL = os.environ.get("SGLANG_MXFP4_W4A8_E8M0_LL", "0") != "0"
_DOT_SCALED_K = 32
_DOT_SCALED_CONTIG_MAXNREG = int(
    os.environ.get("SGLANG_MXFP4_W4A8_CONTIG_MAXNREG", "168")
)
_USE_DOT_SCALED_CONTIG_ALIGNED_NK = (
    os.environ.get("SGLANG_MXFP4_W4A8_CONTIG_ALIGNED_NK", "1") != "0"
)
_USE_HUMMING_NORMAL = (
    os.environ.get("SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL", "0") != "0"
)
_HUMMING_REPLACE_WEIGHTS = (
    os.environ.get(
        "SGLANG_MXFP4_W4A8_HUMMING_REPLACE_WEIGHTS",
        "1" if _USE_HUMMING_NORMAL else "0",
    )
    != "0"
)
_USE_HUMMING_NORMAL = _USE_HUMMING_NORMAL or _HUMMING_REPLACE_WEIGHTS
_ALLOW_HUMMING_NORMAL_FALLBACK = (
    os.environ.get("SGLANG_MXFP4_W4A8_HUMMING_ALLOW_FALLBACK", "0") != "0"
)
_USE_HUMMING_PREFILL_TUNING = (
    os.environ.get("SGLANG_MXFP4_W4A8_HUMMING_PREFILL_TUNING", "1") != "0"
)
_HUMMING_NORMAL_CACHE = {}
_HUMMING_EXPERT_LAYOUT_CACHE = {}
_HUMMING_NORMAL_SKIP_KEYS = set()
_HUMMING_NORMAL_WARNED_REASONS = set()


@dataclass
class HummingMxfp4W4A8Weight:
    layer: Any
    n: int
    k: int
    num_experts: int
    contig_compute_config: dict[str, Any]
    contig_tuning_config: Any
    masked_compute_config: dict[str, Any]
    masked_tuning_config: Any


def should_replace_humming_normal_weights() -> bool:
    return _USE_HUMMING_NORMAL and _HUMMING_REPLACE_WEIGHTS


def _log_humming_warning_once(reason: str, message: str) -> None:
    if reason in _HUMMING_NORMAL_WARNED_REASONS:
        return
    _HUMMING_NORMAL_WARNED_REASONS.add(reason)
    logger.warning(message)


def _handle_humming_unavailable(cache_key, reason: str, message: str) -> bool:
    if not _ALLOW_HUMMING_NORMAL_FALLBACK:
        raise RuntimeError(message)
    _HUMMING_NORMAL_SKIP_KEYS.add(cache_key)
    _log_humming_warning_once(reason, message + " Falling back to Triton.")
    return False


def _humming_entry_cache_key(
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    n: int,
    k: int,
):
    return (
        b_packed.data_ptr(),
        b_scale.data_ptr(),
        tuple(b_packed.shape),
        tuple(b_scale.shape),
        str(b_packed.dtype),
        str(b_scale.dtype),
        b_packed.device.index,
        n,
        k,
    )


def _can_allocate_humming_normal_entry(
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    cache_key,
) -> bool:
    if cache_key in _HUMMING_NORMAL_SKIP_KEYS:
        return False
    if not b_packed.is_contiguous() or not b_scale.is_contiguous():
        return _handle_humming_unavailable(
            cache_key,
            "non_contiguous_weight",
            "Skip Humming MXFP4 W4A8 normal path because weight tensors are not "
            "contiguous.",
        )

    free_bytes, _ = torch.cuda.mem_get_info(b_packed.device)
    # Humming keeps an offline-repacked weight copy and an E8M0 scale copy.
    # During transform there is another transient scale layout copy, so keep
    # a conservative margin to avoid crashing a nearly full serving process.
    # (b_scale is E8M0/uint8 at 1 byte/elem; *4 stays as a safe upper bound.)
    required_bytes = b_packed.nbytes + b_scale.numel() * 4 + 256 * 1024 * 1024
    if free_bytes < required_bytes:
        return _handle_humming_unavailable(
            cache_key,
            "insufficient_free_memory",
            "Skip Humming MXFP4 W4A8 normal path because free GPU memory is "
            "too small for the repacked weight cache "
            f"({free_bytes / 1024**2:.1f} MiB free, "
            f"need about {required_bytes / 1024**2:.1f} MiB).",
        )
    return True


@triton.jit
def _decode_e2m1(nibble: tl.tensor) -> tl.tensor:
    sign_bit = (nibble >> 3) & 1
    exp_bits = (nibble >> 1) & 3
    man_bit = nibble & 1

    is_subnormal = exp_bits == 0
    mantissa = 1.0 + man_bit.to(tl.float32) * 0.5
    exponent = tl.exp2((exp_bits - 1).to(tl.float32))
    value = tl.where(is_subnormal, man_bit.to(tl.float32) * 0.5, mantissa * exponent)
    return tl.where(sign_bit != 0, -value, value)


@triton.jit
def _mxfp4_w4a8_grouped_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_packed_ptr,
    b_scale_ptr,
    c_ptr,
    masked_m_ptr,
    stride_ae: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_ase: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_asg: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk2: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_count = tl.load(masked_m_ptr + expert_id)
    if token_count <= m_block * BLOCK_M:
        if M <= 8:
            tl.store(
                c_ptr
                + expert_id * stride_ce
                + offs_m[:, None] * stride_cm
                + offs_n[None, :] * stride_cn,
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).to(tl.bfloat16),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
        return

    valid_m = offs_m < token_count
    valid_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2
        offs_k_odd = offs_k_even + 1

        a_even = tl.load(
            a_ptr
            + expert_id * stride_ae
            + offs_m[:, None] * stride_am
            + offs_k_even[None, :] * stride_ak,
            mask=valid_m[:, None] & (offs_k_even[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        a_odd = tl.load(
            a_ptr
            + expert_id * stride_ae
            + offs_m[:, None] * stride_am
            + offs_k_odd[None, :] * stride_ak,
            mask=valid_m[:, None] & (offs_k_odd[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        a_scale_even = tl.load(
            a_scale_ptr
            + expert_id * stride_ase
            + offs_m[:, None] * stride_asm
            + (offs_k_even[None, :] // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m[:, None] & (offs_k_even[None, :] < K),
            other=1.0,
        ).to(tl.float32)
        a_scale_odd = tl.load(
            a_scale_ptr
            + expert_id * stride_ase
            + offs_m[:, None] * stride_asm
            + (offs_k_odd[None, :] // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m[:, None] & (offs_k_odd[None, :] < K),
            other=1.0,
        ).to(tl.float32)
        a_even *= a_scale_even
        a_odd *= a_scale_odd

        b_packed = tl.load(
            b_packed_ptr
            + expert_id * stride_be
            + offs_n[:, None] * stride_bn
            + offs_k2[None, :] * stride_bk2,
            mask=valid_n[:, None] & (offs_k2[None, :] < K // 2),
            other=0,
        ).to(tl.int32)
        b_scale = tl.load(
            b_scale_ptr
            + expert_id * stride_bse
            + offs_n[:, None] * stride_bsn
            + ((k_start // 32) + tl.arange(0, BLOCK_K // 2)[None, :] // 16)
            * stride_bsg,
            mask=valid_n[:, None]
            & (((k_start // 32) + tl.arange(0, BLOCK_K // 2)[None, :] // 16) < K // 32),
            other=1.0,
        ).to(tl.float32)

        b_even = _decode_e2m1(b_packed & 0x0F) * b_scale
        b_odd = _decode_e2m1((b_packed >> 4) & 0x0F) * b_scale

        acc += tl.dot(a_even, tl.trans(b_even))
        acc += tl.dot(a_odd, tl.trans(b_odd))

    c = acc.to(tl.bfloat16)
    tl.store(
        c_ptr
        + expert_id * stride_ce
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn,
        tl.where(valid_m[:, None] & valid_n[None, :], c, 0.0),
        mask=(offs_m[:, None] < M) & valid_n[None, :],
    )


@triton.jit
def _mxfp4_w4a8_grouped_gemm_dot_scaled_kernel(
    a_ptr,
    a_scale_ptr,
    b_packed_ptr,
    b_scale_ptr,
    c_ptr,
    masked_m_ptr,
    stride_ae: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_ase: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_asg: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk2: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_count = tl.load(masked_m_ptr + expert_id)
    if token_count <= m_block * BLOCK_M:
        if M <= 8:
            tl.store(
                c_ptr
                + expert_id * stride_ce
                + offs_m[:, None] * stride_cm
                + offs_n[None, :] * stride_cn,
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).to(tl.bfloat16),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
        return

    valid_m = offs_m < token_count
    valid_n = offs_n < N
    offs_k = tl.arange(0, DOT_K)
    offs_k2 = tl.arange(0, DOT_K // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, DOT_K):
        a_raw = tl.load(
            a_ptr
            + expert_id * stride_ae
            + offs_m[:, None] * stride_am
            + (k_start + offs_k[None, :]) * stride_ak,
            mask=valid_m[:, None] & ((k_start + offs_k[None, :]) < K),
            other=0.0,
        )
        b_raw = tl.load(
            b_packed_ptr
            + expert_id * stride_be
            + offs_n[None, :] * stride_bn
            + (k_start // 2 + offs_k2[:, None]) * stride_bk2,
            mask=valid_n[None, :] & ((k_start // 2 + offs_k2[:, None]) < K // 2),
            other=0,
        ).to(tl.uint8)

        raw_acc = tl.dot_scaled(a_raw, None, "e4m3", b_raw, None, "e2m1")
        a_scale = tl.load(
            a_scale_ptr
            + expert_id * stride_ase
            + offs_m * stride_asm
            + (k_start // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m,
            other=1.0,
        ).to(tl.float32)
        b_scale = tl.load(
            b_scale_ptr
            + expert_id * stride_bse
            + offs_n * stride_bsn
            + (k_start // DOT_K) * stride_bsg,
            mask=valid_n,
            other=1.0,
        ).to(tl.float32)
        acc += raw_acc * a_scale[:, None] * b_scale[None, :]

    c = acc.to(tl.bfloat16)
    tl.store(
        c_ptr
        + expert_id * stride_ce
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn,
        tl.where(valid_m[:, None] & valid_n[None, :], c, 0.0),
        mask=(offs_m[:, None] < M) & valid_n[None, :],
    )


@triton.jit
def _mxfp4_w4a8_grouped_gemm_dot_scaled_e8m0_kernel(
    a_ptr,
    a_scale_ptr,
    b_packed_ptr,
    b_scale_e8m0_ptr,
    c_ptr,
    masked_m_ptr,
    stride_ae: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_ase: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_asg: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk2: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_count = tl.load(masked_m_ptr + expert_id)
    if token_count <= m_block * BLOCK_M:
        if M <= 8:
            tl.store(
                c_ptr
                + expert_id * stride_ce
                + offs_m[:, None] * stride_cm
                + offs_n[None, :] * stride_cn,
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).to(tl.bfloat16),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
        return

    valid_m = offs_m < token_count
    valid_n = offs_n < N
    offs_k = tl.arange(0, DOT_K)
    offs_k2 = tl.arange(0, DOT_K // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, DOT_K):
        a_raw = tl.load(
            a_ptr
            + expert_id * stride_ae
            + offs_m[:, None] * stride_am
            + (k_start + offs_k[None, :]) * stride_ak,
            mask=valid_m[:, None] & ((k_start + offs_k[None, :]) < K),
            other=0.0,
        )
        b_raw = tl.load(
            b_packed_ptr
            + expert_id * stride_be
            + offs_n[None, :] * stride_bn
            + (k_start // 2 + offs_k2[:, None]) * stride_bk2,
            mask=valid_n[None, :] & ((k_start // 2 + offs_k2[:, None]) < K // 2),
            other=0,
        ).to(tl.uint8)
        b_scale = tl.load(
            b_scale_e8m0_ptr
            + expert_id * stride_bse
            + offs_n[:, None] * stride_bsn
            + (k_start // DOT_K) * stride_bsg,
            mask=valid_n[:, None],
            other=127,
        ).to(tl.uint8)

        raw_acc = tl.dot_scaled(a_raw, None, "e4m3", b_raw, b_scale, "e2m1")
        a_scale = tl.load(
            a_scale_ptr
            + expert_id * stride_ase
            + offs_m * stride_asm
            + (k_start // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m,
            other=1.0,
        ).to(tl.float32)
        acc += raw_acc * a_scale[:, None]

    c = acc.to(tl.bfloat16)
    tl.store(
        c_ptr
        + expert_id * stride_ce
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn,
        tl.where(valid_m[:, None] & valid_n[None, :], c, 0.0),
        mask=(offs_m[:, None] < M) & valid_n[None, :],
    )


def _check_inputs(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    masked_m: torch.Tensor,
) -> None:
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected FP8 hidden_states, got {hidden_states.dtype}")
    if hidden_states_scale.dtype != torch.float32:
        raise TypeError(
            f"expected float32 hidden_states_scale, got {hidden_states_scale.dtype}"
        )
    if (
        w13_weight_scale.dtype != torch.float32
        or w2_weight_scale.dtype != torch.float32
    ):
        raise TypeError(
            "mxfp4_w4a8 Triton path expects float32 MXFP4 weight scales. "
            f"Got {w13_weight_scale.dtype=} and {w2_weight_scale.dtype=}."
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("hidden_states_scale", hidden_states_scale),
        ("w13_weight", w13_weight),
        ("w2_weight", w2_weight),
        ("w13_weight_scale", w13_weight_scale),
        ("w2_weight_scale", w2_weight_scale),
        ("masked_m", masked_m),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    for name, tensor in (("w13_weight", w13_weight), ("w2_weight", w2_weight)):
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must have stride(-1) == 1 for mxfp4_w4a8 Triton path"
            )


def _launch_grouped_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    n: int,
    k: int,
    num_routed_tokens: int | None = None,
    b_scale_e8m0: torch.Tensor | None = None,
) -> torch.Tensor:
    e, m, _ = a.shape
    if a_scale.shape[:2] != (e, m):
        raise ValueError(
            f"activation scale must start with {(e, m)}, got {tuple(a_scale.shape)}"
        )
    output = torch.empty((e, m, n), device=a.device, dtype=torch.bfloat16)
    a_scale_group_size = k // a_scale.shape[-1]
    if a_scale_group_size <= 0 or k % a_scale_group_size != 0:
        raise ValueError(
            f"invalid activation scale layout: {tuple(a.shape)=}, "
            f"{tuple(a_scale.shape)=}"
        )

    grid_m = m if num_routed_tokens is None else min(m, num_routed_tokens)
    if grid_m <= 0:
        return output

    if m <= 8:
        block_n = 128
        block_k = 128 if k >= n else 64
        if m <= 1:
            block_m = 2 if k >= n else 4
        elif m <= 4:
            block_m = 4
        else:
            block_m = 8
    elif num_routed_tokens is not None and num_routed_tokens <= 32 and m <= 128:
        block_m = 8
        block_n = 128
        block_k = 128 if k >= n else 64
    else:
        block_m = 16
        block_n = 64
        block_k = 64

    grid = (
        e,
        triton.cdiv(grid_m, block_m),
        triton.cdiv(n, block_n),
    )
    launch_args = (
        a,
        a_scale,
        b_packed.view(torch.uint8),
        b_scale,
        output,
        masked_m,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        m,
        n,
        k,
        a_scale_group_size,
    )
    if _USE_DOT_SCALED and _USE_E8M0_WEIGHT_SCALE_LL and b_scale_e8m0 is not None:
        _mxfp4_w4a8_grouped_gemm_dot_scaled_e8m0_kernel[grid](
            a,
            a_scale,
            b_packed.view(torch.uint8),
            b_scale_e8m0,
            output,
            masked_m,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a_scale.stride(0),
            a_scale.stride(1),
            a_scale.stride(2),
            b_packed.stride(0),
            b_packed.stride(1),
            b_packed.stride(2),
            b_scale_e8m0.stride(0),
            b_scale_e8m0.stride(1),
            b_scale_e8m0.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            m,
            n,
            k,
            a_scale_group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            DOT_K=_DOT_SCALED_K,
            num_warps=4,
            num_stages=3,
        )
    elif _USE_DOT_SCALED:
        _mxfp4_w4a8_grouped_gemm_dot_scaled_kernel[grid](
            *launch_args,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            DOT_K=_DOT_SCALED_K,
            num_warps=4,
            num_stages=3,
        )
    else:
        _mxfp4_w4a8_grouped_gemm_kernel[grid](
            *launch_args,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
    return output


@triton.jit
def _mxfp4_w4a8_grouped_gemm_contig_kernel(
    a_ptr,
    a_scale_ptr,
    b_packed_ptr,
    b_scale_ptr,
    c_ptr,
    expert_start_ptr,
    num_tokens_per_expert_ptr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_asg: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk2: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    expert_start = tl.load(expert_start_ptr + expert_id).to(tl.int64)
    token_count = tl.load(num_tokens_per_expert_ptr + expert_id)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    if token_count <= m_block * BLOCK_M:
        return

    global_m = expert_start + offs_m
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_m = offs_m < token_count
    valid_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2
        offs_k_odd = offs_k_even + 1

        a_even = tl.load(
            a_ptr + global_m[:, None] * stride_am + offs_k_even[None, :] * stride_ak,
            mask=valid_m[:, None] & (offs_k_even[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        a_odd = tl.load(
            a_ptr + global_m[:, None] * stride_am + offs_k_odd[None, :] * stride_ak,
            mask=valid_m[:, None] & (offs_k_odd[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        a_scale_even = tl.load(
            a_scale_ptr
            + global_m[:, None] * stride_asm
            + (offs_k_even[None, :] // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m[:, None] & (offs_k_even[None, :] < K),
            other=1.0,
        ).to(tl.float32)
        a_scale_odd = tl.load(
            a_scale_ptr
            + global_m[:, None] * stride_asm
            + (offs_k_odd[None, :] // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m[:, None] & (offs_k_odd[None, :] < K),
            other=1.0,
        ).to(tl.float32)
        a_even *= a_scale_even
        a_odd *= a_scale_odd

        b_packed = tl.load(
            b_packed_ptr
            + expert_id * stride_be
            + offs_n[:, None] * stride_bn
            + offs_k2[None, :] * stride_bk2,
            mask=valid_n[:, None] & (offs_k2[None, :] < K // 2),
            other=0,
        ).to(tl.int32)
        b_scale = tl.load(
            b_scale_ptr
            + expert_id * stride_bse
            + offs_n[:, None] * stride_bsn
            + ((k_start // 32) + tl.arange(0, BLOCK_K // 2)[None, :] // 16)
            * stride_bsg,
            mask=valid_n[:, None]
            & (((k_start // 32) + tl.arange(0, BLOCK_K // 2)[None, :] // 16) < K // 32),
            other=1.0,
        ).to(tl.float32)

        b_even = _decode_e2m1(b_packed & 0x0F) * b_scale
        b_odd = _decode_e2m1((b_packed >> 4) & 0x0F) * b_scale

        acc += tl.dot(a_even, tl.trans(b_even))
        acc += tl.dot(a_odd, tl.trans(b_odd))

    tl.store(
        c_ptr + global_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        tl.where(valid_m[:, None] & valid_n[None, :], acc.to(tl.bfloat16), 0.0),
        mask=valid_m[:, None] & valid_n[None, :],
    )


@triton.jit
def _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel(
    a_ptr,
    a_scale_ptr,
    b_packed_ptr,
    b_scale_ptr,
    c_ptr,
    expert_start_ptr,
    num_tokens_per_expert_ptr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_asg: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk2: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    expert_start = tl.load(expert_start_ptr + expert_id).to(tl.int64)
    token_count = tl.load(num_tokens_per_expert_ptr + expert_id)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    if token_count <= m_block * BLOCK_M:
        return

    global_m = expert_start + offs_m
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_m = offs_m < token_count
    valid_n = offs_n < N
    offs_k = tl.arange(0, DOT_K)
    offs_k2 = tl.arange(0, DOT_K // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, DOT_K):
        a_raw = tl.load(
            a_ptr
            + global_m[:, None] * stride_am
            + (k_start + offs_k[None, :]) * stride_ak,
            mask=valid_m[:, None] & ((k_start + offs_k[None, :]) < K),
            other=0.0,
        )
        b_raw = tl.load(
            b_packed_ptr
            + expert_id * stride_be
            + offs_n[None, :] * stride_bn
            + (k_start // 2 + offs_k2[:, None]) * stride_bk2,
            mask=valid_n[None, :] & ((k_start // 2 + offs_k2[:, None]) < K // 2),
            other=0,
        ).to(tl.uint8)

        raw_acc = tl.dot_scaled(a_raw, None, "e4m3", b_raw, None, "e2m1")
        a_scale = tl.load(
            a_scale_ptr
            + global_m * stride_asm
            + (k_start // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m,
            other=1.0,
        ).to(tl.float32)
        b_scale = tl.load(
            b_scale_ptr
            + expert_id * stride_bse
            + offs_n * stride_bsn
            + (k_start // DOT_K) * stride_bsg,
            mask=valid_n,
            other=1.0,
        ).to(tl.float32)
        acc += raw_acc * a_scale[:, None] * b_scale[None, :]

    tl.store(
        c_ptr + global_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        tl.where(valid_m[:, None] & valid_n[None, :], acc.to(tl.bfloat16), 0.0),
        mask=valid_m[:, None] & valid_n[None, :],
    )


@triton.jit
def _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_aligned_nk_kernel(
    a_ptr,
    a_scale_ptr,
    b_packed_ptr,
    b_scale_ptr,
    c_ptr,
    expert_start_ptr,
    num_tokens_per_expert_ptr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_asg: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk2: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    expert_start = tl.load(expert_start_ptr + expert_id).to(tl.int64)
    token_count = tl.load(num_tokens_per_expert_ptr + expert_id)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    if token_count <= m_block * BLOCK_M:
        return

    global_m = expert_start + offs_m
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_m = offs_m < token_count
    offs_k = tl.arange(0, DOT_K)
    offs_k2 = tl.arange(0, DOT_K // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, DOT_K):
        a_raw = tl.load(
            a_ptr
            + global_m[:, None] * stride_am
            + (k_start + offs_k[None, :]) * stride_ak,
            mask=valid_m[:, None],
            other=0.0,
        )
        b_raw = tl.load(
            b_packed_ptr
            + expert_id * stride_be
            + offs_n[None, :] * stride_bn
            + (k_start // 2 + offs_k2[:, None]) * stride_bk2,
        ).to(tl.uint8)

        raw_acc = tl.dot_scaled(a_raw, None, "e4m3", b_raw, None, "e2m1")
        a_scale = tl.load(
            a_scale_ptr
            + global_m * stride_asm
            + (k_start // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m,
            other=1.0,
        ).to(tl.float32)
        b_scale = tl.load(
            b_scale_ptr
            + expert_id * stride_bse
            + offs_n * stride_bsn
            + (k_start // DOT_K) * stride_bsg,
        ).to(tl.float32)
        acc += raw_acc * a_scale[:, None] * b_scale[None, :]

    tl.store(
        c_ptr + global_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=valid_m[:, None],
    )


_HUMMING_FUSED_E8M0_PATCHED = False


def _patch_humming_disable_fused_e8m0() -> None:
    """Make Humming skip its automatic fp8xfp4xe8m0 -> fused-E8M0 promotion.

    Humming's ``HummingLayerMeta.__post_init__`` auto-selects the *fused*-E8M0
    kernel whenever (a_dtype fp8/int8, fp4 weight, e8m0 weight scale, grouped
    weight scale). That fused path does NOT support per-token-group input scale
    (only a single per-tensor global scale), so with our DeepEP per-token-group
    -128 FP8 activations it silently miscomputes (garbage tokens). The non-fused
    group-scale mainloop natively supports per-group input scale while still
    consuming the 1-byte E8M0 weight scale.

    To keep this entirely on the SGLang side (no Humming source edits) we wrap
    ``HummingLayerMeta.__post_init__``: when SGLANG_HUMMING_DISABLE_FUSED_E8M0 is
    enabled (default), we pre-seed ``use_fused_e8m0_scale = False`` and let the
    original run; the original's auto-promotion is guarded by
    ``if not self.use_fused_e8m0_scale: self.use_fused_e8m0_scale = (...)``, so we
    cannot simply pre-set it. Instead we run the original, then -- if it promoted
    a meta that we want non-fused -- rebuild a fresh meta with the promotion
    suppressed. The rebuild path is exercised only for the exact dtype combo, so
    every other Humming layer is byte-for-byte unaffected.

    Idempotent and gated on SGLANG_HUMMING_DISABLE_FUSED_E8M0 (default "1").
    """
    global _HUMMING_FUSED_E8M0_PATCHED
    if _HUMMING_FUSED_E8M0_PATCHED:
        return

    import dataclasses

    from humming import dtypes as _hd
    from humming.config import WeightScaleType
    from humming.layer import HummingLayerMeta

    _orig_post_init = HummingLayerMeta.__post_init__
    _sentinel = "_sglang_force_nonfused_e8m0"

    def _patched_post_init(self):
        disable = os.environ.get("SGLANG_HUMMING_DISABLE_FUSED_E8M0", "1") != "0"
        # Re-entry guard: when we re-run __post_init__ on a clone (see below) we
        # must not recurse; the clone carries the sentinel so the original runs
        # verbatim with fused promotion already disabled.
        forced = getattr(self, _sentinel, False)
        _orig_post_init(self)
        if not disable or forced:
            return
        # Only intervene for the exact dtype combo Humming auto-promotes to fused.
        would_promote = (
            self.use_fused_e8m0_scale
            and self.a_dtype in [_hd.float8e4m3, _hd.int8]
            and self.weight_scale_group_size > 0
            and self.b_dtype in [_hd.float4e2m1]
            and self.bs_dtype in [_hd.float8e8m0]
        )
        if not would_promote:
            return
        # Undo the fused promotion in place (the instance is frozen after
        # __post_init__ sets _meta_str, so bypass __setattr__ and refresh the
        # cached fields the promotion touched).
        object.__setattr__(self, "use_fused_e8m0_scale", False)
        object.__setattr__(self, "weight_scale_type", WeightScaleType.GROUP)
        object.__setattr__(self, "is_tensor_weight_scale", False)
        object.__delattr__(self, "_meta_str")
        object.__setattr__(self, _sentinel, True)
        object.__setattr__(self, "_meta_str", self.to_str())

    HummingLayerMeta.__post_init__ = _patched_post_init
    _HUMMING_FUSED_E8M0_PATCHED = True


def _import_humming():
    humming_root = os.environ.get("SGLANG_HUMMING_ROOT")
    if humming_root and humming_root not in sys.path:
        sys.path.insert(0, humming_root)

    try:
        from humming.config import GemmType
        from humming.layer import HummingLayer
        from humming.tune import get_heuristics_config
    except ImportError as exc:
        raise RuntimeError(
            "SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL=1 requires Humming to be "
            "importable. Set PYTHONPATH or SGLANG_HUMMING_ROOT to the Humming repo."
        ) from exc
    _patch_humming_disable_fused_e8m0()
    return GemmType, HummingLayer, get_heuristics_config


def _humming_prefill_tuning_overrides(
    n: int,
    k: int,
) -> list[tuple[int, int, dict[str, Any]]]:
    if not _USE_HUMMING_PREFILL_TUNING:
        return []

    tuned_warp_n32_bm48 = {
        "block_shape": (48, 128, 128),
        "warp_shape": (48, 32, 128),
        "use_stream_k": True,
        "use_f16_accum": False,
        "num_sms": 132,
        "num_stages": 3,
        "num_ctas_per_sm": 2,
    }
    if n == 4096 and k == 6144:
        return [(4096, 1 << 30, tuned_warp_n32_bm48)]
    if n == 6144 and k == 2048:
        return [(4096, 1 << 30, tuned_warp_n32_bm48)]
    return []


# Minimum warp-N (warp_shape[1]) for the non-fused E8M0 MoE GEMM path.
#
# The non-fused group-weight-scale WGMMA mainloop indexes the weight-scale
# fragment per warp tile (mainloop_arith.cuh: dq_bs[n*MmaShape::N/8 + inner_n]).
# With an 8-bit (E8M0) weight scale and warp_shape[1] == 16 that index layout is
# wrong and the kernel writes all-zero output (silent garbage, not a crash).
# warp-N >= 32 computes correctly. The bf16-scale path tolerates warp-N == 16,
# which is why this only bit us after switching the weight scale to E8M0.
#
# Humming's H20 heuristic (humming/tune/sm90_h20.py) emits warp-N == 16 for the
# group_size>=128, shape_k>512 case (i.e. our per-token-group-128 activations),
# so every heuristic config for these MoE shapes must be bumped to warp-N 32.
_HUMMING_E8M0_MIN_WARP_N = 32

# Minimum num_ctas_per_sm for the non-fused E8M0 MoE GEMM path.
#
# Separate from the warp-N correctness fix above: this is a *performance* fix.
# Humming's grouped heuristic gives the smallest-M tile (block_m == 8, the
# m[0,4) region) num_ctas_per_sm == 1. On the E8M0 path that tile runs ~1.6x
# slower than with num_ctas_per_sm == 2 (0.040ms vs 0.024ms at M=64, measured),
# because a single CTA/SM cannot hide the per-tile launch + scale-load overhead.
# Bumping to 2 matches every other tile (which the heuristic already gives 2-3)
# and costs nothing on those. SMEM stays within the 227KB budget at the block
# shapes the heuristic emits for these MoE layers (verified: launch succeeds).
_HUMMING_E8M0_MIN_CTAS_PER_SM = 2


def _sanitize_humming_e8m0_warp_n(tuning_config: Any) -> Any:
    """Sanitize Humming heuristic configs for the non-fused E8M0 MoE GEMM path.

    Two fixes, both required for this path:
      * warp_shape[1] (warp-N) >= 32  -- correctness (warp-N 16 -> all-zero output)
      * num_ctas_per_sm >= 2          -- performance (small-M tile otherwise ~1.6x slow)

    Returns a new object; does not mutate the input. Handles both the dict
    (single config) and list ([m_start, m_end, cfg], ...) shapes that
    ``get_heuristics_config`` can return.
    """

    def fix_cfg(cfg: dict) -> dict:
        warp_shape = cfg.get("warp_shape")
        if not warp_shape or len(warp_shape) != 3:
            return cfg
        warp_m, warp_n, warp_k = warp_shape
        ctas = cfg.get("num_ctas_per_sm", 1)
        if warp_n >= _HUMMING_E8M0_MIN_WARP_N and ctas >= _HUMMING_E8M0_MIN_CTAS_PER_SM:
            return cfg
        new_cfg = dict(cfg)
        new_cfg["warp_shape"] = (warp_m, max(warp_n, _HUMMING_E8M0_MIN_WARP_N), warp_k)
        new_cfg["num_ctas_per_sm"] = max(ctas, _HUMMING_E8M0_MIN_CTAS_PER_SM)
        return new_cfg

    if isinstance(tuning_config, dict):
        return fix_cfg(tuning_config)
    if isinstance(tuning_config, list):
        sanitized: list[Any] = []
        for entry in tuning_config:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) == 3
                and isinstance(entry[2], dict)
            ):
                sanitized.append([entry[0], entry[1], fix_cfg(entry[2])])
            else:
                sanitized.append(entry)
        return sanitized
    return tuning_config


def _apply_humming_prefill_tuning(
    tuning_config: Any,
    n: int,
    k: int,
) -> Any:
    overrides = _humming_prefill_tuning_overrides(n, k)
    if not overrides:
        return tuning_config

    if isinstance(tuning_config, dict):
        return overrides[-1][2]
    if not isinstance(tuning_config, list):
        return tuning_config

    min_override_m = min(start for start, _, _ in overrides)
    tuned_config = [
        entry
        for entry in tuning_config
        if len(entry) == 3 and entry[1] <= min_override_m
    ]
    tuned_config.extend([list(item) for item in overrides])
    return tuned_config


def _build_humming_weight_entry(
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    n: int,
    k: int,
) -> HummingMxfp4W4A8Weight:
    GemmType, HummingLayer, get_heuristics_config = _import_humming()

    # The MXFP4 W4A8 checkpoint stores per-32 weight scales as native E8M0
    # exponents. Feed them as uint8/float8_e8m0fnu directly (no fp32/bf16 detour).
    #
    # IMPORTANT: Humming's *fused*-E8M0 path does NOT support per-token-group
    # input scale (it only handles a single per-tensor global scale), which our
    # DeepEP FP8 activations (per-token-group-128) require -> fused gives garbage.
    # We force the *non-fused* group-scale mainloop path instead; it natively
    # supports per-group input scale AND consumes the 1-byte E8M0 weight scale.
    # The forcing is done purely on the SGLang side by _import_humming() ->
    # _patch_humming_disable_fused_e8m0() (no Humming source edits); the env var
    # SGLANG_HUMMING_DISABLE_FUSED_E8M0 (default "1") only gates that patch and is
    # documented here via setdefault so it shows up for operators inspecting env.
    os.environ.setdefault("SGLANG_HUMMING_DISABLE_FUSED_E8M0", "1")
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        raise RuntimeError(
            "Humming MXFP4 W4A8 path requires torch.float8_e8m0fnu support."
        )
    if b_scale.dtype == e8m0_dtype:
        b_scale_e8m0 = b_scale.contiguous()
    elif b_scale.dtype == torch.uint8:
        b_scale_e8m0 = b_scale.contiguous().view(e8m0_dtype)
    else:
        raise TypeError(
            "Humming MXFP4 W4A8 path expects E8M0 (uint8/float8_e8m0fnu) weight "
            f"scale, got {b_scale.dtype}"
        )
    if not b_packed.is_contiguous():
        raise ValueError("Humming MXFP4 W4A8 path requires contiguous weight tensors.")

    num_experts = b_packed.shape[0]
    with torch.cuda.device(b_packed.device):
        layer = HummingLayer(
            shape_n=n,
            shape_k=k,
            num_experts=num_experts,
            weight_config={
                "dtype": "float4e2m1",
                "group_size": 32,
                "scale_dtype": "float8e8m0",
            },
            input_config={"dtype": "float8e4m3", "group_size": 128},
            torch_dtype=torch.bfloat16,
        )
    layer.locks = torch.zeros((1024), dtype=torch.int32, device=b_packed.device)
    for name in ("weight", "weight_scale"):
        if hasattr(layer, name):
            delattr(layer, name)
    layer.weight = torch.nn.Parameter(
        b_packed.view(torch.int32), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        b_scale_e8m0, requires_grad=False
    )
    layer.transform()

    meta = layer.humming_metas[""]
    # Guard: the fused-E8M0 path silently miscomputes with per-group input scale
    # (produces garbage tokens). Ensure the non-fused path was actually selected.
    if getattr(meta, "use_fused_e8m0_scale", False):
        raise RuntimeError(
            "Humming MXFP4 W4A8 selected the fused-E8M0 path, which does not "
            "support per-token-group input scale and will miscompute. Set "
            "SGLANG_HUMMING_DISABLE_FUSED_E8M0=1 before server start."
        )
    contig_compute_config = {
        "use_f16_accum": False,
        "gemm_type": GemmType.GROUPED_CONTIGUOUS.value,
    }
    masked_compute_config = {
        "use_f16_accum": False,
        "gemm_type": GemmType.GROUPED_MASKED.value,
    }
    return HummingMxfp4W4A8Weight(
        layer=layer,
        n=n,
        k=k,
        num_experts=num_experts,
        contig_compute_config=contig_compute_config,
        contig_tuning_config=_sanitize_humming_e8m0_warp_n(
            _apply_humming_prefill_tuning(
                get_heuristics_config(
                    meta=meta,
                    use_f16_accum=False,
                    gemm_type=GemmType.GROUPED_CONTIGUOUS,
                ),
                n,
                k,
            )
        ),
        masked_compute_config=masked_compute_config,
        masked_tuning_config=_sanitize_humming_e8m0_warp_n(
            get_heuristics_config(
                meta=meta,
                use_f16_accum=False,
                gemm_type=GemmType.GROUPED_MASKED,
            )
        ),
    )


def _get_humming_normal_entry(
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    n: int,
    k: int,
):
    cache_key = _humming_entry_cache_key(b_packed, b_scale, n, k)
    entry = _HUMMING_NORMAL_CACHE.get(cache_key)
    if entry is not None:
        return entry
    if not _can_allocate_humming_normal_entry(b_packed, b_scale, cache_key):
        return None

    try:
        entry = _build_humming_weight_entry(b_packed, b_scale, n, k)
    except torch.OutOfMemoryError:
        if not _ALLOW_HUMMING_NORMAL_FALLBACK:
            raise
        _HUMMING_NORMAL_SKIP_KEYS.add(cache_key)
        torch.cuda.empty_cache()
        _log_humming_warning_once(
            "oom_during_repack",
            "Skip Humming MXFP4 W4A8 normal path because repacking the weight "
            "cache ran out of GPU memory; falling back to Triton.",
        )
        return None
    _HUMMING_NORMAL_CACHE[cache_key] = entry
    return entry


def _launch_grouped_gemm_contig_humming(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    expert_start: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    n: int,
    k: int,
    a_scale_group_size: int,
) -> torch.Tensor | None:
    if not _USE_HUMMING_NORMAL:
        return None
    if a_scale_group_size != 128:
        return None
    if a.dtype != torch.float8_e4m3fn or a_scale.dtype != torch.float32:
        return None
    if b_scale.dtype != torch.float32:
        return None
    if n % 128 != 0 or k % 128 != 0:
        return None

    entry = _get_humming_normal_entry(
        b_packed,
        b_scale,
        n,
        k,
    )
    if entry is None:
        return None
    return _launch_grouped_gemm_contig_humming_entry(
        a,
        a_scale,
        entry,
        expert_start,
        num_tokens_per_expert,
        a_scale_group_size,
    )


def _launch_grouped_gemm_contig_humming_entry(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    entry: HummingMxfp4W4A8Weight,
    expert_start: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    a_scale_group_size: int,
) -> torch.Tensor:
    if a_scale_group_size != 128:
        raise ValueError(
            f"Humming MXFP4 W4A8 expects activation scale group size 128, "
            f"got {a_scale_group_size}"
        )
    if a.dtype != torch.float8_e4m3fn or a_scale.dtype != torch.float32:
        raise TypeError(
            "Humming MXFP4 W4A8 expects FP8 E4M3 activations and float32 scales."
        )
    if entry.k != a.shape[1]:
        raise ValueError(f"Humming K mismatch: entry={entry.k}, a={a.shape[1]}")

    layout_key = (
        expert_start.device.index,
        torch.cuda.current_stream(expert_start.device).cuda_stream,
        "torch.int32",
        num_tokens_per_expert.shape[0],
    )
    expert_layout = _HUMMING_EXPERT_LAYOUT_CACHE.get(layout_key)
    if expert_layout is None:
        expert_layout = torch.empty(
            (num_tokens_per_expert.shape[0] + 1,),
            device=expert_start.device,
            dtype=torch.int32,
        )
        _HUMMING_EXPERT_LAYOUT_CACHE[layout_key] = expert_layout
    expert_layout[:-1].copy_(expert_start)
    expert_layout[-1] = a.shape[0]
    return entry.layer(
        inputs=a,
        input_scale=a_scale,
        expert_layout=expert_layout,
        compute_config=entry.contig_compute_config,
        tuning_config=entry.contig_tuning_config,
        top_k=1,
    )


def _launch_grouped_gemm_masked_humming_entry(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    entry: HummingMxfp4W4A8Weight,
    masked_m: torch.Tensor,
) -> torch.Tensor:
    e, m, k = a.shape
    if e != entry.num_experts:
        raise ValueError(
            f"Humming expert mismatch: entry={entry.num_experts}, input={e}"
        )
    if k != entry.k:
        raise ValueError(f"Humming K mismatch: entry={entry.k}, input={k}")
    if a.dtype != torch.float8_e4m3fn or a_scale.dtype != torch.float32:
        raise TypeError(
            "Humming MXFP4 W4A8 expects FP8 E4M3 activations and float32 scales."
        )
    if a_scale.shape[:2] != (e, m):
        raise ValueError(
            f"activation scale must start with {(e, m)}, got {tuple(a_scale.shape)}"
        )
    a_scale_group_size = k // a_scale.shape[-1]
    if a_scale_group_size != 128:
        raise ValueError(
            f"Humming MXFP4 W4A8 expects activation scale group size 128, "
            f"got {a_scale_group_size}"
        )
    if masked_m.shape[0] != e:
        raise ValueError(
            f"masked_m shape {tuple(masked_m.shape)} does not match {e=}"
        )

    output = entry.layer(
        inputs=a.reshape(e * m, k),
        input_scale=a_scale.reshape(e * m, a_scale.shape[-1]),
        expert_layout=masked_m.contiguous(),
        compute_config=entry.masked_compute_config,
        tuning_config=entry.masked_tuning_config,
        top_k=1,
    )
    return output.view(e, m, entry.n)


def prepare_humming_normal_weight_cache(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> None:
    if not _USE_HUMMING_NORMAL:
        return

    gateup_size = w13_weight.shape[1]
    hidden_size = w13_weight.shape[2] * 2
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    if w2_weight.shape[1] != hidden_size or w2_weight.shape[2] * 2 != intermediate_size:
        raise ValueError(
            f"w2 shape mismatch for Humming cache: {tuple(w2_weight.shape)=}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    _get_humming_normal_entry(
        w13_weight,
        w13_weight_scale,
        n=gateup_size,
        k=hidden_size,
    )
    _get_humming_normal_entry(
        w2_weight,
        w2_weight_scale,
        n=hidden_size,
        k=intermediate_size,
    )


def replace_mxfp4_w4a8_weights_with_humming(layer: torch.nn.Module) -> None:
    if not should_replace_humming_normal_weights():
        return

    gateup_size = layer.w13_weight.shape[1]
    hidden_size = layer.w13_weight.shape[2] * 2
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    w2_weight_shape = tuple(layer.w2_weight.shape)
    if (
        layer.w2_weight.shape[1] != hidden_size
        or layer.w2_weight.shape[2] * 2 != intermediate_size
    ):
        raise ValueError(
            f"w2 shape mismatch for Humming replacement: {w2_weight_shape=}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    w13_entry = _build_humming_weight_entry(
        layer.w13_weight,
        layer.w13_weight_scale_inv,
        n=gateup_size,
        k=hidden_size,
    )
    layer.w13_humming_layer = w13_entry.layer
    layer.w13_humming_weight = w13_entry
    delattr(layer, "w13_weight")
    delattr(layer, "w13_weight_scale_inv")
    gc.collect()
    torch.cuda.empty_cache()

    w2_entry = _build_humming_weight_entry(
        layer.w2_weight,
        layer.w2_weight_scale_inv,
        n=hidden_size,
        k=intermediate_size,
    )
    layer.w2_humming_layer = w2_entry.layer
    layer.w2_humming_weight = w2_entry
    delattr(layer, "w2_weight")
    delattr(layer, "w2_weight_scale_inv")
    for name in ("w13_weight_scale_e8m0", "w2_weight_scale_e8m0"):
        if hasattr(layer, name):
            delattr(layer, name)
    gc.collect()
    torch.cuda.empty_cache()


def _launch_grouped_gemm_contig(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    expert_start: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    n: int,
    k: int,
    max_m: int | None = None,
) -> torch.Tensor:
    total_m = a.shape[0]
    if a_scale.shape[0] != total_m:
        raise ValueError(
            f"activation scale first dim must be {total_m}, got {tuple(a_scale.shape)}"
        )
    output = torch.empty((total_m, n), device=a.device, dtype=torch.bfloat16)
    if total_m == 0:
        return output

    a_scale_group_size = k // a_scale.shape[-1]
    if a_scale_group_size <= 0 or k % a_scale_group_size != 0:
        raise ValueError(
            f"invalid activation scale layout: {tuple(a.shape)=}, "
            f"{tuple(a_scale.shape)=}"
        )

    humming_output = _launch_grouped_gemm_contig_humming(
        a,
        a_scale,
        b_packed,
        b_scale,
        expert_start,
        num_tokens_per_expert,
        n,
        k,
        a_scale_group_size,
    )
    if humming_output is not None:
        return humming_output

    if max_m is None:
        max_m = int(num_tokens_per_expert.max().item())
    if max_m <= 8:
        block_m = 8
        block_n = 128
        block_k = 64
    elif max_m <= 16:
        block_m = 16
        block_n = 128
        block_k = 64
    elif max_m <= 32:
        block_m = 32
        block_n = 128
        block_k = 64
    else:
        block_m = 64
        block_n = 128
        block_k = 64

    grid = (
        num_tokens_per_expert.shape[0],
        triton.cdiv(max_m, block_m),
        triton.cdiv(n, block_n),
    )
    launch_args = (
        a,
        a_scale,
        b_packed.view(torch.uint8),
        b_scale,
        output,
        expert_start,
        num_tokens_per_expert,
        a.stride(0),
        a.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        output.stride(0),
        output.stride(1),
        n,
        k,
        a_scale_group_size,
    )
    if _USE_DOT_SCALED:
        launch_kwargs = dict(
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            DOT_K=_DOT_SCALED_K,
            num_warps=4,
            num_stages=3,
        )
        if _DOT_SCALED_CONTIG_MAXNREG > 0:
            launch_kwargs["maxnreg"] = _DOT_SCALED_CONTIG_MAXNREG
        kernel = _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel
        if (
            _USE_DOT_SCALED_CONTIG_ALIGNED_NK
            and n % block_n == 0
            and k % _DOT_SCALED_K == 0
        ):
            kernel = _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_aligned_nk_kernel
        kernel[grid](
            *launch_args,
            **launch_kwargs,
        )
    else:
        _mxfp4_w4a8_grouped_gemm_contig_kernel[grid](
            *launch_args,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
    return output


def mxfp4_w4a8_deepep_ll_humming(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    masked_m: torch.Tensor,
    num_routed_tokens: int,
    w13_weight: HummingMxfp4W4A8Weight,
    w2_weight: HummingMxfp4W4A8Weight,
) -> torch.Tensor:
    del num_routed_tokens
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected FP8 hidden_states, got {hidden_states.dtype}")
    if hidden_states_scale.dtype != torch.float32:
        raise TypeError(
            f"expected float32 hidden_states_scale, got {hidden_states_scale.dtype}"
        )
    if not hidden_states.is_cuda or not hidden_states_scale.is_cuda:
        raise ValueError("hidden_states and hidden_states_scale must be CUDA tensors")

    num_experts, expected_m, hidden_size = hidden_states.shape
    if masked_m.shape[0] != num_experts:
        raise ValueError(
            f"masked_m shape {tuple(masked_m.shape)} does not match {num_experts=}"
        )
    if w13_weight.num_experts != num_experts or w2_weight.num_experts != num_experts:
        raise ValueError("weight expert dimension does not match DeepEP dispatch")
    if w13_weight.k != hidden_size:
        raise ValueError(
            f"w13 K mismatch: weight={w13_weight.k}, {hidden_size=}"
        )
    gateup_size = w13_weight.n
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    if intermediate_size % 128 != 0:
        raise ValueError(
            f"intermediate size must be divisible by FP8 group size 128, "
            f"got {intermediate_size}"
        )
    if w2_weight.n != hidden_size or w2_weight.k != intermediate_size:
        raise ValueError(
            f"w2 shape mismatch: n={w2_weight.n}, k={w2_weight.k}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    gateup_output = _launch_grouped_gemm_masked_humming_entry(
        hidden_states,
        hidden_states_scale,
        w13_weight,
        masked_m,
    )

    down_input = torch.empty(
        (num_experts, expected_m, intermediate_size),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    down_input_scale = torch.empty(
        (num_experts, expected_m, intermediate_size // 128),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    silu_and_mul_masked_post_quant_fwd(
        gateup_output,
        down_input,
        down_input_scale,
        128,
        masked_m,
    )
    del gateup_output

    return _launch_grouped_gemm_masked_humming_entry(
        down_input,
        down_input_scale,
        w2_weight,
        masked_m,
    )


def mxfp4_w4a8_deepep_ll_triton(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    masked_m: torch.Tensor,
    num_routed_tokens: int,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_weight_scale_e8m0: torch.Tensor | None = None,
    w2_weight_scale_e8m0: torch.Tensor | None = None,
) -> torch.Tensor:
    _check_inputs(
        hidden_states,
        hidden_states_scale,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        masked_m,
    )

    num_experts, expected_m, hidden_size = hidden_states.shape
    if masked_m.shape[0] != num_experts:
        raise ValueError(
            f"masked_m shape {tuple(masked_m.shape)} does not match {num_experts=}"
        )
    if w13_weight.shape[0] != num_experts or w2_weight.shape[0] != num_experts:
        raise ValueError("weight expert dimension does not match DeepEP dispatch")

    gateup_size = w13_weight.shape[1]
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    if intermediate_size % 128 != 0:
        raise ValueError(
            f"intermediate size must be divisible by FP8 group size 128, "
            f"got {intermediate_size}"
        )
    if w13_weight.shape[2] * 2 != hidden_size:
        raise ValueError(
            f"w13 K mismatch: packed={tuple(w13_weight.shape)}, {hidden_size=}"
        )
    if w2_weight.shape[1] != hidden_size or w2_weight.shape[2] * 2 != intermediate_size:
        raise ValueError(
            f"w2 shape mismatch: packed={tuple(w2_weight.shape)}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    gateup_output = _launch_grouped_gemm(
        hidden_states,
        hidden_states_scale,
        w13_weight,
        w13_weight_scale,
        masked_m,
        n=gateup_size,
        k=hidden_size,
        num_routed_tokens=num_routed_tokens,
        b_scale_e8m0=w13_weight_scale_e8m0,
    )

    down_input = torch.empty(
        (num_experts, expected_m, intermediate_size),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    down_input_scale = torch.empty(
        (num_experts, expected_m, intermediate_size // 128),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    silu_and_mul_masked_post_quant_fwd(
        gateup_output,
        down_input,
        down_input_scale,
        128,
        masked_m,
    )

    return _launch_grouped_gemm(
        down_input,
        down_input_scale,
        w2_weight,
        w2_weight_scale,
        masked_m,
        n=hidden_size,
        k=intermediate_size,
        num_routed_tokens=num_routed_tokens,
        b_scale_e8m0=w2_weight_scale_e8m0,
    )


def mxfp4_w4a8_deepep_normal_humming(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    expert_start: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    w13_weight: HummingMxfp4W4A8Weight,
    w2_weight: HummingMxfp4W4A8Weight,
    max_tokens_per_expert: int | None = None,
) -> torch.Tensor:
    del max_tokens_per_expert
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected FP8 hidden_states, got {hidden_states.dtype}")
    if hidden_states_scale.dtype != torch.float32:
        raise TypeError(
            f"expected float32 hidden_states_scale, got {hidden_states_scale.dtype}"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("hidden_states_scale", hidden_states_scale),
        ("expert_start", expert_start),
        ("num_tokens_per_expert", num_tokens_per_expert),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    total_m, hidden_size = hidden_states.shape
    if hidden_states_scale.shape[0] != total_m:
        raise ValueError(
            f"hidden_states_scale shape {tuple(hidden_states_scale.shape)} "
            f"does not match {tuple(hidden_states.shape)}"
        )
    num_experts = num_tokens_per_expert.shape[0]
    if expert_start.shape[0] != num_experts:
        raise ValueError("expert_start and num_tokens_per_expert shape mismatch")
    if w13_weight.num_experts != num_experts or w2_weight.num_experts != num_experts:
        raise ValueError("weight expert dimension does not match DeepEP dispatch")
    if w13_weight.k != hidden_size:
        raise ValueError(
            f"w13 K mismatch: weight={w13_weight.k}, {hidden_size=}"
        )
    gateup_size = w13_weight.n
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    if intermediate_size % 128 != 0:
        raise ValueError(
            f"intermediate size must be divisible by FP8 group size 128, "
            f"got {intermediate_size}"
        )
    if w2_weight.n != hidden_size or w2_weight.k != intermediate_size:
        raise ValueError(
            f"w2 shape mismatch: n={w2_weight.n}, k={w2_weight.k}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    a_scale_group_size = hidden_size // hidden_states_scale.shape[-1]
    gateup_output = _launch_grouped_gemm_contig_humming_entry(
        hidden_states,
        hidden_states_scale,
        w13_weight,
        expert_start,
        num_tokens_per_expert,
        a_scale_group_size,
    )

    down_input_3d = torch.empty(
        (1, total_m, intermediate_size),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    down_input_scale_3d = torch.empty(
        (1, total_m, intermediate_size // 128),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    all_tokens_mask = torch.empty((1,), device=hidden_states.device, dtype=torch.int32)
    all_tokens_mask.fill_(total_m)
    silu_and_mul_masked_post_quant_fwd(
        gateup_output.unsqueeze(0),
        down_input_3d,
        down_input_scale_3d,
        128,
        all_tokens_mask,
    )
    del gateup_output

    a_scale_group_size = intermediate_size // down_input_scale_3d.shape[-1]
    return _launch_grouped_gemm_contig_humming_entry(
        down_input_3d.squeeze(0),
        down_input_scale_3d.squeeze(0),
        w2_weight,
        expert_start,
        num_tokens_per_expert,
        a_scale_group_size,
    )


def mxfp4_w4a8_deepep_normal_triton(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    expert_start: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    max_tokens_per_expert: int | None = None,
) -> torch.Tensor:
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected FP8 hidden_states, got {hidden_states.dtype}")
    if hidden_states_scale.dtype != torch.float32:
        raise TypeError(
            f"expected float32 hidden_states_scale, got {hidden_states_scale.dtype}"
        )
    if (
        w13_weight_scale.dtype != torch.float32
        or w2_weight_scale.dtype != torch.float32
    ):
        raise TypeError(
            "mxfp4_w4a8 Triton path expects float32 MXFP4 weight scales. "
            f"Got {w13_weight_scale.dtype=} and {w2_weight_scale.dtype=}."
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("hidden_states_scale", hidden_states_scale),
        ("expert_start", expert_start),
        ("num_tokens_per_expert", num_tokens_per_expert),
        ("w13_weight", w13_weight),
        ("w2_weight", w2_weight),
        ("w13_weight_scale", w13_weight_scale),
        ("w2_weight_scale", w2_weight_scale),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    total_m, hidden_size = hidden_states.shape
    if hidden_states_scale.shape[0] != total_m:
        raise ValueError(
            f"hidden_states_scale shape {tuple(hidden_states_scale.shape)} "
            f"does not match {tuple(hidden_states.shape)}"
        )
    num_experts = num_tokens_per_expert.shape[0]
    if expert_start.shape[0] != num_experts:
        raise ValueError("expert_start and num_tokens_per_expert shape mismatch")
    if w13_weight.shape[0] != num_experts or w2_weight.shape[0] != num_experts:
        raise ValueError("weight expert dimension does not match DeepEP dispatch")

    gateup_size = w13_weight.shape[1]
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    if intermediate_size % 128 != 0:
        raise ValueError(
            f"intermediate size must be divisible by FP8 group size 128, "
            f"got {intermediate_size}"
        )
    if w13_weight.shape[2] * 2 != hidden_size:
        raise ValueError(
            f"w13 K mismatch: packed={tuple(w13_weight.shape)}, {hidden_size=}"
        )
    if w2_weight.shape[1] != hidden_size or w2_weight.shape[2] * 2 != intermediate_size:
        raise ValueError(
            f"w2 shape mismatch: packed={tuple(w2_weight.shape)}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    gateup_output = _launch_grouped_gemm_contig(
        hidden_states,
        hidden_states_scale,
        w13_weight,
        w13_weight_scale,
        expert_start,
        num_tokens_per_expert,
        n=gateup_size,
        k=hidden_size,
        max_m=max_tokens_per_expert,
    )

    down_input_3d = torch.empty(
        (1, total_m, intermediate_size),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    down_input_scale_3d = torch.empty(
        (1, total_m, intermediate_size // 128),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    all_tokens_mask = torch.empty((1,), device=hidden_states.device, dtype=torch.int32)
    all_tokens_mask.fill_(total_m)
    silu_and_mul_masked_post_quant_fwd(
        gateup_output.unsqueeze(0),
        down_input_3d,
        down_input_scale_3d,
        128,
        all_tokens_mask,
    )
    del gateup_output

    return _launch_grouped_gemm_contig(
        down_input_3d.squeeze(0),
        down_input_scale_3d.squeeze(0),
        w2_weight,
        w2_weight_scale,
        expert_start,
        num_tokens_per_expert,
        n=hidden_size,
        k=intermediate_size,
        max_m=max_tokens_per_expert,
    )
