from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd

_USE_DOT_SCALED = os.environ.get("SGLANG_MXFP4_W4A8_DOT_SCALED", "1") != "0"
_USE_E8M0_WEIGHT_SCALE_LL = os.environ.get("SGLANG_MXFP4_W4A8_E8M0_LL", "0") != "0"
_DOT_SCALED_K = 32


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
        _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel[grid](
            *launch_args,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            DOT_K=_DOT_SCALED_K,
            num_warps=4,
            num_stages=3,
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
