from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass

import torch
import triton
import triton.language as tl


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8_deepep_triton import (  # noqa: E402
    _launch_grouped_gemm_contig,
    _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel,
)


@triton.jit
def _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_b_trans_load_kernel(
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
        b_raw_nk = tl.load(
            b_packed_ptr
            + expert_id * stride_be
            + offs_n[:, None] * stride_bn
            + (k_start // 2 + offs_k2[None, :]) * stride_bk2,
            mask=valid_n[:, None] & ((k_start // 2 + offs_k2[None, :]) < K // 2),
            other=0,
        ).to(tl.uint8)
        b_raw = tl.trans(b_raw_nk)

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
def _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_e8m0_kernel(
    a_ptr,
    a_scale_ptr,
    b_packed_ptr,
    b_scale_e8m0_ptr,
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
            + global_m * stride_asm
            + (k_start // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m,
            other=1.0,
        ).to(tl.float32)
        acc += raw_acc * a_scale[:, None]

    tl.store(
        c_ptr + global_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        tl.where(valid_m[:, None] & valid_n[None, :], acc.to(tl.bfloat16), 0.0),
        mask=valid_m[:, None] & valid_n[None, :],
    )


@triton.jit
def _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_a_scale_reuse_kernel(
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

    for group_start in range(0, K, A_SCALE_GROUP_SIZE):
        a_scale = tl.load(
            a_scale_ptr
            + global_m * stride_asm
            + (group_start // A_SCALE_GROUP_SIZE) * stride_asg,
            mask=valid_m,
            other=1.0,
        ).to(tl.float32)

        for k_offset in range(0, A_SCALE_GROUP_SIZE, DOT_K):
            k_start = group_start + k_offset
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


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int


@dataclass(frozen=True)
class Meta:
    block_m: int
    block_n: int
    dot_k: int
    num_warps: int
    num_stages: int


SHAPES = {
    "w13": Shape("w13", n=4096, k=6144),
    "w2": Shape("w2", n=6144, k=2048),
}

BASELINE_META = Meta(block_m=64, block_n=128, dot_k=32, num_warps=4, num_stages=3)


def make_counts(num_experts: int, total_m: int, distribution: str, seed: int) -> list[int]:
    if total_m < num_experts:
        raise ValueError("total_m must be >= num_experts for this harness")

    if distribution == "uniform":
        base = total_m // num_experts
        rem = total_m % num_experts
        return [base + (1 if i < rem else 0) for i in range(num_experts)]

    rng = random.Random(seed)
    weights = [rng.random() ** 2.2 for _ in range(num_experts)]
    total_w = sum(weights)
    counts = [max(1, int(total_m * w / total_w)) for w in weights]
    diff = total_m - sum(counts)
    order = sorted(range(num_experts), key=lambda i: weights[i], reverse=True)
    idx = 0
    while diff != 0:
        i = order[idx % num_experts]
        if diff > 0:
            counts[i] += 1
            diff -= 1
        elif counts[i] > 1:
            counts[i] -= 1
            diff += 1
        idx += 1
    return counts


def make_inputs(
    shape: Shape,
    num_experts: int,
    total_m: int,
    distribution: str,
    seed: int,
    scale_mode: str,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    counts = make_counts(num_experts, total_m, distribution, seed)
    starts = [0]
    for c in counts[:-1]:
        starts.append(starts[-1] + c)

    device = torch.device("cuda")
    a_fp32 = torch.randn((total_m, shape.k), device=device, dtype=torch.float32) * 0.25
    a = a_fp32.to(torch.float8_e4m3fn)
    del a_fp32
    a_scale = torch.rand((total_m, shape.k // 128), device=device, dtype=torch.float32)
    a_scale = a_scale * 0.02 + 0.001

    gen = torch.Generator(device=device).manual_seed(seed + 17)
    b_packed = torch.randint(
        0,
        256,
        (num_experts, shape.n, shape.k // 2),
        device=device,
        dtype=torch.uint8,
        generator=gen,
    )
    b_scale_e8m0 = None
    if scale_mode == "e8m0":
        b_scale_e8m0 = torch.randint(
            119,
            124,
            (num_experts, shape.n, shape.k // 32),
            device=device,
            dtype=torch.uint8,
            generator=gen,
        )
        b_scale = torch.exp2(b_scale_e8m0.to(torch.float32) - 127.0)
    else:
        b_scale = torch.rand(
            (num_experts, shape.n, shape.k // 32),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        b_scale = b_scale * 0.02 + 0.001

    expert_start = torch.tensor(starts, device=device, dtype=torch.int32)
    num_tokens_per_expert = torch.tensor(counts, device=device, dtype=torch.int32)
    return (
        a,
        a_scale,
        b_packed,
        b_scale,
        b_scale_e8m0,
        expert_start,
        num_tokens_per_expert,
        counts,
    )


def launch_with_meta(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    b_scale_e8m0: torch.Tensor | None,
    expert_start: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    shape: Shape,
    meta: Meta,
    max_m: int,
    variant: str = "baseline",
    maxnreg: int | None = None,
) -> torch.Tensor:
    total_m = a.shape[0]
    output = torch.empty((total_m, shape.n), device=a.device, dtype=torch.bfloat16)
    a_scale_group_size = shape.k // a_scale.shape[-1]
    grid = (
        num_tokens_per_expert.shape[0],
        triton.cdiv(max_m, meta.block_m),
        triton.cdiv(shape.n, meta.block_n),
    )
    kernel = (
        _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_b_trans_load_kernel
        if variant == "b_trans_load"
        else _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_kernel
    )
    scale_arg = b_scale
    if variant == "a_scale_reuse":
        kernel = _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_a_scale_reuse_kernel
    if variant == "aligned_nk":
        kernel = _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_aligned_nk_kernel
    if variant == "e8m0_b_scale":
        if b_scale_e8m0 is None:
            raise ValueError("e8m0_b_scale variant requires --scale-mode e8m0")
        kernel = _mxfp4_w4a8_grouped_gemm_contig_dot_scaled_e8m0_kernel
        scale_arg = b_scale_e8m0
    launch_kwargs = {
        "BLOCK_M": meta.block_m,
        "BLOCK_N": meta.block_n,
        "BLOCK_K": meta.dot_k,
        "DOT_K": meta.dot_k,
        "num_warps": meta.num_warps,
        "num_stages": meta.num_stages,
    }
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg
    kernel[grid](
        a,
        a_scale,
        b_packed.view(torch.uint8),
        scale_arg,
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
        scale_arg.stride(0),
        scale_arg.stride(1),
        scale_arg.stride(2),
        output.stride(0),
        output.stride(1),
        shape.n,
        shape.k,
        a_scale_group_size,
        **launch_kwargs,
    )
    return output


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    denom = bf.abs().clamp_min(1e-6)
    rel = diff / denom
    return {
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "max_rel_err": float(rel.max().item()),
        "mean_rel_err": float(rel.mean().item()),
        "allclose_1e-2": bool(torch.allclose(af, bf, atol=1e-2, rtol=1e-2)),
    }


def time_call(fn, warmup: int, iters: int) -> list[float]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    if out is not None:
        del out

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del out
    torch.cuda.synchronize()
    return times


def summarize_times(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "mean_ms": float(statistics.mean(ordered)),
        "p50_ms": float(statistics.median(ordered)),
        "p90_ms": float(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.90) - 1)]),
        "p99_ms": float(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.99) - 1)]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def parse_meta(text: str) -> Meta:
    parts = [int(x) for x in text.split(",")]
    if len(parts) != 5:
        raise ValueError("meta must be block_m,block_n,dot_k,num_warps,num_stages")
    return Meta(*parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--total-m", type=int, default=16384)
    parser.add_argument("--num-experts", type=int, default=48)
    parser.add_argument("--distribution", choices=["uniform", "skew"], default="skew")
    parser.add_argument("--scale-mode", choices=["float", "e8m0"], default="float")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        help="candidate meta as block_m,block_n,dot_k,num_warps,num_stages",
    )
    parser.add_argument(
        "--candidate-variant",
        choices=[
            "baseline",
            "b_trans_load",
            "e8m0_b_scale",
            "a_scale_reuse",
            "aligned_nk",
        ],
        default="baseline",
    )
    parser.add_argument("--candidate-maxnreg", type=int, default=0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    shape = SHAPES[args.shape]
    (
        a,
        a_scale,
        b_packed,
        b_scale,
        b_scale_e8m0,
        expert_start,
        num_tokens_per_expert,
        counts,
    ) = make_inputs(
        shape,
        args.num_experts,
        args.total_m,
        args.distribution,
        args.seed,
        args.scale_mode,
    )
    max_m = max(counts)

    torch.cuda.synchronize()
    baseline_out = _launch_grouped_gemm_contig(
        a,
        a_scale,
        b_packed,
        b_scale,
        expert_start,
        num_tokens_per_expert,
        n=shape.n,
        k=shape.k,
        max_m=max_m,
    )
    direct_baseline_out = launch_with_meta(
        a,
        a_scale,
        b_packed,
        b_scale,
        None,
        expert_start,
        num_tokens_per_expert,
        shape,
        BASELINE_META,
        max_m,
        "baseline",
        None,
    )
    torch.cuda.synchronize()
    baseline_check = compare(direct_baseline_out, baseline_out)
    del direct_baseline_out

    baseline_times = time_call(
        lambda: _launch_grouped_gemm_contig(
            a,
            a_scale,
            b_packed,
            b_scale,
            expert_start,
            num_tokens_per_expert,
            n=shape.n,
            k=shape.k,
            max_m=max_m,
        ),
        args.warmup,
        args.iters,
    )

    result = {
        "shape": asdict(shape),
        "total_m": args.total_m,
        "num_experts": args.num_experts,
        "distribution": args.distribution,
        "scale_mode": args.scale_mode,
        "counts": {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
            "nonzero": sum(1 for c in counts if c > 0),
        },
        "baseline_meta": asdict(BASELINE_META),
        "baseline_grid": [
            args.num_experts,
            triton.cdiv(max_m, BASELINE_META.block_m),
            triton.cdiv(shape.n, BASELINE_META.block_n),
        ],
        "baseline_direct_check": baseline_check,
        "baseline_latency": summarize_times(baseline_times),
        "candidates": [],
    }

    for meta_text in args.meta:
        meta = parse_meta(meta_text)
        out = launch_with_meta(
            a,
            a_scale,
            b_packed,
            b_scale,
            b_scale_e8m0,
            expert_start,
            num_tokens_per_expert,
            shape,
            meta,
            max_m,
            args.candidate_variant,
            args.candidate_maxnreg or None,
        )
        torch.cuda.synchronize()
        correctness = compare(out, baseline_out)
        del out
        times = time_call(
            lambda m=meta: launch_with_meta(
                a,
                a_scale,
                b_packed,
                b_scale,
                b_scale_e8m0,
                expert_start,
                num_tokens_per_expert,
                shape,
                m,
                max_m,
                args.candidate_variant,
                args.candidate_maxnreg or None,
            ),
            args.warmup,
            args.iters,
        )
        result["candidates"].append(
            {
                "meta": asdict(meta),
                "variant": args.candidate_variant,
                "maxnreg": args.candidate_maxnreg or None,
                "grid": [
                    args.num_experts,
                    triton.cdiv(max_m, meta.block_m),
                    triton.cdiv(shape.n, meta.block_n),
                ],
                "correctness": correctness,
                "latency": summarize_times(times),
                "speedup_vs_baseline_mean": result["baseline_latency"]["mean_ms"]
                / summarize_times(times)["mean_ms"],
            }
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
