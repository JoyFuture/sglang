from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict

import torch
import triton


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
HUMMING_ROOT = os.environ.get("HUMMING_ROOT", "/sgl-workspace/humming-main")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)
if HUMMING_ROOT not in sys.path:
    sys.path.insert(0, HUMMING_ROOT)

from bench_contig import SHAPES, compare, make_inputs, summarize_times  # noqa: E402
from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8_deepep_triton import (  # noqa: E402
    _launch_grouped_gemm_contig,
)

from humming import dtypes  # noqa: E402
from humming.config import GemmType  # noqa: E402
from humming.layer import HummingLayer  # noqa: E402
from humming.tune import get_heuristics_config  # noqa: E402


def _time_call(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return times


def _make_humming_layer(
    *,
    shape_n: int,
    shape_k: int,
    num_experts: int,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    b_scale_dtype: str,
) -> HummingLayer:
    if b_scale_dtype == "bf16":
        scale_dtype = "bfloat16"
        scale_tensor = b_scale.to(torch.bfloat16)
    elif b_scale_dtype == "e8m0":
        scale_dtype = "float8e8m0"
        scale_tensor = b_scale.to(torch.float8_e8m0fnu)
    else:
        raise ValueError(f"unsupported b_scale_dtype={b_scale_dtype}")

    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        weight_config={
            "dtype": "float4e2m1",
            "group_size": 32,
            "scale_dtype": scale_dtype,
        },
        input_config={"dtype": "float8e4m3", "group_size": 128},
        torch_dtype=torch.bfloat16,
    ).to(b_packed.device)

    for name in ("weight", "weight_scale"):
        if hasattr(layer, name):
            delattr(layer, name)
    layer.weight = torch.nn.Parameter(
        b_packed.contiguous().view(torch.int32), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        scale_tensor.contiguous(), requires_grad=False
    )
    layer.transform()
    return layer


def _make_expert_layout(num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    counts = num_tokens_per_expert.to(torch.int64)
    layout = torch.empty((counts.numel() + 1,), device=counts.device, dtype=torch.int64)
    layout[0] = 0
    layout[1:] = torch.cumsum(counts, dim=0)
    return layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--total-m", type=int, default=16384)
    parser.add_argument("--num-experts", type=int, default=48)
    parser.add_argument("--distribution", choices=["uniform", "skew"], default="skew")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--b-scale-dtype", choices=["bf16", "e8m0"], default="bf16")
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
        _,
        expert_start,
        num_tokens_per_expert,
        counts,
    ) = make_inputs(
        shape,
        args.num_experts,
        args.total_m,
        args.distribution,
        args.seed,
        "float",
    )
    max_m = max(counts)
    expert_layout = _make_expert_layout(num_tokens_per_expert)

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
    torch.cuda.synchronize()

    layer = _make_humming_layer(
        shape_n=shape.n,
        shape_k=shape.k,
        num_experts=args.num_experts,
        b_packed=b_packed,
        b_scale=b_scale,
        b_scale_dtype=args.b_scale_dtype,
    )
    meta = layer.humming_metas[""]
    tuning_config = get_heuristics_config(
        meta=meta,
        use_f16_accum=False,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    compute_config = {
        "use_f16_accum": False,
        "gemm_type": GemmType.GROUPED_CONTIGUOUS.value,
    }

    def run_humming() -> torch.Tensor:
        return layer(
            inputs=a,
            input_scale=a_scale,
            expert_layout=expert_layout,
            compute_config=compute_config,
            tuning_config=tuning_config,
            top_k=1,
        )

    humming_out = run_humming()
    torch.cuda.synchronize()
    correctness = compare(humming_out, baseline_out)

    baseline_times = _time_call(
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
    humming_times = _time_call(run_humming, args.warmup, args.iters)

    result = {
        "shape": asdict(shape),
        "total_m": args.total_m,
        "num_experts": args.num_experts,
        "distribution": args.distribution,
        "counts": {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
            "stdev": statistics.pstdev(counts),
            "nonzero": sum(1 for c in counts if c > 0),
        },
        "b_scale_dtype": args.b_scale_dtype,
        "triton_baseline_latency": summarize_times(baseline_times),
        "humming_latency": summarize_times(humming_times),
        "speedup_vs_triton_mean": summarize_times(baseline_times)["mean_ms"]
        / summarize_times(humming_times)["mean_ms"],
        "correctness": correctness,
        "humming_tuning_config": tuning_config,
        "triton_grid": [
            args.num_experts,
            triton.cdiv(
                max_m,
                64 if max_m > 32 else max(8, 2 ** ((max_m - 1).bit_length())),
            ),
            triton.cdiv(shape.n, 128),
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
