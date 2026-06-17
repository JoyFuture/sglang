from __future__ import annotations

import argparse
import json
import os
import sys

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
HUMMING_ROOT = os.environ.get("HUMMING_ROOT", "/sgl-workspace/humming-main")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)
if HUMMING_ROOT not in sys.path:
    sys.path.insert(0, HUMMING_ROOT)

from bench_contig import SHAPES, compare, make_inputs, summarize_times  # noqa: E402
from humming.config import GemmType  # noqa: E402
from humming.layer import HummingLayer  # noqa: E402
from humming.tune import get_heuristics_config  # noqa: E402


def _make_layer(shape, b_packed, b_scale):
    layer = HummingLayer(
        shape_n=shape.n,
        shape_k=shape.k,
        num_experts=48,
        weight_config={
            "dtype": "float4e2m1",
            "group_size": 32,
            "scale_dtype": "bfloat16",
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
        b_scale.to(torch.bfloat16).contiguous(), requires_grad=False
    )
    layer.transform()
    return layer


def _layout(counts):
    counts = counts.to(torch.int32)
    out = torch.empty((counts.numel() + 1,), device=counts.device, dtype=torch.int32)
    out[0] = 0
    out[1:] = torch.cumsum(counts, dim=0)
    return out


def _time_call(fn, warmup: int, iters: int):
    for _ in range(warmup):
        out = fn()
        del out
    torch.cuda.synchronize()
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
    return summarize_times(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--total-m", type=int, default=16384)
    parser.add_argument("--bm", type=int, required=True)
    parser.add_argument("--bn", type=int, default=128)
    parser.add_argument("--bk", type=int, default=128)
    parser.add_argument("--wn", type=int, required=True)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--ctas", type=int, default=2)
    parser.add_argument("--sms", type=int, default=132)
    parser.add_argument("--stream-k", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    shape = SHAPES[args.shape]
    a, a_scale, b_packed, b_scale, _, _, counts_gpu, _ = make_inputs(
        shape, 48, args.total_m, "skew", 0, "float"
    )
    expert_layout = _layout(counts_gpu)
    layer = _make_layer(shape, b_packed, b_scale)
    meta = layer.humming_metas[""]
    compute_config = {
        "use_f16_accum": False,
        "gemm_type": GemmType.GROUPED_CONTIGUOUS.value,
    }
    ref_config = get_heuristics_config(
        meta=meta,
        shape_m=args.total_m,
        use_f16_accum=False,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    config = {
        "block_shape": (args.bm, args.bn, args.bk),
        "warp_shape": (args.bm, args.wn, args.bk),
        "use_stream_k": bool(args.stream_k),
        "use_f16_accum": False,
        "num_sms": args.sms,
        "num_stages": args.stages,
        "num_ctas_per_sm": args.ctas,
    }

    def run(config_):
        return layer(
            inputs=a,
            input_scale=a_scale,
            expert_layout=expert_layout,
            compute_config=compute_config,
            tuning_config=config_,
            top_k=1,
        )

    ref = run(ref_config)
    torch.cuda.synchronize()
    out = run(config)
    torch.cuda.synchronize()
    result = {
        "shape": args.shape,
        "total_m": args.total_m,
        "config": config,
        "correctness": compare(out, ref),
        "latency": _time_call(lambda: run(config), args.warmup, args.iters),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
