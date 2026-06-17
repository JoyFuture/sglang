from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from typing import Any

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


def _time_call(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
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
    return times


def _make_humming_layer(
    *,
    shape_n: int,
    shape_k: int,
    num_experts: int,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
) -> HummingLayer:
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
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


def _make_expert_layout(num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    counts = num_tokens_per_expert.to(torch.int64)
    layout = torch.empty((counts.numel() + 1,), device=counts.device, dtype=torch.int64)
    layout[0] = 0
    layout[1:] = torch.cumsum(counts, dim=0)
    return layout


def _candidate_configs(base: dict[str, Any], shape_n: int, shape_k: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add(label: str, **updates: Any) -> None:
        config = copy.deepcopy(base)
        config.update(updates)
        if shape_n % config["block_shape"][1] != 0:
            return
        if shape_k % config["block_shape"][2] != 0:
            return
        config["_label"] = label
        if config not in candidates:
            candidates.append(config)

    add("heuristic")

    for num_sms in (96, 112, 132, 160, 192, 224, 264, 320, 396):
        add(f"num_sms_{num_sms}", num_sms=num_sms)

    for block_m in (32, 48, 64, 96):
        add(
            f"bm{block_m}_bn128_bk128_wn16_c2_s4",
            block_shape=(block_m, 128, 128),
            warp_shape=(block_m, 16, 128),
            num_ctas_per_sm=2,
            num_stages=4,
        )
        add(
            f"bm{block_m}_bn128_bk128_wn32_c2_s4",
            block_shape=(block_m, 128, 128),
            warp_shape=(block_m, 32, 128),
            num_ctas_per_sm=2,
            num_stages=4,
        )

    for block_m in (32, 48, 64):
        add(
            f"bm{block_m}_bn64_bk128_wn16_c2_s4",
            block_shape=(block_m, 64, 128),
            warp_shape=(block_m, 16, 128),
            num_ctas_per_sm=2,
            num_stages=4,
        )
        add(
            f"bm{block_m}_bn256_bk128_wn32_c2_s4",
            block_shape=(block_m, 256, 128),
            warp_shape=(block_m, 32, 128),
            num_ctas_per_sm=2,
            num_stages=4,
        )

    for num_ctas_per_sm in (1, 2, 3, 4):
        add(f"ctas_{num_ctas_per_sm}", num_ctas_per_sm=num_ctas_per_sm)

    for num_stages in (3, 4, 5):
        add(f"stages_{num_stages}", num_stages=num_stages)

    for use_stream_k in (False, True):
        add(f"stream_k_{int(use_stream_k)}", use_stream_k=use_stream_k)

    for block_k, warp_k in ((64, 64), (128, 128), (256, 128)):
        add(
            f"bk{block_k}_wk{warp_k}",
            block_shape=(base["block_shape"][0], base["block_shape"][1], block_k),
            warp_shape=(base["warp_shape"][0], base["warp_shape"][1], warp_k),
        )

    cleaned: list[dict[str, Any]] = []
    seen = set()
    for config in candidates:
        label = config.pop("_label")
        key = json.dumps(config, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        config["_label"] = label
        cleaned.append(config)
    return cleaned


def _strip_label(config: dict[str, Any]) -> dict[str, Any]:
    clean = dict(config)
    clean.pop("_label", None)
    return clean


def _record_output(output_json: str, payload: dict[str, Any]) -> None:
    if not output_json:
        return
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--total-m", type=int, default=16384)
    parser.add_argument("--num-experts", type=int, default=48)
    parser.add_argument("--distribution", choices=["uniform", "skew"], default="skew")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=0)
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
        _expert_start,
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
    expert_layout = _make_expert_layout(num_tokens_per_expert)

    layer = _make_humming_layer(
        shape_n=shape.n,
        shape_k=shape.k,
        num_experts=args.num_experts,
        b_packed=b_packed,
        b_scale=b_scale,
    )
    meta = layer.humming_metas[""]
    heuristic_ranges = get_heuristics_config(
        meta=meta,
        use_f16_accum=False,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    base_config = get_heuristics_config(
        meta=meta,
        shape_m=args.total_m,
        use_f16_accum=False,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    compute_config = {
        "use_f16_accum": False,
        "gemm_type": GemmType.GROUPED_CONTIGUOUS.value,
    }

    def run_with_config(config: dict[str, Any]) -> torch.Tensor:
        return layer(
            inputs=a,
            input_scale=a_scale,
            expert_layout=expert_layout,
            compute_config=compute_config,
            tuning_config=_strip_label(config),
            top_k=1,
        )

    base_labeled = copy.deepcopy(base_config)
    base_labeled["_label"] = "heuristic_reference"
    reference = run_with_config(base_labeled)
    torch.cuda.synchronize()

    candidates = _candidate_configs(base_config, shape.n, shape.k)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    payload: dict[str, Any] = {
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
        "base_config": base_config,
        "heuristic_ranges": heuristic_ranges,
        "results": [],
    }

    for index, config in enumerate(candidates):
        label = str(config["_label"])
        clean_config = _strip_label(config)
        started = time.time()
        result: dict[str, Any] = {
            "index": index,
            "label": label,
            "config": clean_config,
        }
        try:
            out = run_with_config(config)
            torch.cuda.synchronize()
            result["correctness"] = compare(out, reference)
            del out
            times = _time_call(
                lambda config=config: run_with_config(config),
                args.warmup,
                args.iters,
            )
            result["latency"] = summarize_times(times)
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001
            torch.cuda.synchronize()
            result["ok"] = False
            result["error"] = repr(exc)
        result["elapsed_s"] = time.time() - started
        payload["results"].append(result)
        _record_output(args.output_json, payload)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    ok_results = [r for r in payload["results"] if r.get("ok")]
    if ok_results:
        best = min(ok_results, key=lambda r: r["latency"]["mean_ms"])
        payload["best"] = best
        print("BEST", json.dumps(best, indent=2, ensure_ascii=False), flush=True)
    _record_output(args.output_json, payload)


if __name__ == "__main__":
    main()
