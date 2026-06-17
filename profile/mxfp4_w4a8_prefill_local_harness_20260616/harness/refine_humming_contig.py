from __future__ import annotations

import argparse
import copy
import json
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
    return times


def _make_humming_layer(
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


def _strip_label(config: dict[str, Any]) -> dict[str, Any]:
    clean = dict(config)
    clean.pop("_label", None)
    return clean


def _add_candidate(
    candidates: list[dict[str, Any]],
    seen: set[str],
    shape_n: int,
    shape_k: int,
    label: str,
    **config: Any,
) -> None:
    if shape_n % config["block_shape"][1] != 0:
        return
    if shape_k % config["block_shape"][2] != 0:
        return
    if config["block_shape"][0] % config["warp_shape"][0] != 0:
        return
    if config["block_shape"][1] % config["warp_shape"][1] != 0:
        return
    if config["block_shape"][2] % config["warp_shape"][2] != 0:
        return
    payload = dict(config)
    key = json.dumps(payload, sort_keys=True)
    if key in seen:
        return
    seen.add(key)
    payload["_label"] = label
    candidates.append(payload)


def _candidate_configs(shape_name: str, shape_n: int, shape_k: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    num_sms_values = (96, 112, 132, 144, 160, 176, 192, 224, 264, 320, 396)
    block_m_values = (32, 40, 48, 56, 64, 72, 80)
    warp_n_values = (16, 32, 64)
    ctas_values = (1, 2)
    stage_values = (3, 4, 5)

    for block_m in block_m_values:
        for block_n in (64, 128, 256):
            for block_k in (64, 128):
                for warp_n in warp_n_values:
                    for num_ctas_per_sm in ctas_values:
                        for num_stages in stage_values:
                            for use_stream_k in (False, True):
                                for num_sms in num_sms_values:
                                    _add_candidate(
                                        candidates,
                                        seen,
                                        shape_n,
                                        shape_k,
                                        (
                                            f"bm{block_m}_bn{block_n}_bk{block_k}"
                                            f"_wn{warp_n}_c{num_ctas_per_sm}"
                                            f"_s{num_stages}_sk{int(use_stream_k)}"
                                            f"_sms{num_sms}"
                                        ),
                                        block_shape=(block_m, block_n, block_k),
                                        warp_shape=(block_m, warp_n, block_k),
                                        use_stream_k=use_stream_k,
                                        use_f16_accum=False,
                                        num_sms=num_sms,
                                        num_stages=num_stages,
                                        num_ctas_per_sm=num_ctas_per_sm,
                                    )

    if shape_name == "w13":
        preferred = ("bm64_bn128_bk128_wn32_c2_s4_sk1_sms132", "bm48_bn128_bk128_wn32_c2_s4_sk1_sms132")
    else:
        preferred = ("bm48_bn128_bk128_wn32_c2_s4_sk1_sms132", "bm64_bn128_bk128_wn32_c2_s4_sk1_sms132")
    rank = {label: index for index, label in enumerate(preferred)}
    candidates.sort(key=lambda item: rank.get(item["_label"], len(rank)))
    return candidates


def _record(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--total-m", type=int, default=16384)
    parser.add_argument("--num-experts", type=int, default=48)
    parser.add_argument("--distribution", choices=("uniform", "skew"), default="skew")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    shape = SHAPES[args.shape]
    (
        a,
        a_scale,
        b_packed,
        b_scale,
        _,
        _,
        num_tokens_per_expert,
        counts,
    ) = make_inputs(shape, args.num_experts, args.total_m, args.distribution, args.seed, "float")
    expert_layout = _make_expert_layout(num_tokens_per_expert)
    layer = _make_humming_layer(shape.n, shape.k, args.num_experts, b_packed, b_scale)
    meta = layer.humming_metas[""]
    heuristic = get_heuristics_config(
        meta=meta,
        shape_m=args.total_m,
        use_f16_accum=False,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    compute_config = {
        "use_f16_accum": False,
        "gemm_type": GemmType.GROUPED_CONTIGUOUS.value,
    }

    def run(config: dict[str, Any]) -> torch.Tensor:
        return layer(
            inputs=a,
            input_scale=a_scale,
            expert_layout=expert_layout,
            compute_config=compute_config,
            tuning_config=_strip_label(config),
            top_k=1,
        )

    reference_config = copy.deepcopy(heuristic)
    reference_config["_label"] = "heuristic"
    reference = run(reference_config)
    torch.cuda.synchronize()

    candidates = _candidate_configs(args.shape, shape.n, shape.k)
    if args.limit > 0:
        candidates = candidates[: args.limit]

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
            "nonzero": sum(1 for count in counts if count > 0),
        },
        "heuristic": heuristic,
        "results": [],
    }

    for index, config in enumerate(candidates):
        started = time.time()
        result: dict[str, Any] = {
            "index": index,
            "label": config["_label"],
            "config": _strip_label(config),
        }
        try:
            out = run(config)
            torch.cuda.synchronize()
            result["correctness"] = compare(out, reference)
            del out
            times = _time_call(lambda config=config: run(config), args.warmup, args.iters)
            result["latency"] = summarize_times(times)
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001
            torch.cuda.synchronize()
            result["ok"] = False
            result["error"] = repr(exc)
        result["elapsed_s"] = time.time() - started
        payload["results"].append(result)
        ok_results = [item for item in payload["results"] if item.get("ok")]
        if ok_results:
            payload["best"] = min(ok_results, key=lambda item: item["latency"]["mean_ms"])
        _record(args.output_json, payload)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    if "best" in payload:
        print("BEST", json.dumps(payload["best"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
