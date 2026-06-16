from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, log_info_on_rank0
from sglang.srt.utils.common import is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


def _mxfp4_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor | None:
    if os.environ.get("SGLANG_MXFP4_W4A8_E8M0_LL", "0") == "0":
        return None
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        return None
    if scale.dtype == e8m0_dtype:
        return scale.contiguous().view(torch.uint8)
    if scale.dtype == torch.uint8:
        return scale.contiguous()
    if scale.dtype == torch.int8:
        return scale.contiguous().view(torch.uint8)
    return scale.to(e8m0_dtype).view(torch.uint8).contiguous()


class Mxfp4W4A8MoEMethod:
    """MXFP4 expert method for the DeepEP low-latency W4A8 prototype path.

    The checkpoint remains packed MXFP4. The first implementation intentionally
    keeps the weight layout unchanged and lets the fused runner decode the
    selected experts during forward.
    """

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        # Import registers ("deepep", "mxfp4_w4a8") in FusedOpPool before
        # MoeRunner looks it up.
        import sglang.srt.layers.moe.moe_runner.mxfp4_w4a8  # noqa: F401

        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MXFP4_W4A8, moe_runner_config)

    def process_weights_after_loading(self, layer: Module) -> None:
        self._fp8.process_weights_after_loading(layer)

        if not is_cuda() or not is_sm90_supported():
            raise RuntimeError("mxfp4_w4a8 MoE runner currently requires Hopper/SM90.")

        layer.w13_weight.data = layer.w13_weight.data.view(torch.int8)
        layer.w2_weight.data = layer.w2_weight.data.view(torch.int8)
        if os.environ.get("SGLANG_MXFP4_W4A8_USE_HUMMING_NORMAL", "0") != "0" or (
            os.environ.get("SGLANG_MXFP4_W4A8_HUMMING_REPLACE_WEIGHTS", "0") != "0"
        ):
            from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8_deepep_triton import (
                replace_mxfp4_w4a8_weights_with_humming,
                should_replace_humming_normal_weights,
            )

            if should_replace_humming_normal_weights():
                replace_mxfp4_w4a8_weights_with_humming(layer)
            else:
                from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8_deepep_triton import (
                    prepare_humming_normal_weight_cache,
                )

                prepare_humming_normal_weight_cache(
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale_inv,
                    layer.w2_weight_scale_inv,
                )

        if hasattr(layer, "w13_humming_weight"):
            layer._dsv4_mxfp4_backend = "mxfp4_w4a8_humming"
            log_info_on_rank0(
                logger,
                f"Using Humming MXFP4 W4A8 repacked weights for MoE layer "
                f"{self.prefix}.",
            )
            return

        layer.register_buffer(
            "w13_weight_scale_e8m0",
            _mxfp4_scale_to_e8m0(layer.w13_weight_scale_inv.data),
            persistent=False,
        )
        layer.register_buffer(
            "w2_weight_scale_e8m0",
            _mxfp4_scale_to_e8m0(layer.w2_weight_scale_inv.data),
            persistent=False,
        )
        layer._dsv4_mxfp4_backend = "mxfp4_w4a8"
        log_info_on_rank0(
            logger,
            f"Using DeepEP MXFP4 W4A8 prototype runner for MoE layer {self.prefix}.",
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8 import (
            Mxfp4W4A8QuantInfo,
        )

        quant_info = Mxfp4W4A8QuantInfo(
            w13_weight=getattr(layer, "w13_weight", None),
            w2_weight=getattr(layer, "w2_weight", None),
            w13_weight_scale=getattr(layer, "w13_weight_scale_inv", None),
            w2_weight_scale=getattr(layer, "w2_weight_scale_inv", None),
            w13_weight_scale_e8m0=getattr(layer, "w13_weight_scale_e8m0", None),
            w2_weight_scale_e8m0=getattr(layer, "w2_weight_scale_e8m0", None),
            humming_w13_weight=getattr(layer, "w13_humming_weight", None),
            humming_w2_weight=getattr(layer, "w2_humming_weight", None),
            swiglu_limit=self.moe_runner_config.swiglu_limit,
        )
        return self.runner.run(dispatch_output, quant_info=quant_info)
