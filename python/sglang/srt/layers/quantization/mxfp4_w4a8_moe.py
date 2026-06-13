from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, log_info_on_rank0
from sglang.srt.utils.common import is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


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
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale_inv,
            w2_weight_scale=layer.w2_weight_scale_inv,
            swiglu_limit=self.moe_runner_config.swiglu_limit,
        )
        return self.runner.run(dispatch_output, quant_info=quant_info)
