# LoRA module for Qwen-Image-2.1
# 2.1 是单流结构：每个 QwenImage21TransformerBlock 里有 attn(to_q/to_k/to_v/to_out.0)
# 和 img_mlp(gate_layer/proj/out) 共 7 个无 bias 线性层。调制参数是全模型共享的一层
# （model.modulation），不在 block 内，所以不需要像 2512 那样排除 _mod_ 模式。

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


QWEN_IMAGE_21_TARGET_REPLACE_MODULES = ["QwenImage21TransformerBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    exclude_patterns = kwargs.get("exclude_patterns", None)
    exclude_patterns = [] if exclude_patterns is None else ast.literal_eval(exclude_patterns)
    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        QWEN_IMAGE_21_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        QWEN_IMAGE_21_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
