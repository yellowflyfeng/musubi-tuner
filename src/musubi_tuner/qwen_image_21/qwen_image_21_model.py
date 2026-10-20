# 把 diffusers 的 QwenImage21Transformer2DModel 接到 musubi 的训练框架上。
# musubi 的 NetworkTrainer 要求 DiT 提供梯度检查点与 block swap 这几个方法；2.1 的 DiT bf16 只有 14.3GB，
# 32GB 卡直接装得下，所以 block swap 只做成显式不支持，不引入额外复杂度。
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def create_musubi_class():
    from diffusers import QwenImage21Transformer2DModel

    class MusubiQwenImage21Transformer2DModel(QwenImage21Transformer2DModel):
        """在 diffusers 模型上补齐 musubi 训练循环需要的接口。"""

        @property
        def device(self) -> torch.device:
            return next(self.parameters()).device

        @property
        def dtype(self) -> torch.dtype:
            return next(self.parameters()).dtype

        def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
            if activation_cpu_offloading:
                raise NotImplementedError("Qwen-Image-2.1 支持尚未实现激活值 CPU 卸载")
            super().enable_gradient_checkpointing()
            logger.info("Qwen-Image-2.1: gradient checkpointing enabled")

        def disable_gradient_checkpointing(self):
            super().disable_gradient_checkpointing()

        def enable_block_swap(self, blocks_to_swap: int, config=None):
            raise NotImplementedError(
                "Qwen-Image-2.1 支持未实现 block swap：模型 bf16 仅 14.3GB，32GB 卡无需换块，请去掉 --blocks_to_swap"
            )

        def move_to_device_except_swap_blocks(self, device):
            self.to(device)

        def prepare_block_swap_before_forward(self):
            pass

        def switch_block_swap_for_inference(self):
            pass

        def switch_block_swap_for_training(self):
            pass

    return MusubiQwenImage21Transformer2DModel


def load_qwen_image_21_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    attn_mode: str = "sdpa",
    split_attn: bool = False,
):
    cls = create_musubi_class()
    logger.info(f"Loading Qwen-Image-2.1 transformer from {model_path} (dtype={dtype}, device={device})")
    model = cls.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    if attn_mode not in ("sdpa", "torch"):
        logger.warning(f"attn_mode={attn_mode} 不受支持，回退到 sdpa（flex 需 torch.compile 才有收益）")
    model = model.to(device)
    model.eval().requires_grad_(False)
    return model
