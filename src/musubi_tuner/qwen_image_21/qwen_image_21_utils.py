# Qwen-Image-2.1 的加载与编码工具。所有行为对齐 diffusers 的 QwenImage21Pipeline（t2i 路径）。
# 与 Qwen-Image(2512) 的差异：单流 DiT 32 层、VAE 16 倍下采样且 z_dim=64、文本编码器换成 Qwen3-VL。
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

VAE_SCALE_FACTOR = 16          # 空间下采样倍数：1024px -> 64 个 latent 格
LATENT_CHANNELS = 64           # VAE z_dim，等于 DiT 的 in_channels
PATCH_SIZE = 1                 # 2.1 不再做 2x2 patchify，latent 直接展平成 token
RESOLUTION_ALIGN = 32          # img_mask 每槽覆盖 2x2 个 latent token，故像素尺寸须被 32 整除
SYS_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n" "<|im_start|>user\n{}<|im_end|>\n" "<|im_start|>assistant\n"
)
# 调度器参数取自 scheduler/scheduler_config.json，与 musubi 的 qwen_shift 完全一致
SCHED_BASE_SEQ_LEN, SCHED_MAX_SEQ_LEN, SCHED_BASE_SHIFT, SCHED_MAX_SHIFT = 256, 8192, 0.5, 0.9


def calculate_shift(image_seq_len: int) -> float:
    m = (SCHED_MAX_SHIFT - SCHED_BASE_SHIFT) / (SCHED_MAX_SEQ_LEN - SCHED_BASE_SEQ_LEN)
    return image_seq_len * m + SCHED_BASE_SHIFT - m * SCHED_BASE_SEQ_LEN


def load_transformer(model_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu", attn_mode: str = "sdpa"):
    from diffusers import QwenImage21Transformer2DModel

    logger.info(f"Loading Qwen-Image-2.1 transformer from {model_path}")
    model = QwenImage21Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    if attn_mode == "flex":  # flex 需要 torch.compile 才快，默认走 sdpa
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor

        model.set_attn_processor(QwenImage21FlexAttnProcessor())
    return model.to(device).eval()


def load_vae(model_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    return vae.to(device).eval()


def load_text_encoder(model_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, subfolder="processor")
    te = Qwen3VLForConditionalGeneration.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
    return processor, te.to(device).eval()


def get_drop_index(processor) -> int:
    """系统提示占几个 token：这些位置的隐状态要丢掉，和 diffusers 对齐。"""
    sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
    return len(processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])


@torch.no_grad()
def encode_prompts(processor, text_encoder, prompts: list[str], device: torch.device, drop_idx: Optional[int] = None):
    """返回每条 prompt 的变长文本嵌入 list[(L, 4096)]。

    取的是**最后一层 decoder 的输出、且在最终 RMSNorm 之前**——DiT 就是按这个训练的。
    transformers 5.x 会把 hidden_states[-1] 绑到归一化后的 last_hidden_state，所以这里挂一个
    forward hook 把 norm 的输出换回它的输入，抵消归一化（与 diffusers 的处理一致）。
    """
    if drop_idx is None:
        drop_idx = get_drop_index(processor)
    prompts = [" " if not p else p for p in prompts]
    texts = [PROMPT_TEMPLATE_T2I.format(p) for p in prompts]
    inputs = processor(text=texts, padding=True, padding_side="left", return_tensors="pt").to(device)

    text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        kwargs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "output_hidden_states": True}
        if hasattr(inputs, "mm_token_type_ids"):
            kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids
        outputs = text_encoder(**kwargs)
    finally:
        handle.remove()

    hidden = outputs.hidden_states[-1]
    mask = inputs.attention_mask.bool()
    return [hidden[i][mask[i]][drop_idx:] for i in range(hidden.shape[0])]


@torch.no_grad()
def vae_encode(vae, pixels: torch.Tensor) -> torch.Tensor:
    """pixels: (B, C, H, W)，值域 [-1, 1]，C=3(RGB) 或 4(RGBA)。返回归一化后的 (B, 64, h, w)。"""
    if pixels.shape[1] == 3:  # 2.1 的 VAE 吃 4 通道，RGB 补一个全不透明的 alpha
        pixels = torch.cat([pixels, torch.ones_like(pixels[:, :1])], dim=1)
    latents = vae.encode(pixels.unsqueeze(2)).latent_dist.mode()  # (B, 64, 1, h, w)
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    return ((latents - mean) / std).squeeze(2)


@torch.no_grad()
def vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """latents: (B, 64, h, w) 归一化后的。返回 (B, 4, H, W)，值域 [-1, 1]。"""
    if latents.dim() == 4:
        latents = latents.unsqueeze(2)
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    return vae.decode(latents * std + mean).sample.squeeze(2)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) -> (B, h*w, C)。2.1 不做 patchify，只是空间展平。"""
    b, c, h, w = latents.shape
    return latents.view(b, c, h * w).transpose(1, 2)


def unpack_latents(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, h*w, C) -> (B, C, h, w)"""
    b, _, c = tokens.shape
    return tokens.transpose(1, 2).reshape(b, c, h, w)


def build_img_mask(text_len: int, image_seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """DiT 的 img_mask 覆盖「文本 + 目标图」整条序列。

    纯文生图时文本段没有图像槽（全 False），后面按每 4 个 latent token 追加一个槽位，与 diffusers 的
    append_target_slots 一致。
    """
    assert image_seq_len % 4 == 0, (
        f"image_seq_len={image_seq_len} 不能被 4 整除：latent 高宽必须是偶数，"
        "检查分桶步长是否为 32"
    )
    text_part = torch.zeros(batch_size, text_len, dtype=torch.bool, device=device)
    target_part = torch.ones(batch_size, image_seq_len // 4, dtype=torch.bool, device=device)
    return torch.cat([text_part, target_part], dim=1)
