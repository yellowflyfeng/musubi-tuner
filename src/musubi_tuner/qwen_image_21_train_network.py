# Qwen-Image-2.1 的 LoRA 训练（纯文生图路径）
#
# 与 Qwen-Image(2512) 的关键差异，都会体现在下面的实现里：
#   · 单流 DiT 32 层，forward 需要 img_mask，且返回 Transformer2DModelOutput 而不是裸张量
#   · VAE 16 倍下采样、z_dim=64，latent 不做 2x2 patchify，token 数 = h*w
#     （所以时刻采样要用 --timestep_sampling qwen21_shift，而不是 qwen_shift）
#   · 文本编码器是 Qwen3-VL，取最后一层 decoder 输出、且在最终 RMSNorm 之前
import argparse
from typing import Optional

import torch
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_QWEN_IMAGE_21, ARCHITECTURE_QWEN_IMAGE_21_FULL
from musubi_tuner.hv_train_network import NetworkTrainer, DiTOutput, clean_memory_on_device, setup_parser_common, read_config_from_file
from musubi_tuner.qwen_image_21 import qwen_image_21_model, qwen_image_21_utils as qi21
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QwenImage21NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_QWEN_IMAGE_21

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_QWEN_IMAGE_21_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0  # 2.1 走 true CFG，不用嵌入式 guidance
        if args.timestep_sampling != "qwen21_shift":
            logger.warning(
                f"--timestep_sampling={args.timestep_sampling}：2.1 的 latent 不做 2x2 打包，"
                "推荐用 qwen21_shift，否则 shift 会按 1/4 的 token 数计算"
            )

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        """训练中出图用的 prompt：先用 Qwen3-VL 编码好，再把文本编码器放掉。"""
        from musubi_tuner.training.sampling_prompts import load_prompts

        prompts = load_prompts(sample_prompts)
        device = accelerator.device
        processor, text_encoder = qi21.load_text_encoder(args.text_encoder, torch.bfloat16, device)
        drop_idx = qi21.get_drop_index(processor)

        sample_parameters = []
        for pd in prompts:
            p = pd.copy()
            embeds = qi21.encode_prompts(processor, text_encoder, [pd.get("prompt", "")], device, drop_idx)
            p["vl21_embed"] = embeds[0].cpu()
            neg = pd.get("negative_prompt", None)
            neg_embeds = qi21.encode_prompts(processor, text_encoder, [neg if neg else " "], device, drop_idx)
            p["negative_vl21_embed"] = neg_embeds[0].cpu()
            sample_parameters.append(p)

        del text_encoder, processor
        clean_memory_on_device(device)
        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """FlowMatchEuler 去噪，和 diffusers 的 QwenImage21Pipeline 一致（true CFG）。"""
        from diffusers import FlowMatchEulerDiscreteScheduler

        device = accelerator.device
        lat_h, lat_w = height // qi21.VAE_SCALE_FACTOR, width // qi21.VAE_SCALE_FACTOR
        seq_len = lat_h * lat_w

        embed = sample_parameter["vl21_embed"].to(device, dtype=dit_dtype).unsqueeze(0)
        neg_embed = sample_parameter["negative_vl21_embed"].to(device, dtype=dit_dtype).unsqueeze(0)
        cfg = cfg_scale if cfg_scale is not None else 4.0
        do_cfg = cfg > 1.0

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.dit, subfolder="scheduler")
        mu = qi21.calculate_shift(seq_len)
        import numpy as np

        sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
        scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
        timesteps = scheduler.timesteps

        latents = torch.randn(1, seq_len, qi21.LATENT_CHANNELS, device=device, dtype=torch.float32, generator=generator)
        img_shapes = [[(1, lat_h, lat_w)]]

        def denoise(lat, ctx, t):
            mask = qi21.build_img_mask(ctx.shape[1], seq_len, 1, device)
            out = transformer(
                hidden_states=lat.to(dit_dtype),
                encoder_hidden_states=ctx,
                timestep=(t.expand(1) / 1000).to(dit_dtype),
                img_shapes=img_shapes,
                img_mask=mask,
                encoder_hidden_states_mask=torch.ones(1, ctx.shape[1], dtype=torch.bool, device=device),
                return_dict=False,
            )[0]
            return out[:, -seq_len:].float()

        with torch.no_grad(), accelerator.autocast():
            for t in timesteps:
                pred = denoise(latents, embed, t)
                if do_cfg:
                    neg_pred = denoise(latents, neg_embed, t)
                    pred = neg_pred + cfg * (pred - neg_pred)
                latents = scheduler.step(pred, t, latents, return_dict=False)[0]

            lat = qi21.unpack_latents(latents, lat_h, lat_w).to(vae.dtype)
            # 训练期间 latent 是预缓存的，VAE 常驻 CPU 省显存；只在这里搬上卡解码，解完立刻搬回去
            vae.to(device)
            try:
                image = qi21.vae_decode(vae, lat)[:, :3]  # 丢掉 alpha 通道
            finally:
                vae.to("cpu")
                clean_memory_on_device(device)

        image = (image.float() / 2 + 0.5).clamp(0, 1).cpu()
        return image.unsqueeze(2)  # (N, C, F=1, H, W)

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        return qi21.load_vae(vae_path, dtype=vae_dtype, device="cpu")

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        if getattr(args, "fp8_scaled", False) or getattr(args, "fp8_base", False):
            raise NotImplementedError("Qwen-Image-2.1 支持尚未实现 fp8；bf16 只有 14.3GB，32GB 卡放得下")
        return qwen_image_21_model.load_qwen_image_21_model(
            dit_path, dtype=dit_weight_dtype or torch.bfloat16, device=loading_device, attn_mode=attn_mode
        )

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(args, transformer, [transformer.transformer_blocks], disable_linear=False)

    def scale_shift_latents(self, latents):
        return latents  # latent 在缓存时已按 mean/std 归一化

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        model = transformer
        latents = batch["latents"]  # B, C, 1, H, W（缓存写的是 3 维，bucket 会 stack 成 4 维 B,C,H,W）
        if latents.dim() == 5:
            assert latents.shape[2] == 1, "Qwen-Image-2.1 只支持单帧"
            latents = latents.squeeze(2)
            noise = noise.squeeze(2)
            noisy_model_input = noisy_model_input.squeeze(2)
        bsize, _, lat_h, lat_w = latents.shape

        # 打包：(B, C, h, w) -> (B, h*w, C)，2.1 不做 patchify
        hidden = qi21.pack_latents(noisy_model_input)
        seq_len = hidden.shape[1]

        # 文本：变长 embed 补齐到最长，并给出有效位掩码
        vl_embed = batch["vl21_embed"]  # list of (L, D)
        txt_lens = [x.shape[0] for x in vl_embed]
        max_len = max(txt_lens)
        vl_embed = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in vl_embed], dim=0)
        vl_mask = torch.zeros(bsize, max_len, dtype=torch.bool, device=vl_embed.device)
        for i, n in enumerate(txt_lens):
            vl_mask[i, :n] = True

        if args.gradient_checkpointing:
            hidden.requires_grad_(True)
            vl_embed.requires_grad_(True)

        hidden = hidden.to(device=accelerator.device, dtype=network_dtype)
        vl_embed = vl_embed.to(device=accelerator.device, dtype=network_dtype)
        vl_mask = vl_mask.to(device=accelerator.device)
        # img_mask 覆盖「文本 + 目标图」整条序列：文本段无图像槽，目标图每 4 个 token 一个槽
        img_mask = qi21.build_img_mask(max_len, seq_len, bsize, accelerator.device)
        img_shapes = [[(1, lat_h, lat_w)]] * bsize

        with accelerator.autocast():
            model_pred = model(
                hidden_states=hidden,
                encoder_hidden_states=vl_embed,
                timestep=timesteps / 1000.0,
                img_shapes=img_shapes,
                img_mask=img_mask,
                encoder_hidden_states_mask=vl_mask,
                return_dict=False,
            )[0]
            model_pred = model_pred[:, -seq_len:]

        model_pred = qi21.unpack_latents(model_pred, lat_h, lat_w)

        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noise = noise.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents
        return DiTOutput(pred=model_pred, target=target)

    # endregion model specific


def qwen_image_21_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder", type=str, default=None, help="Qwen-Image-2.1 快照目录（内含 text_encoder/ 与 processor/）"
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = qwen_image_21_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # 由训练器决定（bf16）
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = QwenImage21NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
