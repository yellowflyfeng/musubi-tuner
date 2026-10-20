# Qwen-Image-2.1 LoRA 训练

给 musubi-tuner 加的 Qwen-Image-2.1（7B 单流 DiT）支持，只覆盖**纯文生图**路径。

## 与 Qwen-Image (2512) 的差异

| | Qwen-Image 2512 | Qwen-Image-2.1 |
|---|---|---|
| DiT | 60 层双流 MMDiT，20.4B | 32 层单流，7B |
| VAE | 8 倍下采样，16 通道 | **16 倍下采样，64 通道**，4 通道 RGBA |
| latent 打包 | 2×2 patchify | **直接空间展平**，patch_size=1 |
| 1024² 的 token 数 | 4096 | 4096（h=w=64，无打包） |
| 文本编码器 | Qwen2.5-VL-7B | **Qwen3-VL-8B** |
| forward | 需要 `txt_seq_lens`、`guidance` | 需要 **`img_mask`**，返回 `Transformer2DModelOutput` |
| 时刻采样 | `--timestep_sampling qwen_shift` | **`--timestep_sampling qwen21_shift`** |

`qwen21_shift` 是新加的分支：`qwen_shift` 按 `(h//2)*(w//2)` 算 token 数（因为 2512 要 2×2 打包），
2.1 不打包，token 数就是 `h*w`，沿用旧分支会让 shift 按 1/4 的序列长度计算。

DiT 实现直接用 diffusers 的 `QwenImage21Transformer2DModel`，musubi 侧只做了一层薄封装补齐
`enable_gradient_checkpointing` / block swap 这几个训练框架要求的方法。**不支持 fp8 和 block swap**：
bf16 权重只有 14.3 GB，32 GB 卡直接装得下。

## 依赖

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5.17" accelerate peft bitsandbytes
pip install git+https://github.com/huggingface/diffusers   # QwenImage21 尚未进稳定版
```

三个参数都指向同一个 HF 快照目录（里面有 `transformer/`、`vae/`、`text_encoder/`、`processor/`、`scheduler/`）：
`--dit`、`--vae`、`--text_encoder`。

## 用法

```bash
# 1) 缓存 latent
python src/musubi_tuner/qwen_image_21_cache_latents.py \
    --dataset_config dataset.toml --vae /path/to/Qwen-Image-2.1 --keep_cache --skip_existing

# 2) 缓存文本编码
python src/musubi_tuner/qwen_image_21_cache_text_encoder_outputs.py \
    --dataset_config dataset.toml --text_encoder /path/to/Qwen-Image-2.1 --batch_size 8 --keep_cache --skip_existing

# 3) 训练
accelerate launch --num_processes 4 --multi_gpu --mixed_precision bf16 \
  src/musubi_tuner/qwen_image_21_train_network.py \
    --dit /path/to/Qwen-Image-2.1 --vae /path/to/Qwen-Image-2.1 --text_encoder /path/to/Qwen-Image-2.1 \
    --dataset_config dataset.toml --sdpa --mixed_precision bf16 \
    --timestep_sampling qwen21_shift --weighting_scheme none \
    --network_module networks.lora_qwen_image_21 --network_dim 64 --network_alpha 64 \
    --optimizer_type bitsandbytes.optim.AdamW8bit --learning_rate 1e-4 --lr_scheduler cosine \
    --gradient_checkpointing --max_train_epochs 4 --save_every_n_steps 2000 \
    --output_dir outputs --output_name qwen21_lora
```

数据集 toml 与其他架构一致（`resolution = [1024, 1024]`、`enable_bucket`、`image_jsonl_file` 等）。
分桶步长是 32（VAE 16 倍 × img_mask 的 2×2 分组）。

## 实现清单

新增：
- `src/musubi_tuner/qwen_image_21/qwen_image_21_utils.py` — 加载、文本编码、VAE 编解码、latent 打包、shift
- `src/musubi_tuner/qwen_image_21/qwen_image_21_model.py` — diffusers 模型的 musubi 封装
- `src/musubi_tuner/qwen_image_21_cache_latents.py`、`..._cache_text_encoder_outputs.py`、`..._train_network.py`
- `src/musubi_tuner/networks/lora_qwen_image_21.py` — 目标块 `QwenImage21TransformerBlock`
- 三个根目录 shim

修改：
- `dataset/architectures.py` — `qi21` / `qwen_image_21`
- `dataset/bucket.py` — 分桶步长 32
- `dataset/cache_io.py` — `save_latent_cache_qwen_image_21`（3 维 `(C,H,W)`）、`save_text_encoder_output_cache_qwen_image_21`（`varlen_vl21_embed_*`）
- `utils/sai_model_spec.py` — 元数据
- `training/trainer_base.py`、`training/parser_common.py` — `qwen21_shift`
- `networks/network_arch.py` — LoHa/LoKr 的架构识别

## LoRA 挂在哪

每个 `QwenImage21TransformerBlock` 里 7 个无 bias 线性层：`attn.to_q/to_k/to_v/to_out.0`、
`img_mlp.gate_layer/proj/out`。调制参数是全模型共享的 `model.modulation`，不在 block 内，
所以不需要像 2512 那样排除 `_mod_`。

每层每 rank 81,920 参数，32 层合计：

| rank | 参数量 | bf16 | 占 7B |
|---|---|---|---|
| 32 | 83.9 M | 168 MB | 1.20% |
| 64 | 167.8 M | 336 MB | 2.40% |
