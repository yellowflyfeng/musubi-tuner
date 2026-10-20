import argparse
import logging
from typing import List

import torch

from musubi_tuner import cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_QWEN_IMAGE_21,
    ItemInfo,
    save_text_encoder_output_cache_qwen_image_21,
)
from musubi_tuner.qwen_image_21 import qwen_image_21_utils as qi21

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(processor, text_encoder, drop_idx: int, batch: List[ItemInfo], device: torch.device):
    embeds = qi21.encode_prompts(processor, text_encoder, [item.caption for item in batch], device, drop_idx)
    for item, embed in zip(batch, embeds):
        save_text_encoder_output_cache_qwen_image_21(item, embed.detach().to(torch.bfloat16).cpu())


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder", type=str, default=None, help="Qwen-Image-2.1 快照目录（内含 text_encoder/ 与 processor/）"
    )
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = setup_parser(parser)
    args = parser.parse_args()

    device = args.device if hasattr(args, "device") and args.device else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if args.text_encoder is None:
        raise ValueError("--text_encoder is required: pass the Qwen-Image-2.1 snapshot directory (holding text_encoder/)")

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_QWEN_IMAGE_21)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    processor, text_encoder = qi21.load_text_encoder(args.text_encoder, dtype=torch.bfloat16, device=device)
    drop_idx = qi21.get_drop_index(processor)
    logger.info(f"Qwen3-VL loaded, dropping {drop_idx} system-prompt tokens per caption")

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(processor, text_encoder, drop_idx, batch, device)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode,
        requires_content=False,
    )
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
