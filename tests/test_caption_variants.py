"""Caption variants: an image dataset may name extra text-encoder cache directories keyed by the
same item basename; each exposure draws one of the caches an item has, by weight, per epoch."""

import argparse
import collections
import json
import multiprocessing
from pathlib import Path
import random
import sys

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint

N_ITEMS, WITH_SHORT = 200, 120  # items 0..119 have a short variant; even ones among them also a medium
BASE_LEN, SHORT_LEN, MEDIUM_LEN = 40, 3, 7  # TE sequence length identifies which cache was loaded


def _write_caches(root: Path) -> dict[str, Path]:
    dirs = {name: root / name for name in ("cache", "cache_te_short", "cache_te_medium")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    with (root / "train.jsonl").open("w") as f:
        for i in range(N_ITEMS):
            key = f"{i:07d}"
            f.write(json.dumps({"image_path": f"/x/images/{key}.jpg", "caption": "long"}) + "\n")
            latent = {"latents_1x64x64_bfloat16": torch.zeros(1, 64, 64, dtype=torch.bfloat16)}
            save_file(latent, str(dirs["cache"] / f"{key}_0512x0512_qi21.safetensors"))
            te = lambda n: {"varlen_vl21_embed_bfloat16": torch.zeros(n, 8, dtype=torch.bfloat16)}
            save_file(te(BASE_LEN), str(dirs["cache"] / f"{key}_qi21_te.safetensors"))
            if i < WITH_SHORT:
                save_file(te(SHORT_LEN), str(dirs["cache_te_short"] / f"{key}_qi21_te.safetensors"))
                if i % 2 == 0:
                    save_file(te(MEDIUM_LEN), str(dirs["cache_te_medium"] / f"{key}_qi21_te.safetensors"))
    return dirs


def _variants(dirs):
    return [
        {"name": "short", "directory": str(dirs["cache_te_short"]), "weight": 0.3},
        {"name": "medium", "directory": str(dirs["cache_te_medium"]), "weight": 0.2},
    ]


def _build(root: Path, dirs, variants=None, seed=1):
    dataset = {"image_jsonl_file": str(root / "train.jsonl"), "cache_directory": str(dirs["cache"]), "num_repeats": 1}
    if variants is not None:
        dataset["text_encoder_cache_base_weight"] = 0.5
        dataset["text_encoder_cache_variants"] = variants
    user_config = {
        "general": {"resolution": [512, 512], "batch_size": 4, "enable_bucket": True, "bucket_no_upscale": False},
        "datasets": [dataset],
    }
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, argparse.Namespace(debug_dataset=False), architecture="qi21")
    random.seed(seed)
    group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True, shared_epoch=multiprocessing.Value("i", 0))
    return group.datasets[0]


def _epoch_lengths(dataset, epoch):
    dataset.current_epoch = epoch
    dataset.shuffle_buckets()
    manager = dataset.batch_manager
    return [len(t) for i in range(len(manager)) for t in manager[i]["vl21_embed"]]


@pytest.fixture()
def caches(tmp_path):
    return tmp_path, _write_caches(tmp_path)


def test_variant_draws_follow_weights_renormalised_over_available_caches(caches):
    root, dirs = caches
    dataset = _build(root, dirs, _variants(dirs))
    counts = collections.Counter()
    for epoch in range(1, 41):
        counts.update(_epoch_lengths(dataset, epoch))
    total = sum(counts.values())
    assert total == 40 * N_ITEMS
    # 80 base-only items; 60 with {base .5, short .3}; 60 with {base .5, short .3, medium .2}
    expected = {BASE_LEN: (80 + 60 * 0.5 / 0.8 + 60 * 0.5) / N_ITEMS, SHORT_LEN: (60 * 0.3 / 0.8 + 60 * 0.3) / N_ITEMS, MEDIUM_LEN: 60 * 0.2 / N_ITEMS}
    for length, share in expected.items():
        assert abs(counts[length] / total - share) < 0.02


def test_choices_agree_across_workers_and_change_across_epochs(caches):
    root, dirs = caches
    first, second = _build(root, dirs, _variants(dirs)), _build(root, dirs, _variants(dirs))
    assert [_epoch_lengths(first, e) for e in (1, 2, 3)] == [_epoch_lengths(second, e) for e in (1, 2, 3)]
    choice_by_key = lambda d: {it.item_key: d.batch_manager.te_choice[r][i] for r, b in d.batch_manager.buckets.items() for i, it in enumerate(b)}
    epoch3 = choice_by_key(first)
    _epoch_lengths(first, 4)
    assert sum(epoch3[k] != v for k, v in choice_by_key(first).items()) > 20


def test_items_without_variants_keep_the_base_cache(caches):
    root, dirs = caches
    dataset = _build(root, dirs, _variants(dirs))
    _epoch_lengths(dataset, 1)
    for reso, bucket in dataset.batch_manager.buckets.items():
        for position, item in enumerate(bucket):
            if int(item.item_key) >= WITH_SHORT:
                assert item.te_variant_paths is None and dataset.batch_manager.te_choice[reso][position] == 0


def test_without_variants_order_and_captions_are_unchanged(caches):
    root, dirs = caches
    plain, varied = _build(root, dirs), _build(root, dirs, _variants(dirs))
    assert plain.batch_manager.has_te_variants is False
    assert set(_epoch_lengths(plain, 2)) == {BASE_LEN}
    _epoch_lengths(varied, 2)
    order = lambda d: [it.item_key for r in d.batch_manager.bucket_resos for it in d.batch_manager.buckets[r]]
    assert order(plain) == order(varied)
    assert plain.batch_manager.bucket_batch_indices == varied.batch_manager.bucket_batch_indices


def test_variant_metadata_is_recorded(caches):
    root, dirs = caches
    metadata = _build(root, dirs, _variants(dirs)).get_metadata()
    assert metadata["text_encoder_cache_base_weight"] == 0.5
    assert [v["name"] for v in metadata["text_encoder_cache_variants"]] == ["short", "medium"]


def test_variant_directory_must_differ_from_cache_directory(caches):
    root, dirs = caches
    with pytest.raises(ValueError, match="must differ"):
        _build(root, dirs, [{"name": "bad", "directory": str(dirs["cache"]), "weight": 1}])
