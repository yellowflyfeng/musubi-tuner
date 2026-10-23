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


# --- cycle schedule: every group is visited before any group repeats ---------------------------------

GROUP_LENGTHS = {"long": 40, "short": 3, "medium_1": 7, "medium_2": 8, "medium_3": 9, "tiny": 2}
MEDIUM = {GROUP_LENGTHS[m] for m in ("medium_1", "medium_2", "medium_3")}
group_of = lambda n: "medium" if n in MEDIUM else n


def _write_grouped_caches(root: Path, n_items: int = 120) -> dict[str, Path]:
    dirs = {name: root / ("cache" if name == "long" else f"cache_te_{name}") for name in GROUP_LENGTHS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    with (root / "train.jsonl").open("w") as f:
        for i in range(n_items):
            key = f"{i:07d}"
            f.write(json.dumps({"image_path": f"/x/images/{key}.jpg", "caption": "long"}) + "\n")
            save_file({"latents_1x64x64_bfloat16": torch.zeros(1, 64, 64, dtype=torch.bfloat16)}, str(dirs["long"] / f"{key}_0512x0512_qi21.safetensors"))
            for name, length in GROUP_LENGTHS.items():
                if name == "medium_3" and i % 3 == 0:
                    continue  # some items lack a medium (omitted by caption QA)
                te = {"varlen_vl21_embed_bfloat16": torch.zeros(length, 8, dtype=torch.bfloat16)}
                save_file(te, str(dirs[name] / f"{key}_qi21_te.safetensors"))
    return dirs


def _build_cycle(root: Path, dirs, num_repeats=1, tiny=True):
    dataset = {
        "image_jsonl_file": str(root / "train.jsonl"),
        "cache_directory": str(dirs["long"]),
        "num_repeats": num_repeats,
        "text_encoder_cache_schedule": "cycle",
        "text_encoder_cache_variants": [
            {"name": "short", "directory": str(dirs["short"])},
            *({"name": m, "directory": str(dirs[m]), "group": "medium"} for m in ("medium_1", "medium_2", "medium_3")),
            *([{"name": "tiny", "directory": str(dirs["tiny"])}] if tiny else []),
        ],
    }
    user_config = {"general": {"resolution": [512, 512], "batch_size": 4, "enable_bucket": True, "bucket_no_upscale": False}, "datasets": [dataset]}
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, argparse.Namespace(debug_dataset=False), architecture="qi21")
    random.seed(7)
    group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True, shared_epoch=multiprocessing.Value("i", 0))
    return group.datasets[0]


def _lengths_by_item(dataset, epoch):
    dataset.current_epoch = epoch
    dataset.shuffle_buckets()
    manager, out = dataset.batch_manager, collections.defaultdict(list)
    for i in range(len(manager)):
        reso, batch_idx = manager.bucket_batch_indices[i]
        keys = [it.item_key for it in manager.buckets[reso][batch_idx * manager.batch_size : (batch_idx + 1) * manager.batch_size]]
        for key, te in zip(keys, manager[i]["vl21_embed"]):
            out[key].append(len(te))
    return out


def _groups_seen(dataset, epochs):
    seen = collections.defaultdict(list)
    for epoch in epochs:
        for key, lengths in _lengths_by_item(dataset, epoch).items():
            seen[key] += [group_of(n) for n in lengths]
    return seen


@pytest.mark.parametrize("tiny", [True, False])
def test_cycle_covers_every_group_once_per_round(tmp_path, tiny):
    # long / short / medium / tiny: four epochs see each group exactly once; without tiny, three epochs do
    dataset = _build_cycle(tmp_path, _write_grouped_caches(tmp_path), tiny=tiny)
    n_groups = 4 if tiny else 3
    for key, groups in _groups_seen(dataset, range(1, n_groups + 1)).items():
        assert len(groups) == n_groups and len(set(groups)) == n_groups


def test_cycle_mediums_take_turns_across_rounds(tmp_path):
    dataset = _build_cycle(tmp_path, _write_grouped_caches(tmp_path), tiny=False)
    seen = collections.defaultdict(list)
    for epoch in range(1, 7):
        for key, lengths in _lengths_by_item(dataset, epoch).items():
            seen[key] += [n for n in lengths if n in MEDIUM]
    for key, mediums in seen.items():
        assert len(mediums) == 2 and mediums[0] != mediums[1]


@pytest.mark.parametrize("tiny", [True, False])
def test_cycle_is_reproducible_and_balanced_per_epoch(tmp_path, tiny):
    dirs = _write_grouped_caches(tmp_path)
    first, second = _build_cycle(tmp_path, dirs, tiny=tiny), _build_cycle(tmp_path, dirs, tiny=tiny)
    n_groups = 4 if tiny else 3
    for epoch in (1, 2, 3, 4):
        a, b = _lengths_by_item(first, epoch), _lengths_by_item(second, epoch)
        assert a == b
        groups = collections.Counter(group_of(v[0]) for v in a.values())
        # every epoch mixes all groups: about 1/n_groups of the 120 items each
        assert len(groups) == n_groups and all(120 / n_groups / 2 <= c <= 120 / n_groups * 1.6 for c in groups.values())


def test_cycle_repeats_take_successive_positions(tmp_path):
    dataset = _build_cycle(tmp_path, _write_grouped_caches(tmp_path, n_items=40), num_repeats=2)
    for key, lengths in _lengths_by_item(dataset, 1).items():
        assert len(lengths) == 2 and lengths[0] != lengths[1]


def test_cycle_batches_mix_caption_lengths_in_every_epoch(tmp_path):
    dataset = _build_cycle(tmp_path, _write_grouped_caches(tmp_path))
    for epoch in (1, 2, 3, 4):
        dataset.current_epoch = epoch
        dataset.shuffle_buckets()
        manager = dataset.batch_manager
        batches = [{group_of(len(te)) for te in manager[i]["vl21_embed"]} for i in range(len(manager))]
        uniform = sum(len(b) == 1 for b in batches)
        assert uniform <= len(batches) // 4  # independent per-item positions: ~7% of 4-image batches by chance
