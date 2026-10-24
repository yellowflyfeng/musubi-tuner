"""Tests for the `flux2_infer_shift` timestep-sampling mode.

flux2_infer_shift draws t from the distribution FLUX.2 inference visits: a uniform t warped by the time shift
exp(mu), where mu = flux2_utils.compute_empirical_mu(h * w, steps) depends on the token count and the step count.
A uniform draw q therefore lands exactly on the inference schedule value for 1 - q (the schedule runs 1 -> 0).

See flux2_utils.get_schedule (inference) and trainer_base.compute_sampling_timesteps.
"""

import argparse
import math
from types import SimpleNamespace

import torch

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.training import parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer


def _args(steps=50):
    return SimpleNamespace(timestep_sampling="flux2_infer_shift", flux2_infer_steps=steps, min_timestep=None, max_timestep=None,
                           preserve_distribution_shape=False, sigmoid_scale=1.0)


def _shifted(latent_h, latent_w, qs, steps=50):
    latents = torch.zeros(len(qs), 128, latent_h, latent_w, dtype=torch.float32)  # FLUX.2 latents are already patchified
    noisy, _ = NetworkTrainer().get_noisy_model_input_and_timesteps(
        _args(steps), torch.ones_like(latents), latents, list(qs), None, torch.device("cpu"), torch.float32
    )
    return noisy.flatten(1)[:, 0]  # noise=1, latents=0 -> noisy_model_input == t


def test_parser_accepts_flux2_infer_shift():
    parser = argparse.ArgumentParser()
    parser_common._add_timestep_args(parser)
    args = parser.parse_args(["--timestep_sampling", "flux2_infer_shift", "--flux2_infer_steps", "28"])
    assert args.timestep_sampling == "flux2_infer_shift" and args.flux2_infer_steps == 28


def test_flux2_infer_shift_matches_inference_schedule():
    h, w, steps = 77, 53, 50
    schedule = flux2_utils.get_schedule(steps, h * w)
    qs = [1 - i / steps for i in range(1, steps)]  # interior grid points of linspace(1, 0, steps + 1)
    t = _shifted(h, w, qs, steps)
    assert torch.allclose(t, torch.tensor(schedule[1:steps]), atol=1e-5)


def test_flux2_infer_shift_depends_on_step_count():
    q = [0.5]
    assert _shifted(64, 64, q, steps=50).item() < _shifted(64, 64, q, steps=10).item()  # fewer steps -> larger shift
    mu = flux2_utils.compute_empirical_mu(64 * 64, 50)
    assert math.isclose(_shifted(64, 64, q, steps=50).item(), math.exp(mu) / (math.exp(mu) + 1), rel_tol=1e-5)
