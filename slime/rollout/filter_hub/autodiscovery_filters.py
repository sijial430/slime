"""Dynamic-sampling filter for the autodiscovery surprisal reward.

Drops prompt groups that carry no learning signal before they reach training.
Under GRPO/GSPO the per-sample advantage is ``(reward - group_mean) /
group_std``, so a group whose ``n_samples_per_prompt`` hypotheses all scored the
same surprise contributes zero gradient. The common case is a group where every
experiment failed and collapsed to ``FAILURE_REWARD`` -- but a group of equally
(un)surprising hypotheses is just as signal-free.

This is the DAPO-style group filter (same core test as
``dynamic_sampling_filters.check_reward_nonzero_std``); it only adds
surprise-specific reason labels so the drops are visible in rollout metrics.
Note we filter at the GROUP level, never per-sample: a mixed group where some
experiments failed and some surprised is exactly the contrast GRPO needs, so it
is kept.

Wire with::

    --dynamic-sampling-filter-path slime.rollout.filter_hub.autodiscovery_filters.surprise_signal_filter
"""

import os

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.rm_hub.autodiscovery import FAILURE_REWARD
from slime.utils.types import Sample

__all__ = ["surprise_signal_filter"]

# Minimum within-group reward std for a group to be kept. Groups below this
# collapse to ~zero advantage under GRPO/GSPO normalization. Matches the 1e-6
# used by check_reward_nonzero_std; override to require a larger spread.
_MIN_STD = float(os.environ.get("AUTODISCOVERY_MIN_REWARD_STD", "1e-6"))


def surprise_signal_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    rewards = [sample.get_reward_value(args) for sample in samples]
    std = torch.tensor(rewards, dtype=torch.float64).std().item()
    if std > _MIN_STD:
        return DynamicFilterOutput(keep=True, reason=None)

    # No within-group spread -> no gradient. Label why for rollout telemetry.
    if all(abs(r - FAILURE_REWARD) < 1e-9 for r in rewards):
        reason = "all_failed"
    else:
        reason = f"uniform_surprise_{round(rewards[0], 2)}"
    return DynamicFilterOutput(keep=False, reason=reason)
