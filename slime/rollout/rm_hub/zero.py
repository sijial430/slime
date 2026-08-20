"""Control reward: every sample scores 0.0.

Null-signal ablation for the AutoDiscovery RL setup: no experiment is run, no
reward server is contacted, and the reward is uniform across ALL samples
(including truncated ones) — so GRPO advantages are exactly zero and any
policy drift observed in such a run is attributable to the KL/optimizer
dynamics, not the reward.

Wiring::

    --custom-rm-path slime.rollout.rm_hub.zero.reward

The per-rollout dump (``AUTODISCOVERY_RM_DUMP``) still receives one record per
sample, tagged ``"control": "zero"``, so runs stay auditable.
"""

import asyncio

from slime.utils.types import Sample

from .reward_utils import dump_record, extract_hypothesis, sample_base_record

CONTROL = "zero"
REWARD_VALUE = 0.0


async def reward(args, sample: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(sample, list):
        return await asyncio.gather(*(reward(args, s, **kwargs) for s in sample))
    dump_record({
        **sample_base_record(sample, CONTROL),
        "hypothesis": extract_hypothesis(sample.response or "") or None,
        "reward": REWARD_VALUE,
        "success": None,
        "error": None,
    })
    return REWARD_VALUE
