"""Control reward: every sample scores 1.0.

Null-signal ablation for the AutoDiscovery RL setup: no experiment is run, no
reward server is contacted, and the reward is uniform across ALL samples
(including truncated ones) — so GRPO advantages are exactly zero and any
policy drift observed in such a run is attributable to the KL/optimizer
dynamics, not the reward. Identical to the ``zero`` control except for the
constant, which distinguishes reward-scale effects from reward-signal effects.

Wiring::

    --custom-rm-path slime.rollout.rm_hub.one.reward
"""

import asyncio

from slime.utils.types import Sample

from .reward_utils import dump_record, extract_hypothesis, sample_base_record

CONTROL = "one"
REWARD_VALUE = 1.0


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
