"""Control reward: each sample independently scores 1.0 with probability p.

Noise-signal ablation for the AutoDiscovery RL setup: rewards are i.i.d.
Bernoulli(``AUTODISCOVERY_COIN_P``, default 0.5) coin flips in {0, 1},
completely uncorrelated with the hypothesis. Unlike the ``zero``/``one``
controls this DOES produce nonzero GRPO advantages — pure noise gradients —
so it upper-bounds how much reward-hacking-shaped drift the optimizer can
extract from a signal-free reward at the configured lr/KL.

No experiment is run and no reward server is contacted; the flip applies
uniformly to all samples, including truncated ones.

Wiring::

    --custom-rm-path slime.rollout.rm_hub.coin.reward
"""

import asyncio
import os
import random

from slime.utils.types import Sample

from .reward_utils import dump_record, extract_hypothesis, sample_base_record

CONTROL = "coin"
COIN_P = float(os.environ.get("AUTODISCOVERY_COIN_P", "0.5"))


async def reward(args, sample: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(sample, list):
        return await asyncio.gather(*(reward(args, s, **kwargs) for s in sample))
    rew = 1.0 if random.random() < COIN_P else 0.0
    dump_record({
        **sample_base_record(sample, CONTROL),
        "hypothesis": extract_hypothesis(sample.response or "") or None,
        "reward": rew,
        "success": None,
        "error": None,
        "coin_p": COIN_P,
    })
    return rew
