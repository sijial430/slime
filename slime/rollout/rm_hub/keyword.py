"""Control reward: 1.0 iff the hypothesis matches a keyword pattern.

Learnable-signal sanity check for the AutoDiscovery RL setup: unlike ``coin``
(pure noise) this reward is deterministic in the policy's output — reward 1.0
when the extracted hypothesis matches ``AUTODISCOVERY_KEYWORD_PATTERN``
(case-insensitive regex, default ``pottery\\s*form`` so "pottery form(s)"
matches), else 0.0. A working GRPO loop should push the policy toward
mentioning the keyword within a few epochs; failure to climb isolates
optimizer/plumbing problems from reward-model problems.

Samples with no extractable hypothesis score 0.0 — a degenerate response does
not mention the keyword in a usable hypothesis.

No experiment is run and no reward server is contacted.

Wiring::

    --custom-rm-path slime.rollout.rm_hub.keyword.reward
"""

import asyncio
import os
import re

from slime.utils.types import Sample

from .reward_utils import dump_record, extract_hypothesis, sample_base_record

CONTROL = "keyword"
PATTERN = re.compile(os.environ.get("AUTODISCOVERY_KEYWORD_PATTERN", r"pottery\s*form"), re.IGNORECASE)


async def reward(args, sample: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(sample, list):
        return await asyncio.gather(*(reward(args, s, **kwargs) for s in sample))
    hypothesis = extract_hypothesis(sample.response or "") or None
    matched = bool(hypothesis and PATTERN.search(hypothesis))
    rew = 1.0 if matched else 0.0
    dump_record({
        **sample_base_record(sample, CONTROL),
        "hypothesis": hypothesis,
        "reward": rew,
        "success": None,
        "error": None,
        "keyword_pattern": PATTERN.pattern,
        "keyword_matched": matched,
    })
    return rew
