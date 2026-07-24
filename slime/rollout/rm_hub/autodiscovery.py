"""Bayesian-surprise reward via an asta-autodiscovery reward server.

The policy generates a single experiment hypothesis per rollout. This RM posts
the hypothesis to a long-lived autodiscovery reward server, which plans and
executes the experiment (planner -> code execution -> analysis -> review),
elicits prior vs. posterior beliefs given the evidence, and returns the scalar
surprisal reward already shaped server-side (``get_self_value``). One rollout ==
one experiment node; no MCTS tree is involved.

slime needs no autodiscovery dependency: it only speaks HTTP to the server
(``python -m autodiscovery.slime_reward``). Contract::

    POST <args.rm_url>          (point --rm-url at the server's /reward route)
    request:  {"hypothesis": str, "dataset_id": str, "request_id": str}
    response: {"reward": float, "success": bool, "surprising": bool | None,
               "belief_change": float | None, "kl_divergence": float | None,
               "prior_mean": ..., "posterior_mean": ..., "error": str | None}

The server is idempotent on ``request_id`` (scoring runs a real, multi-minute
experiment), so the HTTP retries built into ``post`` never re-execute it.

Wiring::

    --custom-rm-path slime.rollout.rm_hub.autodiscovery.autodiscovery_rm
    --rm-url http://<reward-server>:<port>/reward

The reward is the scalar the server returns under ``reward`` (the server owns
reward shaping and the failure reward via its own config). Prompt data must
carry a ``dataset_id`` in each sample's metadata (``--metadata-key``) so the
server can route to the right dataset.
"""

import asyncio
import json
import logging
import os
import re
import uuid

from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Reward returned when we can't even reach the server with a valid request
# (truncated/empty response, missing dataset_id, transport error). Distinct
# from a scored-but-failed experiment, whose reward the server assigns.
FAILURE_REWARD = float(os.environ.get("AUTODISCOVERY_FAILURE_REWARD", "0.0"))

# Scoring runs a real experiment per request; keep retries low and rely on the
# server-side request_id dedupe rather than hammering it.
MAX_RETRIES = int(os.environ.get("AUTODISCOVERY_RM_MAX_RETRIES", "3"))

_THINK_CLOSE = "</think>"
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_hypothesis(response: str) -> str:
    """Pull the hypothesis text out of a raw model response.

    Handles reasoning-model output (drops everything up to the last
    ``</think>``) and the theorizer's ``HypothesisList`` JSON schema
    (``{"hypotheses": [...]}}`` with string or ``{"hypothesis": ...}`` elements,
    first entry wins). Falls back to the stripped raw text so that early,
    format-unaware checkpoints still receive a meaningful score.
    """
    text = response
    if _THINK_CLOSE in text:
        text = text.rsplit(_THINK_CLOSE, 1)[1]
    text = text.strip()

    match = _JSON_BLOCK_RE.search(text)
    if match is not None:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("hypothesis"), str) and parsed["hypothesis"].strip():
                return parsed["hypothesis"].strip()
            hypotheses = parsed.get("hypotheses")
            if isinstance(hypotheses, list) and hypotheses:
                first = hypotheses[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict) and isinstance(first.get("hypothesis"), str):
                    return first["hypothesis"].strip()
    return text


async def autodiscovery_rm(args, sample: Sample | list[Sample], **kwargs) -> float | list[float]:
    # batched_async_rm (--group-rm / fan-out generate) calls the custom RM with
    # the whole list; fan back out so both arities work.
    if isinstance(sample, list):
        return await asyncio.gather(*(autodiscovery_rm(args, s, **kwargs) for s in sample))

    # A truncated response has no complete hypothesis; don't burn minutes of
    # sandbox time scoring it.
    if sample.status == Sample.Status.TRUNCATED:
        return FAILURE_REWARD

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    dataset_id = metadata.get("dataset_id")
    if not dataset_id:
        raise ValueError(
            "autodiscovery_rm requires sample.metadata['dataset_id']; add a "
            "metadata column with the dataset id to the prompt data (--metadata-key)."
        )

    hypothesis = extract_hypothesis(sample.response)
    if not hypothesis:
        return FAILURE_REWARD

    payload = {
        "hypothesis": hypothesis,
        "dataset_id": dataset_id,
        # Stable across the HTTP retries in post() so the server dedupes instead
        # of re-running the experiment.
        "request_id": f"{sample.rollout_id if sample.rollout_id is not None else sample.index}-{uuid.uuid4().hex[:8]}",
    }
    try:
        result = await post(args.rm_url, payload, max_retries=MAX_RETRIES)
    except Exception as e:  # noqa: BLE001 - a reward failure must not crash rollout
        logger.warning(f"autodiscovery_rm: scoring request failed, using failure reward: {e}")
        return FAILURE_REWARD

    # The server owns reward shaping and its own failed-experiment reward, so
    # trust whatever scalar it returns under "reward".
    reward = result.get("reward")
    if reward is None:
        logger.warning(f"autodiscovery_rm: server returned no reward: {result}")
        return FAILURE_REWARD
    return float(reward)
