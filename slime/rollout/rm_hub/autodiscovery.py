"""Bayesian-surprise reward via an asta-autodiscovery reward server.

The policy generates a single experiment hypothesis per rollout. This RM posts
the hypothesis to a long-lived autodiscovery scoring service, which plans and
executes the experiment (planner -> code execution -> analysis -> review) and
returns Bayesian surprise (prior vs. posterior belief shift given the
experimental evidence). One rollout == one node; no MCTS tree is involved.

Contract with the reward server::

    POST <args.rm_url>          (e.g. http://host:port/score)
    request:  {"hypothesis": str, "dataset_id": str, "request_id": str}
    response: {"success": bool,
               "normalized_surprisal": float | None,   # signed, ~[-1, 1]; boolean_cat only
               "belief_change": float | None,          # |posterior - prior|
               "kl_divergence": float | None,
               "self_value": float | None,
               "failure_reason": str | None}

The server must be idempotent on ``request_id``: scoring runs a real experiment
(minutes of sandbox + LLM calls), so HTTP retries here must not re-execute it.

Wiring::

    --custom-rm-path slime.rollout.rm_hub.autodiscovery.autodiscovery_rm
    --rm-url http://<reward-server>:<port>/score
    --reward-key reward

The returned reward is a dict; ``reward`` resolves to ``normalized_surprisal``
when available, else ``self_value``, else the failure reward.
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

# Reward assigned when the experiment fails (code execution error, unfaithful
# plan, or belief elicitation failure). Distinct from "ran fine but not
# surprising" (which scores ~0), so consider a negative value.
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
    (``{"hypotheses": [...]}} `` with string or ``{"hypothesis": ...}``
    elements, first entry wins). Falls back to the stripped raw text so that
    early, format-unaware checkpoints still receive a meaningful score.
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
            hypotheses = parsed.get("hypotheses")
            if isinstance(hypotheses, list) and hypotheses:
                first = hypotheses[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict) and isinstance(first.get("hypothesis"), str):
                    return first["hypothesis"].strip()
    return text


def _failure_reward_dict(reason: str) -> dict:
    return {
        "reward": FAILURE_REWARD,
        "success": 0.0,
        "normalized_surprisal": None,
        "belief_change": None,
        "kl_divergence": None,
        "self_value": None,
        "failure_reason": reason,
    }


async def autodiscovery_rm(args, sample: Sample | list[Sample], **kwargs):
    # batched_async_rm (--group-rm / fan-out generate) calls the custom RM with
    # the whole list; fan back out so both arities work.
    if isinstance(sample, list):
        return await asyncio.gather(*(autodiscovery_rm(args, s, **kwargs) for s in sample))

    # A truncated response has no complete hypothesis; don't burn minutes of
    # sandbox time scoring it.
    if sample.status == Sample.Status.TRUNCATED:
        return _failure_reward_dict("truncated_response")

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    dataset_id = metadata.get("dataset_id")
    if not dataset_id:
        raise ValueError(
            "autodiscovery_rm requires sample.metadata['dataset_id']; "
            "add a metadata column with the dataset id to the prompt data."
        )

    hypothesis = extract_hypothesis(sample.response)
    if not hypothesis:
        return _failure_reward_dict("empty_hypothesis")

    payload = {
        "hypothesis": hypothesis,
        "dataset_id": dataset_id,
        # Stable across HTTP retries so the server can dedupe instead of
        # re-running the experiment.
        "request_id": f"{sample.rollout_id if sample.rollout_id is not None else sample.index}-{uuid.uuid4().hex[:8]}",
    }
    result = await post(args.rm_url, payload, max_retries=MAX_RETRIES)

    if not result.get("success"):
        return _failure_reward_dict(result.get("failure_reason") or "experiment_failed")

    normalized_surprisal = result.get("normalized_surprisal")
    self_value = result.get("self_value")
    # normalized_surprisal is only defined for belief_mode="boolean_cat";
    # fall back to self_value rather than silently rewarding 0.
    if normalized_surprisal is not None:
        reward = float(normalized_surprisal)
    elif self_value is not None:
        reward = float(self_value)
    else:
        return _failure_reward_dict("no_score_returned")

    return {
        "reward": reward,
        "success": 1.0,
        "normalized_surprisal": normalized_surprisal,
        "belief_change": result.get("belief_change"),
        "kl_divergence": result.get("kl_divergence"),
        "self_value": self_value,
        "failure_reason": None,
    }
