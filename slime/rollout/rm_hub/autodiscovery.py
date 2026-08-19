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
experiment), so transient (5xx/network) retries never re-execute it. A 4xx
(e.g. unknown ``dataset_id``) is permanent: it is not retried, logs a grep-able
``MISSING_DATASET dataset_id=<id>`` line, and falls back to the failure reward.

Wiring::

    --custom-rm-path slime.rollout.rm_hub.autodiscovery.autodiscovery_rm
    --rm-url http://<reward-server>:<port>/reward

The reward is the scalar the server returns under ``reward`` (the server owns
reward shaping and the failure reward via its own config). Prompt data must
carry a ``dataset_id`` in each sample's metadata (``--metadata-key``) so the
server can route to the right dataset.
"""

import asyncio
import fcntl
import json
import logging
import os
import random
import re
import uuid

import aiohttp

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Optional per-rollout record dump. When AUTODISCOVERY_RM_DUMP is set, every
# scored (or failed) sample appends one JSON line with the reward, surprise
# diagnostics, the execution log (only if the server returns one, i.e. it was
# started with --include_execution_log), and the identifying keys
# (rollout_id, dataset_id, data_fmt). Rollout workers are separate processes
# all appending to the same file, so guard each write with an flock.
_DUMP_PATH = os.environ.get("AUTODISCOVERY_RM_DUMP") or None
# execution_log can be enormous (the server's full multi-agent trace). Cap it in
# the dump so /results doesn't balloon to GBs; 0 disables the cap.
_DUMP_MAXLOG = int(os.environ.get("AUTODISCOVERY_RM_DUMP_MAXLOG", "100000"))


def _dump_record(record: dict) -> None:
    """Append one JSON line to the dump file (cross-process safe). Never raises."""
    if not _DUMP_PATH:
        return
    try:
        # Cap the large free-text fields so /results doesn't balloon to GBs.
        for key in ("execution_log", "reasoning"):
            v = record.get(key)
            if _DUMP_MAXLOG and isinstance(v, str) and len(v) > _DUMP_MAXLOG:
                record = {**record, key: v[:_DUMP_MAXLOG], f"{key}_truncated_from": len(v)}
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(_DUMP_PATH, "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception as e:  # noqa: BLE001 - dumping must never break a rollout
        logger.warning(f"[autodiscovery_rm] dump failed: {e}")

# Reward returned when we can't even reach the server with a valid request
# (truncated/empty response, missing dataset_id, transport error). Distinct
# from a scored-but-failed experiment, whose reward the server assigns.
FAILURE_REWARD = float(os.environ.get("AUTODISCOVERY_FAILURE_REWARD", "0.0"))

# Scoring runs a real experiment per request; keep retries low and rely on the
# server-side request_id dedupe rather than hammering it. Only transient
# (5xx/network) errors are retried; 4xx are permanent and never retried.
MAX_RETRIES = int(os.environ.get("AUTODISCOVERY_RM_MAX_RETRIES", "3"))
# Reward calls are minutes long -> no total timeout, but bound each socket read
# so a hung server can't wedge a rollout forever.
REWARD_SOCK_READ_TIMEOUT = float(os.environ.get("AUTODISCOVERY_RM_SOCK_TIMEOUT", "1200"))

_THINK_CLOSE = "</think>"
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_session: aiohttp.ClientSession | None = None


def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, enable_cleanup_closed=True),
            timeout=aiohttp.ClientTimeout(total=None, sock_read=REWARD_SOCK_READ_TIMEOUT),
        )
    return _session


class _PermanentRewardError(Exception):
    """A 4xx from the reward server: retrying cannot help (e.g. unknown dataset_id)."""


async def _post_reward(url: str, payload: dict) -> dict:
    """POST the reward request. Retry 5xx/network up to MAX_RETRIES; never retry 4xx."""
    session = _get_session()
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status < 300:
                    return await resp.json()
                body = (await resp.text())[:300]
                if 400 <= resp.status < 500:
                    raise _PermanentRewardError(f"HTTP {resp.status}: {body}")
                last_exc = RuntimeError(f"HTTP {resp.status}: {body}")  # 5xx -> retry
        except _PermanentRewardError:
            raise
        except Exception as e:  # noqa: BLE001 - network/5xx: retry
            last_exc = e
        if attempt + 1 < MAX_RETRIES:
            await asyncio.sleep(min(2**attempt, 30) + random.random())
    raise last_exc if last_exc is not None else RuntimeError("reward request failed")


# --- Rollout feedback: fold prior-rollout learnings into the next generation ---
# When AUTODISCOVERY_FEEDBACK_STATE is set (a JSONL path on shared disk), every
# scored sample appends {dataset_id, hypothesis, reward, learnings}, where
# "learnings" is an LLM summary of the experiment's Analysis section (fallback:
# truncated raw text). The paired --custom-generate-function-path target
# `generate_with_feedback` then injects the best entries from the PREVIOUS
# rollout step into the user prompt before delegating to slime's default
# generate. Off (both sides no-op) when the env var is unset.
_FEEDBACK_STATE = os.environ.get("AUTODISCOVERY_FEEDBACK_STATE") or None
_FEEDBACK_K = int(os.environ.get("AUTODISCOVERY_FEEDBACK_K", "1"))
_FEEDBACK_SUMMARY_MODEL = os.environ.get("AUTODISCOVERY_FEEDBACK_SUMMARY_MODEL", "gpt-5-mini")
# Fallback truncation length for learnings when no API key / summary failure.
_FEEDBACK_MAXCHARS = int(os.environ.get("AUTODISCOVERY_FEEDBACK_MAXCHARS", "1000"))

_SUMMARIZE_SYS = (
    "You compress data-analysis reports into task-specific learnings for the next "
    "experiment on the SAME dataset. From the report, extract at most 3 short "
    "bullet points covering: data quirks that must be handled (columns, types, "
    "cleaning steps), what analysis choice worked or failed, and the substantive "
    "verdict (supported / refuted / inconclusive, with the key number if any). "
    "Under 100 words total. Output only the bullets."
)


def _extract_analysis(execution_log: str | None) -> str:
    """The Analysis section of the server's experiment trace ('' if absent)."""
    if not execution_log or "\nAnalysis:\n" not in execution_log:
        return ""
    seg = execution_log.split("\nAnalysis:\n", 1)[1]
    return seg.split("\nReview:\n", 1)[0].strip()


async def _summarize_learnings(analysis: str, error: str | None) -> str:
    """LLM-compress an analysis into learnings; degrade gracefully to truncation."""
    if not analysis:
        return f"(no analysis produced{'; ' + error if error else ''})"
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            session = _get_session()
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": _FEEDBACK_SUMMARY_MODEL,
                    "messages": [
                        {"role": "system", "content": _SUMMARIZE_SYS},
                        {"role": "user", "content": analysis[:12000]},
                    ],
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status < 300:
                    data = await resp.json()
                    text = (data["choices"][0]["message"]["content"] or "").strip()
                    if text:
                        return text
                logger.warning(f"[feedback] summarizer HTTP {resp.status}; falling back to truncation")
        except Exception as e:  # noqa: BLE001 - feedback must never break a rollout
            logger.warning(f"[feedback] summarizer failed ({e}); falling back to truncation")
    return analysis[:_FEEDBACK_MAXCHARS]


def _append_feedback(entry: dict) -> None:
    """Append one JSONL feedback entry (cross-process safe). Never raises."""
    if not _FEEDBACK_STATE:
        return
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with open(_FEEDBACK_STATE, "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[feedback] state append failed: {e}")


async def _record_feedback(dataset_id, hypothesis, reward, execution_log=None, error=None) -> None:
    if not _FEEDBACK_STATE or not dataset_id:
        return
    if error in ("truncated_response", "empty_hypothesis"):
        learnings = "(previous response exceeded the token limit or contained no parseable hypothesis — no experiment was run; keep the response focused and within budget)"
    else:
        learnings = await _summarize_learnings(_extract_analysis(execution_log), error)
    _append_feedback({
        "dataset_id": dataset_id,
        "hypothesis": (hypothesis or "")[:600],
        "reward": None if reward is None else float(reward),
        "learnings": learnings,
    })


def _recent_feedback(dataset_id: str, group_size: int, k: int) -> list[dict]:
    """Top-k entries (by reward, then recency) among the last `group_size`
    feedback lines for this dataset — i.e. the previous rollout step's group."""
    if not _FEEDBACK_STATE or not os.path.exists(_FEEDBACK_STATE):
        return []
    try:
        entries = []
        with open(_FEEDBACK_STATE, encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("dataset_id") == dataset_id:
                    entries.append(e)
        tail = entries[-group_size:]
        tail.sort(key=lambda e: (e.get("reward") or 0.0), reverse=True)
        return tail[:k]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[feedback] state read failed: {e}")
        return []


_GEN_SUFFIX_MARKER = os.environ.get("AUTODISCOVERY_FEEDBACK_GEN_MARKER", "<|im_start|>assistant")


def _format_feedback_block(entries: list[dict]) -> str:
    lines = ["Feedback from your previous attempt(s) on this task (best first):"]
    for i, e in enumerate(entries, 1):
        rew = e.get("reward")
        lines.append(f"{i}. prior hypothesis: {e.get('hypothesis', '')}")
        lines.append(f"   prior reward: {'n/a' if rew is None else f'{rew:.3f}'}")
        lines.append(f"   learnings from its analysis: {e.get('learnings', '')}")
    lines.append(
        "Apply these learnings (especially data-handling quirks), and propose a "
        "hypothesis that is NOT a repeat of the prior ones."
    )
    return "\n".join(lines)


def _inject_feedback_into_prompt(prompt, block: str):
    """Insert the feedback block at the end of the final user message.

    Handles both the chat-templated string form (insert before the trailing
    generation suffix, e.g. Qwen's `<|im_start|>assistant`) and the raw
    message-list form (append to the last user message)."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = f"{msg.get('content', '')}\n\n{block}"
                return prompt
        return prompt
    idx = prompt.rfind(_GEN_SUFFIX_MARKER)
    if idx < 0:
        return prompt + "\n\n" + block
    j = prompt.rfind("<|im_end|>", 0, idx)
    insert_at = j if j >= 0 else idx
    return prompt[:insert_at] + "\n\n" + block + prompt[insert_at:]


async def generate_with_feedback(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """``--custom-generate-function-path`` target.

    Injects the previous rollout step's best (hypothesis, reward, learnings)
    into the prompt, then delegates to slime's default generate. Eval prompts
    are left untouched so eval stays comparable across steps."""
    from slime.rollout.sglang_rollout import generate  # lazy: avoid import cycle

    try:
        if not evaluation and _FEEDBACK_STATE:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            dataset_id = metadata.get("dataset_id")
            if dataset_id:
                group = int(getattr(args, "n_samples_per_prompt", 1) or 1)
                entries = _recent_feedback(dataset_id, group, _FEEDBACK_K)
                if entries:
                    sample.prompt = _inject_feedback_into_prompt(
                        sample.prompt, _format_feedback_block(entries)
                    )
                    sample.tokens = []  # force re-encoding of the modified prompt
    except Exception as e:  # noqa: BLE001 - feedback must never break generation
        logger.warning(f"[feedback] prompt injection failed ({e}); generating without it")
    return await generate(args, sample, sampling_params)


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


def extract_reasoning(response: str) -> str:
    """Return the model's reasoning trace: everything before the last ``</think>``.

    The hypothesis is the text *after* the last ``</think>`` (see
    ``extract_hypothesis``); this returns what precedes it, with a leading
    ``<think>`` stripped. Empty string when the response has no think block.
    """
    if _THINK_CLOSE not in response:
        return ""
    reasoning = response.rsplit(_THINK_CLOSE, 1)[0].lstrip()
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>"):]
    return reasoning.strip()


async def autodiscovery_rm(args, sample: Sample | list[Sample], **kwargs) -> float | list[float]:
    # batched_async_rm (--group-rm / fan-out generate) calls the custom RM with
    # the whole list; fan back out so both arities work.
    if isinstance(sample, list):
        return await asyncio.gather(*(autodiscovery_rm(args, s, **kwargs) for s in sample))

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    dataset_id = metadata.get("dataset_id")
    # Identifying keys carried on every dump record so the JSONL can be joined
    # back to the prompt data / training step.
    rollout_id = sample.rollout_id if sample.rollout_id is not None else None
    base = {
        "rollout_id": rollout_id,
        "index": sample.index,
        "dataset_id": dataset_id,
        "data_fmt": metadata.get("format"),
        # The model's reasoning trace (tokens before the last </think>), logged
        # alongside the hypothesis + reward so the two can be inspected together.
        "reasoning": extract_reasoning(sample.response),
    }

    # A truncated response has no complete hypothesis; don't burn minutes of
    # sandbox time scoring it.
    if sample.status == Sample.Status.TRUNCATED:
        _dump_record({**base, "reward": FAILURE_REWARD, "success": False, "error": "truncated_response"})
        await _record_feedback(dataset_id, None, FAILURE_REWARD, error="truncated_response")
        return FAILURE_REWARD

    if not dataset_id:
        raise ValueError(
            "autodiscovery_rm requires sample.metadata['dataset_id']; add a "
            "metadata column with the dataset id to the prompt data (--metadata-key)."
        )

    hypothesis = extract_hypothesis(sample.response)
    if not hypothesis:
        _dump_record({**base, "reward": FAILURE_REWARD, "success": False, "error": "empty_hypothesis"})
        await _record_feedback(dataset_id, None, FAILURE_REWARD, error="empty_hypothesis")
        return FAILURE_REWARD
    base["hypothesis"] = hypothesis

    payload = {
        "hypothesis": hypothesis,
        "dataset_id": dataset_id,
        # Stable across the HTTP retries in post() so the server dedupes instead
        # of re-running the experiment.
        "request_id": f"{rollout_id if rollout_id is not None else sample.index}-{uuid.uuid4().hex[:8]}",
    }
    try:
        result = await _post_reward(args.rm_url, payload)
    except _PermanentRewardError as e:
        # 4xx: the server cannot score this dataset_id (most often it is not in
        # the reward-server registry). Do NOT retry. Log grep-ably so the missing
        # datasets can be collected (`grep MISSING_DATASET`) and added back to the
        # registry / S3 later.
        logger.warning(
            f"[autodiscovery_rm] MISSING_DATASET dataset_id={dataset_id} "
            f"(reward server rejected 4xx: {e}); returning failure reward"
        )
        _dump_record({**base, "reward": FAILURE_REWARD, "success": False, "error": f"missing_dataset: {e}"})
        return FAILURE_REWARD
    except Exception as e:  # noqa: BLE001 - a reward failure must not crash rollout
        logger.warning(f"autodiscovery_rm: scoring request failed after retries, failure reward: {e}")
        _dump_record({**base, "reward": FAILURE_REWARD, "success": False, "error": f"request_failed: {e}"})
        return FAILURE_REWARD

    # The server owns reward shaping and its own failed-experiment reward, so
    # trust whatever scalar it returns under "reward".
    reward = result.get("reward")
    if reward is None:
        logger.warning(f"autodiscovery_rm: server returned no reward: {result}")
        _dump_record({**base, "reward": FAILURE_REWARD, "success": False, "error": "no_reward_in_response"})
        return FAILURE_REWARD

    # Surface the server's experiment trace + surprise diagnostics on the sample
    # so a training run can log them per rollout or fold the execution log into
    # training. Stored under metadata; harmless if the server omits them.
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    if result.get("execution_log") is not None:
        sample.metadata["execution_log"] = result["execution_log"]
    sample.metadata["autodiscovery_reward"] = {
        k: result.get(k)
        for k in ("reward", "success", "surprising", "belief_change", "normalized_surprisal",
                  "kl_divergence", "error",
                  # Server-side gate flags: did the experiment hit the code-exec
                  # timeout / GroupChat round cap, and how many rounds it ran.
                  # Read back by the custom rollout/eval log funcs below.
                  "code_timeout_hit", "max_rounds_hit", "n_rounds")
    }
    sample.metadata["reasoning"] = base["reasoning"]

    _dump_record({
        **base,
        "reward": float(reward),
        **{k: result.get(k) for k in ("success", "surprising", "belief_change", "normalized_surprisal",
                                       "kl_divergence", "prior_mean", "posterior_mean", "error",
                                       "code_timeout_hit", "max_rounds_hit", "n_rounds")},
        "execution_log": result.get("execution_log"),
    })
    await _record_feedback(
        dataset_id, hypothesis, float(reward),
        execution_log=result.get("execution_log"), error=result.get("error"),
    )
    return float(reward)


def _experiment_gate_metrics(samples) -> dict:
    """Mean code-timeout / max-rounds hit rates (and mean round count) over a
    batch of scored samples, read from the ``autodiscovery_reward`` metadata the
    RM stashed. Shared by the rollout and eval custom log functions below; empty
    dict when no sample carries the flags (e.g. an all-truncated batch)."""
    flat = [s for grp in samples for s in grp] if samples and isinstance(samples[0], list) else samples
    to_hits, mr_hits, rounds = [], [], []
    for s in flat:
        md = s.metadata if isinstance(getattr(s, "metadata", None), dict) else {}
        ar = md.get("autodiscovery_reward") or {}
        if ar.get("code_timeout_hit") is not None:
            to_hits.append(float(bool(ar["code_timeout_hit"])))
        if ar.get("max_rounds_hit") is not None:
            mr_hits.append(float(bool(ar["max_rounds_hit"])))
        if ar.get("n_rounds") is not None:
            rounds.append(float(ar["n_rounds"]))
    out = {}
    if to_hits:
        out["code_timeout_frac"] = sum(to_hits) / len(to_hits)
    if mr_hits:
        out["max_rounds_frac"] = sum(mr_hits) / len(mr_hits)
    if rounds:
        out["experiment_rounds_mean"] = sum(rounds) / len(rounds)
    return out


def rollout_log_with_experiment_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """``--custom-rollout-log-function-path`` target.

    Replicates slime's default ``_log_rollout_data`` (so all standard ``rollout/``
    and ``perf/`` metrics are preserved) and adds the experiment gate metrics
    (``rollout/code_timeout_frac``, ``rollout/max_rounds_frac``,
    ``rollout/experiment_rounds_mean``). Returns True to take over logging; on any
    error it returns False so slime falls back to its built-in logging.
    """
    try:
        from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix

        if getattr(args, "load_debug_rollout_data", None):
            return True
        log_dict = {**(rollout_extra_metrics or {})}
        log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
        log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
        log_dict |= dict_add_prefix(_experiment_gate_metrics(samples), "rollout/")
        step = compute_rollout_step(args, rollout_id)
        log_dict["rollout/step"] = step
        # Echo to stdout like slime's default _log_rollout_data does; without this
        # the gate metrics reach W&B but are invisible in the job logs (and the
        # default "perf {id}: ..." line would go missing since we took over).
        logger.info(f"perf {rollout_id}: {log_dict}")
        logging_utils.log(args, log_dict, step_key="rollout/step")
        return True
    except Exception as e:  # noqa: BLE001 - never break training over a log line
        logger.warning(f"[autodiscovery] custom rollout log failed, using default: {e}")
        return False


def eval_log_with_experiment_metrics(rollout_id, args, data, extra_metrics=None):
    """``--custom-eval-rollout-log-function-path`` target.

    Replicates slime's default ``_log_eval_rollout_data`` and adds the same
    experiment gate metrics per eval set (``eval/<name>/code_timeout_frac`` etc.).
    Returns True to take over logging; False on error to fall back to the builtin.
    """
    try:
        from slime.ray.rollout import compute_metrics_from_samples
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, dict_add_prefix

        log_dict = extra_metrics or {}
        for key in data.keys():
            rewards = data[key]["rewards"]
            log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
            if (samples := data[key].get("samples")) is not None:
                log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
                log_dict |= dict_add_prefix(_experiment_gate_metrics(samples), f"eval/{key}/")
            if "truncated" in data[key]:
                truncated = data[key]["truncated"]
                log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
            if getattr(args, "log_passrate", False):
                log_dict |= dict_add_prefix(
                    compute_pass_rate(flat_rewards=rewards, group_size=args.n_samples_per_eval_prompt),
                    f"eval/{key}-",
                )
        step = compute_rollout_step(args, rollout_id)
        log_dict["eval/step"] = step
        # Echo to stdout like slime's default _log_eval_rollout_data does.
        logger.info(f"eval {rollout_id}: {log_dict}")
        logging_utils.log(args, log_dict, step_key="eval/step")
        return True
    except Exception as e:  # noqa: BLE001 - never break eval over a log line
        logger.warning(f"[autodiscovery] custom eval log failed, using default: {e}")
        return False


def _log_group_stats(samples, rew, mean, std, adv, n) -> None:
    """Print per-group reward mean/std/min/max + advantage range, once per rollout.

    ``rew``/``mean``/``std``/``adv`` are (num_groups, n) / (num_groups, 1) tensors
    aligned with contiguous groups of ``n`` samples in ``samples``.
    """
    try:
        num_groups = rew.shape[0]
        zero_std = int((std.flatten() < 1e-6).sum())
        step = None
        for s in samples:
            if getattr(s, "rollout_id", None) is not None:
                step = s.rollout_id
                break
        logger.info(
            f"[group_stats] step={step} groups={num_groups} n={n} "
            f"zero_std_groups={zero_std}/{num_groups} "
            f"batch_reward_mean={rew.mean().item():.4f} batch_reward_std={rew.std().item():.4f}"
        )
        for g in range(num_groups):
            grp = samples[g * n] if g * n < len(samples) else None
            ds = (grp.metadata or {}).get("dataset_id") if grp is not None and isinstance(grp.metadata, dict) else None
            gi = getattr(grp, "group_index", None) if grp is not None else None
            r = rew[g]
            logger.info(
                f"[group_stats]   g={g} group_index={gi} dataset_id={ds} "
                f"reward_mean={mean[g].item():.4f} reward_std={std[g].item():.4f} "
                f"reward_min={r.min().item():.4f} reward_max={r.max().item():.4f} "
                f"adv_min={adv[g].min().item():.4f} adv_max={adv[g].max().item():.4f}"
            )
    except Exception as e:  # noqa: BLE001 - logging must never break training
        logger.warning(f"[group_stats] logging failed: {e}")


def post_process_rewards(args, samples):
    """Custom reward post-process (``--custom-reward-post-process-path``).

    Replicates slime's built-in GRPO group normalization (subtract group mean,
    optionally divide by group std) so training behaviour is unchanged, and
    additionally prints per-group reward/advantage statistics every rollout.
    Wire with::

        --custom-reward-post-process-path slime.rollout.rm_hub.autodiscovery.post_process_rewards

    Returns ``(raw_rewards, processed_rewards)`` in sample order.
    """
    import torch

    # Accept both flat list[Sample] and grouped list[list[Sample]].
    if samples and isinstance(samples[0], list):
        flat = [s for grp in samples for s in grp]
    else:
        flat = samples

    raw_rewards = [s.get_reward_value(args) for s in flat]
    rewards_for_training = raw_rewards

    # Optional response-length controller.  The surprise score remains the raw
    # reward used for monitoring/evaluation; only the reward used to construct
    # the policy advantage is shaped.  A dead band avoids teaching the policy
    # to hit one exact token count while still opposing sustained length drift.
    target = float(os.environ.get("AUTODISCOVERY_LENGTH_TARGET", "0"))
    coefficient = float(os.environ.get("AUTODISCOVERY_LENGTH_COEF", "0"))
    deadband = float(os.environ.get("AUTODISCOVERY_LENGTH_DEADBAND", "0"))
    if target > 0 and coefficient > 0:
        lengths = [float(s.response_length) for s in flat]
        penalties = [coefficient * max(abs(length - target) - deadband, 0.0) / target for length in lengths]
        rewards_for_training = [reward - penalty for reward, penalty in zip(raw_rewards, penalties, strict=True)]
        logger.info(
            f"[length_control] target={target:.0f} deadband={deadband:.0f} coef={coefficient:.4f} "
            f"response_mean={sum(lengths) / len(lengths):.1f} "
            f"penalty_mean={sum(penalties) / len(penalties):.6f} penalty_max={max(penalties):.6f}"
        )
    est = getattr(args, "advantage_estimator", "grpo")
    do_norm = getattr(args, "rewards_normalization", True)
    if est not in ("grpo", "gspo", "cispo", "reinforce_plus_plus_baseline") or not do_norm:
        return raw_rewards, raw_rewards

    n = args.n_samples_per_prompt
    rew = torch.tensor(rewards_for_training, dtype=torch.float)
    if rew.shape[-1] == n * args.rollout_batch_size:
        rew = rew.reshape(-1, n)
    else:  # unequal group sizes (e.g. partial rollout) — best-effort reshape
        rew = rew.view(-1, rew.shape[-1])
    mean = rew.mean(dim=-1, keepdim=True)
    centered = rew - mean
    std = rew.std(dim=-1, keepdim=True)
    if est in ("grpo", "gspo", "cispo") and getattr(args, "grpo_std_normalization", True):
        adv = centered / (std + 1e-6)
    else:
        adv = centered

    _log_group_stats(flat, rew, mean, std, adv, rew.shape[-1])
    return raw_rewards, adv.flatten().tolist()
