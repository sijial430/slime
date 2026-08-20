"""Shared helpers for the standalone AutoDiscovery reward modules
(``zero.py`` / ``one.py`` / ``coin.py`` / ``codex.py``).

Deliberately self-contained: nothing here imports from ``autodiscovery.py``,
so the pluggable rewards carry no dependency on the reward-server RM. The
conventions (env vars, dump format, feedback-state format) match it exactly,
so downstream tooling reads both the same way:

- ``AUTODISCOVERY_RM_DUMP``       per-sample JSONL record dump (flock'd)
- ``AUTODISCOVERY_FAILURE_REWARD`` reward for unscorable samples (default 0.0)
- ``AUTODISCOVERY_FEEDBACK_STATE`` feedback JSONL consumed by the
  FEEDBACK_CONTEXT prompt-injection wrapper (learnings are LLM-summarized via
  ``AUTODISCOVERY_FEEDBACK_SUMMARY_MODEL``, default gpt-5-mini, with
  truncation fallback)
"""

import fcntl
import json
import logging
import os
import re

import aiohttp

logger = logging.getLogger(__name__)

FAILURE_REWARD = float(os.environ.get("AUTODISCOVERY_FAILURE_REWARD", "0.0"))

_DUMP_PATH = os.environ.get("AUTODISCOVERY_RM_DUMP") or None
_DUMP_MAXLOG = int(os.environ.get("AUTODISCOVERY_RM_DUMP_MAXLOG", "100000"))

_THINK_CLOSE = "</think>"
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def dump_record(record: dict) -> None:
    """Append one JSON line to the rollout dump (cross-process safe). Never raises."""
    if not _DUMP_PATH:
        return
    try:
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
        logger.warning(f"[reward_utils] dump failed: {e}")


def extract_hypothesis(response: str) -> str:
    """Hypothesis text from a raw model response (mirrors the server RM's rules):
    drop everything up to the last ``</think>``, unwrap a ``HypothesisList``-style
    JSON block when present, else the stripped raw text."""
    text = response or ""
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
    """The reasoning trace: everything before the last ``</think>`` ('' if none)."""
    response = response or ""
    if _THINK_CLOSE not in response:
        return ""
    reasoning = response.rsplit(_THINK_CLOSE, 1)[0].lstrip()
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>"):]
    return reasoning.strip()


def sample_base_record(sample, control: str) -> dict:
    """The identifying fields every dump record carries."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return {
        "rollout_id": sample.rollout_id,
        "index": sample.index,
        "dataset_id": metadata.get("dataset_id"),
        "data_fmt": metadata.get("format"),
        "reasoning": extract_reasoning(sample.response or ""),
        "control": control,
    }


# --- feedback-state integration (same file format as the server RM path) -----

_FEEDBACK_STATE = os.environ.get("AUTODISCOVERY_FEEDBACK_STATE") or None
_FEEDBACK_SUMMARY_MODEL = os.environ.get("AUTODISCOVERY_FEEDBACK_SUMMARY_MODEL", "gpt-5-mini")
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
    if not execution_log or "\nAnalysis:\n" not in execution_log:
        return ""
    seg = execution_log.split("\nAnalysis:\n", 1)[1]
    return seg.split("\nReview:\n", 1)[0].strip()


async def _summarize_learnings(analysis: str, error: str | None) -> str:
    if not analysis:
        return f"(no analysis produced{'; ' + error if error else ''})"
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            async with aiohttp.ClientSession() as session:
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
                    logger.warning(f"[reward_utils] summarizer HTTP {resp.status}; truncating instead")
        except Exception as e:  # noqa: BLE001 - feedback must never break a rollout
            logger.warning(f"[reward_utils] summarizer failed ({e}); truncating instead")
    return analysis[:_FEEDBACK_MAXCHARS]


async def record_feedback(dataset_id, hypothesis, reward, execution_log=None, error=None) -> None:
    """Append a feedback entry for the FEEDBACK_CONTEXT prompt wrapper. No-op
    unless AUTODISCOVERY_FEEDBACK_STATE is set. Never raises."""
    if not _FEEDBACK_STATE or not dataset_id:
        return
    try:
        if error in ("truncated_response", "empty_hypothesis"):
            learnings = (
                "(previous response exceeded the token limit or contained no parseable "
                "hypothesis — no experiment was run; keep the response focused and within budget)"
            )
        else:
            learnings = await _summarize_learnings(_extract_analysis(execution_log), error)
        entry = {
            "dataset_id": dataset_id,
            "hypothesis": (hypothesis or "")[:600],
            "reward": None if reward is None else float(reward),
            "learnings": learnings,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with open(_FEEDBACK_STATE, "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[reward_utils] feedback append failed: {e}")
