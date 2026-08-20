"""Codex-scored surprise reward: the Codex CLI runs the whole experiment.

For each rollout hypothesis, this RM hands the plan -> execute -> analyze loop
to the OpenAI Codex CLI (``codex exec``) running in a fresh per-request work
dir with the dataset's files symlinked in — so its scripts see the real data
by relative path, exactly like the reward server's sandboxed executor. Codex
must finish with a JSON verdict containing its plan, key code outputs, full
analysis, a review verdict, and its prior/posterior probability that the
hypothesis is true. The reward is binary surprise, mirroring the server's
``use_binary_reward`` rule::

    reward = 1.0  iff  the experiment succeeded and
                       |posterior_prob - prior_prob| >= AUTODISCOVERY_CODEX_SURPRISAL_WIDTH

(default width 0.2; falls back to Codex's boolean ``surprising`` field when it
omits the probabilities). Failures of any kind (truncated response, codex
crash/timeout, unparseable verdict) score ``AUTODISCOVERY_FAILURE_REWARD``.

Requirements inside the job: the ``codex`` binary on PATH (the entry script
installs it when ``REWARD_TYPE=codex``), ``OPENAI_API_KEY`` in the env, the
datasets on local disk, and ``AUTODISCOVERY_CODEX_REGISTRY`` pointing at the
same ``registry.json`` the reward server uses
(``{dataset_id: {"dataset_metadata": path, "dataset_metadata_type": ...}}``).

Env knobs::

    AUTODISCOVERY_CODEX_REGISTRY         registry.json path (required)
    AUTODISCOVERY_CODEX_MODEL            model override (default: codex default)
    AUTODISCOVERY_CODEX_TIMEOUT          seconds per experiment (default 1800)
    AUTODISCOVERY_CODEX_CONCURRENCY      max concurrent codex processes (default 8)
    AUTODISCOVERY_CODEX_SURPRISAL_WIDTH  binary surprise threshold (default 0.2)
    AUTODISCOVERY_CODEX_WORKROOT         where per-request work dirs live

Wiring::

    --custom-rm-path slime.rollout.rm_hub.codex.reward
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile

from slime.utils.types import Sample

from .reward_utils import (
    FAILURE_REWARD,
    dump_record,
    extract_hypothesis,
    record_feedback,
    sample_base_record,
)

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.environ.get("AUTODISCOVERY_CODEX_REGISTRY") or os.environ.get(
    "AUTODS_DATASET_REGISTRY"
)
CODEX_MODEL = os.environ.get("AUTODISCOVERY_CODEX_MODEL") or None
CODEX_TIMEOUT = float(os.environ.get("AUTODISCOVERY_CODEX_TIMEOUT", "1800"))
CODEX_CONCURRENCY = int(os.environ.get("AUTODISCOVERY_CODEX_CONCURRENCY", "8"))
SURPRISAL_WIDTH = float(os.environ.get("AUTODISCOVERY_CODEX_SURPRISAL_WIDTH", "0.2"))
WORKROOT = os.environ.get("AUTODISCOVERY_CODEX_WORKROOT") or tempfile.gettempdir()

_DATA_EXTS = (".csv", ".tsv", ".txt", ".json", ".dta", ".xlsx", ".parquet")

_INSTRUCTIONS = """\
You are a rigorous research scientist. Test the hypothesis below against the
dataset files in the current directory (real data — inspect them first).
1. PLAN a statistically sound experiment: state operationalizations and the test you will use.
2. EXECUTE it: write and run Python here in the workspace. Handle data quirks
   (e.g. comma-decimal strings) by coercing to numeric. Do not fabricate data.
3. ANALYZE the results: effect sizes, test statistics, p-values, and whether the
   evidence supports, refutes, or is inconclusive about the hypothesis.
Your FINAL message must be ONLY a JSON object (no prose, no code fence) with keys:
  success        bool   - experiment ran end-to-end and produced an interpretable result
  plan           string - short summary of the experiment design
  code_output    string - the key numeric outputs of your scripts
  analysis       string - your full analysis of what the results show
  review         string - one-paragraph verdict: supported / refuted / inconclusive, and why
  prior_prob     float  - before seeing the results: probability (0-1) a knowledgeable
                          researcher would assign to the hypothesis being true
  posterior_prob float  - after seeing the results: that probability now
  surprising     bool   - |posterior_prob - prior_prob| >= {width}
"""

_registry_cache: dict | None = None
_sem: asyncio.Semaphore | None = None


def _registry() -> dict:
    global _registry_cache
    if _registry_cache is None:
        if not REGISTRY_PATH or not os.path.exists(REGISTRY_PATH):
            raise RuntimeError(
                "AUTODISCOVERY_CODEX_REGISTRY must point at a registry.json "
                f"(got {REGISTRY_PATH!r})"
            )
        with open(REGISTRY_PATH, encoding="utf-8") as fh:
            _registry_cache = json.load(fh)
    return _registry_cache


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(CODEX_CONCURRENCY)
    return _sem


def _dataset_files(dataset_id: str) -> tuple[str, list[str]]:
    """Resolve (metadata_path, data_file_paths) for *dataset_id*.

    Prefers the file names listed in the metadata's ``datasets`` entries
    (DiscoveryBench convention); falls back to every data-looking file next to
    the metadata (BLADE keeps ``data.csv`` beside ``info.json``).
    """
    entry = _registry()[dataset_id]
    meta_path = entry["dataset_metadata"] if isinstance(entry, dict) else entry
    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    files: list[str] = []
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        for ds in meta.get("datasets") or []:
            name = ds.get("name") if isinstance(ds, dict) else None
            if name and os.path.exists(os.path.join(meta_dir, name)):
                files.append(os.path.join(meta_dir, name))
    except Exception:  # noqa: BLE001 - fall back to directory scan
        pass
    if not files:
        for fname in sorted(os.listdir(meta_dir)):
            if fname.lower().endswith(_DATA_EXTS) and not fname.startswith("metadata_"):
                files.append(os.path.join(meta_dir, fname))
    return meta_path, files


def _run_codex(dataset_id: str, hypothesis: str) -> dict:
    """Blocking: run one codex experiment; returns the verdict record."""
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        return {"error": "codex CLI not found on PATH"}

    meta_path, files = _dataset_files(dataset_id)
    workdir = tempfile.mkdtemp(prefix=f"codexrm_{dataset_id}_", dir=WORKROOT)
    for fpath in files:
        dst = os.path.join(workdir, os.path.basename(fpath))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(fpath), dst)

    with open(meta_path, encoding="utf-8") as fh:
        meta_text = fh.read()
    prompt = (
        _INSTRUCTIONS.format(width=SURPRISAL_WIDTH)
        + "\n##### DATASET METADATA (JSON) #####\n"
        + meta_text[:8000]
        + "\n\n##### DATA FILES IN THIS DIRECTORY #####\n"
        + "\n".join(os.path.basename(f) for f in files)
        + "\n\n##### HYPOTHESIS #####\n"
        + hypothesis
    )
    out_path = os.path.join(workdir, "_codex_last_message.txt")
    cmd = [
        codex_bin, "exec",
        "--sandbox", "workspace-write",
        "--skip-git-repo-check",
        "-C", workdir,
        "--output-last-message", out_path,
    ]
    if CODEX_MODEL:
        cmd += ["-m", CODEX_MODEL]
    cmd.append(prompt)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CODEX_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"error": f"codex timed out after {CODEX_TIMEOUT:.0f}s", "code_timeout_hit": True}
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or proc.stdout or "")[-500:]
        return {"error": f"codex exited {proc.returncode}: {tail}"}

    with open(out_path, encoding="utf-8") as fh:
        last = fh.read().strip()
    verdict = None
    try:
        verdict = json.loads(last)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", last, re.DOTALL)
        if m:
            try:
                verdict = json.loads(m.group(0))
            except json.JSONDecodeError:
                verdict = None
    if not isinstance(verdict, dict) or "analysis" not in verdict:
        return {"error": f"codex final message not parseable as verdict JSON: {last[:300]}"}
    return verdict


def _binary_surprise(verdict: dict) -> tuple[float, bool | None, float | None]:
    """(reward, surprising, belief_change) from a codex verdict."""
    if not verdict.get("success"):
        return FAILURE_REWARD, None, None
    try:
        prior = float(verdict["prior_prob"])
        posterior = float(verdict["posterior_prob"])
        change = abs(posterior - prior)
        surprising = change >= SURPRISAL_WIDTH
    except (KeyError, TypeError, ValueError):
        change = None
        surprising = bool(verdict.get("surprising", False))
    return (1.0 if surprising else 0.0), surprising, change


def _execution_log(hypothesis: str, verdict: dict) -> str:
    """Sectioned trace matching the reward server's format, so downstream
    consumers (feedback summarizer, log browsers) parse it identically."""
    return (
        f"Hypothesis: {hypothesis}\n\n"
        f"Experiment objective: Test the hypothesis via Codex (plan -> execute -> analyze).\n\n"
        f"Steps for the programmer:\n{verdict.get('plan') or '(codex-planned)'}\n\n"
        f"Code Output:\n{verdict.get('code_output') or '(none)'}\n\n"
        f"Analysis:\n{verdict.get('analysis') or '(none)'}\n\n"
        f"Review:\n{verdict.get('review') or '(none)'}"
    )


async def reward(args, sample: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(sample, list):
        return await asyncio.gather(*(reward(args, s, **kwargs) for s in sample))

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    dataset_id = metadata.get("dataset_id")
    base = sample_base_record(sample, "codex")
    if sample.status == Sample.Status.TRUNCATED:
        dump_record({**base, "reward": FAILURE_REWARD, "success": False, "error": "truncated_response"})
        await record_feedback(dataset_id, None, FAILURE_REWARD, error="truncated_response")
        return FAILURE_REWARD
    if not dataset_id:
        raise ValueError("codex reward requires sample.metadata['dataset_id']")
    hypothesis = extract_hypothesis(sample.response)
    if not hypothesis:
        dump_record({**base, "reward": FAILURE_REWARD, "success": False, "error": "empty_hypothesis"})
        await record_feedback(dataset_id, None, FAILURE_REWARD, error="empty_hypothesis")
        return FAILURE_REWARD
    base["hypothesis"] = hypothesis

    async with _semaphore():
        try:
            verdict = await asyncio.to_thread(_run_codex, dataset_id, hypothesis)
        except Exception as e:  # noqa: BLE001 - a reward failure must not crash rollout
            verdict = {"error": f"{type(e).__name__}: {e}"}

    if verdict.get("error"):
        logger.warning(f"[codex_rm] {dataset_id}: {verdict['error']}")
        dump_record({
            **base, "reward": FAILURE_REWARD, "success": False, "error": verdict["error"],
            "code_timeout_hit": bool(verdict.get("code_timeout_hit")),
        })
        await record_feedback(dataset_id, hypothesis, FAILURE_REWARD, error=verdict["error"])
        return FAILURE_REWARD

    reward, surprising, change = _binary_surprise(verdict)
    log = _execution_log(hypothesis, verdict)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["execution_log"] = log
    sample.metadata["autodiscovery_reward"] = {
        "reward": reward, "success": bool(verdict.get("success")), "surprising": surprising,
        "belief_change": change, "normalized_surprisal": None, "kl_divergence": None,
        "error": None, "code_timeout_hit": False, "max_rounds_hit": False, "n_rounds": 1,
    }
    sample.metadata["reasoning"] = base["reasoning"]
    dump_record({
        **base, "reward": reward, "success": bool(verdict.get("success")),
        "surprising": surprising, "belief_change": change,
        "prior_mean": verdict.get("prior_prob"), "posterior_mean": verdict.get("posterior_prob"),
        "error": None, "execution_log": log,
    })
    await record_feedback(dataset_id, hypothesis, reward, execution_log=log)
    return reward
