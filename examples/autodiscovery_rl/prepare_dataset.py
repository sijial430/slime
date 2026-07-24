#!/usr/bin/env python3
"""Build the slime prompt dataset for the AutoDiscovery (Bayesian-surprise) RLVR run.

Each emitted row is ONE hypothesis-generation prompt for one scientific dataset.
The scientific data itself (CSV + metadata.json, "asta"/"dbench" format) is NOT
part of slime's training data -- it is read only by the reward server. slime's
training data is just these prompts. The reward RM
(``slime.rollout.rm_hub.autodiscovery.autodiscovery_rm``) routes each rollout to
the right dataset via ``metadata.dataset_id``, so every row carries that id.

Output schema (jsonl, one object per line), matching slime's Dataset class::

    {
      "messages": [{"role": "user", "content": "<hypothesis-generation prompt>"}],
      "metadata": {"dataset_id": "<id>", "dataset_name": "<id>", "source_name": "autodiscovery"}
    }

Launch slime with ``--input-key messages --metadata-key metadata --apply-chat-template``
(mirrors the gsm8k example: a list-of-messages prompt + chat template). No
``--label-key``: the reward is remote (the autodiscovery server), not a per-row label.

The prompt asks for output in the shape the reward-side ``extract_hypothesis``
accepts: a JSON object ``{"hypothesis": "..."}``. That function also tolerates
plain text, a ``{"hypotheses": [...]}`` list, and a leading reasoning
``</think>`` prefix, so format-unaware early checkpoints still score.

This script imports NOTHING from the heavy autodiscovery package: it reads each
metadata.json directly, so it runs in the slime env.

Usage::

    python prepare_dataset.py \
        --config datasets.example.json \
        --out-dir ./data \
        --num-per-dataset 64 \
        --eval-per-dataset 8

To also emit a concrete registry.json (weka placeholders expanded) for the
reward server::

    WEKA_ROOT=/weka/nora-default/sijial/autodiscovery/datasets \
    python prepare_dataset.py --config datasets.example.json --out-dir ./data \
        --emit-registry registry.example.json
"""

import argparse
import json
import os
import random
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset-description rendering (asta / dbench). Mirrors
# autodiscovery.dataset.get_asta_description / get_dataset_description so we do
# not need to import the heavy autodiscovery package.
# ---------------------------------------------------------------------------


def render_description(metadata: dict, dataset_metadata_type: str = "asta") -> str:
    lines = ["##### DATASET DESCRIPTION #####"]
    if dataset_metadata_type == "asta":
        lines.append(metadata.get("description", ""))
    lines.append("\n### DATASETS: ###\n")
    for dataset in metadata.get("datasets", []):
        lines.append(f"Dataset Name: {dataset.get('name', 'Unnamed Dataset')}")
        if dataset.get("description"):
            lines.append(f"Dataset Description: {dataset['description']}")
        cols = dataset.get("columns", {}).get("raw", [])
        if cols:
            lines.append("\n### COLUMNS: ###")
            for col in cols:
                lines.append(f"\n{col.get('name', 'Unnamed')}:")
                lines.append(f"  {col.get('description', 'No description available.')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction. The instruction paraphrases below give light per-row
# variation so a rollout batch is not byte-identical, while keeping the core
# task (one falsifiable hypothesis, JSON-out) fixed.
# ---------------------------------------------------------------------------

_PERSONA = (
    "You are a research scientist doing open-ended, data-driven research on the "
    "dataset(s) described below."
)

_INSTRUCTION_VARIANTS = [
    "Propose exactly ONE novel, falsifiable scientific hypothesis that can be "
    "sufficiently tested by a statistical experiment using only these dataset(s).",
    "Generate a single creative, falsifiable hypothesis that a rigorous "
    "statistical experiment on only these dataset(s) could confirm or refute.",
    "Come up with one interesting, testable hypothesis about this data. It must "
    "be falsifiable and answerable with a statistical experiment on only the "
    "provided dataset(s).",
    "State exactly ONE original, falsifiable hypothesis about relationships in "
    "these dataset(s) that could be verified with a robust statistical test.",
]

_GUIDELINES = (
    "Guidelines:\n"
    "1. Use ONLY the provided dataset(s); do not invent columns or synthetic data. "
    "You may derive composite variables from existing columns.\n"
    "2. Pick an interesting context (e.g. a subset defined by categorical values), "
    "interesting variables, and an interesting relationship between them.\n"
    "3. The hypothesis must be verifiable with a robust statistical test.\n"
    "4. Make it specific and self-contained: name the outcome variable(s), the "
    "explanatory variable(s), and the expected direction of the relationship."
)

_OUTPUT_INSTRUCTION = (
    "Respond with a single JSON object and nothing else:\n"
    '{"hypothesis": "<your one-sentence falsifiable hypothesis>"}'
)


def build_prompt(description: str, rng: random.Random) -> str:
    instruction = rng.choice(_INSTRUCTION_VARIANTS)
    return (
        f"{_PERSONA} {instruction}\n\n"
        f"{description}\n\n"
        f"{_GUIDELINES}\n\n"
        f"{_OUTPUT_INSTRUCTION}"
    )


def make_row(prompt_text: str, dataset_id: str) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt_text}],
        "metadata": {
            "dataset_id": dataset_id,
            "dataset_name": dataset_id,
            "source_name": "autodiscovery",
        },
    }


# ---------------------------------------------------------------------------
# Config / registry helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> list[dict]:
    with open(config_path) as f:
        cfg = json.load(f)
    datasets = cfg["datasets"] if isinstance(cfg, dict) else cfg
    if not datasets:
        raise ValueError(f"{config_path} contains no datasets")
    return datasets


def resolve_metadata_path(metadata_path: str, config_path: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(metadata_path)))
    if not p.is_absolute():
        p = (config_path.parent / p).resolve()
    return p


def emit_registry(template_path: Path, out_path: Path) -> None:
    """Expand ${WEKA_ROOT} (and any env vars) in a registry template -> concrete JSON."""
    weka_root = os.environ.get("WEKA_ROOT")
    if not weka_root:
        raise SystemExit(
            "WEKA_ROOT is not set; export it before --emit-registry, e.g.\n"
            "  export WEKA_ROOT=/weka/nora-default/sijial/autodiscovery/datasets"
        )
    with open(template_path) as f:
        raw = f.read()
    expanded = json.loads(os.path.expandvars(raw))
    with open(out_path, "w") as f:
        json.dump(expanded, f, indent=2)
        f.write("\n")
    print(f"[registry] wrote {out_path} (WEKA_ROOT={weka_root})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def rand_suffix(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "datasets.example.json")
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    ap.add_argument("--num-per-dataset", type=int, default=64, help="Total prompt rows per dataset (train+eval).")
    ap.add_argument("--eval-per-dataset", type=int, default=8, help="Rows per dataset held out for eval.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--emit-registry",
        type=Path,
        default=None,
        help="Path to a registry template (e.g. registry.example.json) whose ${WEKA_ROOT} "
        "placeholders are expanded into <out-dir>/registry.json for the reward server.",
    )
    args = ap.parse_args()

    if args.eval_per_dataset >= args.num_per_dataset:
        raise SystemExit("--eval-per-dataset must be smaller than --num-per-dataset")

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_config(args.config)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    for entry in datasets:
        dataset_id = entry["dataset_id"]
        md_type = entry.get("dataset_metadata_type", "asta")
        md_path = resolve_metadata_path(entry["metadata_path"], args.config)
        if md_path.exists():
            with open(md_path) as f:
                metadata = json.load(f)
            description = render_description(metadata, md_type)
        else:
            print(f"[warn] {dataset_id}: metadata not found at {md_path}; embedding id only.")
            description = f"##### DATASET DESCRIPTION #####\nDataset id: {dataset_id} (metadata unavailable at prep time)."

        rows = [make_row(build_prompt(description, rng), dataset_id) for _ in range(args.num_per_dataset)]
        rng.shuffle(rows)
        eval_rows.extend(rows[: args.eval_per_dataset])
        train_rows.extend(rows[args.eval_per_dataset :])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    print(f"[data] wrote {len(train_rows)} train rows -> {train_path}")
    print(f"[data] wrote {len(eval_rows)} eval rows  -> {eval_path}")

    if args.emit_registry is not None:
        emit_registry(args.emit_registry, args.out_dir / "registry.json")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
