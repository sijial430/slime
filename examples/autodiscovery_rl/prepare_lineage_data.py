#!/usr/bin/env python3
"""Turn an autodiscovery lineage parquet into slime GRPO prompt data.

The lineage parquet (built by asta-autodiscovery
`build_lineage_training_data.py`) already uses slime's prompt schema — one row
per node with ``messages`` (list of {role, content}) and ``metadata``
(carrying ``dataset_id`` = the run's job_id). This script only:

  1. splits the rows into train / eval **by job_id** (so lineage steps from one
     run never straddle the split — no leakage), and
  2. emits a ``registry.json`` mapping every ``dataset_id`` present to a dataset
     metadata path, which the autodiscovery reward server routes on.

Each node is one GRPO task: the policy sees the node's prompt and generates
``--n-samples-per-prompt`` hypotheses, each scored by the reward server.

Registry paths: with ``--weka-root`` set, entries point at
``<weka-root>/<dataset_id>/metadata.json`` (what the real reward server reads).
Without it, entries are placeholder paths — fine for the mock reward server,
which never opens the files.

Usage:
    python prepare_lineage_data.py \
        --input /path/to/asta/data/lineage_fmt2_pairs_open.parquet \
        --out-dir ./data --eval-frac 0.05 --seed 0
    # -> data/train.parquet, data/eval.parquet, data/registry.json
"""

from __future__ import annotations

import argparse
import json
import os

import pyarrow.parquet as pq


def read_rows(path: str) -> list[dict]:
    """Read a slime-format parquet into a list of {messages, metadata} dicts."""
    rows: list[dict] = []
    for batch in pq.ParquetFile(path).iter_batches():
        rows.extend(batch.to_pylist())
    return rows


def split_by_job(rows: list[dict], eval_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Assign whole jobs to eval so no run's lineage straddles the split."""
    jobs = sorted({r["metadata"]["job_id"] for r in rows})
    # Deterministic hash-free shuffle: seeded index permutation.
    import random

    rng = random.Random(seed)
    rng.shuffle(jobs)
    n_eval = max(1, round(len(jobs) * eval_frac)) if eval_frac > 0 else 0
    eval_jobs = set(jobs[:n_eval])
    train = [r for r in rows if r["metadata"]["job_id"] not in eval_jobs]
    ev = [r for r in rows if r["metadata"]["job_id"] in eval_jobs]
    return train, ev


def write_parquet(rows: list[dict], path: str) -> None:
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def build_registry(rows: list[dict], weka_root: str | None, dtype: str) -> dict:
    """dataset_id -> metadata path (real weka path, or placeholder for the mock)."""
    ids = sorted({r["metadata"]["job_id"] for r in rows})
    registry: dict[str, object] = {}
    for did in ids:
        if weka_root:
            path = f"{weka_root.rstrip('/')}/{did}/metadata.json"
        else:
            path = f"PLACEHOLDER/{did}/metadata.json"  # mock scorer never opens it
        registry[did] = path if dtype == "asta" else {"dataset_metadata": path, "dataset_metadata_type": dtype}
    return registry


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Lineage parquet (slime-format rows).")
    ap.add_argument("--out-dir", default="./data", help="Output directory. Default: %(default)s")
    ap.add_argument("--eval-frac", type=float, default=0.05, help="Fraction of JOBS held out for eval.")
    ap.add_argument("--seed", type=int, default=0, help="Split seed.")
    ap.add_argument("--max-train", type=int, default=None, help="Cap train rows (smoke tests).")
    ap.add_argument("--max-eval", type=int, default=None, help="Cap eval rows (smoke tests).")
    ap.add_argument(
        "--weka-root",
        default=None,
        help="If set, registry paths become <weka-root>/<dataset_id>/metadata.json "
        "(real reward server). Omit for placeholder paths (mock server).",
    )
    ap.add_argument(
        "--dataset-metadata-type",
        default="asta",
        choices=["asta", "dbench", "blade", "ai2"],
        help="Registry metadata type. Default: %(default)s",
    )
    args = ap.parse_args(argv)

    rows = read_rows(args.input)
    train, ev = split_by_job(rows, args.eval_frac, args.seed)
    if args.max_train is not None:
        train = train[: args.max_train]
    if args.max_eval is not None:
        ev = ev[: args.max_eval]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.parquet")
    eval_path = os.path.join(args.out_dir, "eval.parquet")
    reg_path = os.path.join(args.out_dir, "registry.json")
    write_parquet(train, train_path)
    write_parquet(ev, eval_path)
    # registry must cover every dataset_id that can appear in train or eval
    registry = build_registry(rows, args.weka_root, args.dataset_metadata_type)
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")

    n_jobs = len({r["metadata"]["job_id"] for r in rows})
    print(f"input rows: {len(rows):,} across {n_jobs:,} jobs")
    print(f"  train: {len(train):,} rows -> {train_path}")
    print(f"  eval : {len(ev):,} rows -> {eval_path}")
    print(f"  registry: {len(registry):,} dataset_ids -> {reg_path}"
          + (f"  (weka_root={args.weka_root})" if args.weka_root else "  (placeholder paths for mock server)"))


if __name__ == "__main__":
    main()
