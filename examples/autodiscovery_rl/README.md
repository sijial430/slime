# AutoDiscovery RLVR: prompt-data prep

This example builds the **slime prompt dataset** for an RLVR run whose reward is
Bayesian *surprise*. Each rollout the policy is given a prompt and must
**generate one falsifiable scientific hypothesis** for a given scientific
dataset. The scientific data itself (CSV + `metadata.json`, asta/dbench format)
is **not** part of slime's training data — it is read only by an external
autodiscovery reward server. slime's training data is just the
hypothesis-generation **prompts** produced here.

Reward wiring (already built, not part of this example):
- Custom RM: `slime/rollout/rm_hub/autodiscovery.py` (`autodiscovery_rm`) posts
  `{hypothesis, dataset_id, request_id}` to the reward server and returns the
  scalar reward.
- Reward server: `python -m autodiscovery.slime_reward` (in the
  `asta-autodiscovery` repo), routed per request by `dataset_id` via a
  `registry.json`.

## Contents

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Generates the slime prompt dataset (jsonl) + optionally a concrete `registry.json`. |
| `datasets.example.json` | Prep-time config: `dataset_id` + local `metadata.json` path (for description embedding) + type (asta/dbench). |
| `registry.example.json` | `dataset_id -> weka metadata.json path` map the **reward server** consumes. Uses `${WEKA_ROOT}`. |
| `example_metadata/<id>/metadata.json` | Illustrative asta/dbench metadata (schema/description only — no CSV) so the example runs standalone. |
| `sample_prompts.jsonl` | A handful of generated rows, committed so the output format is inspectable. |

## Prompt-data schema

One JSON object per line (jsonl). Keys match slime's `Dataset` class:

```json
{
  "messages": [{"role": "user", "content": "<hypothesis-generation prompt incl. dataset description>"}],
  "metadata": {"dataset_id": "tcga-breast-cancer", "dataset_name": "tcga-breast-cancer", "source_name": "autodiscovery"}
}
```

- `messages` (`--input-key messages`): a list-of-messages prompt, used with
  `--apply-chat-template` (mirrors the gsm8k example).
- `metadata.dataset_id` (`--metadata-key metadata`): **required** — the RM raises
  without it. Its value **must equal a key in `registry.json`**; that is the only
  thing tying a prompt row to the scientific dataset the reward server scores
  against.
- No `--label-key`: the reward is remote (the server), not a per-row label.

The prompt asks for output as `{"hypothesis": "..."}`. The reward-side
`extract_hypothesis` also accepts plain text, a `{"hypotheses": [...]}` list, and
a leading reasoning `</think>` prefix, so early format-unaware checkpoints still
score.

## WEKA path convention (and the GCP -> WEKA migration)

The scientific datasets currently live on GCP. **Prerequisite ops step (not done
here):** sync them to a WEKA folder on the AI2/Beaker cluster, one directory per
`dataset_id`, each containing its `metadata.json` and CSV(s):

```
${WEKA_ROOT}/<dataset_id>/metadata.json
${WEKA_ROOT}/<dataset_id>/<dataset>.csv
```

`WEKA_ROOT` is parameterized via an env var. Example default (confirm the real
path with ops):

```bash
export WEKA_ROOT=/weka/nora-default/sijial/autodiscovery/datasets
```

`registry.example.json` uses `${WEKA_ROOT}` placeholders. The reward server does
not expand env vars in its registry JSON, so render a concrete one first (see
below).

## Run it

1. Generate prompt data (and, optionally, the concrete registry):

```bash
cd examples/autodiscovery_rl
export WEKA_ROOT=/weka/nora-default/sijial/autodiscovery/datasets

python prepare_dataset.py \
    --config datasets.example.json \
    --out-dir ./data \
    --num-per-dataset 64 \
    --eval-per-dataset 8 \
    --emit-registry registry.example.json
# -> data/train.jsonl, data/eval.jsonl, data/registry.json
```

`--metadata_path` in the config is read **at prep time** (locally) only to embed
each dataset's schema/description into the prompt; point it at a locally-readable
copy (the bundled `example_metadata/`, a GCP-synced copy, or the mounted weka
path). It is independent of the weka path the reward server uses at train time.

2. Start the reward server (in the `asta-autodiscovery` repo, on a node with the
   weka mount), pointing it at the expanded registry:

```bash
python -m autodiscovery.slime_reward \
    --dataset_registry /path/to/data/registry.json \
    --host 0.0.0.0 --port 8000 --concurrency 8
```

3. Launch slime pointing at the prompt data + the reward server. The
   autodiscovery-relevant flags:

```bash
   --prompt-data       /path/to/data/train.jsonl \
   --input-key         messages \
   --metadata-key      metadata \
   --apply-chat-template \
   --custom-rm-path    slime.rollout.rm_hub.autodiscovery.autodiscovery_rm \
   --rm-url            http://<reward-server-host>:8000/reward \
   --eval-prompt-data  autodiscovery /path/to/data/eval.jsonl
```

Do **not** pass `--label-key` (reward is remote) and do **not** pass `--rm-type`
(the custom RM path is used instead).

## Expanded training set: DiscoveryBench + BLADE (29 real datasets)

Beyond the illustrative tcga/nls examples above, this dir ships a ready-to-use
set of **29 real datasets** mined into GCS (14 DiscoveryBench `real` + 15 BLADE;
see `asta-autodiscovery/scripts/dataset_mining`). Every one is verified complete
(metadata + all referenced data files present in the bucket).

- `dbench_blade_metadata/<id>/{metadata_0.json,info.json}` — the real metadata,
  read at prep time to embed each dataset's description/columns into the prompt.
- `datasets_dbench_blade.json` — prep config (29 entries, `dataset_metadata_type`
  = `dbench` or `blade`).
- `registry_dbench_blade.example.json` — the reward-server registry
  (`${WEKA_ROOT}`-parameterized; 29 `dataset_id -> {dataset_metadata, type}`).
- `dbench_blade_sample.jsonl` — 6 inspectable prompt rows (dbench + blade).

Generate the expanded prompt data:

```bash
python prepare_dataset.py --config datasets_dbench_blade.json \
    --out-dir data_dbench_blade --num-per-dataset 32 --eval-per-dataset 4
# -> 812 train + 116 eval rows across 29 dataset_ids
```

Then point the reward server at the expanded registry (after `${WEKA_ROOT}`
expansion / syncing the GCS data to that root) and slime at
`data_dbench_blade/{train,eval}.jsonl`, exactly as above. `blade` metadata is a
single dataset under `data_desc`; `dbench` metadata lists `datasets[]` with
columns — `prepare_dataset.py` renders both.

## Assumptions to confirm

- **`WEKA_ROOT`**: `/weka/nora-default/sijial/autodiscovery/datasets` on the
  AI2/Beaker cluster; sync the datasets there before starting the reward server.
- **Datasets included**: `tcga-breast-cancer`, `tcga-melanoma` (asta), `nls_raw`
  (dbench). Adjust `datasets.example.json` / `registry.example.json` to the real
  set; keys must stay identical between the two files.
- **Example metadata** under `example_metadata/` is illustrative (plausible
  columns, no real CSV). Replace with the real `metadata.json` for accurate
  descriptions, or point `metadata_path` at the weka/GCP copies.
