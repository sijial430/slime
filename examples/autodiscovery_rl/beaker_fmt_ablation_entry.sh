#!/bin/bash
# RL (GRPO) fmt-ablation on the DBB lineage. One job per FORMAT; the 24 DBB
# datasets are split by DATASET into disjoint train/val/test sets (built by
# scripts, uploaded to s3://.../slime_prompt_data/dbb_split/). Live surprise
# reward from the autodiscovery reward server (all 24 datasets servable). Runs
# INSIDE slimerl/slime.
#
#   FORMAT in {fmt1_hyps_open, fmt2_pairs_open, fmt3_pairs_related}
#   train  = 16 datasets' lineage samples   -> --prompt-data
#   val    = 4 held-out datasets            -> eval set "valid"
#   test   = 4 held-out datasets            -> eval set "test"
#
# Same hyperparameters as the fmt-compare run; only NUM_ROLLOUT=100 and the data.
# Knobs: FORMAT [fmt1_hyps_open], NUM_ROLLOUT [100], REWARD_CONCURRENCY [48],
#   ROLLOUT_BATCH_SIZE [8], N_SAMPLES_PER_PROMPT [8], ASTA_REF [sijia/generator-dev].
# Required env-secret: OPENAI_API_KEY, AWS_*, WANDB_KEY, GITHUB_TOKEN, HF_TOKEN.
set -ex

FORMAT="${FORMAT:-fmt1_hyps_open}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

DATASET_ROOT="${DATASET_ROOT:-/root/autods_datasets}"
ASTA_DIR="${ASTA_DIR:-/root/asta-autodiscovery}"
ASTA_REF="${ASTA_REF:-sijia/generator-dev}"
S3="s3://ai2-asta-workspaces/autods/datasets"

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export PATH="$HOME/.local/bin:$PATH"
AWS="uvx --from awscli aws"
# gantry's miniconda python lacks torch; restore the image's torch python.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -viE conda | paste -sd: -)"
python3 -c 'import torch' 2>/dev/null && echo "[entry] torch python: $(command -v python3)" || echo "[entry] WARN: no torch"

# --- 1. clone asta-autodiscovery (reward server) ---
[ -d "$ASTA_DIR/.git" ] || git clone --depth 1 --branch "$ASTA_REF" \
    "https://github.com/allenai/asta-autodiscovery.git" "$ASTA_DIR"

# --- 2. fetch the DBB datasets the reward server scores (discoverybench + blade) ---
mkdir -p "$DATASET_ROOT"
$AWS s3 sync "$S3/discoverybench/real" "$DATASET_ROOT/discoverybench/real" --only-show-errors
$AWS s3 sync "$S3/blade"               "$DATASET_ROOT/blade"               --only-show-errors --exclude "LICENSE"

# --- 3. build the reward registry from what we fetched (dbench + blade) ---
python3 - "$DATASET_ROOT" <<'PY'
import glob, json, os, re, sys
root = sys.argv[1]; reg = {}
for split in ("train", "test"):
    base = f"{root}/discoverybench/real/{split}"
    for ds in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        metas = sorted(glob.glob(f"{base}/{ds}/metadata_*.json"),
                       key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))
        if metas:
            reg[ds] = {"dataset_metadata": metas[0], "dataset_metadata_type": "dbench"}
bbase = f"{root}/blade"
for ds in sorted(os.listdir(bbase)) if os.path.isdir(bbase) else []:
    if os.path.isfile(f"{bbase}/{ds}/info.json"):
        reg[ds] = {"dataset_metadata": f"{bbase}/{ds}/info.json", "dataset_metadata_type": "blade"}
json.dump(reg, open(f"{root}/registry.json", "w"), indent=2)
print(f"registry: {len(reg)} datasets -> {root}/registry.json")
PY

# --- 4. fetch the FORMAT's train/val split parquets ---
PROMPT_DIR="$DATASET_ROOT/prompt_data"; mkdir -p "$PROMPT_DIR"
for sp in train val; do
    $AWS s3 cp "$S3/slime_prompt_data/dbb_split/${FORMAT}_${sp}.parquet" "$PROMPT_DIR/" --only-show-errors
done

# --- 5. start the REAL reward server (concurrency = REWARD_CONCURRENCY) ---
REWARD_LOG_DIR="${BEAKER_RESULT_DIR:-/results}"; mkdir -p "$REWARD_LOG_DIR" 2>/dev/null || REWARD_LOG_DIR=/root
REWARD_LOG="$REWARD_LOG_DIR/reward_server.log"
( cd "$ASTA_DIR" && uv run --package asta-autodiscovery python -m autodiscovery.slime_reward \
    --dataset_registry "$DATASET_ROOT/registry.json" --host 127.0.0.1 --port 8000 \
    --concurrency "${REWARD_CONCURRENCY:-48}" \
    --execution_model "${EXECUTION_MODEL:-gpt-5-mini}" --belief_model "${BELIEF_MODEL:-gpt-5-mini}" \
    --n_belief_samples "${N_BELIEF_SAMPLES:-5}" --no-include_execution_log \
    --no-run_data_loading ) > "$REWARD_LOG" 2>&1 &
RM_PID=$!
tail -n +1 -F "$REWARD_LOG" 2>/dev/null & TAIL_PID=$!
trap 'kill $RM_PID $TAIL_PID 2>/dev/null || true; echo "===== reward_server.log ====="; cat "$REWARD_LOG" 2>/dev/null || true' EXIT
echo "waiting for reward server..."
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; sleep 2; done
curl -s http://127.0.0.1:8000/health | head -c 300; echo

# --- 6. use the image's version-matched slime; inject only the autodiscovery RM ---
IMG_SLIME="${IMG_SLIME:-/root/slime}"
[ -f "$IMG_SLIME/train.py" ] || IMG_SLIME="$(cd /root && python3 -c 'import slime,os;print(os.path.dirname(list(slime.__path__)[0]))')"
[ -f "$IMG_SLIME/train.py" ] || { echo "cannot locate image slime"; exit 1; }
[ "$IMG_SLIME" != "$REPO_ROOT" ] && cp slime/rollout/rm_hub/autodiscovery.py "$IMG_SLIME/slime/rollout/rm_hub/autodiscovery.py"
SGL="$IMG_SLIME/slime/backends/sglang_utils/arguments.py"
[ -f "$SGL" ] && sed -i -E 's/^([[:space:]]*)args\.(sglang_[a-z0-9_]+) = args\.(sglang_[a-z0-9_]+)$/\1args.\2 = getattr(args, "\3", args.\2)/' "$SGL" || true
export SLIME_ROOT="$IMG_SLIME"

# --- 7. model: Qwen3.5-9B (download + torch_dist convert) ---
TRAIN_MODEL="${TRAIN_MODEL:-qwen3.5-9B}"; HF_REPO="${HF_REPO:-Qwen/Qwen3.5-9B}"
MODEL_DIR="/root/$(basename "$HF_REPO")"; TD_DIR="${MODEL_DIR}_torch_dist"
if [ "$IMG_SLIME" != "$REPO_ROOT" ]; then
    [ -f "$IMG_SLIME/scripts/models/${TRAIN_MODEL}.sh" ] || cp "$REPO_ROOT/scripts/models/${TRAIN_MODEL}.sh" "$IMG_SLIME/scripts/models/${TRAIN_MODEL}.sh"
    cp -rn "$REPO_ROOT/slime_plugins/." "$IMG_SLIME/slime_plugins/" 2>/dev/null || true
fi
[ -f "$MODEL_DIR/config.json" ] || hf download "$HF_REPO" --local-dir "$MODEL_DIR"
source "$IMG_SLIME/scripts/models/${TRAIN_MODEL}.sh"
[ -d "$TD_DIR" ] || PYTHONPATH="$MEGATRON_PATH:$IMG_SLIME" python "$IMG_SLIME/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" --hf-checkpoint "$MODEL_DIR" --save "$TD_DIR"

# --- 8. GRPO: train on the FORMAT's train split, eval on valid + test ---
#   Same hyperparameters as fmt-compare; NUM_ROLLOUT=100. train/valid/test surprise
#   reward is logged to wandb by slime (rollout reward + eval/<name> reward).
NUM_ROLLOUT="${NUM_ROLLOUT:-100}" \
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}" \
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}" \
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}" MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}" \
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-18432}" \
    MODEL="$TRAIN_MODEL" NUM_GPUS="${NUM_GPUS:-8}" TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}" \
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}" \
    DO_EVAL=1 EVAL_INTERVAL="${EVAL_INTERVAL:-20}" N_SAMPLES_PER_EVAL_PROMPT=1 SAVE_INTERVAL=9999 \
    RM_URL=http://127.0.0.1:8000/reward \
    HF_CHECKPOINT="$MODEL_DIR/" REF_LOAD="$TD_DIR/" \
    PROMPT_DATA="$PROMPT_DIR/${FORMAT}_train.parquet" \
    EVAL_PROMPT_DATA="valid $PROMPT_DIR/${FORMAT}_val.parquet" \
    MEGATRON_PATH="$MEGATRON_PATH" \
    USE_WANDB=1 WANDB_KEY="${WANDB_KEY:-${SIJIAL_WANDB_API_KEY:-}}" \
    WANDB_PROJECT=autodiscovery-rl WANDB_GROUP="fmt-ablation-${FORMAT}" \
    bash examples/autodiscovery_rl/train_grpo.sh

echo "=== fmt-ablation FORMAT=$FORMAT finished OK ==="
