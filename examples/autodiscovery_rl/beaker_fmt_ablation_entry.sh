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
# Knobs: FORMAT [fmt1_hyps_open], NUM_ROLLOUT [100], REWARD_CONCURRENCY [128],
#   N_BELIEF_SAMPLES [3] (normalized-surprisal theoretical max adapts to N automatically),
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
# SPLIT_SUBDIR selects the prompt-data set under slime_prompt_data/ (default dbb_split);
# set e.g. SPLIT_SUBDIR=dbb_valid_split to train on the valid-surprise+verdict-filtered data.
PROMPT_DIR="$DATASET_ROOT/prompt_data"; mkdir -p "$PROMPT_DIR"
for sp in train val; do
    $AWS s3 cp "$S3/slime_prompt_data/${SPLIT_SUBDIR:-dbb_split}/${FORMAT}_${sp}.parquet" "$PROMPT_DIR/" --only-show-errors
done

# --- 5. start the REAL reward server (concurrency = REWARD_CONCURRENCY) ---
REWARD_LOG_DIR="${BEAKER_RESULT_DIR:-/results}"; mkdir -p "$REWARD_LOG_DIR" 2>/dev/null || REWARD_LOG_DIR=/root
REWARD_LOG="$REWARD_LOG_DIR/reward_server.log"
# Per-rollout record dump (reward + surprise diagnostics + execution_log +
# dataset_id/rollout_id/data_fmt). Written by the slime-side custom RM in the
# ray workers; train_grpo.sh forwards AUTODISCOVERY_RM_DUMP into their env.
export AUTODISCOVERY_RM_DUMP="${AUTODISCOVERY_RM_DUMP:-$REWARD_LOG_DIR/rollout_records_${FORMAT}.jsonl}"
# Full per-step sample dump (slime native --save-debug-rollout-data): one .pt per
# rollout with every generated sample (prompt/response/reward/metadata/group_index
# /index/rollout_id). {rollout_id} is a literal placeholder slime fills per step.
# NOTE: assign in a plain statement, NOT inside ${VAR:-default}: the literal '}'
# in {rollout_id} would close the parameter expansion early and mangle the
# template to '{rollout_id.pt}' (str.format then does attribute access -> crash).
if [ -z "${SAVE_ROLLOUT_DATA:-}" ]; then
    SAVE_ROLLOUT_DATA="$REWARD_LOG_DIR/rollout_data_${FORMAT}/{rollout_id}.pt"
fi
export SAVE_ROLLOUT_DATA
# Return the experiment trace so the dump can capture it (INCLUDE_EXEC_LOG=0 to
# opt out and keep dumps small).
if [ "${INCLUDE_EXEC_LOG:-1}" = "1" ]; then EXEC_LOG_FLAG=--include_execution_log; else EXEC_LOG_FLAG=--no-include_execution_log; fi
# Reward signal: REWARD_SIGNAL=norm_surprisal (default) -> reward = |normalized
# surprisal|; REWARD_SIGNAL=belief_change -> reward = belief_change / width.
if [ "${REWARD_SIGNAL:-norm_surprisal}" = "belief_change" ]; then RS_FLAG=--no-use_normalized_surprisal; else RS_FLAG=--use_normalized_surprisal; fi
# REWARD_TYPE=codex hands the plan->execute->analyze loop to the Codex CLI
# inside the rollout workers (see rm_hub/codex.py):
# install the codex binary and point the workers at the dataset registry.
# The zero/one/coin control rewards need neither codex nor the reward server.
if [ "${REWARD_TYPE:-server}" = "codex" ]; then
    if ! command -v codex >/dev/null 2>&1; then
        echo "=== installing codex CLI ==="
        (npm install -g @openai/codex 2>/dev/null) || {
            curl -LsSf -o /tmp/codex.tar.gz \
                https://github.com/openai/codex/releases/latest/download/codex-x86_64-unknown-linux-musl.tar.gz \
            && tar -xzf /tmp/codex.tar.gz -C /tmp \
            && install -m 755 /tmp/codex-x86_64-unknown-linux-musl /usr/local/bin/codex; }
    fi
    codex --version || { echo "codex CLI install failed"; exit 1; }
    # codex does NOT pick up OPENAI_API_KEY for its own Responses-API calls
    # (exec fails with 401 "Missing bearer"); it needs an explicit API-key login.
    printf '%s' "$OPENAI_API_KEY" | codex login --with-api-key \
        || codex login --api-key "$OPENAI_API_KEY" \
        || { echo "codex login failed"; exit 1; }
    codex login status || true
    export AUTODISCOVERY_CODEX_REGISTRY="${AUTODISCOVERY_CODEX_REGISTRY:-$DATASET_ROOT/registry.json}"
fi
# The reward server only runs for REWARD_TYPE=server: control rewards never
# score, and the codex reward executes experiments inside the rollout workers.
if [ "${REWARD_TYPE:-server}" = "server" ]; then
( cd "$ASTA_DIR" && uv run --package asta-autodiscovery python -m autodiscovery.slime_reward \
    --dataset_registry "$DATASET_ROOT/registry.json" --host 127.0.0.1 --port 8000 \
    --concurrency "${REWARD_CONCURRENCY:-128}" \
    --execution_model "${EXECUTION_MODEL:-gpt-5.6-luna}" --belief_model "${BELIEF_MODEL:-gpt-5-mini}" \
    --n_belief_samples "${N_BELIEF_SAMPLES:-3}" "$EXEC_LOG_FLAG" "$RS_FLAG" \
    --no-run_data_loading ) > "$REWARD_LOG" 2>&1 &
RM_PID=$!
tail -n +1 -F "$REWARD_LOG" 2>/dev/null & TAIL_PID=$!
trap 'kill $RM_PID $TAIL_PID 2>/dev/null || true; echo "===== reward_server.log ====="; cat "$REWARD_LOG" 2>/dev/null || true' EXIT
echo "waiting for reward server..."
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; sleep 2; done
curl -s http://127.0.0.1:8000/health | head -c 300; echo
else
    echo "[entry] REWARD_TYPE=${REWARD_TYPE}: reward server not started"
fi

# --- 6. use the image's version-matched slime; inject only the autodiscovery RM ---
IMG_SLIME="${IMG_SLIME:-/root/slime}"
[ -f "$IMG_SLIME/train.py" ] || IMG_SLIME="$(cd /root && python3 -c 'import slime,os;print(os.path.dirname(list(slime.__path__)[0]))')"
[ -f "$IMG_SLIME/train.py" ] || { echo "cannot locate image slime"; exit 1; }
[ "$IMG_SLIME" != "$REPO_ROOT" ] && cp slime/rollout/rm_hub/{autodiscovery,reward_utils,zero,one,coin,codex}.py "$IMG_SLIME/slime/rollout/rm_hub/"
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

# --- 8. GRPO: EPOCHS passes over the train split; checkpoint periodically ---
#   steps/epoch = ceil(train_rows / batch); total = EPOCHS * steps_per_epoch.
#   Eval (valid) at each epoch boundary and save every CKPT_INTERVAL steps
#   (default 50). train/valid surprise reward is logged to wandb by slime.
EPOCHS="${EPOCHS:-3}"
BATCH="${ROLLOUT_BATCH_SIZE:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-50}"
TRAIN_ROWS=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('$PROMPT_DIR/${FORMAT}_train.parquet').num_rows)")
STEPS_PER_EPOCH=$(( (TRAIN_ROWS + BATCH - 1) / BATCH ))
TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
CKPT_DIR="${BEAKER_RESULT_DIR:-/results}/ckpt/${FORMAT}"; mkdir -p "$CKPT_DIR"
echo "[entry] train_rows=$TRAIN_ROWS batch=$BATCH steps/epoch=$STEPS_PER_EPOCH epochs=$EPOCHS total_steps=$TOTAL_STEPS ckpt_interval=$CKPT_INTERVAL -> ckpt $CKPT_DIR"

NUM_ROLLOUT="$TOTAL_STEPS" \
    ROLLOUT_BATCH_SIZE="$BATCH" \
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}" \
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}" MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}" \
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-18432}" \
    MODEL="$TRAIN_MODEL" NUM_GPUS="${NUM_GPUS:-8}" TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}" \
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}" \
    DO_EVAL=1 EVAL_INTERVAL="$STEPS_PER_EPOCH" N_SAMPLES_PER_EVAL_PROMPT=1 \
    SAVE_INTERVAL="$CKPT_INTERVAL" SAVE="$CKPT_DIR" \
    RM_URL=http://127.0.0.1:8000/reward \
    HF_CHECKPOINT="$MODEL_DIR/" REF_LOAD="$TD_DIR/" \
    PROMPT_DATA="$PROMPT_DIR/${FORMAT}_train.parquet" \
    EVAL_PROMPT_DATA="valid $PROMPT_DIR/${FORMAT}_val.parquet" \
    MEGATRON_PATH="$MEGATRON_PATH" \
    USE_WANDB=1 WANDB_KEY="${WANDB_KEY:-${SIJIAL_WANDB_API_KEY:-}}" \
    WANDB_PROJECT=autodiscovery-rl WANDB_GROUP="${REWARD_SIGNAL:-norm_surprisal}-${FORMAT}" \
    bash examples/autodiscovery_rl/train_grpo.sh

echo "=== fmt-ablation FORMAT=$FORMAT finished OK ==="
