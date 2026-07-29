#!/bin/bash
# Two co-located REAL-reward GRPO test jobs comparing training data, both
# evaluated on the dbench+blade cold-start task set. Runs INSIDE slimerl/slime.
#
#   JOB=1  train on dbench_blade_fmt1 (29 dataset-tasks)         eval on dbench_blade_fmt1
#   JOB=2  train on coldstart3_baseline   (3 tcga/nls cold-start)    eval on dbench_blade_fmt1
# 
# Example gantry run command:
# gantry run --workspace ai2/autodiscovery --cluster ai2/jupiter --budget ai2/asta \
#   --gpus 8 --no-python --no-conda --docker-image slimerl/slime:latest \
#   --priority urgent --timeout 0 --task-timeout 6h --yes \
#   --gh-token-secret SIJIAL_AI2_GH_TOKEN \
#   --env CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes \
#   --env-secret WANDB_KEY=SIJIAL_WANDB_API_KEY --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
#   --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
#   --env-secret GITHUB_TOKEN=SIJIAL_AI2_GH_TOKEN --env-secret HF_TOKEN=HF_TOKEN \
#   --env JOB=1 --name autods-fmtcmp-job1-v5 \
#   -- bash examples/autodiscovery_rl/beaker_fmt_compare_entry.sh
# 
# Both post generated hypotheses to the SAME real reward server, whose registry
# is the UNION of every dataset either job needs to score (31 = 29 dbench+blade
# + 2 tcga; nls_raw is shared). Same hyperparams/setup across the two jobs; only
# PROMPT_DATA / EVAL_DATA differ, so a difference in the dbench_blade eval reward
# isolates the effect of the training data.
#
# Data (all on S3, staged out-of-band):
#   discoverybench/real + blade  -> the 29 dbench_blade datasets
#   tcga-breast-cancer, tcga-melanoma (clinical+mutations)  -> the 2 tcga datasets
#   slime_prompt_data/dbench_blade_fmt1_hyps_open.parquet   -> the eval (+ job1 train) set
#   coldstart3_baseline/{train,eval}.jsonl  -> job2 train (committed in this fork)
#
# Required env (gantry --env-secret): OPENAI_API_KEY, AWS_ACCESS_KEY_ID,
#   AWS_SECRET_ACCESS_KEY, WANDB_KEY, GITHUB_TOKEN, HF_TOKEN.
# Knobs: JOB [1], NUM_ROLLOUT [3], ROLLOUT_BATCH_SIZE [8], N_SAMPLES_PER_PROMPT [8],
#   REWARD_CONCURRENCY [8], ASTA_REF [sijia/generator-dev].
set -ex

JOB="${JOB:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

DATASET_ROOT="${DATASET_ROOT:-/root/autods_datasets}"
ASTA_DIR="${ASTA_DIR:-/root/asta-autodiscovery}"
ASTA_REF="${ASTA_REF:-sijia/generator-dev}"
S3="s3://ai2-asta-workspaces/autods/datasets"

# --- install uv (asta 3.13 env + awscli); the slime image has neither ---
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export PATH="$HOME/.local/bin:$PATH"
AWS="uvx --from awscli aws"

# gantry prepends its own miniconda to PATH, and that python lacks torch. The
# slime image installs torch into its system python (pip, cp310), so strip conda
# from PATH to restore python/python3 -> the image's torch-enabled interpreter
# (used by convert_hf_to_torch_dist.py and train.py). uv (~/.local/bin) is kept.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -viE conda | paste -sd: -)"
python3 -c 'import torch' 2>/dev/null \
    && echo "[entry] torch python: $(command -v python3)" \
    || echo "[entry] WARN: python3 still lacks torch at $(command -v python3)"

# --- 1. clone asta-autodiscovery (has autodiscovery.slime_reward) ---
if [ ! -d "$ASTA_DIR/.git" ]; then
    git clone --depth 1 --branch "$ASTA_REF" \
        "https://github.com/allenai/asta-autodiscovery.git" "$ASTA_DIR"
fi

# --- 2. fetch every dataset the reward server must score (union of both jobs) ---
mkdir -p "$DATASET_ROOT"
# 29 dbench+blade: mirror the S3 subtrees verbatim (layout the loaders require)
$AWS s3 sync "$S3/discoverybench/real" "$DATASET_ROOT/discoverybench/real" --only-show-errors
$AWS s3 sync "$S3/blade"               "$DATASET_ROOT/blade"               --only-show-errors --exclude "LICENSE"
# 2 tcga (asta): trimmed metadata (committed) + clinical/mutations from S3
for id in tcga-breast-cancer tcga-melanoma; do
    mkdir -p "$DATASET_ROOT/$id"
    cp "$SCRIPT_DIR/reward_metadata/$id/metadata.json" "$DATASET_ROOT/$id/metadata.json"
    $AWS s3 cp "$S3/$id/cleaned_clinical_data.csv" "$DATASET_ROOT/$id/" --only-show-errors
    $AWS s3 cp "$S3/$id/cleaned_mutations.csv"     "$DATASET_ROOT/$id/" --only-show-errors
done

# --- 3. build the union registry (dbench + blade + asta) from what we fetched ---
python3 - "$DATASET_ROOT" <<'PY'
import glob, json, os, re, sys
root = sys.argv[1]
reg = {}
for split in ("train", "test"):
    base = f"{root}/discoverybench/real/{split}"
    for ds in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        metas = sorted(glob.glob(f"{base}/{ds}/metadata_*.json"),
                       key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))
        if metas:
            reg[ds] = {"dataset_metadata": metas[0], "dataset_metadata_type": "dbench"}
bbase = f"{root}/blade"
for ds in sorted(os.listdir(bbase)) if os.path.isdir(bbase) else []:
    info = f"{bbase}/{ds}/info.json"
    if os.path.isfile(info):
        reg[ds] = {"dataset_metadata": info, "dataset_metadata_type": "blade"}
for ds in ("tcga-breast-cancer", "tcga-melanoma"):
    m = f"{root}/{ds}/metadata.json"
    if os.path.isfile(m):
        reg[ds] = {"dataset_metadata": m, "dataset_metadata_type": "asta"}
json.dump(reg, open(f"{root}/registry.json", "w"), indent=2)
print(f"registry: {len(reg)} datasets -> {root}/registry.json")
PY

# --- 4. fetch the dbench_blade eval/train parquet from S3 ---
PROMPT_DIR="$DATASET_ROOT/prompt_data"; mkdir -p "$PROMPT_DIR"
$AWS s3 cp "$S3/slime_prompt_data/dbench_blade_fmt1_hyps_open.parquet" "$PROMPT_DIR/" --only-show-errors
DBENCH_BLADE_PARQUET="$PROMPT_DIR/dbench_blade_fmt1_hyps_open.parquet"

# --- 5. start the REAL reward server (uv/py3.13), localhost:8000 ---
REWARD_LOG_DIR="${BEAKER_RESULT_DIR:-/results}"; mkdir -p "$REWARD_LOG_DIR" 2>/dev/null || REWARD_LOG_DIR=/root
REWARD_LOG="$REWARD_LOG_DIR/reward_server.log"
( cd "$ASTA_DIR" && uv run --package asta-autodiscovery python -m autodiscovery.slime_reward \
    --dataset_registry "$DATASET_ROOT/registry.json" --host 127.0.0.1 --port 8000 \
    --concurrency "${REWARD_CONCURRENCY:-8}" \
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
if [ "$IMG_SLIME" != "$REPO_ROOT" ]; then
    cp slime/rollout/rm_hub/autodiscovery.py "$IMG_SLIME/slime/rollout/rm_hub/autodiscovery.py"
fi
SGL_ARGS_PY="$IMG_SLIME/slime/backends/sglang_utils/arguments.py"
[ -f "$SGL_ARGS_PY" ] && sed -i -E 's/^([[:space:]]*)args\.(sglang_[a-z0-9_]+) = args\.(sglang_[a-z0-9_]+)$/\1args.\2 = getattr(args, "\3", args.\2)/' "$SGL_ARGS_PY" || true
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

# --- 8. pick train/eval data by JOB; run GRPO (small test sizing) ---
if [ "$JOB" = "1" ]; then
    PROMPT_DATA="$DBENCH_BLADE_PARQUET"; WANDB_GROUP="fmtcmp-job1-train-dbench_blade"
elif [ "$JOB" = "2" ]; then
    PROMPT_DATA="$SCRIPT_DIR/coldstart3_baseline/train.jsonl"; WANDB_GROUP="fmtcmp-job2-train-coldstart3"
else
    echo "JOB must be 1 or 2"; exit 1
fi
EVAL_DATA="$DBENCH_BLADE_PARQUET"   # both jobs evaluate on the dbench_blade tasks

NUM_ROLLOUT="${NUM_ROLLOUT:-3}" \
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}" \
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}" \
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}" MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}" \
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-18432}" \
    MODEL="$TRAIN_MODEL" NUM_GPUS="${NUM_GPUS:-8}" TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}" \
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}" \
    DO_EVAL=1 EVAL_INTERVAL="${EVAL_INTERVAL:-3}" N_SAMPLES_PER_EVAL_PROMPT=1 SAVE_INTERVAL=9999 \
    RM_URL=http://127.0.0.1:8000/reward \
    HF_CHECKPOINT="$MODEL_DIR/" REF_LOAD="$TD_DIR/" \
    PROMPT_DATA="$PROMPT_DATA" EVAL_DATA="$EVAL_DATA" \
    MEGATRON_PATH="$MEGATRON_PATH" \
    USE_WANDB=1 WANDB_KEY="${WANDB_KEY:-${SIJIAL_WANDB_API_KEY:-}}" \
    WANDB_PROJECT=autodiscovery-rl WANDB_GROUP="$WANDB_GROUP" \
    bash examples/autodiscovery_rl/train_grpo.sh

echo "=== fmt-compare JOB=$JOB finished OK ==="
