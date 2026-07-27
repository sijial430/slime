#!/bin/bash
# In-job entry for the CO-LOCATED REAL-reward AutoDiscovery GRPO smoke on Beaker.
#
# Runs INSIDE the slime training image (slimerl/slime:latest). Unlike the mock
# entry, the reward here is the ACTUAL autodiscovery SurpriseRewardScorer: it
# plans + executes a real experiment per hypothesis and elicits prior/posterior
# beliefs. Because autodiscovery needs Python 3.13 (the slime image is 3.10), the
# reward server runs in a SEPARATE uv-managed env and slime talks to it over
# localhost HTTP -- no dependency conflict.
#
# Pipeline:
#   1. clone asta-autodiscovery (has autodiscovery.slime_reward) via GITHUB_TOKEN,
#   2. fetch the 3 scoreable datasets (tcga-breast/melanoma clinical+mutations,
#      nls_raw) from S3 into $DATASET_ROOT, alongside the committed trimmed
#      metadata, and write the reward registry,
#   3. start the real reward server (uv run --package asta-autodiscovery
#      python -m autodiscovery.slime_reward) on localhost:8000,
#   4. re-point slime at THIS fork, download + convert Qwen2.5-0.5B,
#   5. run train_grpo.sh on the cold-start prompts (data_smoke_cold), posting
#      each sampled hypothesis to the real reward server. NUM_ROLLOUT steps, wandb.
#
# Required env (gantry --env-secret): OPENAI_API_KEY, AWS_ACCESS_KEY_ID,
#   AWS_SECRET_ACCESS_KEY, WANDB_KEY, GITHUB_TOKEN (to clone the private asta repo).
# Knobs: NUM_ROLLOUT [5], ROLLOUT_BATCH_SIZE [1], N_SAMPLES_PER_PROMPT [8],
#   ASTA_REF [sijia/generator-dev], REWARD_CONCURRENCY [4].
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

DATASET_ROOT="${DATASET_ROOT:-/root/autods_datasets}"
ASTA_DIR="${ASTA_DIR:-/root/asta-autodiscovery}"
ASTA_REF="${ASTA_REF:-sijia/generator-dev}"
S3_DATASETS="s3://ai2-asta-workspaces/autods/datasets"

# --- install uv (for the asta 3.13 env + awscli); the slime image has neither ---
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export PATH="$HOME/.local/bin:$PATH"
AWS="uvx --from awscli aws"

# --- 1. clone asta-autodiscovery (beaker's git credential helper provides auth) ---
if [ ! -d "$ASTA_DIR/.git" ]; then
    git clone --depth 1 --branch "$ASTA_REF" \
        "https://github.com/allenai/asta-autodiscovery.git" "$ASTA_DIR"
fi

# --- 2. fetch the 3 datasets + write registry ---
mkdir -p "$DATASET_ROOT"
fetch() {  # <id> <s3-subdir> <file>...
    local id="$1" sub="$2"; shift 2
    mkdir -p "$DATASET_ROOT/$id"
    cp "$SCRIPT_DIR/reward_metadata/$id/metadata.json" "$DATASET_ROOT/$id/metadata.json"
    for f in "$@"; do $AWS s3 cp "$S3_DATASETS/$sub/$f" "$DATASET_ROOT/$id/"; done
}
# tcga: clinical + mutations only (trimmed metadata skips the multi-hundred-MB gene file)
fetch tcga-breast-cancer tcga-breast-cancer  cleaned_clinical_data.csv cleaned_mutations.csv
fetch tcga-melanoma      tcga-melanoma        cleaned_clinical_data.csv cleaned_mutations.csv
fetch nls_raw            discoverybench/real/test/nls_raw  nls_raw.csv

python3 - "$DATASET_ROOT" <<'PY'
import json, sys
root = sys.argv[1]
reg = {
    "tcga-breast-cancer": f"{root}/tcga-breast-cancer/metadata.json",
    "tcga-melanoma":      f"{root}/tcga-melanoma/metadata.json",
    "nls_raw": {"dataset_metadata": f"{root}/nls_raw/metadata.json", "dataset_metadata_type": "dbench"},
}
json.dump(reg, open(f"{root}/registry.json", "w"), indent=2)
print("wrote", f"{root}/registry.json")
PY

# --- 3. start the REAL reward server (separate uv/py3.13 env), localhost:8000 ---
# Persist the reward server's execution logs (experiment runs + belief elicitation
# per hypothesis): write into the Beaker result dir so they're saved as a
# downloadable result-dataset artifact, and dump the file to stdout on exit so
# it always lands in `beaker experiment logs` too (even on timeout/crash).
REWARD_LOG_DIR="${BEAKER_RESULT_DIR:-/results}"
mkdir -p "$REWARD_LOG_DIR" 2>/dev/null || REWARD_LOG_DIR=/root
REWARD_LOG="$REWARD_LOG_DIR/reward_server.log"
# Execution logs are still PERSISTED server-side (the $REWARD_LOG above), we just
# don't return them per-request to fold into training for now -> default off.
# Flip INCLUDE_EXEC_LOG=1 to have the server attach execution_log to each reward.
if [ "${INCLUDE_EXEC_LOG:-0}" = "1" ]; then EXEC_LOG_FLAG=--include_execution_log; else EXEC_LOG_FLAG=--no-include_execution_log; fi
( cd "$ASTA_DIR" && uv run --package asta-autodiscovery python -m autodiscovery.slime_reward \
    --dataset_registry "$DATASET_ROOT/registry.json" --host 127.0.0.1 --port 8000 \
    --concurrency "${REWARD_CONCURRENCY:-16}" \
    --execution_model "${EXECUTION_MODEL:-gpt-5-mini}" --belief_model "${BELIEF_MODEL:-gpt-5-mini}" \
    --n_belief_samples "${N_BELIEF_SAMPLES:-5}" "$EXEC_LOG_FLAG" \
    --no-run_data_loading ) > "$REWARD_LOG" 2>&1 &
RM_PID=$!
# Stream the reward server log to stdout (beaker logs) live so gpt rate-limit
# errors ([reward-error]) and per-hypothesis belief spread ([belief]) are visible
# during the run, not only in the persisted file at exit.
tail -n +1 -F "$REWARD_LOG" 2>/dev/null &
TAIL_PID=$!
trap 'kill $RM_PID $TAIL_PID 2>/dev/null || true; echo "===== BEGIN reward_server.log ($REWARD_LOG) ====="; cat "$REWARD_LOG" 2>/dev/null || true; echo "===== END reward_server.log ====="' EXIT
echo "waiting for real reward server (first request builds agents; datasets load lazily)..."
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; sleep 2; done
curl -s http://127.0.0.1:8000/health | head -c 200; echo

# --- 4. use the IMAGE's version-matched slime (train.py + tools), and only
#        inject the autodiscovery RM into it. The fork's slime is not
#        version-matched to the image's bundled sglang (arg-name skew), so we do
#        NOT pip install -e the fork. The fork's example scripts + data (this
#        checkout) are still used; only the slime PACKAGE comes from the image. ---
# The image installs slime at /root/slime (see docker/Dockerfile). Its editable
# install has slime.__file__ == None, so detect via __path__ with a hard fallback.
IMG_SLIME="${IMG_SLIME:-/root/slime}"
if [ ! -f "$IMG_SLIME/train.py" ]; then
    IMG_SLIME="$(cd /root && python3 -c 'import slime, os; print(os.path.dirname(list(slime.__path__)[0]))')"
fi
[ -f "$IMG_SLIME/train.py" ] || { echo "cannot locate image slime (train.py) at '$IMG_SLIME'"; exit 1; }
if [ "$IMG_SLIME" != "$REPO_ROOT" ]; then
    cp slime/rollout/rm_hub/autodiscovery.py "$IMG_SLIME/slime/rollout/rm_hub/autodiscovery.py"
fi
# Patch a slime<->sglang arg-name skew in slimerl/slime:latest: validate_args
# has a block of `args.sglang_<short> = args.sglang_<long>_parallel_size` aliases,
# but this image's sglang only defines the short names -> AttributeError. Rewrite
# every such alias to fall back to the existing (short) attribute.
SGL_ARGS_PY="$IMG_SLIME/slime/backends/sglang_utils/arguments.py"
if [ -f "$SGL_ARGS_PY" ]; then
    sed -i -E 's/^([[:space:]]*)args\.(sglang_[a-z0-9_]+) = args\.(sglang_[a-z0-9_]+)$/\1args.\2 = getattr(args, "\3", args.\2)/' "$SGL_ARGS_PY" || true
fi
export SLIME_ROOT="$IMG_SLIME"
echo "using image slime at $IMG_SLIME (autodiscovery RM injected)"

# --- model: Qwen3.5-9B. Qwen3.5 is a slime-supported family (scripts/models/ +
#     slime_plugins mbridge/models qwen3_5). It's a VLM; the mbridge maps the
#     `model.language_model.*` text tower into Megatron and we train text-only.
#     Prefer the image's native qwen3.5 support; if the image predates it, fill
#     in the arch script + slime_plugins (qwen3_5 mbridge + megatron spec) from
#     this fork so `--spec slime_plugins.models.qwen3_5` and the bridge resolve.
TRAIN_MODEL="${TRAIN_MODEL:-qwen3.5-9B}"
HF_REPO="${HF_REPO:-Qwen/Qwen3.5-9B}"
MODEL_DIR="/root/$(basename "$HF_REPO")"
TD_DIR="${MODEL_DIR}_torch_dist"
if [ "$IMG_SLIME" != "$REPO_ROOT" ]; then
    [ -f "$IMG_SLIME/scripts/models/${TRAIN_MODEL}.sh" ] || \
        cp "$REPO_ROOT/scripts/models/${TRAIN_MODEL}.sh" "$IMG_SLIME/scripts/models/${TRAIN_MODEL}.sh"
    # -n: only add files the image is missing, never clobber the image's own.
    cp -rn "$REPO_ROOT/slime_plugins/." "$IMG_SLIME/slime_plugins/" 2>/dev/null || true
fi
[ -f "$MODEL_DIR/config.json" ] || hf download "$HF_REPO" --local-dir "$MODEL_DIR"
source "$IMG_SLIME/scripts/models/${TRAIN_MODEL}.sh"
[ -d "$TD_DIR" ] || PYTHONPATH="$MEGATRON_PATH:$IMG_SLIME" python "$IMG_SLIME/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" --hf-checkpoint "$MODEL_DIR" --save "$TD_DIR"

# --- 5. GRPO on cold-start prompts, real reward, wandb ---
# NOTE: cold-start prompt data is jsonl; the reward is minutes/hypothesis, so keep
# rollout-batch-size small.
NUM_ROLLOUT="${NUM_ROLLOUT:-5}" \
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}" \
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}" \
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}" \
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}" \
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-18432}" \
    MODEL="$TRAIN_MODEL" \
    NUM_GPUS="${NUM_GPUS:-8}" TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}" \
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}" \
    DO_EVAL=0 SAVE_INTERVAL=9999 \
    RM_URL=http://127.0.0.1:8000/reward \
    HF_CHECKPOINT="$MODEL_DIR/" REF_LOAD="$TD_DIR/" \
    DATA_DIR="$SCRIPT_DIR/data_smoke_cold" \
    PROMPT_DATA="$SCRIPT_DIR/data_smoke_cold/train.jsonl" \
    MEGATRON_PATH="$MEGATRON_PATH" \
    USE_WANDB=1 WANDB_KEY="${WANDB_KEY:-${SIJIAL_WANDB_API_KEY:-}}" \
    WANDB_PROJECT=autodiscovery-rl WANDB_GROUP="real-reward-${TRAIN_MODEL}-${NUM_ROLLOUT}steps" \
    bash examples/autodiscovery_rl/train_grpo.sh

echo "=== real-reward co-located smoke finished OK ==="
