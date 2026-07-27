#!/bin/bash
# GRPO training for the AutoDiscovery Bayesian-surprise reward task.
#
# Each row of the lineage prompt data is ONE task: a single MCTS node whose
# context is the root->parent hypothesis lineage. The policy samples
# N_SAMPLES_PER_PROMPT hypotheses for that node (GRPO group), and each sampled
# hypothesis is scored by the AutoDiscovery reward server (it plans + runs the
# experiment and returns the surprise reward). GRPO normalizes rewards within
# the group of N.
#
# Prerequisites:
#   - Base model: an HF checkpoint (HF_CHECKPOINT) and a Megatron torch_dist
#     checkpoint (REF_LOAD, produced by tools/convert_hf_to_torch_dist.py).
#   - Prompt data from prepare_lineage_data.py:
#       DATA_DIR/train.parquet, DATA_DIR/eval.parquet, DATA_DIR/registry.json
#   - A reachable reward server at RM_URL that routes on the dataset_ids in the
#     data. Real: `python -m autodiscovery.slime_reward --dataset_registry <reg>`.
#     GPU-free smoke: scripts/slime/mock_reward_server.py (same flags).
#
# Env knobs (defaults in []):
#   MODEL [qwen2.5-0.5B]  HF_CHECKPOINT  REF_LOAD  SAVE
#   DATA_DIR [<script>/data]  RM_URL [http://127.0.0.1:8000/reward]  NUM_GPUS [1]
#   N_SAMPLES_PER_PROMPT [8]  ROLLOUT_BATCH_SIZE  NUM_ROLLOUT  MAX_RESPONSE_LEN
#   GLOBAL_BATCH_SIZE [ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT]
#   DO_EVAL [1]  EVAL_INTERVAL [20]
#   USE_WANDB [0]  WANDB_PROJECT [autodiscovery-rl]  WANDB_GROUP  WANDB_TEAM [sijial-ai2]  WANDB_KEY
#   SMOKE [0]  -> 1 runs a single tiny rollout and exits (loop validation)
#
# Example (GPU-free-reward smoke on one GPU):
#   SMOKE=1 RM_URL=http://127.0.0.1:8137/reward USE_WANDB=1 \
#     WANDB_KEY=$SIJIAL_WANDB_API_KEY bash examples/autodiscovery_rl/train_grpo.sh

set -ex

# Clean any leftover ray / sglang from a previous run.
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 2

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# SLIME_ROOT is the slime repo used for train.py + model args. Override it (e.g.
# to the docker image's /root/slime) when the fork's slime is not version-matched
# to the image's bundled sglang.
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# --- Model architecture (sourced from slime/scripts/models/<MODEL>.sh) --------
MODEL="${MODEL:-qwen2.5-0.5B}"
source "${SLIME_ROOT}/scripts/models/${MODEL}.sh"

HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Qwen2.5-0.5B-Instruct/}"
REF_LOAD="${REF_LOAD:-/root/Qwen2.5-0.5B-Instruct_torch_dist/}"
SAVE="${SAVE:-/tmp/autodiscovery_grpo_save/}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
# PROMPT_DATA/EVAL_DATA override the default DATA_DIR/{train,eval}.parquet (e.g.
# cold-start jsonl). slime reads both .parquet and .jsonl.
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/train.parquet}"
EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/eval.parquet}"
RM_URL="${RM_URL:-http://127.0.0.1:8000/reward}"
NUM_GPUS="${NUM_GPUS:-1}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

# --- GRPO group size + rollout sizing ----------------------------------------
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
if [ "${SMOKE:-0}" = "1" ]; then
    NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-9999}"
else
    NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${REF_LOAD}"
    --save "${SAVE}"
    --save-interval "${SAVE_INTERVAL}"
)

# Prompt data: our lineage rows carry `messages` (chat prompt) and `metadata`
# (dataset_id for reward routing). No --label-key (reward is remote) and no
# --rm-type (the custom RM path is used instead).
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --metadata-key metadata
    --apply-chat-template
    --rollout-shuffle

    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --num-steps-per-rollout 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"

    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-temperature 1
)

# AutoDiscovery surprise reward (remote server; already in the slime package).
CUSTOM_RM_ARGS=(
    --custom-rm-path slime.rollout.rm_hub.autodiscovery.autodiscovery_rm
    --rm-url "${RM_URL}"
)

EVAL_ARGS=()
if [ "${DO_EVAL:-1}" = "1" ] && [ "${SMOKE:-0}" != "1" ]; then
    EVAL_ARGS=(
        --eval-prompt-data autodiscovery "${EVAL_DATA}"
        --eval-interval "${EVAL_INTERVAL:-20}"
        --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    )
fi

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.4}"
    # slime runs the co-located engine with memory-saver on (to offload sglang
    # during the training step). This image's sglang defaults the prefill
    # CUDA-graph backend to `breakable`, which refuses to coexist with memory
    # saver ("Breakable CUDA graph is not compatible with memory saver mode").
    # CUDA graphs buy ~nothing for a 0.5B smoke; disable capture entirely.
    --sglang-disable-cuda-graph
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT:-autodiscovery-rl}"
        --wandb-group "${WANDB_GROUP:-grpo-${MODEL}}"
        --wandb-team "${WANDB_TEAM:-sijial-ai2}"
        --wandb-key "${WANDB_KEY:-${SIJIAL_WANDB_API_KEY:-}}"
    )
fi

# --- Launch: ray head + train.py ---------------------------------------------
# cd into SLIME_ROOT so `import slime` (and ray's working_dir upload) resolves to
# the image's version-matched slime, not a fork checkout on the CWD path.
cd "${SLIME_ROOT}"
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{\"env_vars\": {\"PYTHONPATH\": \"${MEGATRON_PATH}\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}" \
    -- python3 "${SLIME_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${CUSTOM_RM_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
