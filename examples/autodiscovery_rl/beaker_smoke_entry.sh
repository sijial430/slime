#!/bin/bash
# In-job entry for the AutoDiscovery GRPO pipeline smoke test on Beaker.
#
# Runs INSIDE the slime training image (slimerl/slime:latest) via gantry:
#   1. re-point the `slime` editable install at THIS fork (which has the
#      autodiscovery reward model + this example) instead of the image's /root/slime,
#   2. download the tiny base model (Qwen2.5-0.5B-Instruct) and convert it to
#      the Megatron torch_dist format slime needs,
#   3. start the dependency-free mock reward server, co-located (localhost),
#   4. run train_grpo.sh SMOKE=1 — one tiny rollout exercising the full
#      rollout -> reward -> policy-update -> wandb loop, then exit.
#
# The mock reward validates the LOOP only (deterministic pseudo-surprise). For
# real training, run `python -m autodiscovery.slime_reward` with the real
# datasets on weka and point RM_URL at it.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export PYTHONUNBUFFERED=1

# 1. Use THIS fork's slime (autodiscovery RM + example), not the image's /root/slime.
pip install -e . --no-deps

MODEL_DIR="${MODEL_DIR:-/root/Qwen2.5-0.5B-Instruct}"
TD_DIR="${TD_DIR:-/root/Qwen2.5-0.5B-Instruct_torch_dist}"

# 2a. Download the base model (idempotent).
if [ ! -f "$MODEL_DIR/config.json" ]; then
    hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODEL_DIR" \
        || huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODEL_DIR"
fi

# 2b. Convert HF -> Megatron torch_dist (idempotent).
source scripts/models/qwen2.5-0.5B.sh
if [ ! -d "$TD_DIR" ]; then
    PYTHONPATH="$MEGATRON_PATH" python tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" --hf-checkpoint "$MODEL_DIR" --save "$TD_DIR"
fi

# 3. Dependency-free mock reward server, co-located.
python examples/autodiscovery_rl/mock_reward_server_standalone.py \
    --registry examples/autodiscovery_rl/data_smoke/registry.json \
    --host 127.0.0.1 --port 8137 &
RM_PID=$!
trap 'kill $RM_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:8137/health >/dev/null 2>&1 && break
    sleep 1
done
echo "reward server health: $(curl -s http://127.0.0.1:8137/health)"

# 4. GRPO smoke: one rollout, group of 8, mock reward, wandb on.
SMOKE=1 \
    RM_URL=http://127.0.0.1:8137/reward \
    HF_CHECKPOINT="$MODEL_DIR/" REF_LOAD="$TD_DIR/" \
    DATA_DIR="$REPO_ROOT/examples/autodiscovery_rl/data_smoke" \
    MEGATRON_PATH="$MEGATRON_PATH" \
    USE_WANDB="${USE_WANDB:-1}" WANDB_KEY="${WANDB_KEY:-${SIJIAL_WANDB_API_KEY:-}}" \
    WANDB_PROJECT="${WANDB_PROJECT:-autodiscovery-rl}" \
    WANDB_GROUP="${WANDB_GROUP:-smoke-qwen0.5b}" \
    bash examples/autodiscovery_rl/train_grpo.sh

echo "=== autodiscovery smoke entry finished OK ==="
