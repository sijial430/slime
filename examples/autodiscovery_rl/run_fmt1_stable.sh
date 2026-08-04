#!/bin/bash
# Conservative fmt1 recipe: reward learning with an explicit response-length
# anchor. This wraps beaker_fmt_ablation_entry.sh; it does not submit a job.
#
# The 7,168-token target is the rounded center of the original fmt1 run
# (roughly 7.1k tokens). Responses inside +/-768 tokens receive no length
# penalty. Outside that band, the small penalty opposes drift without replacing
# normalized surprisal as the optimization target.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export FORMAT="${FORMAT:-fmt1_hyps_open}"
export REWARD_SIGNAL="${REWARD_SIGNAL:-norm_surprisal}"
export EXECUTION_MODEL="${EXECUTION_MODEL:-gpt-5-mini}"

# 4x4 gives 16 scored samples/update and makes the expensive RM sweep practical.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export EPOCHS="${EPOCHS:-3}"

# The local 5e-7 fmt1 baseline was length-stable but reward-flat. Use a larger
# update with enough KL to keep the policy close to the response-length prior.
export LR="${LR:-3e-6}"
export KL_COEF="${KL_COEF:-0.03}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"

export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-12288}"
export LENGTH_CONTROL=1
export AUTODISCOVERY_LENGTH_TARGET="${AUTODISCOVERY_LENGTH_TARGET:-7168}"
export AUTODISCOVERY_LENGTH_DEADBAND="${AUTODISCOVERY_LENGTH_DEADBAND:-768}"
export AUTODISCOVERY_LENGTH_COEF="${AUTODISCOVERY_LENGTH_COEF:-0.05}"

exec bash "$SCRIPT_DIR/beaker_fmt_ablation_entry.sh"
