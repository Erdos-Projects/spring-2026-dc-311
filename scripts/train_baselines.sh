#!/usr/bin/env bash
set -euo pipefail

# Train baseline models for fixed target horizon / rollout settings.

WANDB_OVERRIDE="${WANDB_OVERRIDE:-wandb.enabled=false}"

baseline_models=(
  sarima_random_walk
  sarima_seasonal_random_walk
  naive_yearly
  naive_weekly
)

pairs=(
  "1 1"
  "1 5"
  "7 7"
  "7 11"
)

for pair in "${pairs[@]}"; do
  read -r d h <<< "$pair"

  for model in "${baseline_models[@]}"; do
    echo "Training baseline model=${model} d=${d} h=${h}"

    python3 -m modeling.train \
      "model=${model}" \
      "features.d=${d}" \
      "evaluate.horizon_h=${h}" \
      "${WANDB_OVERRIDE}"
  done
done
