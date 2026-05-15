#!/usr/bin/env bash
set -euo pipefail

# Discover tuned feature configs named:
#   configs/features/best_{model}_d_{d}_h_{evaluate.horizon_h}.yaml
# and train each matching model/config pair.

WANDB_OVERRIDE="${WANDB_OVERRIDE:-wandb.enabled=false}"

shopt -s nullglob
feature_paths=(configs/features/best_*_d_*_h_*.yaml)
shopt -u nullglob

if (( ${#feature_paths[@]} == 0 )); then
  echo "No tuned feature configs found at configs/features/best_*_d_*_h_*.yaml"
  exit 1
fi

for feature_path in "${feature_paths[@]}"; do
  feature_name="$(basename "$feature_path" .yaml)"

  # Example: best_linear_l2_d_7_h_11
  stem="${feature_name#best_}"
  model_key="${stem%_d_*}"
  d_part="${stem#*_d_}"
  d="${d_part%%_h_*}"
  h="${stem##*_h_}"

  case "$model_key" in
    xgb)
      model="xgb"
      ;;
    linear_l1)
      model="linear_l1"
      ;;
    linear_l2)
      model="linear_l2"
      ;;
    glm|negbin_glm)
      model="glm"
      ;;
    glm_poisson|poisson_glm)
      model="glm_poisson"
      ;;
    xgb_sarimax)
      model="xgb_sarimax"
      ;;
    *)
      echo "Skipping unknown model key '${model_key}' from ${feature_name}"
      continue
      ;;
  esac

  echo "Training model=${model} features=${feature_name} d=${d} h=${h}"

  python3 -m modeling.train \
    "model=${model}" \
    "features=${feature_name}" \
    "evaluate.horizon_h=${h}" \
    "${WANDB_OVERRIDE}"
done
