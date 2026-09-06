#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/gi_hsp_training.yaml"
  "configs/flow_only_training.yaml"
  "configs/topology_only_training.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "========================================"
  echo "Running: $config"
  echo "========================================"

  python -m models.proposed.train_gi_hsp \
    --config "$config"

  echo
done
