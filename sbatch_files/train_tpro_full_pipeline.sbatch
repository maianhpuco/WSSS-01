#!/bin/bash
#SBATCH --job-name=train_tpro_pipeline
#SBATCH --output=logs/train_tpro_pipeline.out
#SBATCH --error=logs/train_tpro_pipeline.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

module purge || true

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  conda deactivate || true
fi


cd /project/hnguyen2/mvu9/folder_04_ma/WSSS-01

run_step() {
  local description="$1"
  shift
  echo "[SBATCH][${description}] Starting at $(date)"
  "$@"
  echo "[SBATCH][${description}] Completed at $(date)"
}

run_step "classification" python3 main_train_tpro_cls.py --config configs_maui/tpro_bcss_wsss.yaml
run_step "pseudo-label" python3 main_train_tpro_pseudo.py --config configs_maui/tpro_bcss_wsss.yaml
run_step "segmentation" python3 main_train_tpro_seg.py --config configs_maui/tpro_bcss_wsss.yaml

