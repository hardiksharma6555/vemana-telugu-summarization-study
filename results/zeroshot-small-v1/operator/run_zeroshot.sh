#!/bin/bash
set -uo pipefail
cd ~/vemana-zeroshot
export CUDA_VISIBLE_DEVICES=GPU-2544ad3e-a8af-7b58-55e6-c24c57497b01 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=.venv-zs/bin/python
R=results/zeroshot-small-v1
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > operator_logs/gpu.txt 2>&1
git rev-parse HEAD > operator_logs/commit.txt
$PY scripts/zeroshot_small_models.py generate --model lead --output-dir $R/lead
$PY scripts/zeroshot_small_models.py generate --model indicbart-xlsum --device cuda --output-dir $R/indicbart-xlsum
$PY scripts/zeroshot_small_models.py generate --model sentsum-ss --device cuda --output-dir $R/sentsum-ss
echo ALL_DONE
