#!/usr/bin/env bash
# ── VGD — run inference ────────────────────────────────────────────────────
#
# Setup (one-time):
#   pip install -r requirements.txt
#   pip install -e .          # installs bluestar, vgd, vgd_llava as packages
#
# Usage:
#   bash run.sh
#
# Override any config key on the command line with omegaconf syntax, e.g.:
#   bash run.sh "data_dir=['/path/to/image.png']" "seed=42"
# ──────────────────────────────────────────────────────────────────────────

DEVICES=0

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
  --nproc_per_node=1 \
  --master_port=1235 \
  vgd/inference.py \
  --config=vgd/config/vgd.yaml \
  "wandb.mode=disabled" \
  "data_dir=['/datasets/lexica.art/images/0.png']" \
  "seed=0" \
  "dataset_name=prompt_inversion" \
  "save_dir=./prompt_inversion/" \
  "model.max_length=77" \
  "model.min_length=60" \
  "$@"
