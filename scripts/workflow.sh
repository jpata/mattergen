#!/bin/bash

#uv run python3 mattergen/scripts/generate.py \
#  results --pretrained-name=mattergen_base \
#  --batch_size=16 \
#  --num-batches 1

#create video of the diffusion step
uv run python3 scripts/render_trajectory_parallel.py \
  --name gen_0.extxyz \
  --out results/gen_0.mp4 \
  --fps 50

uv run python3 mattergen/scripts/evaluate.py \
  --structures_path=results/generated_crystals.extxyz \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="results/metrics.json" \
  --save_detailed_as="results/detailed_metrics.json"

