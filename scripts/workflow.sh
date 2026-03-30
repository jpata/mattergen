#!/bin/bash

#1. Generate 16 materials randomly, without any guidance
uv run python3 mattergen/scripts/generate.py \
  results --pretrained-name=mattergen_base \
  --batch_size=16 \
  --num-batches 1
                                                        
#2. Create video of the diffusion step for one trajectory
uv run python3 scripts/render_trajectory_parallel.py \
  --name gen_0.extxyz \
  --out results/gen_0.mp4 \
  --fps 50

# 3. Evaluate structures and produce detailed metrics
uv run python3 mattergen/scripts/evaluate.py \
  --structures_path=results/generated_crystals.extxyz \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="results/metrics.json" \
  --save_detailed_as="results/detailed_metrics.json"

# 4. Generate the comprehensive metrics dashboard
uv run python3 scripts/visualize_metrics.py

# 5. Generate the detailed structure grid (final steps + per-material metrics)
uv run python3 scripts/visualize_grid.py \
  --zip results/generated_trajectories.zip \
  --metrics results/detailed_metrics.json \
  --out results/final_structures_grid_detailed.png
