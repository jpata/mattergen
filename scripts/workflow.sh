#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 40G
#SBATCH --cpus-per-gpu 8
#SBATCH -o logs/slurm-%x-%j-%N.out

set -e
set -x

nvidia-smi -L

RESULTS_DIR="${1:-results}"
mkdir -p $RESULTS_DIR

#1. Generate 16 materials randomly, without any guidance
uv run python3 mattergen/scripts/generate.py \
  $RESULTS_DIR --pretrained-name=mattergen_base \
  --batch_size=16 \
  --num-batches 1
                                                        
#2. Create video of the diffusion step for one trajectory
uv run python3 scripts/render_trajectory_parallel.py \
  --zip $RESULTS_DIR/generated_trajectories.zip \
  --name gen_0.extxyz \
  --out $RESULTS_DIR/gen_0.mp4 \
  --fps 50

# 3. Evaluate structures and produce detailed metrics
uv run python3 mattergen/scripts/evaluate.py \
  --structures_path=$RESULTS_DIR/generated_crystals.extxyz \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_DIR/metrics.json" \
  --save_detailed_as="$RESULTS_DIR/detailed_metrics.json"

# 4. Generate the comprehensive metrics dashboard
uv run python3 scripts/visualize_metrics.py \
  --json $RESULTS_DIR/detailed_metrics.json \
  --out $RESULTS_DIR/detailed_metrics_visualization.png

# 5. Generate the detailed structure grid (final steps + per-material metrics)
uv run python3 scripts/visualize_grid.py \
  --zip $RESULTS_DIR/generated_trajectories.zip \
  --metrics $RESULTS_DIR/detailed_metrics.json \
  --out $RESULTS_DIR/final_structures_grid_detailed.png
