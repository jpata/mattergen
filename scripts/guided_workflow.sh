#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 40G
#SBATCH --cpus-per-gpu 8
#SBATCH -o logs/slurm-%x-%j-%N.out

set -e
set -x

nvidia-smi -L

# This script runs a guided generation workflow using MatterGen.
# It conditions the generation on a specific chemical system.

# Define the results directory
RESULTS_DIR="${1:-results_guided}"
mkdir -p $RESULTS_DIR

# 1. Generate materials conditioned on a chemical system (e.g., Li-Fe-O)
# We use the 'chemical_system' pre-trained model and provide the guidance.
# --diffusion_guidance_factor increases adherence to the condition.
echo "Step 1: Generating materials guided by chemical system Li-Fe-O..."
uv run python3 mattergen/scripts/generate.py \
  $RESULTS_DIR --pretrained-name=chemical_system \
  --batch_size=64 \
  --num-batches 1 \
  --properties_to_condition_on="{'chemical_system':'Li-Fe-O'}" \
  --diffusion_guidance_factor=2.0

# 2. Create video of the diffusion step for the first trajectory
echo "Step 2: Rendering diffusion trajectory video..."
uv run python3 scripts/render_trajectory_parallel.py \
  --zip $RESULTS_DIR/generated_trajectories.zip \
  --name gen_0.extxyz \
  --out $RESULTS_DIR/gen_0.mp4 \
  --fps 50

# 3. Evaluate structures and produce detailed metrics
# This step relaxes the structures and computes stability, novelty, etc.
echo "Step 3: Evaluating structures and computing metrics..."
uv run python3 mattergen/scripts/evaluate.py \
  --structures_path=$RESULTS_DIR/generated_crystals.extxyz \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_DIR/metrics.json" \
  --save_detailed_as="$RESULTS_DIR/detailed_metrics.json"

# 4. Generate the comprehensive metrics dashboard
echo "Step 4: Visualizing evaluation metrics..."
uv run python3 scripts/visualize_metrics.py \
  --json $RESULTS_DIR/detailed_metrics.json \
  --out $RESULTS_DIR/detailed_metrics_visualization.png

# 5. Generate the detailed structure grid (final steps + per-material metrics)
echo "Step 5: Generating detailed structure grid visualization..."
uv run python3 scripts/visualize_grid.py \
  --zip $RESULTS_DIR/generated_trajectories.zip \
  --metrics $RESULTS_DIR/detailed_metrics.json \
  --out $RESULTS_DIR/final_structures_grid_detailed.png

echo "Guided workflow complete. Results are in the '$RESULTS_DIR' directory."
