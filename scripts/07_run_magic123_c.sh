#!/bin/bash
# ============================================================
# 07_run_magic123_c.sh — Single Image to 3D with Magic123 (Object C)
# FIXED VERSION:
#   - Correct text prompt matching the actual object (teapot)
#   - Full two-stage pipeline: coarse NeRF (5000 iters) + DMTet fine (3000 iters)
#   - Proper argument names and values
#   - Outputs textured mesh to outputs/object_c/magic123/
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate task1

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/root/CV/task1/repos/Magic123:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=/root/miniconda3/envs/task1/lib:/root/miniconda3/envs/task1/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /root/CV/task1/repos/Magic123

INPUT_IMAGE=/root/CV/task1/data/object_c/rgba.png
DATA_DIR=/root/CV/task1/data/object_c
OUTPUT_DIR=/root/CV/task1/outputs/object_c/magic123
LOG=/root/CV/task1/logs/07_magic123_c.log
WORKSPACE_COARSE=out/magic123_object_c_coarse
WORKSPACE_FINE=out/magic123_object_c_fine

echo "============================================================" | tee "$LOG"
echo "  Magic123 Single-Image-to-3D for Object C (FIXED)" | tee -a "$LOG"
echo "  Started at: $(date)" | tee -a "$LOG"
echo "  Input: $INPUT_IMAGE" | tee -a "$LOG"
echo "  GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# ── Step 0: Preprocess image (generate depth if needed) ──
if [ -f "$DATA_DIR/depth.png" ] && [ -f "$DATA_DIR/rgba.png" ]; then
    echo "[0] Preprocessing already done (depth.png + rgba.png exist)" | tee -a "$LOG"
else
    echo "[0] Preprocessing image (background removal + depth)..." | tee -a "$LOG"
    python preprocess_image.py --path "$INPUT_IMAGE" 2>&1 | tee -a "$LOG"
    echo "[0] Preprocessing complete." | tee -a "$LOG"
fi

# ── Step 1: Coarse NeRF optimization ──
# Uses SDS (Stable Diffusion) + Zero123 guidance to learn a coarse 3D representation
echo "" | tee -a "$LOG"
echo "[1/2] Coarse NeRF optimization (5000 iterations)..." | tee -a "$LOG"
echo "  Guidance: SD (1.0) + Zero123 (40.0)" | tee -a "$LOG"
echo "  Started at: $(date)" | tee -a "$LOG"

python main.py -O \
    --text "A high-resolution DSLR image of a teapot" \
    --sd_version 1.5 \
    --image "$INPUT_IMAGE" \
    --workspace "$WORKSPACE_COARSE" \
    --optim adam \
    --seed 0 \
    --iters 5000 \
    --guidance SD zero123 \
    --lambda_guidance 1.0 40 \
    --guidance_scale 100 5 \
    --bg_radius -1 \
    --save_mesh \
    --t_range 0.2 0.6 \
    --latent_iter_ratio 0 \
    --known_view_interval 4 \
    --vram_O \
    2>&1 | tee -a "$LOG"

echo "  Coarse stage finished at: $(date)" | tee -a "$LOG"

# ── Step 2: DMTet mesh refinement ──
# Uses the coarse NeRF as initialization and refines with DMTet + SDS
echo "" | tee -a "$LOG"
echo "[2/2] DMTet mesh refinement (3000 iterations)..." | tee -a "$LOG"

# Find the best checkpoint from coarse stage
CKPT="${WORKSPACE_COARSE}/checkpoints/magic123_object_c_coarse.pth"
if [ ! -f "$CKPT" ]; then
    # Try to find any .pth file
    CKPT=$(ls ${WORKSPACE_COARSE}/checkpoints/*.pth 2>/dev/null | head -1)
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "  ERROR: No coarse checkpoint found!" | tee -a "$LOG"
    exit 1
fi

echo "  Using checkpoint: $CKPT" | tee -a "$LOG"
echo "  Guidance: SD only (1.0)" | tee -a "$LOG"
echo "  Started at: $(date)" | tee -a "$LOG"

python main.py -O \
    --text "A high-resolution DSLR image of a teapot" \
    --sd_version 1.5 \
    --image "$INPUT_IMAGE" \
    --workspace "$WORKSPACE_FINE" \
    --dmtet \
    --init_ckpt "$CKPT" \
    --optim adam \
    --seed 0 \
    --iters 3000 \
    --guidance SD \
    --lambda_guidance 1.0 \
    --guidance_scale 100 \
    --bg_radius -1 \
    --save_mesh \
    --t_range 0.02 0.50 \
    --latent_iter_ratio 0 \
    --h 64 --w 64 \
    --dmtet_reso_scale 4 \
    --vram_O \
    2>&1 | tee -a "$LOG"

echo "  DMTet fine stage finished at: $(date)" | tee -a "$LOG"

# ── Step 3: Copy final mesh to output directory ──
echo "" | tee -a "$LOG"
echo "[3/3] Copying final mesh to output directory..." | tee -a "$LOG"
mkdir -p "$OUTPUT_DIR"

# Find the generated mesh
MESH_FILE=$(find "$WORKSPACE_FINE" -name "*.obj" -type f 2>/dev/null | head -1)
if [ -n "$MESH_FILE" ]; then
    cp "$MESH_FILE" "$OUTPUT_DIR/magic123_teapot.obj"
    echo "  Mesh: $OUTPUT_DIR/magic123_teapot.obj" | tee -a "$LOG"
    ls -lh "$OUTPUT_DIR/magic123_teapot.obj" | tee -a "$LOG"
else
    echo "  WARNING: No mesh file found in $WORKSPACE_FINE" | tee -a "$LOG"
    echo "  Checking coarse workspace for mesh..." | tee -a "$LOG"
    MESH_FILE=$(find "$WORKSPACE_COARSE" -name "*.obj" -type f 2>/dev/null | head -1)
    if [ -n "$MESH_FILE" ]; then
        cp "$MESH_FILE" "$OUTPUT_DIR/magic123_teapot.obj"
        echo "  Mesh (from coarse): $OUTPUT_DIR/magic123_teapot.obj" | tee -a "$LOG"
    fi
fi

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  Magic123 for Object C — COMPLETE" | tee -a "$LOG"
echo "  Finished at: $(date)" | tee -a "$LOG"
echo "  Output: $OUTPUT_DIR/" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
