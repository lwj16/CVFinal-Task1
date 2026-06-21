#!/bin/bash
# ============================================================
# run_all.sh — Master Orchestration Script
# Full pipeline: 3D Reconstruction → Fusion → Video → Report
#
# Usage:
#   bash run_all.sh              # Run everything
#   bash run_all.sh --skip-train # Skip training, only fusion + video
#   bash run_all.sh --object-a   # Only Object A
#   bash run_all.sh --background # Only Background
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SKIP_TRAIN=false
ONLY_OBJECT_A=false
ONLY_BACKGROUND=false
for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --object-a) ONLY_OBJECT_A=true ;;
        --background) ONLY_BACKGROUND=true ;;
    esac
done

# Set up environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate task1
export LD_LIBRARY_PATH=/root/miniconda3/envs/task1/lib:/root/miniconda3/envs/task1/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH

mkdir -p logs outputs

echo "=================================================="
echo "  3D Vision Pipeline — Full Orchestration"
echo "  Started at: $(date)"
echo "=================================================="
echo ""
echo "GPU Configuration:"
echo "  GPU 2: Object A (2DGS) → Object B (threestudio)"
echo "  GPU 3: Object C (Magic123)"
echo "  GPU 4: Background (2DGS)"
echo ""

# ── Phase 0: Verify Setup ──
echo "[Phase 0] Verifying setup..."
for tool in colmap ffmpeg; do
    command -v $tool >/dev/null 2>&1 || { echo "ERROR: $tool not found!"; exit 1; }
done
echo "  COLMAP: $(colmap --version 2>&1 | head -1 || echo OK)"
echo "  ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
echo ""

# ── Phase 1: Data Check ──
echo "[Phase 1] Checking data..."
check_data() {
    local name=$1 path=$2 expected=$3
    local count=$(ls "$path" 2>/dev/null | wc -l)
    if [ "$count" -ge "$expected" ]; then
        echo "  $name: OK ($count files)"
    else
        echo "  $name: MISSING (found $count, need $expected)"
        return 1
    fi
}
check_data "Object A images" "data/object_a/images/*.png" 50 || true
check_data "Background images" "data/background/images/*.JPG" 100 || true
echo ""

# ── Phase 2: Training (Parallel) ──
if [ "$SKIP_TRAIN" = false ]; then
    echo "[Phase 2] Starting training pipelines..."

    if [ "$ONLY_OBJECT_A" = true ]; then
        bash scripts/04_run_colmap_a.sh &
        wait
        bash scripts/05_run_2dgs_a.sh
    elif [ "$ONLY_BACKGROUND" = true ]; then
        bash scripts/08_run_colmap_bg.sh &
        wait
        bash scripts/09_run_2dgs_bg.sh
    else
        # Full pipeline with GPU parallelism
        echo "  Launching Object A (GPU 2) + Object C (GPU 3) + Background (GPU 4)..."

        # COLMAP steps (CPU, run first)
        bash scripts/04_run_colmap_a.sh &
        bash scripts/08_run_colmap_bg.sh &
        wait
        echo "  COLMAP complete."

        # Training steps (GPU, run in parallel)
        bash scripts/05_run_2dgs_a.sh &     # GPU 2
        PID_A=$!
        bash scripts/07_run_magic123_c.sh &  # GPU 3
        PID_C=$!
        bash scripts/09_run_2dgs_bg.sh &     # GPU 4
        PID_BG=$!

        wait $PID_A $PID_C $PID_BG
        echo "  Object A + C + Background training complete."

        # Object B (GPU 2, after Object A)
        bash scripts/06_run_threestudio_b.sh
        echo "  Object B complete."
    fi
else
    echo "[Phase 2] SKIPPED (--skip-train)"
fi

echo ""

# ── Phase 3: Fusion Render ──
echo "[Phase 3] Fusion rendering..."
python scripts/10_fusion_render.py 2>&1 | tee logs/10_fusion.log

# ── Phase 4: Video ──
echo "[Phase 4] Generating video..."
bash scripts/11_generate_video.sh

echo ""
echo "=================================================="
echo "  Pipeline Complete!"
echo "  Final video: outputs/fusion/flythrough.mp4"
echo "  Finished at: $(date)"
echo "=================================================="
