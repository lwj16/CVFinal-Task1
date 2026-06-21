#!/bin/bash
# ============================================================
# 08_run_colmap_bg.sh — COLMAP for Background (Mip-NeRF 360 bonsai)
# The bonsai dataset already includes a pre-computed sparse model.
# This script uses it directly if available, or runs COLMAP fresh.
# ============================================================
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate task1

DATASET=/root/CV/task1/data/background
BONSAI_SOURCE=/root/CV/task1/data/background/bonsai
LOG=/root/CV/task1/logs/08_colmap_bg.log

echo "=== COLMAP for Background (bonsai) ===" | tee $LOG
echo "Started at: $(date)" | tee -a $LOG

# Check if pre-computed sparse model exists in bonsai directory
if [ -f "$BONSAI_SOURCE/sparse/0/cameras.bin" ] && [ -f "$BONSAI_SOURCE/sparse/0/images.bin" ]; then
    echo "Using pre-computed COLMAP sparse model from Mip-NeRF 360." | tee -a $LOG
    mkdir -p $DATASET/sparse
    cp -r $BONSAI_SOURCE/sparse/0 $DATASET/sparse/0
    echo "Copied to $DATASET/sparse/0/" | tee -a $LOG
    ls -la $DATASET/sparse/0/ | tee -a $LOG
    echo "Done at: $(date)" | tee -a $LOG
    exit 0
fi

# Otherwise, run COLMAP from scratch (slow on 292 images)
echo "No pre-computed model found. Running COLMAP from scratch..." | tee -a $LOG
echo "WARNING: This may take 1-3 hours on CPU-only COLMAP." | tee -a $LOG

# Step 1: Feature extraction
echo "[1/3] Feature extraction..." | tee -a $LOG
colmap feature_extractor \
    --database_path $DATASET/database.db \
    --image_path $DATASET/images \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_num_features 8192 \
    2>&1 | tee -a $LOG

# Step 2: Sequential matcher (bonsai is forward-facing)
echo "[2/3] Sequential matching..." | tee -a $LOG
colmap sequential_matcher \
    --database_path $DATASET/database.db \
    --SiftMatching.use_gpu 1 \
    --SequentialMatching.overlap 10 \
    2>&1 | tee -a $LOG

# Step 3: Mapper
echo "[3/3] Sparse reconstruction..." | tee -a $LOG
mkdir -p $DATASET/sparse
colmap mapper \
    --database_path $DATASET/database.db \
    --image_path $DATASET/images \
    --output_path $DATASET/sparse \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_extra_params 1 \
    2>&1 | tee -a $LOG

echo "Done at: $(date)" | tee -a $LOG
