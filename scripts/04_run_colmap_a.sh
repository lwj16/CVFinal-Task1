#!/bin/bash
# ============================================================
# 04_run_colmap_a.sh — COLMAP for Object A (Lego from NeRF Synthetic)
# Since we already converted transforms_train.json to COLMAP format,
# we just need to run COLMAP feature extraction and matching to get
# the 3D points that 2DGS expects.
# ============================================================
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate task1

DATASET=/root/CV/task1/data/object_a
LOG=/root/CV/task1/logs/04_colmap_a.log

echo "=== COLMAP SfM for Object A (Lego) ===" | tee $LOG
echo "Started at: $(date)" | tee -a $LOG

# Check if we already have a COLMAP sparse model
if [ -f "$DATASET/sparse/0/cameras.bin" ] && [ -f "$DATASET/sparse/0/images.bin" ]; then
    # We have pre-converted poses; check if we need 3D points
    if [ -f "$DATASET/sparse/0/points3D.bin" ] && [ "$(stat -c%s $DATASET/sparse/0/points3D.bin)" -gt 50 ]; then
        echo "COLMAP model already complete. Skipping." | tee -a $LOG
        exit 0
    fi
    echo "Have camera poses from transforms.json, running COLMAP mapper only to get 3D points..." | tee -a $LOG
    SKIP_FEATURES=true
fi

if [ "$SKIP_FEATURES" != "true" ]; then
    # Step 1: Feature extraction
    echo "[1/3] Feature extraction..." | tee -a $LOG
    colmap feature_extractor \
        --database_path $DATASET/database.db \
        --image_path $DATASET/images \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 1 \
        2>&1 | tee -a $LOG

    # Step 2: Exhaustive matching (100 images, manageable)
    echo "[2/3] Feature matching..." | tee -a $LOG
    colmap exhaustive_matcher \
        --database_path $DATASET/database.db \
        --SiftMatching.use_gpu 1 \
        2>&1 | tee -a $LOG

    # Step 3: Mapper
    echo "[3/3] Sparse reconstruction..." | tee -a $LOG
    mkdir -p $DATASET/sparse/0
    colmap mapper \
        --database_path $DATASET/database.db \
        --image_path $DATASET/images \
        --output_path $DATASET/sparse \
        2>&1 | tee -a $LOG
else
    # We have cameras from transforms but need feature points for 2DGS
    # Run feature extraction then import the existing camera poses
    echo "[1/2] Feature extraction for 3D point triangulation..." | tee -a $LOG
    colmap feature_extractor \
        --database_path $DATASET/database.db \
        --image_path $DATASET/images \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 1 \
        2>&1 | tee -a $LOG

    echo "[2/2] Feature matching + triangulation with known poses..." | tee -a $LOG
    colmap exhaustive_matcher \
        --database_path $DATASET/database.db \
        --SiftMatching.use_gpu 1 \
        2>&1 | tee -a $LOG

    # Use point triangulator with known poses
    colmap point_triangulator \
        --database_path $DATASET/database.db \
        --image_path $DATASET/images \
        --input_path $DATASET/sparse/0 \
        --output_path $DATASET/sparse/0 \
        2>&1 | tee -a $LOG
fi

# Verify output
echo "=== Verification ===" | tee -a $LOG
ls -la $DATASET/sparse/0/ | tee -a $LOG
echo "Done at: $(date)" | tee -a $LOG
