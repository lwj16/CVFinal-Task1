#!/bin/bash
# ============================================================
# 05_run_2dgs_a.sh — 2D Gaussian Splatting for Object A (Lego)
# ============================================================
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate task1

export LD_LIBRARY_PATH=/root/miniconda3/envs/task1/lib:/root/miniconda3/envs/task1/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2

cd /root/CV/task1/repos/2d-gaussian-splatting

SOURCE_PATH=/root/CV/task1/data/object_a
MODEL_PATH=/root/CV/task1/outputs/object_a/2dgs
LOG=/root/CV/task1/logs/05_2dgs_a.log

echo "=== 2DGS Training for Object A (Lego) ===" | tee $LOG
echo "Started at: $(date)" | tee -a $LOG
echo "Source: $SOURCE_PATH" | tee -a $LOG
echo "Model: $MODEL_PATH" | tee -a $LOG

# Verify input
ls $SOURCE_PATH/sparse/0/cameras.bin > /dev/null 2>&1 || { echo "ERROR: No COLMAP sparse model!"; exit 1; }
echo "Images: $(ls $SOURCE_PATH/images/*.png | wc -l)" | tee -a $LOG

python train.py \
    --source_path $SOURCE_PATH \
    --model_path $MODEL_PATH \
    --images images \
    --resolution 1 \
    --iterations 30000 \
    --position_lr_init 0.00016 \
    --position_lr_final 0.0000016 \
    --position_lr_delay_mult 0.01 \
    --position_lr_max_steps 30000 \
    --feature_lr 0.0025 \
    --opacity_lr 0.05 \
    --scaling_lr 0.005 \
    --rotation_lr 0.001 \
    --percent_dense 0.01 \
    --lambda_dssim 0.2 \
    --densification_interval 100 \
    --opacity_reset_interval 3000 \
    --densify_from_iter 500 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.0002 \
    --white_background \
    2>&1 | tee -a $LOG

echo "=== Training Complete ===" | tee -a $LOG
echo "Done at: $(date)" | tee -a $LOG
echo "Checkpoint: $MODEL_PATH/point_cloud/iteration_30000/point_cloud.ply" | tee -a $LOG
