#!/bin/bash
# ============================================================
# 06_run_threestudio_b.sh — Text-to-3D with threestudio (Object B)
# Uses DreamFusion-style SDS Loss + Stable Diffusion 1.5 (offline)
# ============================================================
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate task1

export LD_LIBRARY_PATH=/root/miniconda3/envs/task1/lib:/root/miniconda3/envs/task1/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/root/CV/task1/repos/threestudio:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /root/CV/task1/repos/threestudio

PROMPT=$(cat /root/CV/task1/data/object_b/prompt.txt)
OUTPUT_DIR=/root/CV/task1/outputs/object_b/threestudio
LOG=/root/CV/task1/logs/06_threestudio_b.log

echo "=== threestudio Text-to-3D for Object B ===" | tee $LOG
echo "Started at: $(date)" | tee -a $LOG
echo "Prompt: $PROMPT" | tee -a $LOG
echo "Output: $OUTPUT_DIR" | tee -a $LOG

# Check if threestudio has the required config files
if [ ! -f "configs/dreamfusion-sd.yaml" ]; then
    echo "WARNING: dreamfusion-sd.yaml not found, searching for available configs..." | tee -a $LOG
    ls configs/ | head -20 | tee -a $LOG
fi

# threestudio uses Hydra/OmegaConf config system
# Use the dreamfusion-sd config (DreamFusion + Stable Diffusion)
python launch.py \
    --config configs/dreamfusion-sd.yaml \
    --train \
    --gpu 0 \
    system.prompt_processor.prompt="$PROMPT" \
    system.prompt_processor.pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    system.guidance.pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    system.guidance.guidance_scale=100 \
    system.guidance.enable_sequential_cpu_offload=true \
    trainer.max_steps=10000 \
    system.exporter_type="mesh-exporter" \
    system.geometry.radius=2.0 \
    system.geometry.pos_encoding_config.n_levels=10 \
    system.geometry.pos_encoding_config.log2_hashmap_size=17 \
    data.width=128 \
    data.height=128 \
    system.loggers.wandb.enable=false \
    name="object_b_ceramic_vase" \
    tag="text_to_3d" \
    2>&1 | tee -a $LOG

echo "=== threestudio Complete ===" | tee -a $LOG
echo "Done at: $(date)" | tee -a $LOG
echo "Check output: $OUTPUT_DIR" | tee -a $LOG
