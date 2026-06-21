#!/bin/bash
# Magic123 Coarse Stage — Single Image to 3D
source /root/miniconda3/etc/profile.d/conda.sh
conda activate task1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/root/miniconda3/envs/task1/lib:/root/miniconda3/envs/task1/lib/python3.8/site-packages/torch/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=/root/CV/task1/repos/Magic123:$PYTHONPATH

cd /root/CV/task1/repos/Magic123
python main.py -O \
    --text "A high-resolution DSLR image of an Iron Man action figure" \
    --sd_version 1.5 \
    --image /root/CV/task1/data/object_c/rgba.png \
    --workspace /root/CV/task1/outputs/object_c/magic123_coarse \
    --optim adam \
    --iters 1000 \
    --guidance SD zero123 \
    --lambda_guidance 1.0 40 \
    --guidance_scale 100 5 \
    --bg_radius -1 \
    --save_mesh \
    --t_range 0.2 0.6 \
    --latent_iter_ratio 0 \
    --vram_O \
    --seed 0 \
    2>&1 | tee /root/CV/task1/logs/07_magic123_c_coarse.log
