#!/bin/bash
# ============================================================
# 11_generate_video.sh — Compose flythrough frames into MP4 video
# ============================================================
set -e

FRAMES_DIR=/root/CV/task1/outputs/fusion/flythrough_frames
OUTPUT_VIDEO=/root/CV/task1/outputs/fusion/flythrough.mp4
LOG=/root/CV/task1/logs/11_video.log

echo "=== Video Generation ===" | tee $LOG
echo "Started at: $(date)" | tee -a $LOG

if [ ! -d "$FRAMES_DIR" ]; then
    echo "ERROR: Frames directory not found: $FRAMES_DIR" | tee -a $LOG
    exit 1
fi

NUM_FRAMES=$(ls $FRAMES_DIR/frame_*.png 2>/dev/null | wc -l)
echo "Found $NUM_FRAMES frames" | tee -a $LOG

if [ "$NUM_FRAMES" -eq 0 ]; then
    echo "ERROR: No frames found!" | tee -a $LOG
    exit 1
fi

# Generate video with ffmpeg
echo "Encoding video..." | tee -a $LOG
ffmpeg -y \
    -framerate 30 \
    -i $FRAMES_DIR/frame_%04d.png \
    -c:v libx264 \
    -preset medium \
    -crf 18 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    $OUTPUT_VIDEO \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Video Complete ===" | tee -a $LOG
echo "Output: $OUTPUT_VIDEO" | tee -a $LOG
echo "Size: $(du -h $OUTPUT_VIDEO | cut -f1)" | tee -a $LOG
echo "Duration: $(ffprobe -v error -show_entries format=duration -of csv=p=0 $OUTPUT_VIDEO 2>/dev/null || echo 'N/A')" | tee -a $LOG
echo "Done at: $(date)" | tee -a $LOG
