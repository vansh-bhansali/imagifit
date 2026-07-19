#!/bin/bash

echo "=========================================="
echo "    Starting Imagifit AI Server...        "
echo "=========================================="

echo "🧹 Cleaning up any old stuck servers..."
# Kills any process using port 5050 (Flask) or 7860 (Gradio) to prevent
# "port already in use" errors and stale servers answering with old code.
lsof -ti:5050 | xargs kill -9 2>/dev/null
lsof -ti:7860 | xargs kill -9 2>/dev/null
# Kill any stale localtunnel processes so the subdomain can be reclaimed
pkill -f localtunnel 2>/dev/null

echo "🗑️  Clearing the received_images folder..."
# Clear all previous received images (recreates folder if needed)
rm -rf "$(dirname "$0")/CatVTON/received_images"/*

# Wait a moment for the old tunnel to fully release the subdomain
sleep 2

echo "🚀 Launching Python App..."
# Move into the correct directory
cd "$(dirname "$0")/CatVTON" || exit

# Run the python app using your CatVTON Conda environment.
# -u = unbuffered output so all prints show up immediately in the terminal.
# The MPS watermark ratios cap allocations at ~60% of RAM so a single
# generation can't starve the OS, and 384x512 is the reduced resolution that
# fits in 18GB unified memory (see PROJECT_NOTES.md "Issues Fixed" #7).
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 \
/opt/homebrew/Caskroom/miniconda/base/envs/catvton/bin/python -u app.py \
    --output_dir=output --mixed_precision=fp16 --width 384 --height 512
