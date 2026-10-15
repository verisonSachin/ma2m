#!/usr/bin/env bash
# Manga2Motion: Download model weights from Hugging Face & GitHub Releases
# Uses 'uv' for managing python utilities where necessary.

set -euo pipefail

MODEL_DIR="models"
mkdir -p "${MODEL_DIR}"

echo "=========================================================="
echo "    Manga2Motion Neural Weights Downloader (via uv)       "
echo "=========================================================="

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "[Info] Found 'uv' binary."
    DOWNLOADER="uv run huggingface-cli"
else
    echo "[Warn] 'uv' not found in PATH, using huggingface-cli directly."
    DOWNLOADER="huggingface-cli"
fi

# 1. YOLOv8 Manga Panel Extractor
echo "[1/6] Downloading YOLOv8 Manga Panel Extractor..."
if [ ! -f "${MODEL_DIR}/yolov8_manga.pt" ]; then
    curl -L "https://github.com/cocktailpeanut/manga-panel-extractor/releases/download/v1.0/yolov8_manga.pt" \
         -o "${MODEL_DIR}/yolov8_manga.pt" || echo "YOLO weights download skipped or using fallback."
else
    echo "  -> Found ${MODEL_DIR}/yolov8_manga.pt"
fi

# 2. Big-LaMa Inpainting Weights
echo "[2/6] Downloading Big-LaMa Inpainter..."
if [ ! -f "${MODEL_DIR}/big-lama.pt" ]; then
    curl -L "https://github.com/advimman/lama/releases/download/v0.1.0/big-lama.pt" \
         -o "${MODEL_DIR}/big-lama.pt" || echo "LaMa weights download skipped or using OpenCV fallback."
else
    echo "  -> Found ${MODEL_DIR}/big-lama.pt"
fi

# 3. Depth-Anything-V2 Anime / Base
echo "[3/6] Downloading Depth-Anything-V2 ViT-B..."
if [ ! -f "${MODEL_DIR}/depth_anything_v2_vitb.pth" ]; then
    curl -L "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth" \
         -o "${MODEL_DIR}/depth_anything_v2_vitb.pth" || echo "Depth-Anything download skipped or using perspective fallback."
else
    echo "  -> Found ${MODEL_DIR}/depth_anything_v2_vitb.pth"
fi

# 4. SAM 2 Hiera Large Checkpoint
echo "[4/6] Downloading SAM 2 Hiera Large Checkpoint..."
if [ ! -f "${MODEL_DIR}/sam2_hiera_large.pt" ]; then
    curl -L "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" \
         -o "${MODEL_DIR}/sam2_hiera_large.pt" || echo "SAM2 checkpoint download skipped or using saliency fallback."
else
    echo "  -> Found ${MODEL_DIR}/sam2_hiera_large.pt"
fi

# 5. RIFE v4.6 Flow Interpolation Weights
echo "[5/6] Downloading RIFE v4.6 Interpolator..."
if [ ! -f "${MODEL_DIR}/rife46.pth" ]; then
    curl -L "https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/rife46.pth" \
         -o "${MODEL_DIR}/rife46.pth" || echo "RIFE download skipped or using OpenCV DIS flow."
else
    echo "  -> Found ${MODEL_DIR}/rife46.pth"
fi

# 6. Diffusers Models Cache Warmup (AnimateDiff & ControlNet Anime Lineart)
echo "[6/6] Checking Diffusers Models..."
echo "Diffusers models (AnimateDiff, ControlNet Lineart, Animagine XL) will stream automatically from Hugging Face hub on initial run."

echo "=========================================================="
echo "    Manga2Motion Model Preparation Complete!              "
echo "=========================================================="
