#!/usr/bin/env bash
set -e  # exit kalau ada error

# Export environment variables
export HF_HOME=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch
export PIP_NO_CACHE_DIR=1

# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('indobenchmark/indobert-base-p2')"
python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"

# Jalankan pipeline utama
python main.py --pipeline classifier --dataset-dir /app/dataset --output_dir /app/outputs
