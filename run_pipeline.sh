#!/usr/bin/env bash
set -e  # exit kalau ada error

# Export environment variables
export HF_HOME=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch
export PIP_NO_CACHE_DIR=1

echo "🔍 Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "❌ GPU not available! Exiting..."
    exit 1
fi

echo "✅ GPU is available and ready"

echo "📥 Pre-downloading models..."
python -c "from transformers import AutoModel; AutoModel.from_pretrained('indobenchmark/indobert-base-p2')"
python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"

echo "🚀 Starting training pipeline..."
python main.py "$@"