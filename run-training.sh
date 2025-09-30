#!/bin/bash

# Script untuk menjalankan training BPJS AntiFraud dengan parameter kustom
# Usage: ./run-training.sh [pipeline] [epochs] [batch_size] [num_workers] [lr_bert] [lr_resnet] [lr_heads]

# Default values
PIPELINE=${1:-"classifier"}
EPOCHS=${2:-"10"}
BATCH_SIZE=${3:-"32"}
NUM_WORKERS=${4:-"4"}
LR_BERT=${5:-"2e-5"}
LR_RESNET=${6:-"1e-4"}
LR_HEADS=${7:-"1e-3"}

echo "🚀 Starting BPJS AntiFraud Training"
echo "=================================="
echo "Pipeline: $PIPELINE"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Num Workers: $NUM_WORKERS"
echo "LR BERT: $LR_BERT"
echo "LR ResNet: $LR_RESNET"
echo "LR Heads: $LR_HEADS"
echo "=================================="

# Build image if not exists
echo "🔨 Building Docker image..."
docker-compose build

# Run training with custom parameters
echo "🏃 Running training..."
docker-compose run --rm antifraud-training python3 main.py \
    --pipeline $PIPELINE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr_bert $LR_BERT \
    --lr_resnet $LR_RESNET \
    --lr_heads $LR_HEADS

echo "✅ Training completed!"
