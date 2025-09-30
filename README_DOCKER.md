# BPJS AntiFraud Training Models - Docker Setup

Dockerfile ini dibuat berdasarkan contoh `Dockerfile.example` dengan penyesuaian untuk sistem training BPJS AntiFraud.

## 🏗️ Struktur Dockerfile

- **Base Image**: Ubuntu 20.04
- **Python Version**: 3.11 (dari deadsnakes PPA)
- **User**: Non-root user `appuser` untuk keamanan
- **Working Directory**: `/app`

## 🚀 Cara Menjalankan Training

### 1. Build Docker Image
```bash
docker build -t bpjs-antifraud-training .
```

### 2. Run dengan Parameter Default
```bash
docker run --rm -v $(pwd)/dataset:/app/dataset -v $(pwd)/outputs:/app/outputs bpjs-antifraud-training
```

### 3. Run dengan Parameter Kustom
```bash
docker run --rm \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  bpjs-antifraud-training \
  python3 main.py \
  --pipeline classifier \
  --epochs 20 \
  --batch_size 16 \
  --num_workers 8 \
  --lr_bert 1e-5 \
  --lr_resnet 5e-5 \
  --lr_heads 5e-4
```

### 4. Menggunakan Docker Compose

#### Training Classifier dengan Parameter Kustom
```bash
# Set environment variables
export EPOCHS=20
export BATCH_SIZE=16
export NUM_WORKERS=8
export LR_BERT=1e-5
export LR_RESNET=5e-5
export LR_HEADS=5e-4

# Run classifier training
docker-compose --profile classifier up classifier-training
```

#### Training Signature
```bash
# Set signature parameters
export SIGNATURE_EPOCHS=100

# Run signature training
docker-compose --profile signature up signature-training
```

### 5. Menggunakan Script Helper
```bash
# Make script executable
chmod +x run-training.sh

# Run dengan parameter default
./run-training.sh

# Run dengan parameter kustom
./run-training.sh classifier 20 16 8 1e-5 5e-5 5e-4
```

## 📋 Parameter Training

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--pipeline` | `classifier` | Pipeline yang dijalankan (classifier/signature/both) |
| `--epochs` | `10` | Jumlah epoch training |
| `--batch_size` | `32` | Ukuran batch |
| `--num_workers` | `4` | Jumlah worker untuk data loading |
| `--lr_bert` | `2e-5` | Learning rate untuk BERT |
| `--lr_resnet` | `1e-4` | Learning rate untuk ResNet |
| `--lr_heads` | `1e-3` | Learning rate untuk classification heads |

## 🔧 Environment Variables

Buat file `.env` dengan konfigurasi berikut:

```bash
# S3 Configuration
S3_BUCKET_NAME=your-bucket-name
S3_ENDPOINT=your-s3-endpoint
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
S3_CLASSIFIER_PREFIX=dataset/classifier/
S3_SIGNATURE_PREFIX=dataset/signature/

# Training Parameters
EPOCHS=20
BATCH_SIZE=16
NUM_WORKERS=8
LR_BERT=1e-5
LR_RESNET=5e-5
LR_HEADS=5e-4

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

## 📁 Volume Mounts

- `./dataset` → `/app/dataset` (Training datasets)
- `./outputs` → `/app/outputs` (Model outputs)
- `./config` → `/app/config` (Configuration files)
- `./logs` → `/app/logs` (Training logs)

## 🎯 Contoh Penggunaan

### Training Classifier dengan GPU
```bash
# Enable GPU support
export CUDA_VISIBLE_DEVICES=0

# Run training
docker run --rm --gpus all \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  -e CUDA_VISIBLE_DEVICES=0 \
  bpjs-antifraud-training \
  python3 main.py \
  --pipeline classifier \
  --epochs 50 \
  --batch_size 64 \
  --num_workers 8 \
  --lr_bert 1e-5 \
  --lr_resnet 1e-4 \
  --lr_heads 1e-3
```

### Training dengan Resource Limits
```bash
docker run --rm \
  --memory=8g \
  --cpus=4 \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  bpjs-antifraud-training \
  python3 main.py \
  --pipeline both \
  --epochs 30 \
  --batch_size 16
```

## 🔍 Monitoring dan Debugging

### View Logs
```bash
# Follow container logs
docker logs -f <container_id>

# Check training progress
docker exec <container_id> tail -f /app/logs/pipeline_*.log
```

### Debug Mode
```bash
# Run interactive shell
docker run -it --rm \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  bpjs-antifraud-training \
  bash
```

## 🛠️ Troubleshooting

### 1. Out of Memory
```bash
# Reduce batch size
docker run --rm \
  --memory=16g \
  bpjs-antifraud-training \
  python3 main.py --batch_size 8
```

### 2. S3 Connection Issues
```bash
# Check S3 configuration
docker run --rm bpjs-antifraud-training python3 -c "
from utils.s3_utils import S3Handler
s3 = S3Handler()
print('S3 connection OK')
"
```

### 3. GPU Not Detected
```bash
# Check GPU availability
docker run --rm --gpus all bpjs-antifraud-training nvidia-smi
```

## 📚 Additional Commands

### Clean Up
```bash
# Remove unused images
docker image prune

# Remove all containers
docker container prune

# Remove all volumes
docker volume prune
```

### Build with Different Tags
```bash
# Build with version tag
docker build -t bpjs-antifraud-training:v1.0.0 .

# Build with latest tag
docker build -t bpjs-antifraud-training:latest .
```
