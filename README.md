# Classifier Training Pipelines

This project exposes three Python entry points for training the BPJS hybrid OCR classifier. Use this guide to decide which script to run, understand their arguments, and wire them into Docker or Airflow.

## Entry Points Overview

- `main.py` - high-level orchestrator that can run either training mode and optionally sync data from S3 before launching a run.
- `train_classifier/main_classifier.py` - standalone full-training pipeline that builds the model from scratch and exports artifacts.
- `train_classifier/main_classifier_TL.py` - standalone transfer-learning pipeline that reuses a pretrained checkpoint and updates the classifier with new data.

## Prerequisites

- Python 3.9 or newer.
- Install dependencies: `pip install -r requirements.txt`.
- Optional but recommended: access to S3 (or compatible storage) for datasets and pretrained artifacts. When using S3, configure:
  - `S3_BUCKET_NAME`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` (or standard AWS credentials).
  - `S3_CLASSIFIER_PREFIX` for the dataset download performed by `main.py` (defaults to `dataset/classifier` inside the script).

## Orchestrator (`main.py`)

```bash
python main.py [options]
```

`main.py` resolves default directories from the config dataclasses, ensures they exist, optionally wipes and re-syncs `dataset/` from S3, and then delegates to the appropriate inner script.

### Core Flags

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `full` | Select `full` (runs `main_classifier.py`) or `transfer` (runs `main_classifier_TL.py`). |
| `--dataset-dir` | `dataset` | Root directory for datasets. |
| `--skip_download` | flag | Skip the S3 dataset sync (avoids deleting `dataset/`). |
| `--epochs` | `50` | Training epochs used when no override is provided (`TrainConfig.epochs` / `TransferConfig.epochs`). |
| `--batch_size` | `16` | Mini-batch size for both modes. |
| `--num_workers` | `8` | PyTorch DataLoader worker count. |
| `--lr_bert` | `2e-5` | Learning rate for the BERT encoder. |
| `--lr_resnet` | `1e-4` | Learning rate for ResNet layers. |
| `--lr_heads` | `1e-4` | Learning rate for fusion/classifier heads. |
| `--early_stop_patience` | `15` | Epochs to wait before early stopping. |
| `--state_dict_s3_key` | `outputs_training/state_dict.pt` | (Transfer mode) Default pretrained weights location. |
| `--label_map_s3_key` | `outputs_training/label2id.json` | (Transfer mode) Default previous label map location. |
| `--freeze_backbone` | `True` | (Transfer mode) Backbones stay frozen unless you modify the transfer config. |

> When an override is omitted, the CLI falls back to these concrete defaults sourced from `TrainConfig` (full) or `TransferConfig` (transfer).

### Examples

Run a clean full-training experiment with custom learning rate and batch size:

```bash
python main.py --mode full --epochs 25 --batch_size 64 --lr_bert 3e-5
```

Reuse existing datasets and launch the transfer-learning pipeline with a frozen backbone:

```bash
python main.py --mode transfer --skip_download --freeze_backbone --epochs 12
```

## Full Training Script (`train_classifier/main_classifier.py`)

Use this script when you want to execute the full training loop directly, bypassing the orchestrator, or when you need finer control over configuration files.

```bash
python train_classifier/main_classifier.py [options]
```

### Flags

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `None` | Path to a JSON file that mirrors `TrainConfig` fields. |
| `--epochs` | `50` | Override epochs at runtime. |
| `--batch_size` | `16` | Override batch size. |
| `--num_workers` | `8` | Override CPU worker count. |
| `--lr_bert` | `2e-5` | Adjust encoder learning rate. |
| `--lr_resnet` | `1e-4` | Adjust CNN learning rate. |
| `--lr_heads` | `1e-4` | Adjust classifier head learning rate. |
| `--early_stop_patience` | `15` | Set early-stopping patience. |

The script reads `TrainConfig`, applies overrides (CLI takes precedence over JSON), builds datasets, trains the hybrid model, exports state dict, label map, ONNX graph, and a JSON training report. Artifacts are timestamped and sent to S3 via `S3Handler` when credentials are present.

**Example**

```bash
python train_classifier/main_classifier.py \
  --config configs/full_training.json \
  --epochs 40 \
  --batch_size 32
```

## Transfer Learning Script (`train_classifier/main_classifier_TL.py`)

Run this entry point to fine-tune an existing classifier on new data while preserving previous classes.

```bash
python train_classifier/main_classifier_TL.py [options]
```

### Flags

| Argument | Default | Description |
|----------|---------|-------------|
| `--state_dict_s3_key` | `outputs_training/state_dict.pt` | Download pretrained weights from S3 when local paths are missing. |
| `--label_map_s3_key` | `outputs_training/label2id.json` | Download the previous label map from S3 when missing locally. |
| `--epochs` | `50` | Fine-tuning epochs. |
| `--batch_size` | `16` | Mini-batch size. |
| `--num_workers` | `8` | DataLoader and OCR worker count. |
| `--lr_bert` | `2e-5` | Learning rate for the text encoder. |
| `--lr_resnet` | `1e-4` | Learning rate for the vision backbone. |
| `--lr_heads` | `1e-4` | Learning rate for newly initialised heads. |
| `--early_stop_patience` | `15` | Early-stopping patience. |
| `--freeze_backbone` | `True` | Disable gradients for BERT and ResNet unless you edit the config. |

The script merges old and new label mappings, reuses pretrained weights, trains with optional class-weight balancing, exports ONNX (when enabled in config), writes a JSON summary, and uploads artifacts to the transfer-specific S3 prefixes.

**Example**

```bash
python train_classifier/main_classifier_TL.py \
  --state_dict_s3_key outputs_training/state_dict.pt \
  --label_map_s3_key outputs_training/label2id.json \
  --freeze_backbone \
  --epochs 10
```

## Docker Workflow

`docker-compose.yaml` defines a GPU-enabled service (`training-app-pytorch`) built from `Dockerfile.pytorch` (target `runtime-gpu`).

```bash
docker compose build training-app-pytorch
docker compose up training-app-pytorch
```

The default command is `python main.py --mode full --epochs 5`. Adjust behaviour by:

- Editing the service `command` (e.g. `python train_classifier/main_classifier.py --epochs 30`).
- Overriding environment variables (`MODE`, `CLASSIFIER_EXTRA_ARGS`, `DATASET_DIR`, `OUTPUT_DIR`, etc.).
- Supplying credentials via `.env`, automatically consumed by Compose.
- Tweaking GPU allocation in the `deploy.resources` section (defaults to GPU index 2).

Mounted volumes map `./dataset` to `/app/dataset`, `./outputs` to `/app/outputs`, and host caches to `/root/.cache/{huggingface,torch}`.

## Airflow (KubernetesPodOperator)

Package the training image (`antifraud-training-full` or `antifraud-training-transfer`) and launch with KubernetesPodOperator:

```python
train_full = KubernetesPodOperator(
    task_id="train_classifier_full",
    name="classifier-full",
    namespace="ml",
    image="registry.example.com/antifraud-training-full:latest",
    env_vars={
        "MODE": "full",
        "DATASET_DIR": "/app/dataset",
        "OUTPUT_DIR": "/app/outputs",
        "CLASSIFIER_EXTRA_ARGS": "--epochs 20 --batch_size 48",
    },
    get_logs=True,
    is_delete_operator_pod=True,
    startup_timeout_seconds=600,
    resources={"limit_gpu": 1},
)
```

For transfer learning, switch the image or set `MODE=transfer` and provide artifact locations:

```python
train_transfer = KubernetesPodOperator(
    task_id="train_classifier_transfer",
    name="classifier-transfer",
    namespace="ml",
    image="registry.example.com/antifraud-training-transfer:latest",
    env_vars={
        "MODE": "transfer",
        "DATASET_DIR": "/app/dataset",
        "IMAGE_ROOT_DIR": "/app/dataset/dataset_TL",
        "DATA_CSV_PATH": "/app/dataset/dataset_TL.csv",
        "STATE_DICT_S3_KEY": "outputs_training/state_dict.pt",
        "LABEL_MAP_S3_KEY": "outputs_training/label2id.json",
        "CLASSIFIER_EXTRA_ARGS": "--freeze_backbone --epochs 8",
        "SKIP_MODEL_DOWNLOADS": "true",
    },
    secrets=[...],  # mount S3 credentials / AWS secrets
    get_logs=True,
    is_delete_operator_pod=True,
    resources={"limit_gpu": 1},
)
```

Mount persistent volumes for `/app/dataset` and `/app/outputs` as required by your cluster setup.

## Logs and Outputs

- Logs stream to stdout with timestamps.
- Artifacts (state dicts, label maps, ONNX exports, reports) are written under the configured `output_dir` with run-specific timestamps.
- Transfer-learning runs merge the historical label map with newly observed classes before exporting artifacts.

## Troubleshooting

- Ensure S3 credentials are configured when relying on automatic downloads or uploads.
- Use `--skip_download` (or `SKIP_MODEL_DOWNLOADS=true` in containerised runs) to avoid deleting local datasets.
- Confirm GPU availability with `torch.cuda.is_available()` when training on hardware accelerators.
- Validate custom CSVs and image layouts match the expectations in `train_classifier.dataset`.
