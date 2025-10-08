# Classifier Training Pipeline

This document explains how to run `main.py`, the orchestrator for the BPJS classifier training pipelines. It covers local execution, Docker workflows, Airflow integration, and the supported command-line arguments.

## Prerequisites

- Python 3.9 or newer.
- Project dependencies installed locally: `pip install -r requirements.txt`.
- Access to pretrained artifacts and datasets stored in S3 (optional but recommended).
- Environment variables (used when downloading from S3):
  - `S3_CLASSIFIER_PREFIX` – S3 prefix containing the full-training dataset (e.g. `dataset/classifier/`).
  - `S3_BUCKET_NAME`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` (or standard AWS credentials) consumed by `boto3`.

## Running `main.py` Locally

```bash
python main.py [options]
```

The pipeline supports two modes via `--mode`:

- `full` (default): runs the original classifier training pipeline from `train_classifier/main_classifier.py`.
- `transfer`: runs the transfer-learning pipeline from `train_classifier/main_classifier_TL.py`.

Output artifacts are written to the directories defined in `TrainConfig` (full) or
`TransferConfig` (transfer). Adjust those dataclasses or provide config files when you
need to customise file locations.

### Common Arguments

These flags are available in both modes:

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `full` | Select `full` or `transfer` pipeline. |
| `--dataset-dir` | `dataset` | Root directory for local datasets. |
| `--skip_download` | _flag_ | Skip S3 dataset sync (only meaningful for `full`). |
| `--epochs` | `None` | Number of training epochs (falls back to config defaults). |
| `--batch_size` | `None` | Mini-batch size for dataloaders. |
| `--num_workers` | `None` | DataLoader worker count. |
| `--lr_bert` | `None` | Learning rate for the BERT encoder. |
| `--lr_resnet` | `None` | Learning rate for the ResNet backbone. |
| `--lr_heads` | `None` | Learning rate for fusion/classifier heads. |
| `--state_dict_s3_key` | `None` | Override pretrained classifier weights S3 key. |
| `--label_map_s3_key` | `None` | Override previous label-map S3 key. |
| `--early_stop_patience` | `None` | Override early-stopping patience. |
| `--freeze_backbone` | _flag_ | Freeze BERT & ResNet during fine-tuning. |

> Any `None` values fall back to the defaults defined in the underlying training dataclasses.

### Full-Training Mode

`main.py --mode full`:

- Downloads the dataset from S3 unless `--skip_download` is provided.
- Launches `train_classifier/main_classifier.py` with the arguments listed above.

**Example:**

```bash
python main.py \
  --mode full \
  --dataset-dir dataset \
  --epochs 25 \
  --batch_size 64 \
  --lr_bert 3e-5
```

### Transfer-Learning Mode

`main.py --mode transfer` triggers `train_classifier/main_classifier_TL.py`. In addition to the common flags, it accepts the following options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--state_dict_s3_key` | `outputs_training/state_dict.pt` | S3 key to download the pretrained weights when no local artifact is configured. |
| `--label_map_s3_key` | `outputs_training/label2id.json` | S3 key for the previous label-map. |
| `--early_stop_patience` | `None` | Override early-stopping patience (defaults to 5). |
| `--freeze_backbone` | _flag_ | Freeze BERT & ResNet during fine-tuning. |

**Examples:**

Minimal run relying on S3 artifacts and defaults:

```bash
python main.py --mode transfer
```

Custom artifact locations and hyper-parameters:

```bash
python main.py \
  --mode transfer \
  --state_dict_s3_key outputs_training/state_dict.pt \
  --label_map_s3_key outputs_training/label2id.json \
  --freeze_backbone \
  --epochs 12 \
  --batch_size 24
```

## Running in Docker

The repository ships with a single GPU-capable service in `docker-compose.yaml` named `training-app-pytorch`. The service builds from `Dockerfile.pytorch` (target `runtime-gpu`) and mounts local folders for datasets and outputs so you can inspect artifacts directly on the host.

### Build the image

```bash
docker compose build training-app-pytorch
```

### Run the default pipeline

```bash
docker compose up training-app-pytorch
```

By default the service executes `python main.py --mode full --epochs 5`. The compose file maps:

- `./dataset` → `/app/dataset`
- `./outputs` → `/app/outputs`
- model caches to `/root/.cache/{huggingface,torch}`

You can override the behaviour in a few ways:

- Update `command:` in `docker-compose.yaml` (for example `python main.py --mode transfer --epochs 10`).
- Adjust the environment variables already defined in the compose file (`MODE`, `DATASET_DIR`, `OUTPUT_DIR`, etc.). Additional overrides map directly to CLI flags, e.g. add `EPOCHS=30` or `LR_BERT=2e-5`.
- Supply secrets or credentials via `.env`, which is automatically loaded by the service.
- Modify the `deploy.resources.reservations.devices` block if you need to expose a different GPU (the sample compose file pins the container to GPU index `2`).

If you need separate long-running services for full and transfer training, duplicate the service entry in `docker-compose.yaml` with different names, commands, and environment overrides.

## Running from Airflow (KubernetesPodOperator)

Use the built Docker images (`antifraud-training-full` or `antifraud-training-transfer`) with the KubernetesPodOperator. Example:

```python
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

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
    resources={
        "limit_gpu": 1,
    },
)
```

For transfer learning, switch `image` to `antifraud-training-transfer` (or keep the same image and set `MODE=transfer`) and provide the relevant artifacts:

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
    resources={
        "limit_gpu": 1,
    },
)
```

Mount persistent volumes or object stores as needed for `/app/dataset` and `/app/outputs`. GPU scheduling depends on your Kubernetes runtime (e.g. NVIDIA device plugin).

## Logs and Outputs

- Logs stream to stdout and are timestamped for readability.
- Artifacts (state dicts, ONNX exports, label maps) are stored in the run folder inside `--output_dir`.
- Transfer-learning runs maintain previous class weights by merging the supplied label map with new classes.

## Troubleshooting

- Ensure all required environment variables are set when using S3 downloads.
- Confirm GPU availability with `torch.cuda.is_available()` inside your environment if you expect GPU training.
- Use `--skip_download` (or `SKIP_MODEL_DOWNLOADS=true` in Docker/Airflow) when warm-up downloads are unnecessary.
- When providing custom CSV or image directories, verify that file paths and label columns follow the expected format (`label/filename`).

For additional customization, inspect `train_classifier/main_classifier.py` (full training) and `train_classifier/main_classifier_TL.py` (transfer learning).
