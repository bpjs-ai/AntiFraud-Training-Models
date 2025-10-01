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

The pipeline supports two modes via `--classifier_mode`:

- `full` (default): runs the original classifier training pipeline from `train_classifier/main_classifier.py`.
- `transfer`: runs the transfer-learning pipeline from `train_classifier/main_classifier_TL.py`.

Output artifacts are written to `--output_dir/<mode>/<timestamp>`.

### Common Arguments

These flags are available in both modes:

| Argument | Default | Description |
|----------|---------|-------------|
| `--classifier_mode` | `full` | Select `full` or `transfer` pipeline. |
| `--dataset-dir` | `dataset` | Root directory for local datasets. |
| `--output_dir` | `outputs` | Root directory for training outputs and logs. |
| `--skip_download` | _flag_ | Skip S3 dataset sync (only meaningful for `full`). |
| `--data_csv_path` | `None` | Override path to the CSV describing the dataset. |
| `--image_root_dir` | `None` | Override root folder for images. |
| `--val_size` | `None` | Validation split fraction (`0.2` used when unset). |
| `--epochs` | `None` | Number of training epochs (falls back to config defaults). |
| `--batch_size` | `None` | Mini-batch size for dataloaders. |
| `--num_workers` | `None` | DataLoader worker count. |
| `--lr_bert` | `None` | Learning rate for the BERT encoder. |
| `--lr_resnet` | `None` | Learning rate for the ResNet backbone. |
| `--lr_heads` | `None` | Learning rate for fusion/classifier heads. |
| `--bert_model_name` | `None` | Override HuggingFace model name/path. |

> Any `None` values fall back to the defaults defined in the underlying training script (`TrainConfig` for full training, CLI defaults for transfer learning).

### Full-Training Mode

`main.py --classifier_mode full`:

- Downloads the dataset from S3 unless `--skip_download` is provided.
- Launches `train_classifier/main_classifier.py` with the arguments listed above.

**Example:**

```bash
python main.py \
  --classifier_mode full \
  --dataset-dir dataset \
  --output_dir outputs \
  --epochs 25 \
  --batch_size 64 \
  --lr_bert 3e-5
```

### Transfer-Learning Mode

`main.py --classifier_mode transfer` triggers `train_classifier/main_classifier_TL.py`. In addition to the common flags, it accepts the following options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--state_dict_path` | `None` | Local path to a pretrained `state_dict`. |
| `--state_dict_s3_key` | `outputs_training/state_dict.pt` | S3 key to download the pretrained weights when `--state_dict_path` is not supplied. |
| `--previous_label_map_path` | `None` | Local label-map (JSON) to merge existing classes. |
| `--label_map_s3_key` | `outputs_training/label2id.json` | S3 key for the previous label-map (used when no local file is passed). |
| `--artifact_dir` | `pretrained_artifacts` | Local directory for downloaded artifacts. |
| `--max_length` | `None` | Tokenizer max sequence length (defaults to 512 in TL script). |
| `--image_size` | `None` | Image resize target as `H,W` (defaults to `224,224`). |
| `--early_stop_patience` | `None` | Override early-stopping patience (defaults to 5). |
| `--freeze_backbone` | _flag_ | Freeze BERT & ResNet during fine-tuning. |
| `--export_onnx` | _flag_ | Export the best checkpoint to ONNX after training. |

**Examples:**

Minimal run relying on S3 artifacts and defaults:

```bash
python main.py --classifier_mode transfer
```

Custom artifact locations and hyper-parameters:

```bash
python main.py \
  --classifier_mode transfer \
  --state_dict_path outputs/classifier/state_dict.pt \
  --previous_label_map_path outputs/classifier/label2id.json \
  --image_root_dir dataset/dataset_TL \
  --data_csv_path dataset/dataset_TL.csv \
  --freeze_backbone \
  --epochs 12 \
  --batch_size 24
```

## Running in Docker

Two GPU-enabled services are provided in `docker-compose.yaml`:

- `training-full-pytorch` – full training pipeline.
- `training-transfer-pytorch` – transfer-learning pipeline.

Both services share the same entrypoint (`run_pipeline.sh`) and can be configured via environment variables.

### Build Images

```bash
docker compose build training-full-pytorch training-transfer-pytorch
```

### Run Full Training in Docker

```bash
docker compose up training-full-pytorch
```

Key environment variables (see `docker-compose.yaml`):

- `CLASSIFIER_MODE=full`
- `DATASET_DIR=/app/dataset`
- `OUTPUT_DIR=/app/outputs`
- Optional overrides map directly to CLI arguments: set `EPOCHS=30`, `LR_BERT=2e-5`, etc.

### Run Transfer Learning in Docker

```bash
docker compose up training-transfer-pytorch
```

Additional useful variables for transfer mode:

- `IMAGE_ROOT_DIR` (defaults to `/app/dataset/dataset_TL`).
- `DATA_CSV_PATH` (defaults to `/app/dataset/dataset_TL.csv`).
- `STATE_DICT_PATH`, `STATE_DICT_S3_KEY`, `PREVIOUS_LABEL_MAP_PATH`, `LABEL_MAP_S3_KEY`, `ARTIFACT_DIR`.
- Boolean flags can be toggled with `FREEZE_BACKBONE=true` or `EXPORT_ONNX=true`.

### `run_pipeline.sh` Environment Mapping

`run_pipeline.sh` reads environment variables and forwards them to `main.py`. The mapping is one-to-one:

- `DATA_CSV_PATH` -> `--data_csv_path`
- `LR_BERT` -> `--lr_bert`
- `STATE_DICT_S3_KEY` -> `--state_dict_s3_key`
- `CLASSIFIER_EXTRA_ARGS` – additional CLI tokens appended verbatim (`"--epochs 5 --batch_size 16"`).
- `SKIP_MODEL_DOWNLOADS` – set to `true`/`1` to skip the optional warm-up downloads for BERT and ResNet weights.

This allows the same container image to handle both modes without modification.

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
        "CLASSIFIER_MODE": "full",
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

For transfer learning, switch `image` to `antifraud-training-transfer` (or keep the same image and set `CLASSIFIER_MODE=transfer`) and provide the relevant artifacts:

```python
train_transfer = KubernetesPodOperator(
    task_id="train_classifier_transfer",
    name="classifier-transfer",
    namespace="ml",
    image="registry.example.com/antifraud-training-transfer:latest",
    env_vars={
        "CLASSIFIER_MODE": "transfer",
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
