# Classifier Training Pipeline

This document explains how to run `main.py`, the orchestrator for the BPJS classifier training pipelines.
It covers the available modes, runtime requirements, and all supported command-line arguments.

## Prerequisites

- Python 3.9 or newer.
- Project dependencies installed: `pip install -r requirements.txt`.
- Access to pretrained artifacts and datasets stored in S3 (optional but recommended).
- Environment variables (used when downloading from S3):
  - `S3_CLASSIFIER_PREFIX` – S3 prefix containing the full-training dataset (e.g. `dataset/classifier/`).
  - `AWS`/`S3` credentials used by `boto3` (e.g. `S3_BUCKET_NAME`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`).

## Running `main.py`

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

## Logs and Outputs

- Logs stream to stdout and are timestamped for readability.
- Artifacts (state dicts, ONNX exports, label maps) are stored in the run folder inside `--output_dir`.
- Transfer-learning runs maintain previous class weights by merging the supplied label map with new classes.

## Troubleshooting

- Ensure all required environment variables are set when using S3 downloads.
- Confirm GPU availability with `torch.cuda.is_available()` inside your environment if you expect GPU training.
- Use `--skip_download` when working fully offline.
- When providing custom CSV or image directories, verify that file paths and label columns follow the expected format (`label/filename`).

For additional customization, inspect `train_classifier/main_classifier.py` (full training) and `train_classifier/main_classifier_TL.py` (transfer learning).
