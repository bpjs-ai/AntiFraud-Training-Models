#!/usr/bin/env bash
set -e

export HF_HOME=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch
export PIP_NO_CACHE_DIR=1

python -c "from transformers import AutoModel; AutoModel.from_pretrained('indobenchmark/indobert-base-p2')"
python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"

CLASSIFIER_MODE="${CLASSIFIER_MODE:-full}"
ARGS=(
  "--classifier_mode" "$CLASSIFIER_MODE"
  "--dataset-dir" "${DATASET_DIR:-/app/dataset}"
  "--output_dir" "${OUTPUT_DIR:-/app/outputs}"
)

append_if_set() {
  local var_name="$1"
  local flag="$2"
  local value="${!var_name}"
  if [[ -n "$value" ]]; then
    ARGS+=("$flag" "$value")
  fi
}

append_flag_if_truthy() {
  local var_name="$1"
  local flag="$2"
  local value="${!var_name}"
  if [[ "$value" == "1" || "$value" == "true" || "$value" == "True" ]]; then
    ARGS+=("$flag")
  fi
}

append_if_set DATA_CSV_PATH "--data_csv_path"
append_if_set IMAGE_ROOT_DIR "--image_root_dir"
append_if_set VAL_SIZE "--val_size"
append_if_set EPOCHS "--epochs"
append_if_set BATCH_SIZE "--batch_size"
append_if_set NUM_WORKERS "--num_workers"
append_if_set LR_BERT "--lr_bert"
append_if_set LR_RESNET "--lr_resnet"
append_if_set LR_HEADS "--lr_heads"
append_if_set BERT_MODEL_NAME "--bert_model_name"

append_if_set STATE_DICT_PATH "--state_dict_path"
append_if_set STATE_DICT_S3_KEY "--state_dict_s3_key"
append_if_set PREVIOUS_LABEL_MAP_PATH "--previous_label_map_path"
append_if_set LABEL_MAP_S3_KEY "--label_map_s3_key"
append_if_set ARTIFACT_DIR "--artifact_dir"
append_if_set MAX_LENGTH "--max_length"
append_if_set IMAGE_SIZE "--image_size"
append_if_set EARLY_STOP_PATIENCE "--early_stop_patience"
append_flag_if_truthy FREEZE_BACKBONE "--freeze_backbone"
append_flag_if_truthy EXPORT_ONNX "--export_onnx"

python main.py "${ARGS[@]}"
