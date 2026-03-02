import os
import json
from pathlib import Path
from io import BytesIO

import fitz  # PyMuPDF
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, resnet50

import warnings

warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore")


class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        # Text model - IndoBERT
        self.bert = BertModel.from_pretrained("indobenchmark/indobert-base-p2")
        self.text_linear = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Vision model - ResNet50
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.resnet = resnet50(weights=self.weights)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove last fc
        self.image_linear = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.dropout = nn.Dropout(0.5)

        # Multi-Head Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

    def forward(self, input_ids, attention_mask, pixel_values):
        # Process text
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state.mean(dim=1)
        text_features = self.text_linear(text_features)
        text_features = text_features * 1.1
        text_feat_seq = text_features.unsqueeze(1)

        # Process image
        image_features = self.resnet(pixel_values)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_linear(image_features)
        image_features = image_features * 0.9
        image_feat_seq = image_features.unsqueeze(1)

        # Cross attention
        attn_output, _ = self.cross_attn(
            query=text_feat_seq,
            key=image_feat_seq,
            value=image_feat_seq
        )
        text_with_image = text_feat_seq + attn_output
        text_features = text_with_image.squeeze(1)

        # Weighted concatenation
        combined = torch.cat((text_features, image_features), dim=1)
        logits = self.fusion(combined)

        return logits


# --- 1. Load tokenizer & transform ---
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 2. Create dummy input ---
text = "Resume Medis Rawat Inap"
encoding = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=16,
    return_tensors='pt'
)

# Make sure the image file exists or replace with a placeholder
image_path = "(01)0195R0580824V000018_7.png"
if os.path.exists(image_path):
    image = Image.open(image_path).convert("RGB")
else:
    print(f"Warning: Image file not found at '{image_path}'. Using a dummy black image.")
    image = Image.new('RGB', (100, 100)) # Placeholder image

pixel_values = transform(image).unsqueeze(0)

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

STATE_DICT_PATH = Path("outputs/best_model.pt")
if not STATE_DICT_PATH.exists():
    STATE_DICT_PATH = Path("best_model.pt")

LABEL_MAP_PATH = Path("outputs/label2id.json")
if not LABEL_MAP_PATH.exists():
    LABEL_MAP_PATH = Path("label2id.json")

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    raw_class_names = json.load(f)

class_names = {str(v): k for k, v in raw_class_names.items()}

# --- 3. Load model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HybridModel(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(str(STATE_DICT_PATH), map_location=device))
model.eval()

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
pixel_values = pixel_values.to(device)

# --- 4. Export TorchScript ---
print("Exporting to TorchScript...")
traced_model = torch.jit.trace(model, (input_ids, attention_mask, pixel_values))
torch.jit.save(traced_model, "hybrid_model.pt")
print("TorchScript model saved to hybrid_model.pt")

# --- 5. Export ONNX ---
print("\nExporting to ONNX...")
torch.onnx.export(
    model,
    (input_ids, attention_mask, pixel_values),
    "hybrid_model.onnx",
    input_names=["input_ids", "attention_mask", "pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=18
)
print("ONNX model saved to hybrid_model.onnx")

# --- 6. Compare outputs ---
print("\n--- Comparing Model Outputs ---")

# PyTorch and TorchScript inference
# Using torch.no_grad() disables gradient calculation, which is essential for inference
# It prevents the error and reduces memory usage.
with torch.no_grad():
    # Original PyTorch model
    out_pt = model(input_ids, attention_mask, pixel_values).cpu().numpy()

    # Load and run TorchScript model
    ts_model = torch.jit.load("hybrid_model.pt").to(device)
    ts_out = ts_model(input_ids, attention_mask, pixel_values).cpu().numpy()

# ONNX inference
onnx_sess = ort.InferenceSession("hybrid_model.onnx", providers=['CPUExecutionProvider'])
onnx_out = onnx_sess.run(
    None,
    {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
        "pixel_values": pixel_values.cpu().numpy()
    }
)[0]

# --- 7. Calculate and print differences ---
diff_pt_ts = np.max(np.abs(out_pt - ts_out))
diff_pt_onnx = np.max(np.abs(out_pt - onnx_out))

print(f"\nMax absolute difference (PyTorch vs. TorchScript): {diff_pt_ts}")
print(f"Max absolute difference (PyTorch vs. ONNX):      {diff_pt_onnx}")

# A small difference is expected due to floating-point arithmetic variations
if diff_pt_ts < 1e-5 and diff_pt_onnx < 1e-5:
    print("\n✅ Outputs are consistent across all three formats.")
else:
    print("\n⚠️ Outputs show significant differences. Review the export process.")
