from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel


class HybridModel(nn.Module):
    def __init__(self, num_classes: int, bert_model_name: str = "indobenchmark/indobert-base-p2") -> None:
        super().__init__()

        # Text encoder (BERT)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.text_linear = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Vision encoder (ResNet50)
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = resnet50(weights=self.weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.image_linear = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Cross-attention and fusion head
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        # Text path
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state.mean(dim=1)
        text_features = self.text_linear(text_features)
        text_features = text_features * 1.1
        text_seq = text_features.unsqueeze(1)

        # Image path
        image_features = self.resnet(pixel_values)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_linear(image_features)
        image_features = image_features * 0.9
        image_seq = image_features.unsqueeze(1)

        # Cross attention (text queries over image keys/values)
        attn_output, _ = self.cross_attn(query=text_seq, key=image_seq, value=image_seq)
        fused_text = text_seq + attn_output
        fused_text = fused_text.squeeze(1)

        # Concatenate and classify
        combined = torch.cat((fused_text, image_features), dim=1)
        logits = self.fusion(combined)
        return logits


