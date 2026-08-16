"""Frozen-MotionBERT correctness model for REHAB24 Ex5 Lunge."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from backbone.motionbert import MotionBERT
from experts.lunge_expert import LungeExpert
from models.pushup_correctness import PushupCorrectnessHead


class LungeCorrectnessModel(nn.Module):
    def __init__(self, motionbert_checkpoint: str | Path | None = None, *, backbone: Optional[nn.Module] = None, dropout: float = .25) -> None:
        super().__init__(); self.backbone=backbone or MotionBERT(checkpoint_path=motionbert_checkpoint)
        self.expert=LungeExpert(dropout=dropout); self.correctness_head=PushupCorrectnessHead(dropout=dropout); self.set_backbone_frozen(True)

    def set_backbone_frozen(self, frozen: bool=True) -> None:
        self.backbone_frozen=bool(frozen)
        for parameter in self.backbone.parameters(): parameter.requires_grad_(not frozen)
        if frozen: self.backbone.eval()

    def train(self, mode: bool=True):
        super().train(mode)
        if self.backbone_frozen: self.backbone.eval()
        return self

    def forward(self, values: torch.Tensor, temporal_mask: torch.Tensor|None=None):
        if values.ndim != 4 or values.shape[2:] != (17,3): raise ValueError(f"Expected (B,T,17,3), got {tuple(values.shape)}")
        with torch.no_grad() if self.backbone_frozen else torch.enable_grad(): features=self.backbone(values)
        return self.forward_features(features,temporal_mask)

    def forward_features(self, features: torch.Tensor, temporal_mask: torch.Tensor|None=None):
        expert=self.expert(features,temporal_mask=temporal_mask); embedding=expert["global_embedding"]
        if embedding is None: raise RuntimeError("LungeExpert returned no global embedding")
        logits=self.correctness_head(embedding)
        return {"logits":logits,"correct_probability":torch.softmax(logits,dim=-1)[:,1],"global_embedding":embedding}
