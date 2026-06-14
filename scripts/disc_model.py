#!/usr/bin/env python3
"""Coherence discriminator: model definition + inference wrapper.

A small BiLSTM sentence classifier P(real | sentence). Trained (train_disc.py) to
separate real corpus sentences (positives) from two negative classes:
  1. minimal-edit corruptions of real sentences (token-level selectional fit), and
  2. generator-output junk (sentence-level pipeline-junk signature).

Used as a corpus-derived, no-LLM coherence energy. Inference stays in torch
tensors / Python lists (the env's numpy 2.x breaks torch<->numpy interop).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"


class DiscriminatorModel(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 96, hidden: int = 96,
                 dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x, lengths):
        e = self.drop(self.embed(x))                       # B,T,E
        out, _ = self.lstm(e)                              # B,T,2H
        mask = (x != 0).unsqueeze(-1).float()              # B,T,1
        out = out * mask
        summed = out.sum(1)
        meanp = summed / mask.sum(1).clamp(min=1)          # B,2H
        very_neg = out.masked_fill(mask == 0, -1e9)
        maxp = very_neg.max(1).values                      # B,2H
        feat = torch.cat([meanp, maxp], dim=-1)            # B,4H
        return self.head(self.drop(feat)).squeeze(-1)      # B (logit)


class Discriminator:
    def __init__(self, model: DiscriminatorModel, stoi: Dict[str, int], device: str = "cpu"):
        self.model = model.to(device).eval()
        self.stoi = stoi
        self.device = device
        self.unk = stoi[UNK]

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> Optional["Discriminator"]:
        p = Path(path)
        if not p.exists():
            return None
        ckpt = torch.load(p, map_location=device)
        cfg, stoi = ckpt["config"], ckpt["stoi"]
        m = DiscriminatorModel(len(stoi), cfg["emb"], cfg["hidden"], 0.0, stoi[PAD])
        m.load_state_dict(ckpt["state_dict"])
        return cls(m, stoi, device)

    def _encode(self, tokens: Sequence[str]) -> List[int]:
        ids = [self.stoi.get(t.lower(), self.unk) for t in tokens if t and t not in (BOS, EOS)]
        return ids or [self.unk]

    @torch.no_grad()
    def score(self, tokens: Sequence[str]) -> float:
        ids = self._encode(tokens)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logit = self.model(x, [len(ids)])
        return float(torch.sigmoid(logit)[0])


_CACHE: Dict[str, Discriminator] = {}


def load_discriminator(path: Path, device: str = "cpu") -> Optional[Discriminator]:
    key = f"{path}@{device}"
    if key in _CACHE:
        return _CACHE[key]
    d = Discriminator.load(path, device)
    if d is not None:
        _CACHE[key] = d
    return d
