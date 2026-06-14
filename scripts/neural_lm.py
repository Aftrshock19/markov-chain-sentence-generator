#!/usr/bin/env python3
"""Small word-level LSTM language model — model definition + inference wrapper.

A "learned from data, no-LLM" neural LM trained on the same clean corpus as the
KN trigram. Its job is to give a FULL-left-context sentence score so the
generator can demote locally-fluent-but-globally-incoherent candidates
("El vino es al mismo tiempo.") that a 2-token trigram window cannot see.

Inference avoids torch<->numpy interop (the repo's numpy is 2.x but torch was
built against 1.x), so everything stays in torch tensors / Python lists.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
PAD = "<pad>"


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 256, hidden: int = 512,
                 layers: int = 2, dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x, hc=None):
        e = self.drop(self.embed(x))
        out, hc = self.lstm(e, hc)
        logits = self.proj(self.drop(out))
        return logits, hc


class NeuralLM:
    """Inference wrapper: load a checkpoint and score sentences."""

    def __init__(self, model: LSTMLanguageModel, stoi: Dict[str, int], device: str = "cpu"):
        self.model = model.to(device).eval()
        self.stoi = stoi
        self.device = device
        self.unk = stoi[UNK]
        self.bos = stoi[BOS]
        self.eos = stoi[EOS]

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> Optional["NeuralLM"]:
        p = Path(path)
        if not p.exists():
            return None
        ckpt = torch.load(p, map_location=device)
        cfg = ckpt["config"]
        stoi = ckpt["stoi"]
        model = LSTMLanguageModel(
            vocab_size=len(stoi), emb=cfg["emb"], hidden=cfg["hidden"],
            layers=cfg["layers"], dropout=0.0, pad_idx=stoi[PAD],
        )
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, stoi, device=device)

    def _encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(t.lower(), self.unk) for t in tokens]

    @torch.no_grad()
    def sentence_logprob(self, tokens: Sequence[str]) -> float:
        """Sum_t log p(w_t | w_<t) over the real tokens plus </s>, with a single
        <s> priming the context. Mirrors KNLanguageModel.sentence_logprob's shape
        so the two scores are interchangeable in the scorer."""
        ids = self._encode([t for t in tokens if t and t not in (BOS, EOS)])
        if not ids:
            return 0.0
        seq = [self.bos] + ids + [self.eos]
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=self.device)
        tgt = seq[1:]
        logits, _ = self.model(x)
        logp = torch.log_softmax(logits[0], dim=-1)
        total = 0.0
        for i, t in enumerate(tgt):
            total += float(logp[i, t])
        return total

    def logp_per_token(self, tokens: Sequence[str]) -> float:
        toks = [t for t in tokens if t and t not in (BOS, EOS)]
        if not toks:
            return 0.0
        return self.sentence_logprob(toks) / (len(toks) + 1)  # +1 for </s>


_CACHE: Dict[str, NeuralLM] = {}


def load_neural_lm(path: Path, device: str = "cpu") -> Optional[NeuralLM]:
    key = f"{path}@{device}"
    if key in _CACHE:
        return _CACHE[key]
    m = NeuralLM.load(path, device=device)
    if m is not None:
        _CACHE[key] = m
    return m
