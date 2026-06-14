#!/usr/bin/env python3
"""Train the coherence discriminator (scripts/disc_model.py) on disc_data.jsonl."""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disc_model import DiscriminatorModel, PAD, UNK, BOS, EOS  # noqa: E402


def build_vocab(rows, max_vocab):
    c = Counter()
    for r in rows:
        c.update(r["tokens"])
    itos = [PAD, UNK] + [w for w, _ in c.most_common(max_vocab)]
    return {w: i for i, w in enumerate(itos)}


def encode(tokens, stoi):
    unk = stoi[UNK]
    return [stoi.get(t, unk) for t in tokens] or [unk]


def batches(data, bs, device, shuffle=True):
    order = list(range(len(data)))
    if shuffle:
        random.shuffle(order)
    for s in range(0, len(order), bs):
        chunk = [data[i] for i in order[s:s + bs]]
        maxlen = max(len(x[0]) for x in chunk)
        xb, yb, ln = [], [], []
        for ids, lab in chunk:
            xb.append(ids + [0] * (maxlen - len(ids)))
            yb.append(lab); ln.append(len(ids))
        yield (torch.tensor(xb, dtype=torch.long, device=device),
               torch.tensor(yb, dtype=torch.float, device=device), ln)


def auc(y, p):
    pairs = [(pi, yi) for pi, yi in zip(p, y)]
    pairs.sort()
    ranks = {}
    # average-rank tie handling
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        r = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = r
        i = j
    sum_pos = sum(ranks[k] for k, (_, yi) in enumerate(pairs) if yi == 1)
    npos = sum(1 for yi in y if yi == 1); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (sum_pos - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_clean/disc_data.jsonl")
    ap.add_argument("--out", default="data_clean/disc.pt")
    ap.add_argument("--max-vocab", type=int, default=12000)
    ap.add_argument("--emb", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    random.shuffle(rows)
    stoi = build_vocab(rows, args.max_vocab)
    data = [(encode(r["tokens"], stoi), r["label"]) for r in rows]
    nval = max(2000, len(data) // 20)
    val, train = data[:nval], data[nval:]
    npos = sum(1 for _, l in train if l == 1)
    print(f"[disc] vocab={len(stoi)} train={len(train)} val={len(val)} "
          f"train_pos={npos} train_neg={len(train)-npos}", file=sys.stderr, flush=True)

    device = "cpu"
    model = DiscriminatorModel(len(stoi), args.emb, args.hidden, 0.3, stoi[PAD]).to(device)
    # class-balanced BCE
    pos_w = torch.tensor([(len(train) - npos) / max(1, npos)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); run = seen = 0
        for bi, (xb, yb, ln) in enumerate(batches(train, args.batch_size, device)):
            opt.zero_grad()
            logit = model(xb, ln)
            loss = crit(logit, yb)
            loss.backward(); opt.step()
            run += float(loss); seen += 1
            if bi % 100 == 0:
                print(f"  ep{ep} b{bi} loss={run/max(1,seen):.3f}", file=sys.stderr, flush=True)
        # val
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for xb, yb, ln in batches(val, args.batch_size, device, shuffle=False):
                p = torch.sigmoid(model(xb, ln))
                ps += p.tolist(); ys += yb.tolist()
        a = auc(ys, ps)
        acc = sum(1 for pi, yi in zip(ps, ys) if (pi >= 0.5) == (yi == 1)) / len(ys)
        print(f"[disc] epoch {ep} {time.time()-t0:.0f}s  val_auc={a:.4f} val_acc={acc:.3f}", file=sys.stderr, flush=True)
        if a > best:
            best = a
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "stoi": stoi, "config": {"emb": args.emb, "hidden": args.hidden},
                        "val_auc": a}, args.out)
            print(f"[disc] saved {args.out} (val_auc={a:.4f})", file=sys.stderr, flush=True)
    print(f"[disc] best val_auc={best:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
