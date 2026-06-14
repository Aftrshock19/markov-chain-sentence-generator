#!/usr/bin/env python3
"""Train the small LSTM language model (scripts/neural_lm.py) on the clean corpus.

Same corpus as the KN trigram (data_clean/good_corpus.txt). Trains on Apple MPS
if available. Avoids torch<->numpy interop (numpy 2.x / torch built vs 1.x), so
batching is done with plain Python + torch tensors.
"""
import argparse
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neural_lm import LSTMLanguageModel, BOS, EOS, UNK, PAD  # noqa: E402


def build_vocab(corpus_path: Path, max_vocab: int):
    c = Counter()
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            c.update(line.split())
    itos = [PAD, BOS, EOS, UNK] + [w for w, _ in c.most_common(max_vocab)]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def load_encoded(corpus_path: Path, stoi, max_sents: int = 0):
    unk = stoi[UNK]
    bos, eos = stoi[BOS], stoi[EOS]
    data = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            ids = [bos] + [stoi.get(t, unk) for t in toks] + [eos]
            if len(ids) >= 3:
                data.append(ids)
            if max_sents and len(data) >= max_sents:
                break
    return data


def batches(data, batch_size, pad_idx, device, shuffle=True):
    order = list(range(len(data)))
    if shuffle:
        random.shuffle(order)
    for s in range(0, len(order), batch_size):
        chunk = [data[i] for i in order[s:s + batch_size]]
        maxlen = max(len(x) for x in chunk)
        xb, yb = [], []
        for ids in chunk:
            padded = ids + [pad_idx] * (maxlen - len(ids))
            xb.append(padded[:-1])
            yb.append(padded[1:])
        yield (torch.tensor(xb, dtype=torch.long, device=device),
               torch.tensor(yb, dtype=torch.long, device=device))


def evaluate(model, data, batch_size, pad_idx, device, crit):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for xb, yb in batches(data, batch_size, pad_idx, device, shuffle=False):
            logits, _ = model(xb)
            loss = crit(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            ntok = int((yb != pad_idx).sum())
            tot_loss += float(loss) * ntok
            tot_tok += ntok
    model.train()
    return math.exp(tot_loss / max(1, tot_tok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data_clean/good_corpus.txt")
    ap.add_argument("--out", default="data_clean/neural_lm.pt")
    ap.add_argument("--max-vocab", type=int, default=20000)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--max-sents", type=int, default=0, help="cap training sentences (0=all)")
    ap.add_argument("--threads", type=int, default=0, help="CPU torch threads (0=default)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "cpu" and args.threads:
        torch.set_num_threads(args.threads)
    print(f"[nlm] device={device} threads={torch.get_num_threads()}", file=sys.stderr, flush=True)

    corpus = Path(args.corpus)
    stoi, itos = build_vocab(corpus, args.max_vocab)
    print(f"[nlm] vocab={len(stoi):,}", file=sys.stderr, flush=True)
    data = load_encoded(corpus, stoi, max_sents=args.max_sents)
    random.shuffle(data)
    n_val = max(1000, len(data) // 50)
    val, train = data[:n_val], data[n_val:]
    print(f"[nlm] train={len(train):,}  val={len(val):,}", file=sys.stderr)

    pad_idx = stoi[PAD]
    model = LSTMLanguageModel(len(stoi), args.emb, args.hidden, args.layers,
                             args.dropout, pad_idx=pad_idx).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[nlm] params={nparams/1e6:.1f}M", file=sys.stderr)
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ppl = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run, seen = 0.0, 0
        for bi, (xb, yb) in enumerate(batches(train, args.batch_size, pad_idx, device)):
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss); seen += 1
            if bi % args.log_every == 0:
                print(f"  ep{ep} batch {bi} loss={run/max(1,seen):.3f}", file=sys.stderr, flush=True)
        ppl = evaluate(model, val, args.batch_size, pad_idx, device, crit)
        print(f"[nlm] epoch {ep} done in {time.time()-t0:.0f}s  val_ppl={ppl:.2f}", file=sys.stderr)
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "stoi": stoi,
                "config": {"emb": args.emb, "hidden": args.hidden, "layers": args.layers},
                "val_ppl": ppl,
            }, args.out)
            print(f"[nlm] saved {args.out} (val_ppl={ppl:.2f})", file=sys.stderr)

    print(f"[nlm] best val_ppl={best_ppl:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
