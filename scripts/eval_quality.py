#!/usr/bin/env python3
"""Score two generator output CSVs with the trained reranker.

Reports mean reranker logit, mean predicted-positive probability, % estimated
good (logit > 0), and side-by-side sentence dumps per lemma.
"""
import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import featurize, tokenize  # noqa: E402
from kn_lm import KNLanguageModel  # noqa: E402


def load_rank_by_word(path: Path):
    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lemma = (row.get("lemma") or row.get("word") or "").strip().lower()
            if not lemma:
                continue
            try:
                rank = int(float(row.get("rank") or 999999))
            except ValueError:
                rank = 999999
            out.setdefault(lemma, rank)
    return out


def load_reranker(path: Path):
    with path.open("rb") as f:
        m = pickle.load(f)
    return m


def score(features: dict, reranker_model: dict) -> float:
    z = reranker_model["intercept"]
    for name, mean, scale, coef in zip(
        reranker_model["feature_names"],
        reranker_model["scaler_mean"],
        reranker_model["scaler_scale"],
        reranker_model["coef"],
    ):
        v = (features.get(name, 0.0) - mean) / (scale if scale else 1.0)
        z += v * coef
    return z


def score_file(path: Path, lm: KNLanguageModel, rank_by_word, reranker):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sent = row.get("sentence", "")
            lemma = (row.get("lemma") or "").strip().lower()
            try:
                rank = int(float(row.get("rank") or 999999))
            except ValueError:
                rank = 999999
            toks = tokenize(sent)
            if not toks:
                rows.append((lemma, rank, sent, None, None, None))
                continue
            lm_lp = lm.sentence_logprob(toks)
            feats = featurize(toks, lemma, rank, lm_lp, rank_by_word)
            z = score(feats, reranker)
            p = 1.0 / (1.0 + math.exp(-z))
            rows.append((lemma, rank, sent, z, p, lm_lp))
    return rows


def summarize(rows, name: str):
    total = len(rows)
    nonempty = [r for r in rows if r[2]]
    scored = [r for r in rows if r[3] is not None]
    empty = total - len(nonempty)
    mean_logit = sum(r[3] for r in scored) / max(1, len(scored))
    mean_prob = sum(r[4] for r in scored) / max(1, len(scored))
    pct_positive = sum(1 for r in scored if r[3] > 0) / max(1, len(scored))
    mean_lm = sum(r[5] for r in scored) / max(1, len(scored))
    mean_lm_per_tok = sum(r[5] / max(1, len(tokenize(r[2]))) for r in scored) / max(1, len(scored))
    print(f"=== {name} ===")
    print(f"  total:            {total}")
    print(f"  empty:            {empty}")
    print(f"  mean reranker z:  {mean_logit:+.3f}")
    print(f"  mean p(good):     {mean_prob:.3f}")
    print(f"  %(z > 0):         {100*pct_positive:.1f}%")
    print(f"  mean LM logprob:  {mean_lm:.2f}")
    print(f"  mean LM/token:    {mean_lm_per_tok:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--lm", default="data_clean/kn_lm.pkl")
    ap.add_argument("--reranker", default="data_clean/reranker.pkl")
    ap.add_argument("--lexicon", default="stg_words_spa.csv")
    ap.add_argument("--diff-out", default=None)
    args = ap.parse_args()

    lm = KNLanguageModel(Path(args.lm))
    reranker = load_reranker(Path(args.reranker))
    rank_by_word = load_rank_by_word(Path(args.lexicon))

    results = {p: score_file(Path(p), lm, rank_by_word, reranker) for p in args.files}
    for p, rows in results.items():
        summarize(rows, p)
        print()

    if args.diff_out and len(args.files) == 2:
        a, b = args.files
        ra, rb = results[a], results[b]
        by_lemma_a = {(r[0], r[1]): r for r in ra}
        by_lemma_b = {(r[0], r[1]): r for r in rb}
        keys = sorted(set(by_lemma_a) & set(by_lemma_b), key=lambda k: k[1])
        with open(args.diff_out, "w", encoding="utf-8") as out:
            out.write(f"rank\tlemma\tz_{Path(a).stem}\tz_{Path(b).stem}\tsent_{Path(a).stem}\tsent_{Path(b).stem}\n")
            for lemma, rank in keys:
                ra_row = by_lemma_a[(lemma, rank)]
                rb_row = by_lemma_b[(lemma, rank)]
                out.write(
                    f"{rank}\t{lemma}\t"
                    f"{ra_row[3] if ra_row[3] is not None else ''}\t"
                    f"{rb_row[3] if rb_row[3] is not None else ''}\t"
                    f"{ra_row[2]}\t{rb_row[2]}\n"
                )
        print(f"diff table -> {args.diff_out}")


if __name__ == "__main__":
    main()
