#!/usr/bin/env python3
"""Train a modified Kneser-Ney trigram language model on a clean corpus.

The output is a pickle containing the counts and discount params needed to
compute p_KN(w | w_{-2}, w_{-1}) at decode time. Trigram counts are pruned to
keep the model compact.

References: Chen & Goodman (1999), Heafield (2013) for modified KN.
"""
import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path


BOS = "<s>"
EOS = "</s>"


def iter_ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i + n])


def build_counts(corpus_path: Path):
    uni = Counter()
    bi = Counter()
    tri = Counter()
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            toks = [BOS, BOS] + toks + [EOS]
            for w in toks:
                uni[w] += 1
            for ng in iter_ngrams(toks, 2):
                bi[ng] += 1
            for ng in iter_ngrams(toks, 3):
                tri[ng] += 1
    return uni, bi, tri


def prune(counts: Counter, min_count: int) -> Counter:
    if min_count <= 1:
        return counts
    return Counter({k: v for k, v in counts.items() if v >= min_count})


def compute_discounts(counts: Counter):
    """Modified-KN discounts from n1..n4."""
    histogram = Counter(counts.values())
    n1 = histogram[1] or 1
    n2 = histogram[2] or 1
    n3 = histogram[3] or 1
    n4 = histogram[4] or 1
    Y = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.5
    d1 = 1 - 2 * Y * n2 / n1 if n1 > 0 else 0.5
    d2 = 2 - 3 * Y * n3 / n2 if n2 > 0 else 0.75
    d3 = 3 - 4 * Y * n4 / n3 if n3 > 0 else 1.0
    d1 = max(0.1, min(0.99, d1))
    d2 = max(0.1, min(1.99, d2))
    d3 = max(0.1, min(2.99, d3))
    return d1, d2, d3


def build_continuation_counts(bi: Counter, tri: Counter):
    """Continuation counts used as lower-order estimates in KN.

    cc1[w] = #distinct contexts u where (u, w) was seen
    cc2[(u, w)] = #distinct contexts (v, u) where (v, u, w) was seen? No — we need:
      for bigram-to-trigram: cc_bi[w_1, w_2] = #distinct v such that (v, w_1, w_2) seen
      for unigram-from-bigram: cc_uni[w] = #distinct u such that (u, w) seen
    """
    cc_uni = Counter()
    seen_uni_pairs = set()
    for (u, w) in bi:
        if u in (BOS,):
            pass
        key = (u, w)
        if key not in seen_uni_pairs:
            seen_uni_pairs.add(key)
            cc_uni[w] += 1

    cc_bi = Counter()
    seen_bi_triples = set()
    for (v, u, w) in tri:
        key = (v, u, w)
        if key not in seen_bi_triples:
            seen_bi_triples.add(key)
            cc_bi[(u, w)] += 1

    total_cc_uni = sum(cc_uni.values())
    return cc_uni, cc_bi, total_cc_uni


def build_context_aux(bi: Counter, tri: Counter):
    """Aux maps for gamma computation.

    gamma(u, v) uses counts of trigrams starting with (u,v) stratified by how
    often they occurred. We need, per bigram (u,v), how many distinct w
    followed at count 1, 2, 3+.
    """
    per_bi_strata = defaultdict(lambda: [0, 0, 0])  # n1, n2, n3+
    bi_followers_total = Counter()  # sum c(u,v,*)
    for (u, v, w), c in tri.items():
        if c == 1:
            per_bi_strata[(u, v)][0] += 1
        elif c == 2:
            per_bi_strata[(u, v)][1] += 1
        else:
            per_bi_strata[(u, v)][2] += 1
        bi_followers_total[(u, v)] += c

    per_uni_strata = defaultdict(lambda: [0, 0, 0])  # per left-context u, counts of (u, *) stratified
    uni_followers_total = Counter()
    for (u, v), c in bi.items():
        if c == 1:
            per_uni_strata[u][0] += 1
        elif c == 2:
            per_uni_strata[u][1] += 1
        else:
            per_uni_strata[u][2] += 1
        uni_followers_total[u] += c
    return per_bi_strata, bi_followers_total, per_uni_strata, uni_followers_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data_clean/good_corpus.txt")
    ap.add_argument("--out", default="data_clean/kn_lm.pkl")
    ap.add_argument("--min-tri", type=int, default=2, help="prune trigrams seen fewer times")
    ap.add_argument("--min-bi", type=int, default=1)
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    print(f"[kn-lm] counting ngrams from {corpus_path}", file=sys.stderr)
    uni, bi, tri = build_counts(corpus_path)
    print(f"  unigrams={len(uni):,}  bigrams={len(bi):,}  trigrams={len(tri):,}", file=sys.stderr)

    tri = prune(tri, args.min_tri)
    bi = prune(bi, args.min_bi)
    print(f"  after prune: bigrams={len(bi):,}  trigrams={len(tri):,}", file=sys.stderr)

    d1_tri, d2_tri, d3_tri = compute_discounts(tri)
    d1_bi, d2_bi, d3_bi = compute_discounts(bi)
    print(f"  discounts tri: D1={d1_tri:.3f} D2={d2_tri:.3f} D3={d3_tri:.3f}", file=sys.stderr)
    print(f"  discounts bi:  D1={d1_bi:.3f} D2={d2_bi:.3f} D3={d3_bi:.3f}", file=sys.stderr)

    cc_uni, cc_bi, total_cc_uni = build_continuation_counts(bi, tri)
    per_bi_strata, bi_followers_total, per_uni_strata, uni_followers_total = build_context_aux(bi, tri)

    model = dict(
        uni=dict(uni),
        bi=dict(bi),
        tri=dict(tri),
        cc_uni=dict(cc_uni),
        cc_bi=dict(cc_bi),
        total_cc_uni=total_cc_uni,
        per_bi_strata={k: list(v) for k, v in per_bi_strata.items()},
        bi_followers_total=dict(bi_followers_total),
        per_uni_strata={k: list(v) for k, v in per_uni_strata.items()},
        uni_followers_total=dict(uni_followers_total),
        d_tri=(d1_tri, d2_tri, d3_tri),
        d_bi=(d1_bi, d2_bi, d3_bi),
        BOS=BOS,
        EOS=EOS,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(model, f, protocol=4)
    print(f"[kn-lm] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
