#!/usr/bin/env python3
"""Build training data for the coherence discriminator (two negative classes).

Positives  : real corpus sentences (data_clean/good_corpus.txt).
Negatives 1: minimal-edit CORRUPTIONS of real sentences — replace ONE content
             word with a substitute that is LOCALLY plausible (a frequent bigram
             follower of the preceding word) yet GLOBALLY incoherent (low/zero
             PMI with the sentence's other content words). The low-PMI filter is
             what excludes accidentally-valid same-class swaps (madre/padre):
             if the substitute actually fits, its PMI is high and we reject it as
             a corruption. Content words only.
Negatives 2: GENERATOR junk — generator outputs not labeled 'good'.

All frozen-eval sentences are excluded from every split (held-out).

Output: data_clean/disc_data.jsonl  {tokens, label(1=real,0=neg), src}
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kn_lm import KNLanguageModel  # noqa: E402
from collocations import load_collocations  # noqa: E402
from build_collocations import is_content  # noqa: E402


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def load_frozen(path: Path):
    if not path.exists():
        return set()
    d = json.loads(path.read_text())
    out = set()
    for k in ("good", "nonsense"):
        for s in d.get(k, []):
            out.add(norm(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data_clean/good_corpus.txt")
    ap.add_argument("--lm", default="data_clean/kn_lm.pkl")
    ap.add_argument("--colloc", default="data_clean/collocations.pkl")
    ap.add_argument("--frozen", default="data_clean/disc_eval_frozen.json")
    ap.add_argument("--gen-csv", nargs="*", default=[
        "outputs/markov_ml_regen.csv", "outputs/markov_ml_500.csv",
        "outputs/ab_neural_off.csv", "outputs/ab_neural_on.csv",
    ])
    ap.add_argument("--gen-labels", default="data_clean/regen2_labels.csv",
                    help="per-row labels (lemma,sentence,label) to source clean generator-junk")
    ap.add_argument("--out", default="data_clean/disc_data.jsonl")
    ap.add_argument("--n-pos", type=int, default=60000)
    ap.add_argument("--n-corrupt", type=int, default=45000)
    ap.add_argument("--min-freq", type=int, default=80, help="only corrupt/replace reasonably frequent content words")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    lm = KNLanguageModel(Path(args.lm))
    col = load_collocations(Path(args.colloc))
    if col is None:
        raise SystemExit("collocation table required for the corruptor")
    frozen = load_frozen(Path(args.frozen))
    print(f"[disc-data] frozen held-out: {len(frozen)} sentences", file=sys.stderr)

    # Load corpus sentences (token lists), excluding frozen.
    sents = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if 3 <= len(t) <= 18 and norm(" ".join(t)) not in frozen:
                sents.append(t)
    random.shuffle(sents)
    print(f"[disc-data] corpus sentences: {len(sents)}", file=sys.stderr)

    # content-word frequency table (for frequency-matched substitutes)
    cfreq = Counter()
    for t in sents:
        for w in t:
            if is_content(w):
                cfreq[w] += 1
    content_vocab = [w for w, c in cfreq.items() if c >= args.min_freq]
    content_weights = [cfreq[w] for w in content_vocab]
    print(f"[disc-data] content vocab (freq>={args.min_freq}): {len(content_vocab)}", file=sys.stderr)

    def stem(w):  # crude: collapse simple plural / gender variants to avoid morph-variant swaps
        for suf in ("es", "s", "a", "o"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
        return w

    def low_pmi(sub, others):
        """True if `sub` is globally incoherent with the sentence's other content
        words (mostly zero / negative PMI) -> a genuine corruption, not a valid swap."""
        vals = []
        for o in others:
            p = col.window_pmi(sub, o)
            if p is not None:
                vals.append(p)
        if not vals:
            return True  # no evidence of fit -> treat as incoherent
        return (sum(vals) / len(vals)) <= -0.5

    def corrupt(tokens):
        idxs = [i for i, w in enumerate(tokens) if is_content(w) and cfreq.get(w, 0) >= args.min_freq]
        random.shuffle(idxs)
        for i in idxs:
            w = tokens[i]
            others = [tokens[j] for j in range(len(tokens)) if j != i and is_content(tokens[j])]
            prev = tokens[i - 1] if i > 0 else lm.BOS
            # locally-fluent candidates: frequent bigram followers of prev, content-only
            foll = lm.bi_followers(prev) if prev else {}
            cands = [w2 for w2 in foll if is_content(w2) and cfreq.get(w2, 0) >= args.min_freq
                     and stem(w2) != stem(w)]
            random.shuffle(cands)
            tried = 0
            for sub in cands:
                if low_pmi(sub, others):
                    return tokens[:i] + [sub] + tokens[i + 1:]
                tried += 1
                if tried > 25:
                    break
            # fallback: frequency-matched random content word that is low-PMI and not same stem
            for _ in range(20):
                sub = random.choices(content_vocab, weights=content_weights, k=1)[0]
                if stem(sub) != stem(w) and low_pmi(sub, others):
                    return tokens[:i] + [sub] + tokens[i + 1:]
        return None

    rows = []
    # positives
    for t in sents[: args.n_pos]:
        rows.append({"tokens": t, "label": 1, "src": "pos"})
    # corruption negatives
    made = 0
    for t in sents:
        if made >= args.n_corrupt:
            break
        c = corrupt(t)
        if c is not None and norm(" ".join(c)) not in frozen:
            rows.append({"tokens": c, "label": 0, "src": "corrupt"})
            made += 1
    print(f"[disc-data] corruption negatives: {made}", file=sys.stderr)

    # generator-junk negatives — BOTH sources, deduped, excluding frozen:
    #  (a) per-row labeled non-good rows (clean junk), oversampled to give the
    #      sentence-level signal weight against the much larger corruption set;
    #  (b) low-reranker-score rows from all generator CSVs (volume).
    seen_junk = set()
    junk_rows = []
    labels_path = Path(args.gen_labels)
    if labels_path.exists():
        for r in csv.DictReader(open(labels_path)):
            if r.get("label") and r["label"] != "good":
                s = r["sentence"]; k = norm(s)
                if k and k not in frozen and k not in seen_junk:
                    seen_junk.add(k)
                    junk_rows.append(s.lower().split())
    n_labeled = len(junk_rows)
    for path in args.gen_csv:
        p = Path(path)
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            try:
                sc = float(r.get("score") or 0)
            except ValueError:
                sc = 0
            s = r.get("sentence", ""); k = norm(s)
            if s and sc < -1.0 and k not in frozen and k not in seen_junk:
                seen_junk.add(k)
                junk_rows.append(s.lower().split())
    # oversample the clean labeled junk x3 so the sentence-level signal isn't
    # drowned by the ~10x larger corruption set.
    for t in junk_rows:
        rows.append({"tokens": t, "label": 0, "src": "genjunk"})
    for t in junk_rows[:n_labeled]:
        rows.append({"tokens": t, "label": 0, "src": "genjunk"}); rows.append({"tokens": t, "label": 0, "src": "genjunk"})
    print(f"[disc-data] generator-junk negatives: {len(junk_rows)} unique "
          f"({n_labeled} labeled, oversampled) -> {sum(1 for r in rows if r['src']=='genjunk')} rows", file=sys.stderr)

    random.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    npos = sum(1 for r in rows if r["label"] == 1)
    print(f"[disc-data] wrote {args.out}: {len(rows)} rows ({npos} pos / {len(rows)-npos} neg)", file=sys.stderr)


if __name__ == "__main__":
    main()
