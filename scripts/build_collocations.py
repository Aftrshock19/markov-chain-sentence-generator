#!/usr/bin/env python3
"""Build a corpus-derived collocation / selectional-preference table.

This is the no-LLM, count-based coherence signal the diagnostic flagged as the
only principled attack on "fluent nonsense" (grammatical but meaningless output
like "Ella puede hablar un poco de agua." or "Las casas son un poco de agua.").

We make a single pass over the same corpus that trains the KN LM
(Tatoeba Spanish + human-reviewed-good rows) and count:

  * unigram   : content-word frequencies
  * window    : unordered content-word co-occurrence within a small token window
                (catches verb->object violations, e.g. hablar + agua)
  * copula    : (left content word, right content noun) around a copula
                (catches subject<->predicate violations, e.g. casas + agua in
                "las casas son agua")

PMI is computed at query time from these counts so the threshold stays tunable.
The result is saved to data_clean/collocations.pkl.

Usage:
    python scripts/build_collocations.py \
        --review outputs/sentances_review.csv \
        --tatoeba spa_sentences.tsv \
        --out data_clean/collocations.pkl
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    ARTICLES, CONTRACTIONS_PREP, OBJECT_CLITICS, PREPOSITIONS,
    SUBJECT_PRONOUNS, VERB_PERSON,
)

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")

# Copula forms that link a subject to a predicate nominal.
COPULAS = {
    "es", "son", "está", "están", "era", "eran", "fue", "fueron",
    "soy", "eres", "somos", "sois", "estoy", "estás", "estamos", "estáis",
    "será", "serán", "sería", "serían",
}

# Function words that are never "content" for collocation purposes.
FUNCTION_WORDS = (
    ARTICLES | PREPOSITIONS | CONTRACTIONS_PREP | OBJECT_CLITICS
    | SUBJECT_PRONOUNS | set(VERB_PERSON) | COPULAS
    | {
        "y", "e", "o", "u", "ni", "que", "si", "no", "se", "lo", "le",
        "como", "más", "menos", "muy", "tan", "ya", "también", "tampoco",
        "pero", "porque", "cuando", "donde", "aunque", "mientras",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "mi", "tu", "su", "mis", "tus", "sus", "del", "al", "un", "una",
        "uno", "unos", "unas", "hay", "ha", "han", "he", "has", "hemos",
        "a", "de", "en", "con", "por", "para", "sin",
    }
)

# Verb-ish forms still count as content for the window relation (we WANT
# "hablar"/"habla" paired with their objects), so don't drop them here.


def is_content(w: str) -> bool:
    return len(w) >= 3 and w not in FUNCTION_WORDS


def tokenize(text: str):
    return [t.lower() for t in WORD_RE.findall(text)]


def iter_corpus(review_path: Path, tatoeba_path: Path, max_tatoeba: int,
                min_len: int, max_len: int):
    """Yield token lists from the same sources build_clean_corpus.py uses."""
    n_good = 0
    if review_path.exists():
        with review_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("grammatical_ok") == "1"
                        and row.get("natural_ok") == "1"
                        and row.get("learner_clear_ok") == "1"):
                    toks = tokenize(row.get("sentence", ""))
                    if min_len <= len(toks) <= max_len:
                        n_good += 1
                        yield toks
    print(f"reviewed-good sentences: {n_good}", file=sys.stderr)
    n_tat = 0
    with tatoeba_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            toks = tokenize(parts[2])
            if min_len <= len(toks) <= max_len:
                n_tat += 1
                yield toks
                if n_tat >= max_tatoeba:
                    break
    print(f"tatoeba sentences:       {n_tat}", file=sys.stderr)


def build(review_path: Path, tatoeba_path: Path, window: int,
          max_tatoeba: int, min_len: int, max_len: int):
    uni: Counter = Counter()
    window_pairs: Counter = Counter()
    copula_pairs: Counter = Counter()
    n_sent = 0

    for toks in iter_corpus(review_path, tatoeba_path, max_tatoeba, min_len, max_len):
        n_sent += 1
        # Content tokens with their positions (for windowing) + unigram counts.
        content = [(i, w) for i, w in enumerate(toks) if is_content(w)]
        for _, w in content:
            uni[w] += 1
        # Windowed content co-occurrence (unordered, dedup per sentence so a
        # repeated word doesn't inflate the pair count).
        seen_pairs = set()
        for a in range(len(content)):
            ia, wa = content[a]
            for b in range(a + 1, len(content)):
                ib, wb = content[b]
                if ib - ia > window:
                    break
                key = (wa, wb) if wa < wb else (wb, wa)
                if key not in seen_pairs and key[0] != key[1]:
                    seen_pairs.add(key)
                    window_pairs[key] += 1
        # Copula relation: nearest content word left of a copula <-> nearest
        # content word right of it.
        for i, w in enumerate(toks):
            if w in COPULAS:
                left = next((toks[j] for j in range(i - 1, -1, -1) if is_content(toks[j])), None)
                right = next((toks[j] for j in range(i + 1, len(toks)) if is_content(toks[j])), None)
                if left and right and left != right:
                    key = (left, right) if left < right else (right, left)
                    copula_pairs[key] += 1

    print(f"total sentences:         {n_sent}", file=sys.stderr)
    print(f"content vocab (raw):     {len(uni)}", file=sys.stderr)
    print(f"window pairs (raw):      {len(window_pairs)}", file=sys.stderr)
    print(f"copula pairs (raw):      {len(copula_pairs)}", file=sys.stderr)
    return n_sent, uni, window_pairs, copula_pairs


def prune(uni: Counter, window_pairs: Counter, copula_pairs: Counter,
          min_uni: int, min_pair: int):
    uni_p = {w: c for w, c in uni.items() if c >= min_uni}
    win_p = {
        k: c for k, c in window_pairs.items()
        if c >= min_pair and k[0] in uni_p and k[1] in uni_p
    }
    cop_p = {
        k: c for k, c in copula_pairs.items()
        if c >= min_pair and k[0] in uni_p and k[1] in uni_p
    }
    return uni_p, win_p, cop_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="outputs/sentances_review.csv")
    ap.add_argument("--tatoeba", default="spa_sentences.tsv")
    ap.add_argument("--out", default="data_clean/collocations.pkl")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--max-tatoeba", type=int, default=400000)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--min-uni", type=int, default=20,
                    help="Drop content words rarer than this (PMI on rare words is noise).")
    ap.add_argument("--min-pair", type=int, default=2,
                    help="Drop pairs seen fewer than this many times.")
    args = ap.parse_args()

    n_sent, uni, win, cop = build(
        Path(args.review), Path(args.tatoeba),
        args.window, args.max_tatoeba, args.min_len, args.max_len,
    )
    uni_p, win_p, cop_p = prune(uni, win, cop, args.min_uni, args.min_pair)
    print(f"content vocab (kept):    {len(uni_p)}", file=sys.stderr)
    print(f"window pairs (kept):     {len(win_p)}", file=sys.stderr)
    print(f"copula pairs (kept):     {len(cop_p)}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "n_sent": n_sent,
        "window": args.window,
        "min_uni": args.min_uni,
        "unigram": uni_p,
        "window_pairs": win_p,
        "copula_pairs": cop_p,
        "total_content_tokens": sum(uni.values()),
    }
    with out_path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
