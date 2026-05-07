#!/usr/bin/env python3
"""Build a clean training corpus for the high-quality Markov generator.

Sources:
  - outputs/sentances_review.csv: rows with grammatical_ok=natural_ok=learner_clear_ok=1
  - spa_sentences.tsv: Tatoeba Spanish sentences (learner-oriented, short, correct)

Outputs:
  - data_clean/good_corpus.txt : one sentence per line, tokenized (lowercase)
  - data_clean/good_corpus.pos.txt : corresponding POS tags
"""
import argparse
import csv
import re
import sys
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9¿¡]+|[.,;:!?()]")


def tokenize(text: str):
    if not text:
        return []
    return [t.lower() for t in WORD_RE.findall(text)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="outputs/sentances_review.csv")
    ap.add_argument("--tatoeba", default="spa_sentences.tsv")
    ap.add_argument("--out", default="data_clean/good_corpus.txt")
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=15)
    ap.add_argument("--max-tatoeba", type=int, default=400000)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        n_good = 0
        with open(args.review, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("grammatical_ok") == "1" and row.get("natural_ok") == "1" and row.get("learner_clear_ok") == "1":
                    toks = tokenize(row.get("sentence", ""))
                    if args.min_len <= len(toks) <= args.max_len:
                        out.write(" ".join(toks) + "\n")
                        n_good += 1
                        written += 1
        n_tat = 0
        with open(args.tatoeba, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                text = parts[2]
                toks = tokenize(text)
                if args.min_len <= len(toks) <= args.max_len:
                    out.write(" ".join(toks) + "\n")
                    n_tat += 1
                    written += 1
                    if n_tat >= args.max_tatoeba:
                        break

    print(f"reviewed-good written: {n_good}", file=sys.stderr)
    print(f"tatoeba written:       {n_tat}", file=sys.stderr)
    print(f"total lines written:   {written}", file=sys.stderr)


if __name__ == "__main__":
    main()
