#!/usr/bin/env python3
"""Build a high-precision gender/number table from corpus article-noun co-occurrence.

The original UD morph table (models_rebuild2/lemma_forms.pkl) is absent, so the
generator falls back to the -a/-o spelling heuristic, which gets -e/-es/-re nouns
wrong (sangre, noche, gente, mujeres -> defaulted masculine -> "El sangre").

This derives gender/number purely from how each word is determined in the clean
corpus: a noun preceded by la/las/una/unas votes feminine, by el/los/un votes
masculine, etc. Majority vote with a confidence margin -> only emit confident
entries (high precision; unknowns fall back to the heuristic, which is fine).

The "el agua" trap: feminine nouns starting with a stressed /a/ take el/un in the
singular ("el agua", "un hacha"). So for a-/ha-initial nouns we DROP the
masculine-singular markers (el/un/del/al) and rely on las/unas/esta(s)/esa(s),
which still mark them feminine ("las aguas", "esta agua").

Output: data_clean/gender_table.pkl = {form: {"Gender": "Fem"|"Masc",
"Number": "Plur"|"Sing"}} (each axis present only when confident).
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# determiner -> (gender, number); gender/number None when the determiner doesn't mark it
MASC_SG = {"el", "un", "del", "al", "este", "ese", "aquel"}
MASC_PL = {"los", "unos", "estos", "esos", "aquellos"}
FEM_SG = {"la", "una", "esta", "esa", "aquella"}
FEM_PL = {"las", "unas", "estas", "esas", "aquellas"}
# masc-singular markers that are unreliable before a stressed-/a/ feminine noun
MASC_SG_AINITIAL_DROP = {"el", "un", "del", "al"}

ALL_DET = MASC_SG | MASC_PL | FEM_SG | FEM_PL


def a_initial(w: str) -> bool:
    return w[:1] == "a" or w[:2] == "ha"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data_clean/good_corpus.txt")
    ap.add_argument("--out", default="data_clean/gender_table.pkl")
    ap.add_argument("--min-votes", type=int, default=4)
    ap.add_argument("--gender-margin", type=float, default=0.80)
    ap.add_argument("--number-margin", type=float, default=0.85)
    args = ap.parse_args()

    gen_votes = defaultdict(lambda: [0, 0])  # word -> [masc, fem]
    num_votes = defaultdict(lambda: [0, 0])  # word -> [sing, plur]

    with Path(args.corpus).open(encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            for i in range(1, len(toks)):
                det, w = toks[i - 1], toks[i]
                if det not in ALL_DET or not w.isalpha() or len(w) < 3:
                    continue
                # gender
                if det in MASC_SG | MASC_PL:
                    if not (a_initial(w) and det in MASC_SG_AINITIAL_DROP):
                        gen_votes[w][0] += 1
                elif det in FEM_SG | FEM_PL:
                    gen_votes[w][1] += 1
                # number
                if det in MASC_SG | FEM_SG:
                    if not (a_initial(w) and det in MASC_SG_AINITIAL_DROP):
                        num_votes[w][0] += 1
                elif det in MASC_PL | FEM_PL:
                    num_votes[w][1] += 1

    table = {}
    words = set(gen_votes) | set(num_votes)
    for w in words:
        entry = {}
        m, fem = gen_votes[w]
        tot = m + fem
        if tot >= args.min_votes:
            if fem / tot >= args.gender_margin:
                entry["Gender"] = "Fem"
            elif m / tot >= args.gender_margin:
                entry["Gender"] = "Masc"
        s, pl = num_votes[w]
        tot = s + pl
        if tot >= args.min_votes:
            if pl / tot >= args.number_margin:
                entry["Number"] = "Plur"
            elif s / tot >= args.number_margin:
                entry["Number"] = "Sing"
        if entry:
            table[w] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("wb") as f:
        pickle.dump(table, f, protocol=4)

    n_fem = sum(1 for e in table.values() if e.get("Gender") == "Fem")
    n_masc = sum(1 for e in table.values() if e.get("Gender") == "Masc")
    print(f"[gender] {len(table):,} confident forms  ({n_fem:,} fem / {n_masc:,} masc)", file=sys.stderr)
    for probe in ["sangre", "noche", "noches", "gente", "mujer", "mujeres", "agua", "aguas",
                  "libro", "casa", "día", "mano", "problema", "vino"]:
        print(f"    {probe:<10} {table.get(probe, '—')}", file=sys.stderr)


if __name__ == "__main__":
    main()
