#!/usr/bin/env python3
"""Regression tests for the corpus-derived coherence signal (collocations.py).

Requires data_clean/collocations.pkl (build it with build_collocations.py). If
the table is absent the test SKIPS rather than fails, since the .pkl is too big
to ship in the repo (same policy as the UD morph table).

Checks:
  1. Known good collocations out-score known nonsense (re-ranking direction).
  2. The hard gate fires on clear verb->object selectional violations.
  3. The hard gate does NOT fire on a curated set of natural sentences
     (precision floor — at most a couple of marginal sentences allowed).

Run: python scripts/test_collocations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import tokenize  # noqa: E402
from collocations import load_collocations  # noqa: E402

TABLE = Path("data_clean/collocations.pkl")

# (nonsense, good) — the nonsense must score strictly lower than the good one.
RANKING_PAIRS = [
    ("Ella puede hablar un poco de agua", "Ella puede beber un poco de agua"),
    ("Las casas son agua", "Las casas son bonitas"),
    ("Quiero hablar un poco de agua", "Quiero beber un poco de agua"),
]

# Sentences whose verb->object pairing is genuinely incoherent: gate must fire.
GATE_SHOULD_FIRE = [
    "Ella puede hablar un poco de agua",
    "Quiero hablar un poco de agua",
]

# Natural sentences the gate must (almost all) leave alone.
GATE_SHOULD_PASS = [
    "Bebo un poco de agua",
    "Quiero comer una manzana",
    "La casa es grande",
    "Mi madre prepara la cena",
    "Leo un libro interesante",
    "Tengo mucho trabajo hoy",
    "Las casas son bonitas",
    "Mis amigos son buenos",
    "La canción es bonita",
    "El día es largo",
    "Ayudo a la gente",
    "Quiero ayudar a la gente",
    "El agua está fría",
]
MAX_GATE_FALSE_POSITIVES = 2  # marginal sentences tolerated (penalty, not reject)


def main() -> int:
    model = load_collocations(TABLE)
    if model is None:
        print(f"SKIP: {TABLE} not found — run scripts/build_collocations.py first.")
        return 0

    failures = 0

    print("\n-- ranking (nonsense must score below good) --")
    for nonsense, good in RANKING_PAIRS:
        sn = model.coherence_score(tokenize(nonsense))
        sg = model.coherence_score(tokenize(good))
        # Combine the soft score with the flat gate penalty the generator applies,
        # so the test mirrors real selection.
        pn = sn - (1.5 if model.incoherent_reason(tokenize(nonsense)) else 0)
        pg = sg - (1.5 if model.incoherent_reason(tokenize(good)) else 0)
        if pn < pg:
            print(f"  ok   {pn:+.2f} < {pg:+.2f}   [{nonsense}] < [{good}]")
        else:
            print(f"  FAIL {pn:+.2f} !< {pg:+.2f}  [{nonsense}] vs [{good}]")
            failures += 1

    print("\n-- gate should FIRE (verb->object nonsense) --")
    for s in GATE_SHOULD_FIRE:
        if model.incoherent_reason(tokenize(s)):
            print(f"  ok   gated: {s}")
        else:
            print(f"  FAIL not gated: {s}")
            failures += 1

    print("\n-- gate should PASS (natural sentences) --")
    fps = [s for s in GATE_SHOULD_PASS if model.incoherent_reason(tokenize(s))]
    for s in GATE_SHOULD_PASS:
        mark = "FP" if s in fps else "ok"
        print(f"  {mark}   {s}")
    if len(fps) > MAX_GATE_FALSE_POSITIVES:
        print(f"  FAIL too many gate false-positives: {len(fps)} > {MAX_GATE_FALSE_POSITIVES}")
        failures += len(fps) - MAX_GATE_FALSE_POSITIVES

    if failures:
        print(f"\n{failures} failure(s).")
        return 1
    print("\nall ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
