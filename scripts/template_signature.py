#!/usr/bin/env python3
"""Compute a function-word skeleton for a sentence.

Two sentences that share the same skeleton (after collapsing articles &
demonstratives to <D> and content words to *) are considered the same
template. The generator uses this signature to cap how many outputs in a
single CSV can share the same template, preventing outputs like
"X está a punto de salir." or "Tengo X libros en mi vida." from dominating.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    ARTICLES, OBJECT_CLITICS, PREPOSITIONS, SUBJECT_PRONOUNS, VERB_PERSON,
)
from validators import (  # noqa: E402
    COORDINATORS, CONTRACTIONS, DEM_POSS_FEM_PL, DEM_POSS_FEM_SG,
    DEM_POSS_MASC_PL, DEM_POSS_MASC_SG, PREP_PRONOUNS, RELATIVE_INTERROGATIVE,
)

DET_LIKE = ARTICLES | DEM_POSS_FEM_PL | DEM_POSS_FEM_SG | DEM_POSS_MASC_PL | DEM_POSS_MASC_SG

FUNCTION_LITERALS = (
    PREPOSITIONS
    | COORDINATORS
    | RELATIVE_INTERROGATIVE
    | OBJECT_CLITICS
    | PREP_PRONOUNS
    | SUBJECT_PRONOUNS
    | CONTRACTIONS
    | set(VERB_PERSON)
    | {
        "no", "nunca", "nada", "nadie", "tampoco", "ni", "jamás",
        "muy", "más", "menos", "tan", "también", "ya", "aún", "solo", "sólo",
        "aquí", "allí", "ahí", "allá", "acá",
        "así", "bien", "mal",
        "qué", "cómo", "cuándo", "dónde", "quién", "cuál",
    }
)


def template_signature(tokens: Sequence[str]) -> Tuple[str, ...]:
    out = []
    for t in tokens:
        tl = t.lower()
        if tl in DET_LIKE:
            out.append("<D>")
        elif tl in FUNCTION_LITERALS:
            out.append(tl)
        else:
            out.append("*")
    return tuple(out)


def skeleton(tokens: Sequence[str], max_len: int = 7) -> Tuple[str, ...]:
    """Short prefix-based fingerprint; useful for detecting "X está a punto de salir"
    style templates that differ only in the subject slot."""
    sig = template_signature(tokens)
    return sig[:max_len]
