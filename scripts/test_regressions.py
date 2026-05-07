#!/usr/bin/env python3
"""Regression tests for known-bad generator outputs.

Every sentence in BAD_EXAMPLES must produce at least one rejection reason
from validate(). Every sentence in GOOD_EXAMPLES must produce zero rejection
reasons. Run with `python scripts/test_regressions.py`; exit code is non-zero
on any failure.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import tokenize  # noqa: E402
from validators import validate  # noqa: E402


# Each entry: (sentence, expected_reason_substring_or_None_to_accept_any_failure)
BAD_EXAMPLES = [
    ("Creo que es verdad que estás haciendo", "gerund_final"),
    ("Del en la casa de mi padre", "contraction_start"),
    ("No sé si es verdad fue", "dangling_final_verb"),
    ("Ella es mí no me gusta mucho", "subj_copula_prep_pronoun"),
    ("Uno hombre es un tipo de cosas", "uno_before_noun"),
    ("Voy a trabajar en un lugar donde", "relative_final"),
    ("Ido a la gente no le gusta", "lone_participle_start"),
    ("Mí es importante para los niños", "prep_pronoun_as_subject"),
    # Second-wave bad examples from the 100-target CSV:
    ("Estoy en casa todo el mundo", "todo_el_mundo_after_np"),
    ("Voy para que no se puede hacer", "para_que_indicative"),
    ("Claro que sí misma no es nada", "si_misma_without_preposition"),
    ("Tengo mi hombre con quien hablar", "possessive_unsafe_noun"),
    ("Ella es algo que ver con esto", "ser_algo_que_ver"),
    ("No estoy seguro de que la gente", "incomplete_de_que"),
    ("No era el único que puede hacer", "bare_modal_inf_final"),
    ("Una vez más personas que no puedo", "bare_modal_final"),
    ("Todos libros son para los niños", "quantifier_without_article"),
    ("No creo que es hora de hacer", "no_creo_que_indicative"),
]

GOOD_EXAMPLES = [
    "Voy a casa de mi padre",
    "No hay nadie aquí",
    "Me gusta mucho el café",
    "Ella está en casa",
    "Los libros están en el mundo",
    "Creo que es verdad",
    "Quiero ir al cine",
    "Tengo una casa bonita",
    "No sé qué hacer",
    "Es bueno para ti",
    "Voy a trabajar en un lugar bonito",
    "Dime si puedes ir",
    # Guards against the second-wave rules
    "Todos los libros son buenos",
    "Todos mis amigos están aquí",
    "Es bueno para todo el mundo",
    "Para que sea feliz",
    "Lo hizo por sí misma",
    "Tengo un amigo en casa",
    "Mi padre está aquí",
    "Tiene algo que ver con eso",
    "No creo que sea verdad",
    "Creo que es verdad",
    "No sé si puede venir",
]


def _load_morph(path: Path):
    if not path.exists():
        return {}
    with path.open("rb") as f:
        lf = pickle.load(f)
    out = {}
    for _, forms in lf.items():
        for entry in forms:
            form = entry.get("form")
            morph = entry.get("morph") or {}
            if form and morph:
                out.setdefault(form.lower(), morph)
    return out


def main() -> int:
    morph = _load_morph(Path("models_rebuild2/lemma_forms.pkl"))
    failures = 0

    print("\n-- BAD examples (must fail validation) --")
    for sent, must_contain in BAD_EXAMPLES:
        toks = tokenize(sent)
        reasons = validate(toks, morph=morph)
        if not reasons:
            print(f"  FAIL: expected rejection but got none: {sent!r}")
            failures += 1
            continue
        if must_contain and must_contain not in reasons:
            print(
                f"  FAIL: expected reason '{must_contain}' for {sent!r}, got {reasons}"
            )
            failures += 1
            continue
        print(f"  ok   [{','.join(reasons)}] {sent}")

    print("\n-- GOOD examples (must pass validation) --")
    for sent in GOOD_EXAMPLES:
        toks = tokenize(sent)
        reasons = validate(toks, morph=morph)
        if reasons:
            print(f"  FAIL: unexpected rejection {reasons}: {sent!r}")
            failures += 1
            continue
        print(f"  ok   {sent}")

    if failures:
        print(f"\n{failures} failure(s).")
        return 1
    print("\nall ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
