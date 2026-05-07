#!/usr/bin/env python3
"""Grammar-safe sentence templates keyed by target POS and morphology.

Each template is a concrete list of tokens that includes the target lemma.
Templates are meant to be tried *before* beam search: if one of them passes
the validators and scores well enough, we ship it. Only if all templates fail
do we fall back to beam-search decoding.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import VERB_PERSON  # noqa: E402


def _noun_templates(lemma: str, fem: bool, plural: bool) -> List[List[str]]:
    if plural:
        if fem:
            return [
                ["tengo", lemma, "nuevas"],
                ["me", "gustan", "las", lemma],
                ["las", lemma, "son", "bonitas"],
                ["no", "tengo", lemma],
                ["veo", "las", lemma],
                ["hay", "muchas", lemma, "aquí"],
                ["necesito", lemma],
            ]
        return [
            ["tengo", lemma, "nuevos"],
            ["me", "gustan", "los", lemma],
            ["los", lemma, "son", "bonitos"],
            ["no", "tengo", lemma],
            ["veo", "los", lemma],
            ["hay", "muchos", lemma, "aquí"],
            ["necesito", lemma],
        ]
    if fem:
        return [
            ["tengo", "una", lemma],
            ["la", lemma, "está", "aquí"],
            ["la", lemma, "es", "bonita"],
            ["no", "tengo", lemma],
            ["esa", "es", "mi", lemma],
            ["me", "gusta", "la", lemma],
            ["veo", "una", lemma],
            ["hay", "una", lemma],
        ]
    return [
        ["tengo", "un", lemma],
        ["el", lemma, "está", "aquí"],
        ["el", lemma, "es", "bueno"],
        ["no", "tengo", lemma],
        ["ese", "es", "mi", lemma],
        ["me", "gusta", "el", lemma],
        ["veo", "un", lemma],
        ["hay", "un", lemma],
    ]


def _adj_templates(lemma: str, fem: bool) -> List[List[str]]:
    if fem:
        return [
            ["ella", "es", lemma],
            ["la", "casa", "es", lemma],
            ["no", "es", lemma],
            ["está", "muy", lemma],
        ]
    return [
        ["él", "es", lemma],
        ["el", "hombre", "es", lemma],
        ["esto", "es", lemma],
        ["no", "es", lemma],
        ["está", "muy", lemma],
    ]


_SPECIAL_VERB_TEMPLATES = {
    # Opinion verbs: always pair with "que + <indicative>" for affirmative and
    # "que + <subjunctive>" for negative. Give safe templates for both polarities.
    "creo": [["creo", "que", "sí"], ["creo", "que", "es", "verdad"], ["no", "lo", "creo"]],
    "crees": [["tú", "crees", "que", "sí"], ["tú", "crees", "en", "él"]],
    "cree": [["ella", "cree", "en", "dios"], ["él", "cree", "que", "es", "cierto"]],
    "creemos": [["creemos", "que", "sí"]],
    "creen": [["ellos", "creen", "en", "ti"]],
    "pienso": [["pienso", "que", "sí"], ["pienso", "mucho", "en", "ti"]],
    "piensa": [["ella", "piensa", "mucho"]],
    "sé": [["no", "lo", "sé"], ["sé", "que", "es", "verdad"], ["yo", "sé", "mucho"]],
    "sabes": [["tú", "sabes", "mucho"], ["tú", "sabes", "la", "verdad"]],
    "sabe": [["ella", "sabe", "la", "verdad"]],
    "gusta": [["me", "gusta", "mucho"], ["no", "le", "gusta", "nada"]],
    "gustan": [["me", "gustan", "las", "flores"]],
    "hay": [["no", "hay", "nadie"], ["hay", "una", "casa"], ["hay", "algo", "aquí"]],
    "dijo": [["ella", "dijo", "la", "verdad"], ["él", "dijo", "que", "sí"]],
    "dije": [["yo", "te", "dije", "la", "verdad"]],
    "hizo": [["él", "lo", "hizo", "bien"]],
    "puede": [["ella", "puede", "venir"], ["no", "puede", "ser"]],
    "pueden": [["ellos", "pueden", "venir"]],
    "quiero": [["yo", "quiero", "ir"], ["no", "quiero", "nada"]],
    "quieres": [["tú", "quieres", "ir"]],
    "quiere": [["ella", "quiere", "venir"]],
    "fue": [["fue", "un", "buen", "día"], ["él", "fue", "a", "casa"]],
    "era": [["era", "un", "buen", "día"], ["era", "muy", "joven"]],
    "eran": [["eran", "buenos", "amigos"]],
    "estaba": [["yo", "estaba", "en", "casa"], ["ella", "estaba", "cansada"]],
    "soy": [["yo", "soy", "estudiante"], ["soy", "de", "aquí"]],
    "eres": [["tú", "eres", "mi", "amigo"]],
    "somos": [["nosotros", "somos", "amigos"]],
    "son": [["ellos", "son", "amigos"], ["son", "las", "tres"]],
    "tengo": [["yo", "tengo", "un", "libro"], ["tengo", "hambre"]],
    "tienes": [["tú", "tienes", "razón"], ["tú", "tienes", "mucho", "tiempo"]],
    "tiene": [["ella", "tiene", "razón"]],
    "voy": [["yo", "voy", "a", "casa"], ["voy", "contigo"]],
    "vas": [["tú", "vas", "a", "casa"]],
    "va": [["ella", "va", "a", "casa"]],
    "vamos": [["nosotros", "vamos", "juntos"]],
    "ha": [["él", "ha", "llegado"], ["ella", "ha", "venido"]],
    "he": [["yo", "he", "llegado"], ["he", "estado", "aquí"]],
    "es": [["es", "muy", "bonito"], ["es", "la", "verdad"]],
    "está": [["ella", "está", "bien"], ["está", "en", "casa"]],
    "están": [["ellos", "están", "aquí"]],
    "estoy": [["yo", "estoy", "bien"], ["estoy", "en", "casa"]],
    "estás": [["tú", "estás", "bien"]],
}


def _verb_templates(lemma: str, verb_person: Optional[str], is_infinitive: bool) -> List[List[str]]:
    if lemma in _SPECIAL_VERB_TEMPLATES:
        return _SPECIAL_VERB_TEMPLATES[lemma]
    if is_infinitive:
        return [
            ["quiero", lemma],
            ["puedo", lemma],
            ["voy", "a", lemma],
            ["necesito", lemma],
            ["tengo", "que", lemma],
            ["no", "puedo", lemma],
            ["me", "gusta", lemma],
        ]
    if verb_person:
        subj = {
            "1s": "yo", "2s": "tú", "3s": "ella", "1p": "nosotros",
            "2p": "vosotros", "3p": "ellos",
        }.get(verb_person, "ella")
        base = [
            [subj, lemma],
            [subj, "no", lemma],
            [subj, lemma, "bien"],
            ["siempre", lemma] if verb_person == "1s" else [subj, "siempre", lemma],
        ]
        return [b for b in base if b]
    return [[lemma]]


def _adv_templates(lemma: str) -> List[List[str]]:
    return [
        ["ella", "está", lemma],
        ["no", "está", lemma],
        ["él", "vive", lemma],
        ["estamos", lemma],
        ["vengo", lemma],
    ]


def _subject_pronoun_templates(lemma: str) -> List[List[str]]:
    verb = {
        "yo": "estoy", "tú": "estás", "él": "está", "ella": "está",
        "usted": "está", "nosotros": "estamos", "nosotras": "estamos",
        "vosotros": "estáis", "vosotras": "estáis",
        "ellos": "están", "ellas": "están", "ustedes": "están",
    }.get(lemma, "está")
    return [
        [lemma, verb, "bien"],
        [lemma, verb, "aquí"],
        [lemma, "no", verb, "bien"],
    ]


def _clitic_templates(lemma: str) -> List[List[str]]:
    if lemma in {"me", "te"}:
        return [
            [lemma, "gusta", "mucho"],
            ["ella", lemma, "ayuda"],
            [lemma, "escucha"],
        ]
    if lemma == "nos":
        return [
            ["ellos", "nos", "ayudan"],
            ["nos", "gusta"],
        ]
    if lemma == "os":
        return [
            ["ellos", "os", "ayudan"],
        ]
    if lemma == "se":
        return [
            ["se", "puede", "hacer"],
            ["no", "se", "sabe"],
        ]
    if lemma in {"lo", "la", "los", "las", "le", "les"}:
        return [
            ["no", lemma, "veo"],
            ["ya", lemma, "tengo"],
            ["ella", lemma, "dice"],
        ]
    return [[lemma, "gusta"]]


def _prep_templates(lemma: str) -> List[List[str]]:
    if lemma == "a":
        return [["voy", "a", "casa"], ["vengo", "a", "verte"], ["ayudo", "a", "mi", "padre"]]
    if lemma == "de":
        return [["soy", "de", "aquí"], ["es", "de", "mi", "padre"], ["vengo", "de", "casa"]]
    if lemma == "en":
        # Keep the complement tight — earlier templates that added "... todo el mundo"
        # produced run-on NPs like "Estoy en casa todo el mundo."
        return [
            ["estoy", "en", "casa"],
            ["vivo", "en", "madrid"],
            ["trabajo", "en", "una", "oficina"],
        ]
    if lemma == "con":
        return [["estoy", "con", "ella"], ["vivo", "con", "mi", "padre"], ["hablo", "con", "él"]]
    if lemma == "por":
        return [["voy", "por", "agua"], ["gracias", "por", "todo"], ["hablamos", "por", "teléfono"]]
    if lemma == "para":
        # Avoid "para que + clause" templates for now — getting the subjunctive right
        # reliably requires morphology we don't have.
        return [
            ["es", "para", "ti"],
            ["trabajo", "para", "ella"],
            ["es", "un", "regalo", "para", "mi", "madre"],
        ]
    if lemma == "sin":
        return [["estoy", "sin", "dinero"], ["vivo", "sin", "ti"]]
    if lemma == "sobre":
        return [["hablamos", "sobre", "el", "tema"], ["es", "sobre", "la", "mesa"]]
    if lemma == "hasta":
        return [["voy", "hasta", "el", "final"], ["hasta", "mañana"]]
    if lemma == "desde":
        return [["vengo", "desde", "casa"], ["vivo", "aquí", "desde", "hace", "años"]]
    if lemma == "entre":
        return [["está", "entre", "nosotros"], ["es", "entre", "amigos"]]
    if lemma == "contra":
        return [["no", "tengo", "nada", "contra", "ti"]]
    if lemma == "hacia":
        return [["voy", "hacia", "casa"], ["camina", "hacia", "la", "puerta"]]
    return [["hablo", lemma, "mi", "padre"], ["estoy", lemma, "casa"]]


def _conj_templates(lemma: str) -> List[List[str]]:
    if lemma == "y":
        # Shorter, fully-complete templates; the previous ones sometimes trailed
        # into "... estamos aquí para ver." which reads as a fragment.
        return [
            ["ella", "y", "yo", "somos", "amigos"],
            ["tú", "y", "yo", "estamos", "bien"],
            ["mi", "padre", "y", "mi", "madre"],
        ]
    if lemma in {"o", "u"}:
        return [
            ["quieres", "ir", "o", "quedarte"],
            ["es", "él", "o", "ella"],
        ]
    if lemma == "pero":
        return [
            ["quiero", "ir", "pero", "no", "puedo", "hoy"],
            ["es", "bueno", "pero", "difícil"],
        ]
    if lemma == "si":
        return [
            ["dime", "si", "puedes", "ir"],
            ["no", "sé", "si", "es", "cierto"],
        ]
    if lemma == "porque":
        return [
            ["estoy", "aquí", "porque", "quiero", "ayudar"],
            ["no", "voy", "porque", "llueve"],
        ]
    if lemma == "cuando":
        return [
            ["vengo", "cuando", "puedo"],
            ["estudio", "cuando", "tengo", "tiempo"],
        ]
    if lemma == "aunque":
        return [
            ["vengo", "aunque", "llueva"],
        ]
    if lemma == "mientras":
        return [
            ["espera", "mientras", "trabajo"],
        ]
    if lemma == "ni":
        return [
            ["no", "tengo", "ni", "idea"],
        ]
    if lemma == "que":
        # Never emit "no creo que es ..." (indicative after negated opinion verb);
        # stick to subjunctive-safe frames.
        return [
            ["creo", "que", "es", "verdad"],
            ["sé", "que", "puedes", "hacerlo"],
            ["espero", "que", "sí"],
        ]
    if lemma == "como":
        return [
            ["trabajo", "como", "maestro"],
            ["es", "como", "tú"],
        ]
    return [["es", "bueno", lemma, "es", "difícil"]]


def _det_num_templates(lemma: str, fem: bool, plural: bool) -> List[List[str]]:
    # "uno" should apocope to "un" before a masc sg noun — avoid that construction.
    if lemma == "uno":
        return [["tengo", "uno"], ["es", "uno"], ["uno", "de", "ellos"]]
    # Possessives get safe, non-human head nouns. "mi hombre" reads as odd in
    # beginner Spanish ("my man"); "mi libro" / "mi casa" are unambiguous.
    POSSESSIVES = {"mi", "tu", "su", "mis", "tus", "sus",
                   "nuestro", "nuestra", "nuestros", "nuestras",
                   "vuestro", "vuestra", "vuestros", "vuestras"}
    # "todos"/"todas" require a determiner between them and the noun
    # ("todos los libros", not "todos libros").
    if lemma in {"todos", "todas", "ambos", "ambas"}:
        if fem or lemma in {"todas", "ambas"}:
            return [["todas", "las", "personas", "son", "buenas"],
                    ["conozco", "a", "todas", "mis", "amigas"]] if lemma in {"todas", "ambas"} else \
                   [["todos", "los", "libros", "son", "buenos"],
                    ["conozco", "a", "todos", "mis", "amigos"]]
        return [["todos", "los", "libros", "son", "buenos"],
                ["conozco", "a", "todos", "mis", "amigos"]]
    if lemma in POSSESSIVES:
        if plural:
            noun = "hermanas" if fem else "libros"
            verb = "son"
            adj = "buenas" if fem else "buenos"
        else:
            noun = "casa" if fem else "libro"
            verb = "es"
            adj = "bonita" if fem else "bueno"
        return [
            [lemma, noun, verb, adj],
            ["veo", lemma, noun],
            ["tengo", lemma, noun],
        ]
    # Generic determiners / numerals — use "libro" / "casa" / "libros" / "casas"
    # as safe head nouns so the sentence is semantically plausible.
    if plural:
        noun = "casas" if fem else "libros"
        verb = "son"
        adj = "bonitas" if fem else "buenos"
        ending_adj = "nuevas" if fem else "nuevos"
    else:
        noun = "casa" if fem else "libro"
        verb = "es"
        adj = "bonita" if fem else "bueno"
        ending_adj = "nueva" if fem else "nuevo"
    return [
        [lemma, noun, verb, adj],
        ["tengo", lemma, noun],
        [lemma, noun, ending_adj],
    ]


def _special_noun_templates(lemma: str) -> Optional[List[List[str]]]:
    special = {
        "gracias": [["muchas", "gracias"], ["gracias", "por", "todo"]],
        "tiempo": [["no", "tengo", "tiempo"], ["el", "tiempo", "vuela"]],
        "vez": [["una", "vez", "más"], ["otra", "vez", "aquí"]],
        "verdad": [["es", "verdad"], ["dime", "la", "verdad"]],
        "hay": [["hay", "una", "casa"], ["no", "hay", "nadie"], ["hay", "algo", "aquí"]],
        "mamá": [["mi", "mamá", "está", "aquí"], ["amo", "a", "mi", "mamá"]],
        "papá": [["mi", "papá", "está", "aquí"], ["amo", "a", "mi", "papá"]],
        # contractions can appear mid-sentence but never at the start
        "al": [["voy", "al", "mercado"], ["vamos", "al", "cine"], ["llego", "al", "final"]],
        "del": [["vengo", "del", "trabajo"], ["es", "la", "casa", "del", "vecino"], ["hablamos", "del", "problema"]],
    }
    return special.get(lemma)


def grammar_safe_templates(
    target_lemma: str,
    pos: str,
    morph_for_lemma: Optional[Dict[str, str]] = None,
    is_infinitive: bool = False,
    verb_person: Optional[str] = None,
) -> List[List[str]]:
    """Return a list of complete, grammatically-safe templates that contain the target.

    The caller is expected to validate and score each template, then pick the
    best. Templates that fail validation or score poorly should be discarded;
    the caller can then fall back to beam-search decoding.
    """
    pos = (pos or "").lower()
    morph = morph_for_lemma or {}
    fem = morph.get("Gender") == "Fem"
    plural = morph.get("Number") == "Plur"

    special = _special_noun_templates(target_lemma)
    if special is not None:
        return special

    if pos == "n":
        return _noun_templates(target_lemma, fem, plural)
    if pos in {"adj", "adjective"}:
        return _adj_templates(target_lemma, fem)
    if pos in {"v", "verb"}:
        if not verb_person and target_lemma in VERB_PERSON:
            verb_person = VERB_PERSON[target_lemma]
        is_inf = is_infinitive or (
            target_lemma.endswith(("ar", "er", "ir"))
            and target_lemma not in VERB_PERSON
        )
        return _verb_templates(target_lemma, verb_person, is_inf)
    if pos in {"adv", "adverb"}:
        return _adv_templates(target_lemma)
    if pos == "pron":
        if target_lemma in {"me", "te", "se", "lo", "la", "los", "las", "le", "les", "nos", "os"}:
            return _clitic_templates(target_lemma)
        return _subject_pronoun_templates(target_lemma)
    if pos in {"prep", "preposition"}:
        return _prep_templates(target_lemma)
    if pos in {"conj", "conjunction"}:
        return _conj_templates(target_lemma)
    if pos in {"art", "determiner", "num"}:
        return _det_num_templates(target_lemma, fem, plural)
    return [[target_lemma]]
