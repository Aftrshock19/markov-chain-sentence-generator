#!/usr/bin/env python3
"""Hard rejection validators for generated Spanish sentences.

validate(tokens, target_lemma) returns a list of reason-name strings. Any
non-empty list means the candidate must be rejected. Reasons are stable
identifiers suitable for regression tests.

Designed to reject the following known failure modes from the ML generator:

  1. Gerund at sentence start or end ("Creo que es verdad que estás haciendo")
  2. Past participle (ido, hecho, ...) at sentence start with no auxiliary
  3. Relative/interrogative pronoun at sentence end ("... un lugar donde")
  4. Contraction (al/del) at sentence start, or double-preposition start
     ("Del en la casa de mi padre")
  5. <subject-pronoun> <copula> <prepositional-pronoun>, e.g. "ella es mí"
  6. "uno" immediately before a masculine singular noun ("uno hombre ...")
  7. Dangling finite verb at the end not preceded by a coordinator/negation
     ("No sé si es verdad fue")
  8. Run-on of two independent clauses without a coordinator
     ("Ella es mí no me gusta mucho")
  9. Article / demonstrative / possessive + noun gender or number mismatch
     (uses UD morph when available, heuristic otherwise)
 10. Ending in a BAD_FINAL token that isn't part of an allowed tail
     (delegates to the list in features.py)
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    ARTICLES, BAD_FINAL, FEMININE_ART, FINITE_VERBS, MASCULINE_ART,
    OBJECT_CLITICS, PLURAL_ART, PREPOSITIONS, SINGULAR_ART,
    SUBJECT_PRONOUNS, VERB_PERSON,
)

PAST_PARTICIPLES = {
    "ido", "sido", "estado", "hecho", "dicho", "visto", "puesto",
    "dado", "muerto", "roto", "abierto", "escrito", "vuelto",
    "cubierto", "resuelto", "impreso", "suelto", "frito",
}

RELATIVE_INTERROGATIVE = {
    "donde", "cuando", "como", "quien", "cual", "cuales", "quienes",
    "cuyo", "cuya", "cuyos", "cuyas", "cuánto", "cuánta", "cuántos", "cuántas",
    "cuál", "cuáles", "cómo", "dónde", "cuándo", "qué",
}

CONTRACTIONS = {"al", "del"}

PREP_PRONOUNS = {"mí", "ti", "sí", "conmigo", "contigo", "consigo"}

# Modal finite verbs that always want an infinitive complement (never stand alone in
# a subordinate clause). They're fine as top-level isolated responses ("Sí, puedo.")
# but inside a larger structure ending here means the complement is missing.
MODAL_FINITE = {
    "puedo", "puedes", "puede", "podemos", "podéis", "pueden",
    "quiero", "quieres", "quiere", "queremos", "queréis", "quieren",
    "debo", "debes", "debe", "debemos", "debéis", "deben",
    "suelo", "sueles", "suele", "solemos", "suelen",
    "necesito", "necesitas", "necesita", "necesitamos", "necesitan",
}

# Bare common TRANSITIVE infinitives: when they appear as the very last token after
# a modal ("puede hacer.", "debe decir.") the object is missing and the clause is
# incomplete. Intransitive infinitives (ir, venir, salir, llegar, volver) are
# intentionally excluded — "Dime si puedes ir." and "No sé si puede venir." are
# fine because those verbs don't take an object.
BARE_INFINITIVES = {
    "hacer", "decir", "tener", "saber",
    "ver", "leer", "escribir",
    "dar", "poner", "querer",
    "comer", "beber", "tomar", "pedir", "buscar", "coger",
    "escuchar", "recordar", "olvidar",
}

# Finite verbs in the indicative that must NOT follow "para que" (which requires subjunctive)
# or "no creo que" / "no pienso que" / "dudo que" (also subjunctive-taking).
INDICATIVE_AFTER_SUBJUNCTIVE_TRIGGER = {
    "es", "está", "son", "están", "soy", "estoy", "eres", "estás",
    "tengo", "tienes", "tiene", "tenemos", "tienen",
    "voy", "vas", "va", "vamos", "van",
    "puedo", "puedes", "puede", "podemos", "pueden",
    "quiero", "quieres", "quiere", "queremos", "quieren",
    "sé", "sabes", "sabe", "sabemos", "saben",
    "digo", "dices", "dice", "decimos", "dicen",
    "hago", "haces", "hace", "hacemos", "hacen",
    "veo", "ves", "ve", "vemos", "ven",
    "creo", "crees", "cree", "creemos", "creen",
    "vengo", "vienes", "viene", "venimos", "vienen",
    "hay",
}

# Possessive determiners.
POSSESSIVES = {
    "mi", "tu", "su", "mis", "tus", "sus",
    "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras",
}

# Human-ish nouns that make bare possessive templates awkward for beginners
# ("mi hombre" = *my man*, weird; "mi persona" almost never used like that).
UNSAFE_POSSESSIVE_HEAD_NOUNS = {"hombre", "mujer", "persona", "gente", "tipo"}

# Plural quantifiers that, in standard Spanish, almost always need a determiner between
# them and the noun ("todos los libros", "todas mis cosas").
# Unlike "muchos/pocos" which accept bare NPs.
QUANTIFIERS_NEEDING_DETERMINER = {"todos", "todas", "ambos", "ambas"}

# Words that legitimately tie clauses / verbs together.
COORDINATORS = {
    "y", "e", "o", "u", "ni", "pero", "sino", "mas",
    "que", "porque", "aunque", "cuando", "si", "mientras",
    "como", "donde", "quien", "cual",
    "también", "tampoco", "luego", "entonces", "por", "para",
}

# Tokens that may precede a past participle grammatically (aux/copula + clitics).
AUX_PRECEDERS = {
    "he", "has", "ha", "hemos", "han",
    "había", "habías", "habíamos", "habían",
    "haya", "hayas", "hayamos", "hayan",
    "hube", "hubo", "hubimos", "hubieron",
    "habré", "habrás", "habrá", "habremos", "habrán",
    "habría", "habrías", "habríamos", "habrían",
    "soy", "eres", "es", "somos", "son",
    "estoy", "estás", "está", "estamos", "están",
    "fue", "fui", "fuiste", "fuimos", "fueron",
    "era", "eras", "éramos", "eran",
    "estaba", "estabas", "estábamos", "estaban",
    "ser", "estar", "haber",
    "sido", "estado", "sea", "seas", "sean",  # compound "haber sido ..."
}

# Demonstratives/possessives that share gender-number agreement with articles.
DEM_POSS_FEM_PL = {"estas", "esas", "aquellas", "mías", "tuyas", "suyas", "nuestras", "vuestras"}
DEM_POSS_FEM_SG = {"esta", "esa", "aquella", "mía", "tuya", "suya", "nuestra", "vuestra",
                   "toda", "alguna", "ninguna", "otra", "cierta", "cualquiera"}
DEM_POSS_MASC_PL = {"estos", "esos", "aquellos", "míos", "tuyos", "suyos", "nuestros", "vuestros",
                    "todos", "algunos", "ningunos", "otros"}
DEM_POSS_MASC_SG = {"este", "ese", "aquel", "mío", "tuyo", "suyo", "nuestro", "vuestro",
                    "todo", "algún", "alguno", "ningún", "ninguno", "otro", "cierto", "cualquier"}

ARTICLE_LIKE_FEM = (FEMININE_ART | DEM_POSS_FEM_PL | DEM_POSS_FEM_SG)
ARTICLE_LIKE_MASC = (MASCULINE_ART | DEM_POSS_MASC_PL | DEM_POSS_MASC_SG)
ARTICLE_LIKE_PL = (PLURAL_ART | DEM_POSS_FEM_PL | DEM_POSS_MASC_PL)
ARTICLE_LIKE_SG = (SINGULAR_ART | DEM_POSS_FEM_SG | DEM_POSS_MASC_SG)
ARTICLE_LIKE_ALL = ARTICLE_LIKE_FEM | ARTICLE_LIKE_MASC


def _is_gerund(w: str) -> bool:
    return len(w) >= 5 and (w.endswith("ando") or w.endswith("iendo") or w.endswith("yendo"))


def _heuristic_fem_noun(w: str) -> bool:
    return (
        (w.endswith("a") and not w.endswith(("ma", "pa", "ta")))
        or w.endswith(("dad", "tad", "tud", "ción", "sión", "umbre", "ez"))
    )


def _heuristic_plural_noun(w: str) -> bool:
    return w.endswith("s") and len(w) > 3 and not w.endswith(("és", "ís", "ús", "ás", "os"))


def _agreement_mismatch(prev: str, nxt: str, morph: Optional[Dict[str, Dict[str, str]]]) -> Optional[str]:
    """Return a reason name if (prev, nxt) is a hard article/demonstrative/noun agreement violation."""
    if prev not in ARTICLE_LIKE_ALL:
        return None
    # Skip when nxt is itself a function word (another det, prep, verb, ...) — those are not nouns.
    if nxt in ARTICLE_LIKE_ALL or nxt in PREPOSITIONS or nxt in FINITE_VERBS or nxt in VERB_PERSON:
        return None
    # Resolve gender and number.
    if morph is not None and nxt in morph:
        g = morph[nxt].get("Gender")
        n = morph[nxt].get("Number")
        fem = (g == "Fem") if g else None
        plur = (n == "Plur") if n else None
    else:
        fem = None
        plur = None
    if fem is None:
        fem = _heuristic_fem_noun(nxt)
    if plur is None:
        plur = _heuristic_plural_noun(nxt)
    prev_fem = prev in ARTICLE_LIKE_FEM
    prev_masc = prev in ARTICLE_LIKE_MASC
    prev_plur = prev in ARTICLE_LIKE_PL
    prev_sing = prev in ARTICLE_LIKE_SG
    if fem and prev_masc:
        return "gender_mismatch"
    if (fem is False) and prev_fem:
        return "gender_mismatch"
    if plur and prev_sing:
        return "number_mismatch"
    if (plur is False) and prev_plur:
        return "number_mismatch"
    return None


def validate(
    tokens: Sequence[str],
    target_lemma: str = "",
    morph: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[str]:
    """Return a list of failure-reason names. Empty list means the sentence is valid."""
    toks = [t.lower() for t in tokens if t]
    reasons: List[str] = []
    if not toks:
        return ["empty"]

    first = toks[0]
    last = toks[-1]
    n = len(toks)

    # --- R1: lone past participle at start ---
    if first in PAST_PARTICIPLES:
        reasons.append("lone_participle_start")

    # --- R1b: gerund at start ---
    if _is_gerund(first):
        reasons.append("gerund_start")

    # --- R2: gerund at end (needs complement/auxiliary) ---
    if _is_gerund(last):
        reasons.append("gerund_final")

    # --- R3: relative/interrogative pronoun at end ---
    if last in RELATIVE_INTERROGATIVE:
        reasons.append("relative_final")

    # --- R4: contraction at start ---
    if first in CONTRACTIONS:
        reasons.append("contraction_start")

    # --- R4b: preposition/contraction + preposition/contraction at start ---
    if n >= 2 and first in (PREPOSITIONS | CONTRACTIONS) and toks[1] in (PREPOSITIONS | CONTRACTIONS):
        reasons.append("double_preposition_start")

    # --- R5: subject-pron + copula + prepositional-pron (e.g., "ella es mí") ---
    COPULAS = {"es", "está", "soy", "eres", "somos", "son",
               "estoy", "estás", "estamos", "están",
               "era", "eras", "éramos", "eran",
               "fue", "fui", "fuiste", "fuimos", "fueron"}
    for i in range(n - 2):
        a, b, c = toks[i], toks[i + 1], toks[i + 2]
        if a in SUBJECT_PRONOUNS and b in COPULAS and c in PREP_PRONOUNS:
            reasons.append("subj_copula_prep_pronoun")
            break

    # --- R6: "uno" immediately before a masculine singular noun ---
    for i in range(n - 1):
        if toks[i] == "uno":
            nxt = toks[i + 1]
            if (
                nxt not in ARTICLES
                and nxt not in PREPOSITIONS
                and nxt not in FINITE_VERBS
                and nxt not in VERB_PERSON
                and nxt not in COORDINATORS
                and len(nxt) >= 3
                and not _heuristic_plural_noun(nxt)
                and not _heuristic_fem_noun(nxt)
            ):
                reasons.append("uno_before_noun")
                break

    # --- Collect finite-verb positions for R7/R8 ---
    verb_positions = [i for i, t in enumerate(toks) if t in FINITE_VERBS]

    # --- R7: dangling finite verb at the end ---
    if last in FINITE_VERBS and len(verb_positions) >= 2:
        # Allow if the tail is a tight known pattern: "no puedo", "también puedo",
        # conjunction + verb, clitic + verb.
        prev_tok = toks[-2] if n >= 2 else ""
        ok_prev = (
            prev_tok in COORDINATORS
            or prev_tok in {"no", "nunca", "tampoco", "jamás"}
            or prev_tok in OBJECT_CLITICS
            or prev_tok in PREPOSITIONS
            or prev_tok in SUBJECT_PRONOUNS
        )
        if not ok_prev:
            reasons.append("dangling_final_verb")

    # --- R8: run-on clauses (two finite verbs with no connector between) ---
    allowed_between = (
        COORDINATORS | OBJECT_CLITICS | PREPOSITIONS | ARTICLES
        | SUBJECT_PRONOUNS | RELATIVE_INTERROGATIVE
        | {"no", "nunca", "tampoco", "jamás", "muy", "más", "menos", "tan", "también", "ya", "aún"}
        | CONTRACTIONS
        | DEM_POSS_FEM_PL | DEM_POSS_FEM_SG | DEM_POSS_MASC_PL | DEM_POSS_MASC_SG
    )
    for j in range(1, len(verb_positions)):
        a_i, b_i = verb_positions[j - 1], verb_positions[j]
        between = toks[a_i + 1:b_i]
        if between and not any(t in allowed_between for t in between):
            reasons.append("run_on_clauses")
            break

    # --- R9: article/demonstrative/possessive agreement mismatch anywhere in the sentence ---
    for i in range(n - 1):
        fail = _agreement_mismatch(toks[i], toks[i + 1], morph)
        if fail:
            reasons.append(fail)
            break

    # --- R10: bad final token (delegates to the BAD_FINAL list, with exceptions) ---
    if last in BAD_FINAL:
        tail_allow = (
            n >= 2
            and toks[-2] in {"con", "para", "de", "a"}
            and last in {"ti", "mí", "sí", "él", "ella", "nosotros", "ellos", "ellas"}
        )
        tail_allow |= (
            n >= 2 and toks[-2] == "no" and last in {"puedo", "quiero", "sé"}
        )
        if not tail_allow:
            reasons.append("bad_final_token")

    # --- R11a: prepositional pronoun (mí/ti/sí/conmigo/...) cannot be a subject ---
    if first in PREP_PRONOUNS:
        reasons.append("prep_pronoun_as_subject")

    # --- R12: "todo el mundo" / "todos los días" dangling after a mid-sentence NP ---
    # Look for adjacent trigrams like "todo el mundo" / "todas las cosas" / "todos los días".
    for i in range(n - 2):
        if toks[i] in {"todo", "todos", "todas", "toda"} and toks[i + 1] in ARTICLES:
            if i == 0:
                continue  # "Todos los libros son buenos." — fine as subject.
            prev = toks[i - 1]
            safe_prev = (
                prev in PREPOSITIONS | CONTRACTIONS
                or prev in COORDINATORS
                or prev in OBJECT_CLITICS
                or prev in FINITE_VERBS
                or prev in VERB_PERSON
                or prev in {"no", "nunca", "tampoco", "jamás"}
            )
            if not safe_prev:
                reasons.append("todo_el_mundo_after_np")
                break

    # --- R13: "para que" followed by an indicative verb (needs subjunctive) ---
    # Skip intervening negation / clitics / subject pronouns when scanning for the
    # first finite verb in the "para que" tail.
    SUBJ_INTRODUCERS = (
        {"no", "nunca", "jamás", "tampoco"}
        | OBJECT_CLITICS
        | SUBJECT_PRONOUNS
    )
    triggered_para_que = False
    for i in range(n - 2):
        if toks[i] == "para" and toks[i + 1] == "que":
            for k in range(i + 2, min(i + 6, n)):
                tok = toks[k]
                if tok in SUBJ_INTRODUCERS:
                    continue
                if tok in INDICATIVE_AFTER_SUBJUNCTIVE_TRIGGER:
                    reasons.append("para_que_indicative")
                    triggered_para_que = True
                break  # first non-skippable token decides
            if triggered_para_que:
                break

    # --- R14: "sí misma/mismo" without a supporting preposition ---
    for i in range(n - 1):
        if toks[i] == "sí" and toks[i + 1] in {"misma", "mismo", "mismas", "mismos"}:
            prev = toks[i - 1] if i >= 1 else ""
            if prev not in {"por", "en", "a", "con", "de", "para", "entre", "sobre", "contra"}:
                reasons.append("si_misma_without_preposition")
                break

    # --- R15: possessive determiner + semantically-unsafe human noun ---
    for i in range(n - 1):
        if toks[i] in POSSESSIVES and toks[i + 1] in UNSAFE_POSSESSIVE_HEAD_NOUNS:
            reasons.append("possessive_unsafe_noun")
            break

    # --- R16: "ser algo que ver" — invalid; valid idiom is "tener algo que ver" ---
    copulas_ser = {"es", "era", "fue", "sea", "son", "eran", "fueron", "sean"}
    for i in range(n - 3):
        if toks[i] in copulas_ser and toks[i + 1] == "algo" and toks[i + 2] == "que" and toks[i + 3] == "ver":
            reasons.append("ser_algo_que_ver")
            break
    # Also reject without the 2-slot "ser algo que ver": "es algo que ver" even if
    # there is something between subject and ser. Check directly for the 4-gram.
    # (The above loop already covers it; no extra work.)

    # --- R17a: "de que" followed by an NP without any finite verb in the tail ---
    for i in range(n - 2):
        if toks[i] == "de" and toks[i + 1] == "que":
            tail = toks[i + 2:]
            if tail and not any(t in FINITE_VERBS for t in tail):
                reasons.append("incomplete_de_que")
                break

    # --- R17b: bare modal at end ("...que no puedo") inside an embedded clause ---
    # The whole sentence may legitimately end in a modal when preceded only by a negation
    # and nothing else ("No quiero."). The failure is when there is a mid-sentence "que"
    # introducing a relative/complement clause, the tail after "que" has no infinitive,
    # and the very last token is a modal.
    if last in MODAL_FINITE:
        # Find the last "que".
        last_que = -1
        for j in range(n - 1, -1, -1):
            if toks[j] == "que":
                last_que = j
                break
        if last_que >= 0 and last_que < n - 1:
            tail_after_que = toks[last_que + 1:]
            # Tail has a modal at end but no infinitive anywhere in the tail.
            if not any(t.endswith(("ar", "er", "ir")) and t not in MODAL_FINITE for t in tail_after_que):
                reasons.append("bare_modal_final")
        # Also reject bare modal final in very short sentences where a complement is
        # obviously missing (like "Una vez más personas que no puedo" caught above).

    # --- R17c: "<modal> <bare-infinitive>" at the very end ---
    if n >= 2 and toks[-1] in BARE_INFINITIVES and toks[-2] in MODAL_FINITE:
        reasons.append("bare_modal_inf_final")

    # --- R18: quantifier (todos/todas/ambos/ambas) + noun without intervening det ---
    for i in range(n - 1):
        if toks[i] in QUANTIFIERS_NEEDING_DETERMINER:
            nxt = toks[i + 1]
            safe_next = (
                nxt in ARTICLES
                or nxt in POSSESSIVES
                or nxt in DEM_POSS_FEM_PL | DEM_POSS_FEM_SG | DEM_POSS_MASC_PL | DEM_POSS_MASC_SG
                or nxt in SUBJECT_PRONOUNS
                or nxt in FINITE_VERBS
                or nxt in VERB_PERSON
                or nxt in PREPOSITIONS
                or nxt in COORDINATORS
                or nxt in {"juntos", "juntas"}
            )
            if not safe_next:
                reasons.append("quantifier_without_article")
                break

    # --- R19: "no creo/creemos/crees/creen que" followed by indicative ---
    # Spanish requires subjunctive after negated opinion verbs ("no creo que sea/tenga").
    NEGATED_OPINION = {"creo", "crees", "cree", "creemos", "creen",
                       "pienso", "piensas", "piensa", "pensamos", "piensan"}
    for i in range(n - 3):
        if toks[i] == "no" and toks[i + 1] in NEGATED_OPINION and toks[i + 2] == "que":
            nxt = toks[i + 3] if i + 3 < n else ""
            if nxt in INDICATIVE_AFTER_SUBJUNCTIVE_TRIGGER:
                reasons.append("no_creo_que_indicative")
                break

    # --- R11: starts with a bare object clitic (non-imperative) ---
    # Clitics should attach to a verb; a sentence cannot start with "me/te/se/nos/os ..."
    # unless the next token is a verb form. We skip "la/los/las" because those are
    # *far* more often articles at sentence start; the article-agreement rules handle
    # any real misuse.
    AMBIGUOUS_WITH_ARTICLES = ARTICLES  # {"el", "la", "los", "las", ...}
    if first in OBJECT_CLITICS and first not in AMBIGUOUS_WITH_ARTICLES:
        nxt = toks[1] if n >= 2 else ""
        if nxt not in FINITE_VERBS and nxt not in VERB_PERSON:
            reasons.append("bad_clitic_start")

    return reasons


def is_valid(tokens: Sequence[str], target_lemma: str = "",
             morph: Optional[Dict[str, Dict[str, str]]] = None) -> bool:
    return not validate(tokens, target_lemma, morph=morph)
