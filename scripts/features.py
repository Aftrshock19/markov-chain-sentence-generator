#!/usr/bin/env python3
"""Hand-engineered features for a logistic-regression sentence quality reranker.

The features capture things that distinguish "good learner Spanish sentences"
from the corpus/Markov garbage the generator currently produces:
  - KN-LM log-probability (whole sentence + per-token)
  - length bucket
  - starts-with-bad-opener flag
  - ends-with-bad-token flag
  - subject-pronoun-with-wrong-verb-person flag
  - contains unsupported "que" clause
  - article-noun gender agreement violations
  - repeated-token / triple-repetition flag
  - fraction of tokens in the top-5k lemma list (simplicity)
  - contains impossible patterns ("nos está", "ella hay", "más ella está")
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")

PUNCT_CHARS = set(".,;:!?¡¿")

CONTEXT_OPENERS = {"si", "que", "cuando", "aunque", "porque", "como", "pero", "y", "o", "ni"}
PREPOSITIONS = {"a", "de", "en", "con", "por", "para", "sin", "sobre", "hasta", "desde", "entre", "contra", "hacia"}
ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
DEFINITE = {"el", "la", "los", "las"}
INDEFINITE = {"un", "una", "unos", "unas"}
FEMININE_ART = {"la", "las", "una", "unas"}
MASCULINE_ART = {"el", "los", "un", "unos"}
PLURAL_ART = {"los", "las", "unos", "unas"}
SINGULAR_ART = {"el", "la", "un", "una"}
SUBJECT_PRONOUNS = {"yo", "tú", "él", "ella", "nosotros", "nosotras", "vosotros", "vosotras", "ellos", "ellas", "usted", "ustedes"}
OBJECT_CLITICS = {"me", "te", "se", "lo", "la", "los", "las", "le", "les", "nos", "os"}
CONTRACTIONS_PREP = {"al", "del"}
DEGREE_ADV_NON_FINAL = {"muy", "tan", "más", "menos", "bastante", "demasiado", "tanto"}
AUX_VERBS = {"he", "has", "ha", "hemos", "han", "había", "habías", "habíamos", "habían", "hube", "hubo", "habré", "habrás", "habrá"}
# Non-standalone verb forms that usually need an object or infinitive following them
NEEDS_COMPLEMENT_VERBS = {"sea", "seas", "sean", "seamos", "fuera", "fueras", "fueran", "siendo", "estando", "habiendo", "haber"}
APOCOPATED_ADJ = {"buen", "mal", "gran", "primer", "tercer", "san", "algún", "ningún", "cualquier"}
BAD_FINAL = (
    PREPOSITIONS
    | {"y", "o", "que", "pero", "ni", "si", "cuando", "porque", "aunque", "mientras", "aun", "como"}
    | ARTICLES
    | SUBJECT_PRONOUNS
    | OBJECT_CLITICS
    | CONTRACTIONS_PREP
    | DEGREE_ADV_NON_FINAL
    | AUX_VERBS
    | NEEDS_COMPLEMENT_VERBS
    | APOCOPATED_ADJ
    | {"mi", "tu", "su", "mis", "tus", "sus", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella", "qué", "cuál", "cuánto", "cuántos", "cuántas"}
)

FIRST_PERSON_SG = {"soy", "estoy", "tengo", "voy", "puedo", "quiero", "creo", "sé", "digo", "hago", "veo", "vivo", "trabajo", "hablo", "vengo", "doy", "pongo", "pienso", "espero", "necesito",
                   "dije", "hice", "fui", "vi", "di", "vine", "tuve", "estuve", "quise", "pude", "puse", "supe",
                   "decía", "hacía", "iba", "venía", "tenía", "era", "estaba", "quería", "podía", "sabía", "veía",
                   "diré", "haré", "iré", "tendré", "sabré", "podré", "querré",
                   "diría", "haría", "iría", "sería", "estaría", "tendría", "podría", "querría",
                   "amo", "miro", "escucho", "leo", "escribo", "como", "bebo", "duermo", "juego", "pago", "compro", "salgo", "encuentro"}
SECOND_PERSON_SG = {"eres", "estás", "tienes", "vas", "puedes", "quieres", "crees", "sabes", "dices", "haces", "ves", "vives", "trabajas", "hablas", "vienes", "das", "pones",
                    "dijiste", "hiciste", "fuiste", "viniste", "tuviste", "estuviste", "quisiste", "pudiste",
                    "decías", "hacías", "ibas", "tenías", "eras", "estabas", "querías", "podías", "sabías"}
THIRD_PERSON_SG = {"es", "está", "tiene", "va", "puede", "quiere", "cree", "sabe", "dice", "hace", "ve", "vive", "trabaja", "habla", "viene", "da", "pone", "hay",
                   "fue", "dijo", "hizo", "vino", "tuvo", "estuvo", "quiso", "pudo", "puso", "supo", "vio", "dio",
                   "decía", "hacía", "iba", "venía", "tenía", "era", "estaba", "quería", "podía", "sabía", "veía",
                   "será", "estará", "tendrá", "dirá", "hará", "irá", "podrá", "querrá", "sabrá",
                   "sería", "estaría", "tendría", "diría", "haría", "iría", "podría", "querría",
                   "gusta", "encanta", "parece", "importa", "duele", "queda", "falta", "pasa", "llega", "sale", "piensa", "ama", "mira", "escucha", "lee", "escribe", "come", "bebe", "duerme", "juega", "paga", "compra", "encuentra"}
FIRST_PERSON_PL = {"somos", "estamos", "tenemos", "vamos", "podemos", "queremos", "creemos", "sabemos", "decimos", "hacemos", "vemos", "vivimos", "trabajamos", "hablamos", "venimos", "damos", "ponemos",
                   "dijimos", "hicimos", "fuimos", "vinimos", "tuvimos", "estuvimos",
                   "decíamos", "hacíamos", "íbamos", "teníamos", "éramos", "estábamos", "queríamos"}
SECOND_PERSON_PL = {"sois", "estáis", "tenéis", "vais", "podéis", "queréis", "creéis", "sabéis", "decís", "hacéis", "veis", "vivís", "trabajáis", "habláis", "venís", "dais", "ponéis"}
THIRD_PERSON_PL = {"son", "están", "tienen", "van", "pueden", "quieren", "creen", "saben", "dicen", "hacen", "ven", "viven", "trabajan", "hablan", "vienen", "dan", "ponen",
                   "dijeron", "hicieron", "fueron", "vinieron", "tuvieron", "estuvieron", "quisieron", "pudieron", "vieron", "dieron",
                   "decían", "hacían", "iban", "venían", "tenían", "eran", "estaban", "querían", "podían", "sabían",
                   "serán", "estarán", "tendrán", "dirán", "harán", "irán", "podrán", "querrán",
                   "serían", "estarían", "tendrían", "dirían", "harían", "irían", "podrían", "querrían",
                   "gustan", "encantan", "parecen", "importan", "duelen"}
AUX_PERFECT = {"he", "has", "ha", "hemos", "han", "había", "habías", "habíamos", "habían",
               "haya", "hayas", "hayamos", "hayan", "habré", "habrás", "habrá", "habremos", "habrán"}
FINITE_VERBS = (FIRST_PERSON_SG | SECOND_PERSON_SG | THIRD_PERSON_SG | FIRST_PERSON_PL | SECOND_PERSON_PL | THIRD_PERSON_PL | AUX_PERFECT)

PRONOUN_PERSON = {
    "yo": "1s", "tú": "2s", "él": "3s", "ella": "3s", "usted": "3s",
    "nosotros": "1p", "nosotras": "1p", "vosotros": "2p", "vosotras": "2p",
    "ellos": "3p", "ellas": "3p", "ustedes": "3p",
}
VERB_PERSON = {}
for v in FIRST_PERSON_SG: VERB_PERSON[v] = "1s"
for v in SECOND_PERSON_SG: VERB_PERSON[v] = "2s"
for v in THIRD_PERSON_SG: VERB_PERSON[v] = "3s"
for v in FIRST_PERSON_PL: VERB_PERSON[v] = "1p"
for v in SECOND_PERSON_PL: VERB_PERSON[v] = "2p"
for v in THIRD_PERSON_PL: VERB_PERSON[v] = "3p"

# A known problematic bigram set learned from review data. If "ella hay" appears, it's almost always bad.
BAD_BIGRAMS = {
    ("ella", "hay"), ("nos", "está"), ("nos", "es"),
    ("más", "ella"), ("le", "hace"), ("lo", "hace"),
    ("a", "yo"), ("a", "tú"), ("a", "él"), ("a", "ella"),
    ("a", "nosotros"), ("a", "ellos"), ("a", "ellas"),
    ("con", "ella", "misma"),  # handled in trigram scan
}


def tokenize(sentence: str) -> List[str]:
    if not sentence:
        return []
    return [t.lower() for t in WORD_RE.findall(sentence)]


def _feminine_noun_heuristic(w: str) -> bool:
    return (
        w.endswith(("a",))
        and not w.endswith(("ma", "pa", "ta"))
    ) or w.endswith(("dad", "tad", "tud", "ción", "sión", "umbre", "ez"))


def _plural_noun_heuristic(w: str) -> bool:
    return w.endswith("s") and len(w) > 3 and not w.endswith(("és", "ís", "ús", "ás", "os"))


def featurize(
    tokens: Sequence[str],
    target_lemma: str,
    target_rank: int,
    lm_score: float,
    rank_by_word: Dict[str, int],
) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    n = max(1, len(tokens))
    feats["bias"] = 1.0
    feats["lm_logp"] = lm_score
    feats["lm_logp_per_tok"] = lm_score / n
    feats["len"] = float(n)
    feats["len_sq"] = float(n * n)
    feats["len_lt3"] = float(n < 3)
    feats["len_3_5"] = float(3 <= n <= 5)
    feats["len_6_8"] = float(6 <= n <= 8)
    feats["len_9_12"] = float(9 <= n <= 12)
    feats["len_gt12"] = float(n > 12)

    first = tokens[0] if tokens else ""
    last = tokens[-1] if tokens else ""
    feats["bad_opener"] = float(first in CONTEXT_OPENERS)
    feats["bad_final"] = float(last in BAD_FINAL)
    feats["ends_prep"] = float(last in PREPOSITIONS)
    feats["ends_conj"] = float(last in {"y", "o", "pero", "ni", "que", "si"})

    has_verb = any(t in FINITE_VERBS for t in tokens)
    feats["no_verb"] = float(not has_verb)
    feats["verb_count"] = sum(1 for t in tokens if t in FINITE_VERBS)
    feats["multi_verb"] = float(feats["verb_count"] >= 2)

    # Subject-verb agreement: first pronoun + first following verb
    agreement_mismatch = 0
    agreement_match = 0
    for i, t in enumerate(tokens):
        if t in PRONOUN_PERSON:
            for j in range(i + 1, min(len(tokens), i + 4)):
                if tokens[j] in VERB_PERSON:
                    if PRONOUN_PERSON[t] != VERB_PERSON[tokens[j]]:
                        agreement_mismatch += 1
                    else:
                        agreement_match += 1
                    break
            break
    feats["sv_mismatch"] = float(agreement_mismatch)
    feats["sv_match"] = float(agreement_match)

    # Article-noun agreement (heuristic)
    art_noun_mismatch = 0
    for i in range(len(tokens) - 1):
        art, nxt = tokens[i], tokens[i + 1]
        if art in ARTICLES:
            noun_fem = _feminine_noun_heuristic(nxt)
            noun_pl = _plural_noun_heuristic(nxt)
            art_fem = art in FEMININE_ART
            art_pl = art in PLURAL_ART
            if nxt not in ARTICLES and nxt not in PREPOSITIONS and nxt not in FINITE_VERBS:
                # Only penalize confident mismatches: both gender and number indicators exist
                if noun_fem and not art_fem:
                    art_noun_mismatch += 1
                if art_pl != noun_pl and noun_pl is True:
                    art_noun_mismatch += 1
    feats["art_noun_mismatch"] = float(art_noun_mismatch)

    # Repeated tokens
    repeated = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    feats["repeat"] = float(repeated)

    # Bad bigrams / trigrams
    bad_bi = 0
    for a, b in zip(tokens, tokens[1:]):
        if (a, b) in BAD_BIGRAMS:
            bad_bi += 1
    feats["bad_bigram"] = float(bad_bi)
    bad_tri = 0
    for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
        if (a, b, c) == ("con", "ella", "misma"):
            bad_tri += 1
    feats["bad_trigram"] = float(bad_tri)

    # target presence
    feats["target_present"] = float(target_lemma in tokens)
    feats["target_first"] = float(tokens[0] == target_lemma if tokens else False)
    feats["target_last"] = float(tokens[-1] == target_lemma if tokens else False)

    # Vocabulary difficulty vs target
    oov = 0
    over_rank = 0
    for t in tokens:
        r = rank_by_word.get(t, 999999)
        if r >= 999999:
            oov += 1
        elif r > max(800, target_rank * 3):
            over_rank += 1
    feats["oov_tokens"] = float(oov)
    feats["oov_tokens_frac"] = oov / n
    feats["over_rank_tokens"] = float(over_rank)

    # "que" without following verb
    que_bad = 0
    for i, t in enumerate(tokens):
        if t == "que" and i < len(tokens) - 1:
            tail = tokens[i + 1:]
            if not any(x in FINITE_VERBS for x in tail) and tail != ["no"]:
                que_bad += 1
    feats["que_no_clause"] = float(que_bad)

    # "a" + subject pronoun = bad
    bad_a = 0
    for i in range(len(tokens) - 1):
        if tokens[i] == "a" and tokens[i + 1] in {"yo", "tú", "él", "ella", "nosotros", "ellos", "ellas"}:
            bad_a += 1
    feats["a_subject"] = float(bad_a)

    # "no" final alone when sentence is long
    feats["no_final_long"] = float(n > 2 and last == "no")

    # verb-final without object after it
    feats["verb_final_no_obj"] = float(last in FINITE_VERBS and not (n >= 2 and tokens[-2] == "no" and last in {"puedo", "quiero", "sé"}))

    # double negation: "nada" without "no"/"nunca"
    feats["neg_without_no"] = float("nada" in tokens and "no" not in tokens and "nunca" not in tokens)

    return feats
