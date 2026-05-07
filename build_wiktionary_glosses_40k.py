#!/usr/bin/env python3
"""
Full production pipeline: Spanish lemma → flashcard-safe English translation.

Stages:
  A  collect_candidates(lemma, rank, entries)    → list[dict]
  B  classify_candidate_bucket(candidate)        → bucket str
     score_candidate(candidate)                  → float
  C  generate_flashcard_translation(...)         → str
     is_flashcard_safe_english(...)              → bool
     compact_to_flashcard_english(...)           → list[str]
  D  run_qa(rows, grouped, candidates_map)       → diagnostics list
     repair_flagged_row(...)                     → (str, str)
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import orjson
from tqdm import tqdm

# ── Sense quality filters ─────────────────────────────────────────────────────

BANNED_SENSE_TAGS = {
    "obsolete", "archaic", "dated", "historical",
    "rare", "uncommon", "superseded", "misspelling", "typo",
}
LOW_PRIORITY_SENSE_TAGS = {
    "slang", "colloquial", "regional", "dialectal",
    "figurative", "literary", "formal",
}
SKIP_POS = {
    "abbrev", "character", "diacritic", "glyph", "han character",
    "ideophone", "symbol", "punctuation",
}

# ── Semantic buckets ──────────────────────────────────────────────────────────

BUCKET_VERB_FORM  = "verb_form"      # inflected / non-finite verb
BUCKET_VERB_LEMMA = "verb_lemma"     # infinitive (base form of verb)
BUCKET_FUNCTION   = "function_word"  # article, prep, conj, particle, contraction
BUCKET_PRONOUN    = "pronoun"
BUCKET_DETERMINER = "determiner"     # det, num, numeral
BUCKET_CONTENT    = "content_word"   # noun, adj, adv, interjection, phrase
BUCKET_META       = "meta"           # only form-of / grammar prose, no usable English
BUCKET_JUNK       = "junk"           # letter names, bad domains, etc.

# Priority tables: higher number = preferred for this rank band
_PRI_TOP1K: dict[str, float] = {
    BUCKET_FUNCTION:   100.0,
    BUCKET_PRONOUN:     98.0,
    BUCKET_DETERMINER:  96.0,
    BUCKET_VERB_LEMMA:  72.0,
    BUCKET_CONTENT:     58.0,
    BUCKET_VERB_FORM:   45.0,
    BUCKET_META:         8.0,
    BUCKET_JUNK:         0.0,
}
_PRI_MID5K: dict[str, float] = {
    BUCKET_FUNCTION:    80.0,
    BUCKET_PRONOUN:     78.0,
    BUCKET_DETERMINER:  76.0,
    BUCKET_VERB_LEMMA:  72.0,
    BUCKET_CONTENT:     68.0,
    BUCKET_VERB_FORM:   45.0,
    BUCKET_META:         8.0,
    BUCKET_JUNK:         0.0,
}
_PRI_REST: dict[str, float] = {
    BUCKET_FUNCTION:    60.0,
    BUCKET_PRONOUN:     58.0,
    BUCKET_DETERMINER:  56.0,
    BUCKET_VERB_LEMMA:  72.0,
    BUCKET_CONTENT:     72.0,
    BUCKET_VERB_FORM:   45.0,
    BUCKET_META:         8.0,
    BUCKET_JUNK:         0.0,
}


def _bucket_priority_table(rank: Optional[int]) -> dict:
    if rank is None or rank > 5000:
        return _PRI_REST
    if rank > 1000:
        return _PRI_MID5K
    return _PRI_TOP1K


# ── Pattern lists ─────────────────────────────────────────────────────────────

FORM_OF_PATTERNS = (
    "inflection of ", "form of ", "past participle of ",
    "present participle of ", "gerund of ", "alternative form of ",
    "alternative spelling of ", "plural of ", "singular of ",
    "feminine singular of ", "masculine singular of ",
    "feminine plural of ", "masculine plural of ",
    "third-person singular", "third-person plural",
    "second-person singular", "second-person plural",
    "first-person singular", "first-person plural",
    "superlative of ", "comparative of ", "diminutive of ",
    "augmentative of ", "vocative of ", "genitive of ",
    "dative of ", "accusative of ", "nominative of ",
    "prepositional of ", "apocopic form",
)

META_PROSE_PATTERNS = (
    "used to ", "used after ", "used before ", "used with ",
    "used only ", "used in ", "expression of ", "title of respect",
    "commonly used", "literally ", "the act of ", "the state of ",
    "denotes a ", "denoting a ", "indicates ", "referring to ",
    "variant of ", "contraction of ", "elision of ",
    "in grammar", "grammatical ", "short for ", "abbreviation of ",
    "interrogative only", "definite article", "indefinite article",
    "personal third person", "subject and disjunctive pronoun",
    "subjectively and after prepositions", "can refer",
    "as opposed to", "with the ", 'with "',
)

GRAMMAR_PROSE_WORDS = frozenset({
    "singular", "plural", "masculine", "feminine", "neuter",
    "nominative", "accusative", "dative", "genitive", "vocative",
    "participle", "gerund", "infinitive", "subjunctive", "imperative",
    "indicative", "conditional", "preterite", "imperfect", "pluperfect",
    "determiner", "reflexive", "disjunctive", "enclitic", "clitic",
    "declension", "conjugation", "morphological", "pronoun",
})

BAD_DOMAIN_PATTERNS = (
    "latin script letter", "greek letter", "hebrew letter",
    "arabic letter", "cyrillic letter", "letter of the latin",
    "letter of the greek", "letter of the hebrew", "letter of the",
    "name of the letter", "the letter ", "letter name",
    "musical note", "solfège", "solfege",
    "psychoanalytic", "psychoanalysis", "freudian", "freud",
)

# ── Irregular English verb forms ──────────────────────────────────────────────

IRREGULAR_ENGLISH_FORMS: dict[str, dict] = {
    "be": {
        ("present","1","sg"):"am",  ("present","2","sg"):"are",
        ("present","3","sg"):"is",  ("present","1","pl"):"are",
        ("present","2","pl"):"are", ("present","3","pl"):"are",
        ("preterite","1","sg"):"was", ("preterite","2","sg"):"were",
        ("preterite","3","sg"):"was", ("preterite","1","pl"):"were",
        ("preterite","2","pl"):"were", ("preterite","3","pl"):"were",
        ("imperfect","1","sg"):"was", ("imperfect","2","sg"):"were",
        ("imperfect","3","sg"):"was", ("imperfect","1","pl"):"were",
        ("imperfect","2","pl"):"were", ("imperfect","3","pl"):"were",
        ("participle",None,None):"been", ("gerund",None,None):"being",
    },
    "have": {
        ("present","3","sg"):"has",
        ("preterite",None,None):"had", ("imperfect",None,None):"had",
        ("participle",None,None):"had", ("gerund",None,None):"having",
    },
    "do": {
        ("present","3","sg"):"does",
        ("preterite",None,None):"did", ("imperfect",None,None):"did",
        ("participle",None,None):"done", ("gerund",None,None):"doing",
    },
    "go": {
        ("present","3","sg"):"goes",
        ("preterite",None,None):"went", ("imperfect",None,None):"went",
        ("participle",None,None):"gone", ("gerund",None,None):"going",
    },
    "say": {
        ("present","3","sg"):"says",
        ("preterite",None,None):"said", ("imperfect",None,None):"said",
        ("participle",None,None):"said", ("gerund",None,None):"saying",
    },
    "see": {
        ("present","3","sg"):"sees",
        ("preterite",None,None):"saw", ("imperfect",None,None):"saw",
        ("participle",None,None):"seen", ("gerund",None,None):"seeing",
    },
    "know": {
        ("present","3","sg"):"knows",
        ("preterite",None,None):"knew", ("imperfect",None,None):"knew",
        ("participle",None,None):"known", ("gerund",None,None):"knowing",
    },
    "come": {
        ("present","3","sg"):"comes",
        ("preterite",None,None):"came", ("imperfect",None,None):"came",
        ("participle",None,None):"come", ("gerund",None,None):"coming",
    },
    "put": {
        ("present","3","sg"):"puts",
        ("preterite",None,None):"put", ("imperfect",None,None):"put",
        ("participle",None,None):"put", ("gerund",None,None):"putting",
    },
    "give": {
        ("present","3","sg"):"gives",
        ("preterite",None,None):"gave", ("imperfect",None,None):"gave",
        ("participle",None,None):"given", ("gerund",None,None):"giving",
    },
    "take": {
        ("present","3","sg"):"takes",
        ("preterite",None,None):"took", ("imperfect",None,None):"took",
        ("participle",None,None):"taken", ("gerund",None,None):"taking",
    },
    "make": {
        ("present","3","sg"):"makes",
        ("preterite",None,None):"made", ("imperfect",None,None):"made",
        ("participle",None,None):"made", ("gerund",None,None):"making",
    },
    "get": {
        ("present","3","sg"):"gets",
        ("preterite",None,None):"got", ("imperfect",None,None):"got",
        ("participle",None,None):"got", ("gerund",None,None):"getting",
    },
    "find": {
        ("present","3","sg"):"finds",
        ("preterite",None,None):"found", ("imperfect",None,None):"found",
        ("participle",None,None):"found", ("gerund",None,None):"finding",
    },
    "think": {
        ("present","3","sg"):"thinks",
        ("preterite",None,None):"thought", ("imperfect",None,None):"thought",
        ("participle",None,None):"thought", ("gerund",None,None):"thinking",
    },
    "tell": {
        ("present","3","sg"):"tells",
        ("preterite",None,None):"told", ("imperfect",None,None):"told",
        ("participle",None,None):"told", ("gerund",None,None):"telling",
    },
    "feel": {
        ("present","3","sg"):"feels",
        ("preterite",None,None):"felt", ("imperfect",None,None):"felt",
        ("participle",None,None):"felt", ("gerund",None,None):"feeling",
    },
    "leave": {
        ("present","3","sg"):"leaves",
        ("preterite",None,None):"left", ("imperfect",None,None):"left",
        ("participle",None,None):"left", ("gerund",None,None):"leaving",
    },
    "keep": {
        ("present","3","sg"):"keeps",
        ("preterite",None,None):"kept", ("imperfect",None,None):"kept",
        ("participle",None,None):"kept", ("gerund",None,None):"keeping",
    },
    "lose": {
        ("present","3","sg"):"loses",
        ("preterite",None,None):"lost", ("imperfect",None,None):"lost",
        ("participle",None,None):"lost", ("gerund",None,None):"losing",
    },
    "bring": {
        ("present","3","sg"):"brings",
        ("preterite",None,None):"brought", ("imperfect",None,None):"brought",
        ("participle",None,None):"brought", ("gerund",None,None):"bringing",
    },
    "begin": {
        ("present","3","sg"):"begins",
        ("preterite",None,None):"began", ("imperfect",None,None):"began",
        ("participle",None,None):"begun", ("gerund",None,None):"beginning",
    },
    "write": {
        ("present","3","sg"):"writes",
        ("preterite",None,None):"wrote", ("imperfect",None,None):"wrote",
        ("participle",None,None):"written", ("gerund",None,None):"writing",
    },
    "read": {
        ("present","3","sg"):"reads",
        ("preterite",None,None):"read", ("imperfect",None,None):"read",
        ("participle",None,None):"read", ("gerund",None,None):"reading",
    },
    "hear": {
        ("present","3","sg"):"hears",
        ("preterite",None,None):"heard", ("imperfect",None,None):"heard",
        ("participle",None,None):"heard", ("gerund",None,None):"hearing",
    },
    "fall": {
        ("present","3","sg"):"falls",
        ("preterite",None,None):"fell", ("imperfect",None,None):"fell",
        ("participle",None,None):"fallen", ("gerund",None,None):"falling",
    },
    "run": {
        ("present","3","sg"):"runs",
        ("preterite",None,None):"ran", ("imperfect",None,None):"ran",
        ("participle",None,None):"run", ("gerund",None,None):"running",
    },
    "speak": {
        ("present","3","sg"):"speaks",
        ("preterite",None,None):"spoke", ("imperfect",None,None):"spoke",
        ("participle",None,None):"spoken", ("gerund",None,None):"speaking",
    },
    "send": {
        ("present","3","sg"):"sends",
        ("preterite",None,None):"sent", ("imperfect",None,None):"sent",
        ("participle",None,None):"sent", ("gerund",None,None):"sending",
    },
    "stand": {
        ("present","3","sg"):"stands",
        ("preterite",None,None):"stood", ("imperfect",None,None):"stood",
        ("participle",None,None):"stood", ("gerund",None,None):"standing",
    },
    "understand": {
        ("present","3","sg"):"understands",
        ("preterite",None,None):"understood", ("imperfect",None,None):"understood",
        ("participle",None,None):"understood", ("gerund",None,None):"understanding",
    },
    "meet": {
        ("present","3","sg"):"meets",
        ("preterite",None,None):"met", ("imperfect",None,None):"met",
        ("participle",None,None):"met", ("gerund",None,None):"meeting",
    },
    "pay": {
        ("present","3","sg"):"pays",
        ("preterite",None,None):"paid", ("imperfect",None,None):"paid",
        ("participle",None,None):"paid", ("gerund",None,None):"paying",
    },
    "show": {
        ("present","3","sg"):"shows",
        ("preterite",None,None):"showed", ("imperfect",None,None):"showed",
        ("participle",None,None):"shown", ("gerund",None,None):"showing",
    },
    "follow": {
        ("present","3","sg"):"follows",
        ("preterite",None,None):"followed", ("imperfect",None,None):"followed",
        ("participle",None,None):"followed", ("gerund",None,None):"following",
    },
    "want": {
        ("present","3","sg"):"wants",
        ("preterite",None,None):"wanted", ("imperfect",None,None):"wanted",
        ("participle",None,None):"wanted", ("gerund",None,None):"wanting",
    },
    "seem": {
        ("present","3","sg"):"seems",
        ("preterite",None,None):"seemed", ("imperfect",None,None):"seemed",
        ("participle",None,None):"seemed", ("gerund",None,None):"seeming",
    },
    "live": {
        ("present","3","sg"):"lives",
        ("preterite",None,None):"lived", ("imperfect",None,None):"lived",
        ("participle",None,None):"lived", ("gerund",None,None):"living",
    },
}

# ── Strategic overrides (~120 entries where the generic system reliably fails) ─

TINY_OVERRIDES: dict[str, tuple[str, str]] = {
    # Articles
    "el": ("the", "article"),   "la": ("the", "article"),
    "los": ("the", "article"),  "las": ("the", "article"),
    "un": ("a; an", "article"), "una": ("a; an", "article"),
    "unos": ("some", "article"),"unas": ("some", "article"),
    # Prepositions
    "de": ("of; from", "prep"),      "a": ("to; at", "prep"),
    "en": ("in; on; at", "prep"),    "para": ("for; to", "prep"),
    "por": ("by; for", "prep"),      "con": ("with", "prep"),
    "sin": ("without", "prep"),      "sobre": ("on; about; over", "prep"),
    "entre": ("between; among", "prep"),
    "hasta": ("until; up to", "prep"),
    "desde": ("from; since", "prep"),
    "hacia": ("toward; towards", "prep"),
    "contra": ("against", "prep"),
    "ante": ("before; in front of", "prep"),
    "bajo": ("under; below", "prep"),
    "tras": ("after; behind", "prep"),
    "durante": ("during", "prep"),
    "según": ("according to", "prep"),
    "mediante": ("by means of; through", "prep"),
    # Conjunctions
    "y": ("and", "conj"),       "o": ("or", "conj"),
    "que": ("that; which", "conj"),  "porque": ("because", "conj"),
    "si": ("if", "conj"),       "pero": ("but", "conj"),
    "ni": ("nor; neither", "conj"),
    "aunque": ("although; even though", "conj"),
    "cuando": ("when", "conj"),
    "como": ("as; like; how", "conj"),
    "donde": ("where", "conj"),
    "mientras": ("while; meanwhile", "conj"),
    "pues": ("well; so; since", "conj"),
    "sino": ("but rather; but instead", "conj"),
    # Personal pronouns
    "yo": ("I", "pron"),           "tú": ("you", "pron"),
    "él": ("he; him", "pron"),     "ella": ("she; her", "pron"),
    "nosotros": ("we; us", "pron"),"nosotras": ("we; us", "pron"),
    "vosotros": ("you all", "pron"),"vosotras": ("you all", "pron"),
    "ellos": ("they; them", "pron"),"ellas": ("they; them", "pron"),
    "usted": ("you", "pron"),      "ustedes": ("you all", "pron"),
    # Object / reflexive pronouns
    "me": ("me", "pron"),    "te": ("you", "pron"),
    "se": ("oneself; himself; herself; itself; themselves", "pron"),
    "nos": ("us; ourselves", "pron"),
    "os": ("you all; yourselves", "pron"),
    "lo": ("it; him", "pron"),
    "le": ("him; her; it", "pron"),
    "les": ("them", "pron"),
    # Possessive determiners
    "mi": ("my", "det"),    "mis": ("my", "det"),
    "tu": ("your", "det"),  "tus": ("your", "det"),
    "su": ("his; her; its; your; their", "det"),
    "sus": ("his; her; its; your; their", "det"),
    "nuestro": ("our", "det"),  "nuestra": ("our", "det"),
    "nuestros": ("our", "det"), "nuestras": ("our", "det"),
    "vuestro": ("your", "det"), "vuestra": ("your", "det"),
    "vuestros": ("your", "det"),"vuestras": ("your", "det"),
    # Demonstratives
    "este": ("this", "det"),  "esta": ("this", "det"), "esto": ("this", "pron"),
    "estos": ("these", "det"),"estas": ("these", "det"),
    "ese": ("that", "det"),   "esa": ("that", "det"),  "eso": ("that", "pron"),
    "esos": ("those", "det"), "esas": ("those", "det"),
    "aquel": ("that; that one over there", "det"),
    "aquella": ("that; that one over there", "det"),
    "aquello": ("that; that thing over there", "pron"),
    "aquellos": ("those over there", "det"),
    "aquellas": ("those over there", "det"),
    # Indefinites / quantifiers
    "nada": ("nothing", "pron"),
    "nadie": ("nobody; no one", "pron"),
    "algo": ("something; somewhat", "pron"),
    "alguien": ("someone; somebody", "pron"),
    "algún": ("some; any", "det"),   "alguna": ("some; any", "det"),
    "algunos": ("some", "det"),      "algunas": ("some", "det"),
    "ningún": ("no; none", "det"),   "ninguna": ("no; none", "det"),
    "todo": ("everything; all", "pron"), "toda": ("all; every", "det"),
    "todos": ("all; everyone", "pron"), "todas": ("all", "pron"),
    # Core adverbs
    "no": ("not", "adv"),       "sí": ("yes", "adv"),
    "más": ("more; most", "adv"),"muy": ("very", "adv"),
    "bien": ("well", "adv"),    "mal": ("badly; poorly", "adv"),
    "ya": ("already; now", "adv"),
    "también": ("also; too", "adv"),
    "tampoco": ("neither; not either", "adv"),
    "nunca": ("never", "adv"),  "jamás": ("never; ever", "adv"),
    "siempre": ("always", "adv"),
    "solo": ("only; alone", "adv"),  "sólo": ("only; just", "adv"),
    "aquí": ("here", "adv"),    "ahí": ("there", "adv"),
    "allí": ("there; over there", "adv"),
    "allá": ("over there; there", "adv"),
    "así": ("like this; so; thus", "adv"),
    "antes": ("before; earlier", "adv"),
    "después": ("after; later", "adv"),
    "ahora": ("now", "adv"),
    "hoy": ("today", "adv"),
    "ayer": ("yesterday", "adv"),
    "pronto": ("soon; quickly", "adv"),
    "quizás": ("maybe; perhaps", "adv"),
    "quizá": ("maybe; perhaps", "adv"),
    "todavía": ("still; yet", "adv"),
    "aún": ("still; yet", "adv"),
    "aun": ("even; still", "adv"),
    "casi": ("almost; nearly", "adv"),
    "tan": ("so; such; as", "adv"),
    "tanto": ("so much; so many", "adv"), "tanta": ("so much", "adv"),
    "tantos": ("so many", "adv"),         "tantas": ("so many", "adv"),
    "demasiado": ("too; too much", "adv"),
    "bastante": ("enough; quite; rather", "adv"),
    "poco": ("little; not much", "adv"),
    "mucho": ("much; a lot", "adv"),
    # Interrogatives / relatives
    "qué": ("what; which", "pron"),
    "quién": ("who; whom", "pron"),
    "quiénes": ("who; whom", "pron"),
    "cuál": ("which; what", "pron"),
    "cuáles": ("which; what", "pron"),
    "cuánto": ("how much; how many", "pron"),
    "cuánta": ("how much", "pron"),
    "cuántos": ("how many", "pron"),
    "cuántas": ("how many", "pron"),
    "cómo": ("how", "adv"),
    "cuándo": ("when", "adv"),
    "dónde": ("where", "adv"),
    # Numbers
    "uno": ("one", "num"),    "dos": ("two", "num"),
    "tres": ("three", "num"), "cuatro": ("four", "num"),
    "cinco": ("five", "num"), "seis": ("six", "num"),
    "siete": ("seven", "num"),"ocho": ("eight", "num"),
    "nueve": ("nine", "num"), "diez": ("ten", "num"),
    "once": ("eleven", "num"),"doce": ("twelve", "num"),
    "trece": ("thirteen", "num"), "catorce": ("fourteen", "num"),
    "quince": ("fifteen", "num"), "veinte": ("twenty", "num"),
    "treinta": ("thirty", "num"), "cuarenta": ("forty", "num"),
    "cincuenta": ("fifty", "num"), "sesenta": ("sixty", "num"),
    "setenta": ("seventy", "num"), "ochenta": ("eighty", "num"),
    "noventa": ("ninety", "num"),
    "cien": ("one hundred", "num"), "ciento": ("one hundred", "num"),
    "mil": ("one thousand", "num"),  "millón": ("one million", "num"),
    "primero": ("first", "adj"),  "primer": ("first", "adj"),
    "segunda": ("second", "adj"), "segundo": ("second", "adj"),
    "tercero": ("third", "adj"),  "tercer": ("third", "adj"),
    # haber impersonals (catastrophic sense-collision)
    "hay": ("there is; there are", "verb"),
    "había": ("there was; there were", "verb"),
    "habrá": ("there will be", "verb"),
    "habría": ("there would be", "verb"),
    "haya": ("there be", "verb"),
    "hubo": ("there was; there were", "verb"),
    # Other catastrophic sense-collision forms
    "he": ("I have", "verb"),
    "son": ("they are; are", "verb"),
    "vamos": ("we go; let's go", "verb"),
    "va": ("he goes; she goes; it goes", "verb"),
    "vale": ("OK; it's worth it", "interjection"),
    "venga": ("come on; come here", "interjection"),
}

# ── Text utilities ────────────────────────────────────────────────────────────

_SMART_QUOTES = re.compile(r"[""'']")
_PAREN       = re.compile(r"\([^)]*\)")
_WHITESPACE  = re.compile(r"\s+")


def norm_space(text) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def clean_text(text: str) -> str:
    text = _SMART_QUOTES.sub('"', norm_space(text))
    return text.strip(" ;,.")


def contains_lemma(text: str, lemma: str, original_lemma: str = "") -> bool:
    padded = " " + clean_text(text).lower() + " "
    tokens = {lemma.lower().strip()}
    if original_lemma:
        tokens.add(original_lemma.lower().strip())
    for token in tokens:
        if token and (f" {token} " in padded or f'"{token}"' in padded):
            return True
    return False


# ── Sense data extraction ─────────────────────────────────────────────────────

def get_tags(sense: dict) -> frozenset:
    tags: set[str] = set()
    for key in ("tags", "raw_tags"):
        val = sense.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    tags.add(item.strip().lower())
    return frozenset(tags)


def get_glosses(sense: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in ("glosses", "raw_glosses"):
        val = sense.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    item = clean_text(item)
                    if item and item.lower() not in seen:
                        seen.add(item.lower())
                        out.append(item)
    return out


def extract_english_translations(sense: dict, max_items: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tr in sense.get("translations", []):
        if not isinstance(tr, dict):
            continue
        code      = norm_space(tr.get("code", "")).lower()
        lang_code = norm_space(tr.get("lang_code", "")).lower()
        lang      = norm_space(tr.get("lang", "")).lower()
        if code != "en" and lang_code != "en" and lang not in {"english", "inglés", "ingles"}:
            continue
        word = clean_text(tr.get("word", ""))
        if word and word.lower() not in seen:
            seen.add(word.lower())
            out.append(word)
            if len(out) >= max_items:
                break
    return out


def _extract_linked_word(items) -> str:
    if not isinstance(items, list):
        return ""
    for item in items:
        if isinstance(item, dict):
            word = norm_space(item.get("word", ""))
        elif isinstance(item, str):
            word = norm_space(item)
        else:
            continue
        if word:
            return word
    return ""


def get_original_lemma_from_sense(sense: dict) -> str:
    return (
        _extract_linked_word(sense.get("form_of"))
        or _extract_linked_word(sense.get("alt_of"))
        or ""
    )


def has_form_link(sense: dict) -> bool:
    return bool(get_original_lemma_from_sense(sense))


# ── Pattern testers ───────────────────────────────────────────────────────────

def looks_like_form_of(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in FORM_OF_PATTERNS)


def is_meta_prose(text: str) -> bool:
    lowered = clean_text(text).lower()
    return any(p in lowered for p in META_PROSE_PATTERNS)


def is_bad_domain(text: str) -> bool:
    lowered = clean_text(text).lower()
    return any(p in lowered for p in BAD_DOMAIN_PATTERNS)


def _all_glosses_are_meta(glosses: list[str]) -> bool:
    if not glosses:
        return True
    return all(looks_like_form_of(g) or is_meta_prose(g) for g in glosses)


def is_segment_grammar_prose(text: str) -> bool:
    """True if a single text segment reads as grammar description rather than vocabulary."""
    lowered = text.lower().strip()
    if not lowered:
        return True
    if any(p in lowered for p in META_PROSE_PATTERNS):
        return True
    if any(p.strip() in lowered for p in FORM_OF_PATTERNS):
        return True
    # Any grammar word in the segment → prose
    words = [w.rstrip(".,;:") for w in lowered.split()]
    if any(w in GRAMMAR_PROSE_WORDS for w in words):
        return True
    # Long text is probably an explanation, not a translation
    if len(text) > 35:
        return True
    return False


# ── Compactor ─────────────────────────────────────────────────────────────────

def compact_to_flashcard_english(raw: str, pos: str, max_options: int = 4) -> list[str]:
    """
    Transform a raw Wiktionary gloss into compact, flashcard-ready English options.
    Returns a list of short clean strings (empty list if nothing usable found).

    Examples:
      "feminine singular definite article; the"  → ["the"]
      "there: used to designate a place nearby"   → ["there"]
      "he, him, masculine personal third person"  → ["he", "him"]
      "what; which (interrogative only)"          → ["what", "which"]
      "reflexive of nosotros: ourselves; each other" → ["ourselves", "each other"]
    """
    text = clean_text(raw)
    if not text:
        return []

    # Colon rule: if "LEFT: RIGHT" and one side is grammar prose, keep the other
    if ":" in text:
        left, right = text.split(":", 1)
        left  = left.strip()
        right = right.strip()
        left_meta  = is_segment_grammar_prose(left)
        right_meta = is_segment_grammar_prose(right)
        if left_meta and not right_meta:
            text = right
        elif right_meta and not left_meta:
            text = left
        elif left_meta and right_meta:
            return []
        # else: neither clearly meta — keep full text for splitting

    # Strip parenthetical qualifiers: (formal), (used before nouns), etc.
    text = _PAREN.sub("", text)
    text = clean_text(text)

    # Remove inline quoted source words
    text = re.sub(r'"[^"]{1,30}"', "", text)
    text = clean_text(text)

    if not text:
        return []

    # Split on common delimiters
    raw_parts = re.split(r"\s*[;,/]\s*", text)

    clean_parts: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        part = clean_text(part)
        if not part:
            continue
        if is_segment_grammar_prose(part):
            continue
        key = part.lower()
        if key not in seen:
            seen.add(key)
            clean_parts.append(part)

    return clean_parts[:max_options]


# ── Sanitizer ─────────────────────────────────────────────────────────────────

def is_flashcard_safe_english(
    text: str, lemma: str, original_lemma: str, pos: str
) -> bool:
    """
    Return True only if `text` is safe to write as the English side of a flashcard.
    Hard rejects: blank, too long, echoes lemma, contains grammar prose,
    contains form-of language, bad domain, no alphabetic content.
    """
    if not text or not text.strip():
        return False
    if len(text) > 70:
        return False
    if contains_lemma(text, lemma, original_lemma):
        return False

    lowered = text.lower()

    if any(p in lowered for p in FORM_OF_PATTERNS):
        return False
    if any(p in lowered for p in META_PROSE_PATTERNS):
        return False
    if is_bad_domain(lowered):
        return False
    if not re.search(r"[a-zA-Z]", text):
        return False

    return True


# ── Stage A: collect_candidates ───────────────────────────────────────────────

def collect_candidates(
    lemma: str, rank: Optional[int], entries: list[dict]
) -> list[dict]:
    """
    Stage A: Extract all plausible sense candidates from Wiktionary entries.
    One candidate dict per (entry, sense) pair. Bucket and score fields
    are left None/0 to be filled in Stage B.
    """
    candidates: list[dict] = []
    for entry in entries:
        pos = norm_space(entry.get("pos", "")).lower()
        if not pos or pos in SKIP_POS:
            continue
        senses = entry.get("senses") or []
        if not isinstance(senses, list):
            continue
        for sense in senses:
            if not isinstance(sense, dict):
                continue
            tags = get_tags(sense)
            if tags & BANNED_SENSE_TAGS:
                continue
            glosses          = get_glosses(sense)
            raw_translations = extract_english_translations(sense, max_items=8)
            original_lemma   = get_original_lemma_from_sense(sense) or lemma
            is_form          = has_form_link(sense) or any(
                looks_like_form_of(g) for g in glosses
            )
            candidates.append({
                "lemma":           lemma,
                "rank":            rank,
                "pos":             pos,
                "sense":           sense,
                "tags":            tags,
                "glosses":         glosses,
                "raw_translations": raw_translations,
                "original_lemma":  original_lemma,
                "is_form":         is_form,
                # filled in Stage B:
                "bucket":          None,
                "total_score":     0.0,
            })
    return candidates


# ── Stage B: classify bucket + score ──────────────────────────────────────────

def classify_candidate_bucket(candidate: dict) -> str:
    """
    Stage B: Assign a semantic bucket to a candidate.
    The bucket controls which rank-aware priority table applies.
    """
    pos             = candidate["pos"]
    glosses         = candidate["glosses"]
    raw_translations = candidate["raw_translations"]
    is_form         = candidate["is_form"]

    all_text = " ".join(glosses + raw_translations).lower()

    # Junk first: bad domains / letter names
    if is_bad_domain(all_text):
        return BUCKET_JUNK
    if any(p in all_text for p in (
        "letter of the", "the letter ", "latin script letter",
        "letter name", "musical note", "solfège", "solfege",
    )):
        return BUCKET_JUNK

    # Meta: every gloss is form-of/grammar prose AND no direct English translation
    if _all_glosses_are_meta(glosses) and not raw_translations:
        return BUCKET_META

    if pos in {"article", "preposition", "conjunction", "particle", "contraction"}:
        return BUCKET_FUNCTION
    if pos == "pronoun":
        return BUCKET_PRONOUN
    if pos in {"determiner", "num", "numeral"}:
        return BUCKET_DETERMINER
    if pos == "verb":
        return BUCKET_VERB_FORM if is_form else BUCKET_VERB_LEMMA

    # Noun, adj, adv, interjection, phrase, proper noun → content
    if _all_glosses_are_meta(glosses) and not raw_translations:
        return BUCKET_META
    return BUCKET_CONTENT


def score_candidate(candidate: dict) -> float:
    """Score combining rank-aware bucket priority with within-bucket quality."""
    rank            = candidate["rank"]
    bucket          = candidate["bucket"]
    tags            = candidate["tags"]
    pos             = candidate["pos"]
    is_form         = candidate["is_form"]
    raw_translations = candidate["raw_translations"]
    glosses         = candidate["glosses"]

    score = _bucket_priority_table(rank).get(bucket, 0.0)

    # Quality within bucket
    score -= 6.0 * len(tags & LOW_PRIORITY_SENSE_TAGS)

    if raw_translations:
        score += 30.0
    elif glosses and not _all_glosses_are_meta(glosses):
        score += 12.0
    else:
        score -= 20.0

    # Penalise proper nouns heavily for high-frequency words
    if pos == "proper noun":
        score -= (50.0 if (rank is not None and rank <= 2000) else 15.0)

    # Verb lemma forms are slightly preferred over verb form entries
    if bucket == BUCKET_VERB_LEMMA:
        score += 5.0

    return score


def _pick_best_candidate(candidates: list[dict]) -> Optional[dict]:
    return max(candidates, key=lambda c: c["total_score"]) if candidates else None


def _pick_ranked_candidates(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda c: -c["total_score"])


# ── Verb utilities ────────────────────────────────────────────────────────────

def _third_person_s(base: str) -> str:
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ies"
    if base.endswith(("o", "ch", "sh", "s", "x", "z")):
        return base + "es"
    return base + "s"


def _regular_past(base: str) -> str:
    if base.endswith("e"):
        return base + "d"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ied"
    return base + "ed"


def _regular_ing(base: str) -> str:
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
        return base[:-1] + "ing"
    return base + "ing"


def _pronouns_for(person: Optional[str], number: Optional[str]) -> list[str]:
    table = {
        ("1", "sg"): ["I"],
        ("2", "sg"): ["you"],
        ("3", "sg"): ["he", "she", "it"],
        ("1", "pl"): ["we"],
        ("2", "pl"): ["you"],
        ("3", "pl"): ["they"],
    }
    return table.get((person, number), [])


def parse_verb_features(sense: dict) -> dict:
    tags    = get_tags(sense)
    glosses = get_glosses(sense)
    text    = " ".join(sorted(tags)) + " " + " ".join(glosses).lower()

    person = (
        "1" if "first-person"  in text else
        "2" if "second-person" in text else
        "3" if "third-person"  in text else None
    )
    number = (
        "sg" if "singular" in text else
        "pl" if "plural"   in text else None
    )
    form = "finite"
    if "past participle" in text:
        form = "participle"
    elif "gerund" in text or "present participle" in text:
        form = "gerund"
    elif "infinitive" in text:
        form = "infinitive"

    tense = (
        "preterite"   if "preterite"   in text else
        "imperfect"   if "imperfect"   in text else
        "future"      if "future"      in text else
        "conditional" if "conditional" in text else
        "present"     if "present"     in text else None
    )
    return {"person": person, "number": number, "form": form, "tense": tense}


def _normalize_to_base_english_verb(text: str) -> str:
    """Extract the bare verb root from glosses like 'to be', 'know', 'to make; do'."""
    text = _PAREN.sub("", clean_text(text)).strip()
    first = re.split(r"[;,/]", text)[0].strip()
    if first.lower().startswith("to "):
        first = first[3:].strip()
    return norm_space(first).lower()


def _english_head_nonfinite(head: str, form: str) -> str:
    irr = IRREGULAR_ENGLISH_FORMS.get(head, {})
    if form == "participle":
        return irr.get(("participle", None, None), _regular_past(head))
    if form == "gerund":
        return irr.get(("gerund", None, None), _regular_ing(head))
    return f"to {head}"


def _english_head_finite(
    head: str,
    tense: Optional[str],
    person: Optional[str],
    number: Optional[str],
) -> str:
    tense = tense or "present"
    if tense == "future":
        return f"will {head}"
    if tense == "conditional":
        return f"would {head}"
    irr   = IRREGULAR_ENGLISH_FORMS.get(head, {})
    exact = irr.get((tense, person, number))
    if exact:
        return exact
    generic = irr.get((tense, None, None))
    if generic:
        return generic
    if tense in {"preterite", "imperfect"}:
        return _regular_past(head)
    if tense == "present" and person == "3" and number == "sg":
        return _third_person_s(head)
    return head


def _infer_base_english_verb(original_lemma: str, grouped: dict) -> str:
    """Look up the English infinitive for `original_lemma` in the Wiktionary index."""
    for entry in grouped.get(original_lemma, []):
        if norm_space(entry.get("pos", "")).lower() != "verb":
            continue
        for sense in entry.get("senses") or []:
            if not isinstance(sense, dict):
                continue
            if get_tags(sense) & BANNED_SENSE_TAGS:
                continue
            if has_form_link(sense):
                continue
            for tr in extract_english_translations(sense, max_items=4):
                base = _normalize_to_base_english_verb(tr)
                if base and 1 <= len(base.split()) <= 3:
                    return base
            for g in get_glosses(sense):
                if looks_like_form_of(g) or is_meta_prose(g):
                    continue
                base = _normalize_to_base_english_verb(g)
                if base and 1 <= len(base.split()) <= 3:
                    return base
    return ""


# ── Stage C: generate flashcard translation ───────────────────────────────────

def _generate_verb_translation(candidate: dict, grouped: dict) -> str:
    """
    Verb-specific generator.

    verb_lemma (infinitive base): return "to {base}"  e.g. ser → to be
    verb_form  (inflected form):  return "I go; we go" etc.
    """
    lemma          = candidate["lemma"]
    original_lemma = candidate["original_lemma"]
    is_form        = candidate["is_form"]
    bucket         = candidate["bucket"]
    sense          = candidate["sense"]

    # ── Infinitive / lemma form ──────────────────────────────────────────────
    if bucket == BUCKET_VERB_LEMMA:
        bases: list[str] = []
        seen_bases: set[str] = set()

        for tr in candidate["raw_translations"]:
            parts = compact_to_flashcard_english(tr, "verb", max_options=2)
            for p in parts:
                base = _normalize_to_base_english_verb(p)
                if base and base not in seen_bases:
                    seen_bases.add(base)
                    bases.append(base)

        if not bases:
            for g in candidate["glosses"]:
                if looks_like_form_of(g) or is_meta_prose(g):
                    continue
                base = _normalize_to_base_english_verb(g)
                if base and base not in seen_bases:
                    seen_bases.add(base)
                    bases.append(base)

        if not bases:
            return ""

        options = [f"to {b}" for b in bases[:2]]
        return "; ".join(options)

    # ── Inflected / non-finite form ───────────────────────────────────────────
    base = _infer_base_english_verb(original_lemma, grouped)
    if not base:
        # Fallback: use direct options if available
        direct: list[str] = []
        for tr in candidate["raw_translations"][:3]:
            parts = compact_to_flashcard_english(tr, "verb", max_options=2)
            direct.extend(parts)
        return "; ".join(direct[:3]) if direct else ""

    head, *tail_parts = base.split(" ", 1)
    tail = tail_parts[0] if tail_parts else ""

    def join_ht(h: str, t: str) -> str:
        return f"{h} {t}".strip() if t else h

    features = parse_verb_features(sense)

    if features["form"] != "finite":
        return join_ht(_english_head_nonfinite(head, features["form"]), tail)

    person = features["person"]
    number = features["number"]
    if person is None or number is None:
        # Can't produce a pronoun+verb, use direct options
        direct = []
        for tr in candidate["raw_translations"][:2]:
            direct.extend(compact_to_flashcard_english(tr, "verb", max_options=2))
        if direct:
            return "; ".join(direct[:3])
        return join_ht(_english_head_finite(head, features["tense"], person, number), tail)

    finite   = join_ht(_english_head_finite(head, features["tense"], person, number), tail)
    pronouns = _pronouns_for(person, number)
    if not pronouns:
        return finite
    return "; ".join(f"{p} {finite}" for p in pronouns)


def generate_flashcard_translation(candidate: dict, grouped: dict) -> str:
    """
    Stage C: Generate a compact, flashcard-safe English translation.
    Verbs use the verb-specific generator.
    All other POS use the compactor on translations then glosses.
    Falls back to base-lemma lookup if the form has no direct content.
    """
    pos            = candidate["pos"]
    lemma          = candidate["lemma"]
    original_lemma = candidate["original_lemma"]

    if pos == "verb":
        return _generate_verb_translation(candidate, grouped)

    options: list[str] = []
    seen: set[str] = set()

    for text in candidate["raw_translations"]:
        for p in compact_to_flashcard_english(text, pos):
            if p.lower() not in seen and is_flashcard_safe_english(p, lemma, original_lemma, pos):
                seen.add(p.lower())
                options.append(p)

    if not options:
        for text in candidate["glosses"]:
            if looks_like_form_of(text):
                continue
            for p in compact_to_flashcard_english(text, pos):
                if p.lower() not in seen and is_flashcard_safe_english(p, lemma, original_lemma, pos):
                    seen.add(p.lower())
                    options.append(p)

    # Base-lemma fallback for inflected non-verb forms
    if not options and original_lemma and original_lemma != lemma:
        base_entries = grouped.get(original_lemma, [])
        if base_entries:
            base_cands = collect_candidates(original_lemma, None, base_entries)
            for bc in base_cands:
                bc["bucket"]      = classify_candidate_bucket(bc)
                bc["total_score"] = score_candidate(bc)
            best_base = _pick_best_candidate(base_cands)
            if best_base is not None:
                base_tr = generate_flashcard_translation(best_base, grouped)
                if base_tr and is_flashcard_safe_english(base_tr, lemma, original_lemma, pos):
                    return base_tr

    # Prefer shorter options (more compact = better flashcard)
    options.sort(key=lambda x: (len(x.split()), len(x)))
    return "; ".join(options[:4])


# ── Stage D: QA + repair ──────────────────────────────────────────────────────

_VERB_INDICATOR_RE = re.compile(
    r"\bto\s+\w|\b(am|is|are|was|were|be|been|being|"
    r"do|does|did|done|doing|have|has|had|having|"
    r"go|goes|went|gone|going|"
    r"will|would|can|could|shall|should|may|might|must)\b"
)

_HARD_FLAGS = {"blank", "echo", "meta", "rare_domain", "letter"}


def flag_translation(
    lemma: str, rank: Optional[int], translation: str, pos: str
) -> list[str]:
    """Return a list of QA flag strings for a (lemma, translation, pos) triple."""
    flags: list[str] = []

    if not translation or not translation.strip():
        flags.append("blank")
        return flags

    lowered = translation.lower()

    if contains_lemma(translation, lemma, ""):
        flags.append("echo")

    if (any(p in lowered for p in FORM_OF_PATTERNS)
            or any(p in lowered for p in META_PROSE_PATTERNS)):
        flags.append("meta")

    if len(translation) > 65:
        flags.append("long")

    if is_bad_domain(lowered):
        flags.append("rare_domain")

    if ("letter" in lowered and
            any(p in lowered for p in ("script letter", "alphabet", "the letter"))):
        flags.append("letter")

    if rank is not None and rank <= 1000 and pos == "proper noun":
        flags.append("bad_rank_pos")

    if pos == "verb" and not _VERB_INDICATOR_RE.search(lowered):
        flags.append("suspicious_verb")

    return flags


def repair_flagged_row(
    lemma: str,
    rank: Optional[int],
    flags: list[str],
    all_candidates: list[dict],
    grouped: dict,
    original_translation: str,
) -> tuple[str, str]:
    """
    Stage D repair: try every candidate in score order to find a clean translation.
    Skips the candidate that produced `original_translation`.
    Returns (repaired_translation, repaired_pos) — empty strings if repair fails.
    """
    if not all_candidates:
        return "", ""

    ranked = _pick_ranked_candidates(all_candidates)

    for candidate in ranked:
        tr = generate_flashcard_translation(candidate, grouped)
        if not tr or tr == original_translation:
            continue
        if not is_flashcard_safe_english(tr, lemma, candidate["original_lemma"], candidate["pos"]):
            continue
        row_flags = flag_translation(lemma, rank, tr, candidate["pos"])
        if not (set(row_flags) - {"suspicious_verb"}):
            return tr, candidate["pos"]

    # Aggressive fallback: try raw glosses with minimal sanitization
    for candidate in ranked:
        for gloss in candidate["glosses"]:
            if looks_like_form_of(gloss) or is_meta_prose(gloss):
                continue
            cleaned = _PAREN.sub("", gloss).strip(" ;,.")
            cleaned = re.sub(r"\s+", " ", cleaned)
            if (cleaned == original_translation
                    or not cleaned
                    or len(cleaned) > 60):
                continue
            if contains_lemma(cleaned, lemma, ""):
                continue
            gflags = flag_translation(lemma, rank, cleaned, candidate["pos"])
            if not (set(gflags) & {"meta", "echo", "blank", "rare_domain", "letter"}):
                return cleaned, candidate["pos"]

    return "", ""


def run_qa(
    rows: list[dict],
    grouped: dict,
    all_candidates_map: dict[str, list[dict]],
) -> list[dict]:
    """
    Stage D: QA + repair pass over all generated rows.
    Mutates rows in-place (updates translation/pos for repaired rows).
    Returns a list of diagnostic dicts (one per row).
    """
    diagnostics: list[dict] = []

    for row in tqdm(rows, desc="QA + repair", unit="row"):
        lemma       = row["lemma"]
        rank        = row["rank"]
        translation = row.get("translation", "")
        pos         = row.get("pos", "")

        # Overrides are pre-verified — skip QA
        if lemma in TINY_OVERRIDES:
            diagnostics.append({
                "lemma": lemma, "rank": rank,
                "translation_pass1": translation, "translation_pass2": "",
                "translation_final": translation, "pos_final": pos,
                "flags_pass1": "", "flags_pass2": "", "pass": 1, "repaired": False,
            })
            continue

        flags = flag_translation(lemma, rank, translation, pos)

        if not flags:
            diagnostics.append({
                "lemma": lemma, "rank": rank,
                "translation_pass1": translation, "translation_pass2": "",
                "translation_final": translation, "pos_final": pos,
                "flags_pass1": "", "flags_pass2": "", "pass": 1, "repaired": False,
            })
            continue

        # Attempt repair
        repaired_tr, repaired_pos = repair_flagged_row(
            lemma, rank, flags,
            all_candidates_map.get(lemma, []),
            grouped,
            translation,
        )

        if repaired_tr:
            repair_flags = flag_translation(lemma, rank, repaired_tr, repaired_pos)
            row["translation"] = repaired_tr
            row["pos"]         = repaired_pos
            diagnostics.append({
                "lemma": lemma, "rank": rank,
                "translation_pass1": translation,
                "translation_pass2": repaired_tr,
                "translation_final": repaired_tr,
                "pos_final":         repaired_pos,
                "flags_pass1":       ",".join(flags),
                "flags_pass2":       ",".join(repair_flags),
                "pass": 2, "repaired": True,
            })
        else:
            # Only wipe translation if the original has a hard flag
            if set(flags) & _HARD_FLAGS:
                row["translation"] = ""
                row["pos"]         = ""
                final_tr, final_pos = "", ""
            else:
                # Soft flags only (long, suspicious_verb, bad_rank_pos) — keep original
                final_tr, final_pos = translation, pos

            diagnostics.append({
                "lemma": lemma, "rank": rank,
                "translation_pass1": translation,
                "translation_pass2": "",
                "translation_final": final_tr,
                "pos_final":         final_pos,
                "flags_pass1":       ",".join(flags),
                "flags_pass2":       "blank" if not final_tr else ",".join(flags),
                "pass": 2, "repaired": False,
            })

    return diagnostics


# ── I/O utilities ─────────────────────────────────────────────────────────────

def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def load_input_rows(path: Path, limit: Optional[int]) -> list[dict]:
    total = _count_csv_rows(path)
    if limit is not None:
        total = min(total, limit)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "lemma" not in reader.fieldnames:
            raise ValueError("Input CSV must have a 'lemma' column")
        rows: list[dict] = []
        for i, row in enumerate(
            tqdm(reader, total=total, desc="Loading lemmas", unit="row")
        ):
            if limit is not None and i >= limit:
                break
            lemma    = norm_space(row.get("lemma", ""))
            rank_raw = norm_space(row.get("rank", ""))
            rank     = int(rank_raw) if rank_raw.isdigit() else None
            if lemma:
                rows.append({"lemma": lemma, "rank": rank})
    return rows


def _open_bytes(path: Path):
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, "rb")
    return path.open("rb")


def index_entries(
    wiktextract_path: Path,
    wanted_words: set[str],
    lang_code: str,
    desc: str,
    expand_refs: bool = False,
) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    total_bytes = (
        None if wiktextract_path.suffix == ".gz"
        else wiktextract_path.stat().st_size
    )
    matched   = 0
    lang_code = lang_code.lower()

    with _open_bytes(wiktextract_path) as f:
        with tqdm(
            total=total_bytes, desc=desc, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for line in f:
                if total_bytes is not None:
                    pbar.update(len(line))
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if (not isinstance(obj.get("lang_code"), str)
                        or obj["lang_code"].lower() != lang_code):
                    continue
                word = obj.get("word")
                if not isinstance(word, str) or word not in wanted_words:
                    continue
                index[word].append(obj)
                matched += 1
                if expand_refs:
                    for sense in obj.get("senses") or []:
                        if isinstance(sense, dict):
                            linked = get_original_lemma_from_sense(sense)
                            if linked:
                                wanted_words.add(linked)
                if matched % 1000 == 0:
                    pbar.set_postfix(matches=matched, lemmas=len(index))

    return dict(index)


def _collect_referenced_lemmas(grouped: dict) -> set[str]:
    refs: set[str] = set()
    for entries in grouped.values():
        for entry in entries:
            for sense in entry.get("senses") or []:
                if isinstance(sense, dict):
                    linked = get_original_lemma_from_sense(sense)
                    if linked:
                        refs.add(linked)
    return refs


def write_output(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lemma", "rank", "translation", "pos"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "lemma":       row["lemma"],
                "rank":        "" if row["rank"] is None else row["rank"],
                "translation": row.get("translation", ""),
                "pos":         row.get("pos", ""),
            })


def write_diagnostics(path: Path, diagnostics: list[dict]) -> None:
    fieldnames = [
        "lemma", "rank",
        "translation_pass1", "translation_pass2", "translation_final",
        "pos_final", "flags_pass1", "flags_pass2", "pass", "repaired",
    ]
    # Flagged rows first, then by ascending rank (most important words at top)
    def sort_key(d: dict):
        flagged = 1 if d.get("flags_pass1") else 0
        r = d["rank"] if d["rank"] is not None else 999_999
        return (-flagged, r)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in sorted(diagnostics, key=sort_key):
            writer.writerow({k: d.get(k, "") for k in fieldnames})


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build flashcard-safe English translations for Spanish lemmas"
    )
    parser.add_argument("--input",       required=True, type=Path,
                        help="CSV with lemma[,rank] columns")
    parser.add_argument("--wiktextract", required=True, type=Path,
                        help="Wiktextract JSONL or JSONL.GZ")
    parser.add_argument("--output",      required=True, type=Path,
                        help="Output CSV (lemma, rank, translation, pos)")
    parser.add_argument("--diagnostics", type=Path, default=None,
                        help="Diagnostics CSV (default: <output-stem>_diagnostics.csv)")
    parser.add_argument("--lang-code",   default="es")
    parser.add_argument("--limit",       type=int, default=None)
    args = parser.parse_args()

    diag_path = args.diagnostics or args.output.with_name(
        args.output.stem + "_diagnostics" + args.output.suffix
    )

    # ── Stage 0: load input ───────────────────────────────────────────────────
    rows         = load_input_rows(args.input, args.limit)
    wanted_words = {row["lemma"] for row in rows}
    print(f"Loaded {len(rows)} rows, {len(wanted_words)} unique lemmas")

    # ── Index entries (pass 1, expand referenced lemmas) ─────────────────────
    grouped = index_entries(
        args.wiktextract, wanted_words, args.lang_code,
        "Scanning entries (with ref expansion)", expand_refs=True,
    )

    missing_refs = _collect_referenced_lemmas(grouped) - set(grouped.keys())
    if missing_refs:
        extra = index_entries(
            args.wiktextract, missing_refs, args.lang_code,
            "Scanning referenced lemmas", expand_refs=False,
        )
        for word, entries in extra.items():
            grouped.setdefault(word, []).extend(entries)

    # ── Stages A + B: collect and classify candidates for every lemma ─────────
    all_candidates_map: dict[str, list[dict]] = {}

    for row in tqdm(rows, desc="Stage A+B: classify candidates", unit="lemma"):
        lemma = row["lemma"]
        if lemma in TINY_OVERRIDES:
            all_candidates_map[lemma] = []
            continue
        candidates = collect_candidates(lemma, row["rank"], grouped.get(lemma, []))
        for c in candidates:
            c["bucket"]      = classify_candidate_bucket(c)
            c["total_score"] = score_candidate(c)
        all_candidates_map[lemma] = candidates

    # ── Stage C: generate translations (pass 1) ───────────────────────────────
    for row in tqdm(rows, desc="Stage C: generating translations", unit="lemma"):
        lemma = row["lemma"]

        if lemma in TINY_OVERRIDES:
            tr, pos        = TINY_OVERRIDES[lemma]
            row["translation"] = tr
            row["pos"]         = pos
            continue

        best = _pick_best_candidate(all_candidates_map.get(lemma, []))
        if best is None:
            row["translation"] = ""
            row["pos"]         = ""
            continue

        tr = generate_flashcard_translation(best, grouped)
        if not is_flashcard_safe_english(tr, lemma, best["original_lemma"], best["pos"]):
            tr = ""

        row["translation"] = tr
        row["pos"]         = best["pos"] if tr else ""

    # ── Stage D: QA + repair ─────────────────────────────────────────────────
    diagnostics = run_qa(rows, grouped, all_candidates_map)

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_output(args.output, rows)
    write_diagnostics(diag_path, diagnostics)

    # ── Summary ───────────────────────────────────────────────────────────────
    total      = len(rows)
    overridden = sum(1 for r in rows if r["lemma"] in TINY_OVERRIDES)
    filled     = sum(1 for r in rows if r.get("translation"))
    blank      = total - filled
    flagged_p1 = sum(1 for d in diagnostics if d["flags_pass1"])
    repaired   = sum(1 for d in diagnostics if d["repaired"])

    print(f"\n{'─' * 50}")
    print(f"Total rows:        {total:>7}")
    print(f"Overridden:        {overridden:>7}")
    print(f"Filled:            {filled:>7}  ({100 * filled / total:.1f}%)")
    print(f"Blank:             {blank:>7}  ({100 * blank / total:.1f}%)")
    print(f"Flagged (pass 1):  {flagged_p1:>7}")
    print(f"Repaired:          {repaired:>7}")
    print(f"Output:            {args.output}")
    print(f"Diagnostics:       {diag_path}")


if __name__ == "__main__":
    main()
