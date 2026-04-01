#!/usr/bin/env python3
"""
rebuild_artifacts.py — Corpus Training Pipeline for Large Corpora
==================================================================
Rebuilds all pickle artifacts from a plain-text corpus (one sentence
per line).  Designed for corpora up to ~50M sentences / 10+ GB.

Two-pass architecture:
  Pass 1 (fast, no spaCy):  n-grams, lemma_contexts, lemma_forms
          Uses regex tokenisation + heuristic lemmatisation.
          ~5-15 min for 50M sentences.

  Pass 2 (selective spaCy): enriched_lexicon + morph backfill
          Runs spaCy only on a reservoir sample of sentences per
          lexicon lemma, for gender/verb-type/prep evidence.
          ~30-60 min depending on sample depth.

REQUIRES:  Python 3.9+, spacy, tqdm
SETUP:     pip install spacy tqdm && python -m spacy download es_core_news_lg

USAGE:
    python rebuild_artifacts.py \
        --corpus models/corpus.txt \
        --lexicon stg_words_spa.csv \
        --outdir models/ \
        --context-reservoir-size 50 \
        --spacy-sample-per-lemma 100

OUTPUTS (all in --outdir):
    lemma_forms.pkl         lemma -> [{form, morph}, ...]
    lemma_contexts.pkl      lemma -> [{sentence, tokens, index, left, right, ...}, ...]
    bigrams.pkl             {next, totals}
    trigrams.pkl            {next, totals}
    pos_transitions.pkl     {next, totals}
    enriched_lexicon.pkl    lemma -> {gender, verb_type, semantic_class, ...}

FLAGS:
    --skip-spacy            Skip pass 2; reuse existing enriched_lexicon.pkl.
                            Useful for fast iteration on n-gram quality.

EXPECTED TIME (50M sentences, M2 MacBook):
    Pass 1:  ~10-20 min
    Pass 2:  ~30-60 min  (or ~0 with --skip-spacy)
EXPECTED RAM:  ~4-6 GB peak (n-gram counters dominate)
"""

import argparse
import csv
import os
import pickle
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ===================================================================
# UTILITIES
# ===================================================================

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
TOKEN_RE = re.compile(
    r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+|[¿¡.,;:!?\"\"\'()\[\]{}…—–\-]"
)
BAD_LINE_RE = re.compile(r"https?://|www\.|@|\||\x00")


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def norm(s: str) -> str:
    return s.strip().lower()


def norm_key(s: str) -> str:
    return strip_accents(norm(s))


def is_word(token: str) -> bool:
    return bool(WORD_RE.fullmatch(token))


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def reservoir_add(reservoir: list, item: Any, max_size: int, n_seen: int) -> None:
    if len(reservoir) < max_size:
        reservoir.append(item)
    else:
        j = random.randint(0, n_seen - 1)
        if j < max_size:
            reservoir[j] = item


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        buf = f.raw.read(1 << 20)
        while buf:
            n += buf.count(b"\n")
            buf = f.raw.read(1 << 20)
    return n


# ===================================================================
# HEURISTIC LEMMATISER
# ===================================================================

_IRREGULAR = {
    "soy": "ser", "eres": "ser", "es": "ser", "somos": "ser", "son": "ser",
    "fui": "ir", "fue": "ser", "fuimos": "ser", "fueron": "ser",
    "era": "ser", "eras": "ser", "éramos": "ser", "eran": "ser",
    "sea": "ser", "seas": "ser", "seamos": "ser", "sean": "ser",
    "sido": "ser", "siendo": "ser", "sería": "ser", "serían": "ser",
    "será": "ser", "serás": "ser", "serán": "ser",
    "fuera": "ser", "fuese": "ser", "fueras": "ser",
    "estoy": "estar", "estás": "estar", "está": "estar",
    "estamos": "estar", "están": "estar",
    "estaba": "estar", "estabas": "estar", "estábamos": "estar", "estaban": "estar",
    "estuvo": "estar", "estuve": "estar", "estuvieron": "estar",
    "esté": "estar", "estés": "estar", "estén": "estar",
    "estaré": "estar", "estará": "estar", "estarán": "estar",
    "estaría": "estar", "estarían": "estar",
    "he": "haber", "has": "haber", "ha": "haber", "hay": "haber",
    "hemos": "haber", "han": "haber",
    "había": "haber", "habían": "haber", "hubo": "haber",
    "haya": "haber", "hayas": "haber", "hayan": "haber",
    "hubiera": "haber", "hubiese": "haber",
    "habrá": "haber", "habría": "haber", "habrían": "haber",
    "voy": "ir", "vas": "ir", "va": "ir", "vamos": "ir", "van": "ir",
    "iba": "ir", "ibas": "ir", "íbamos": "ir", "iban": "ir",
    "vaya": "ir", "vayas": "ir", "vayan": "ir",
    "iré": "ir", "irás": "ir", "irá": "ir", "irán": "ir",
    "iría": "ir", "irían": "ir", "ido": "ir",
    "tengo": "tener", "tienes": "tener", "tiene": "tener",
    "tenemos": "tener", "tienen": "tener",
    "tenía": "tener", "tenías": "tener", "tenían": "tener",
    "tuvo": "tener", "tuve": "tener", "tuvieron": "tener",
    "tenga": "tener", "tengas": "tener", "tengan": "tener",
    "tendré": "tener", "tendrá": "tener", "tendrán": "tener",
    "tendría": "tener", "tendrían": "tener", "tenido": "tener",
    "hago": "hacer", "haces": "hacer", "hace": "hacer",
    "hacemos": "hacer", "hacen": "hacer",
    "hacía": "hacer", "hacían": "hacer",
    "hizo": "hacer", "hice": "hacer", "hiciste": "hacer", "hicieron": "hacer",
    "haga": "hacer", "hagas": "hacer", "hagan": "hacer",
    "haré": "hacer", "hará": "hacer", "harán": "hacer",
    "haría": "hacer", "harían": "hacer", "hecho": "hacer", "haciendo": "hacer",
    "puedo": "poder", "puedes": "poder", "puede": "poder",
    "podemos": "poder", "pueden": "poder",
    "podía": "poder", "podían": "poder",
    "pudo": "poder", "pude": "poder", "pudieron": "poder",
    "pueda": "poder", "puedan": "poder",
    "podré": "poder", "podrá": "poder", "podrán": "poder",
    "podría": "poder", "podrían": "poder", "podrías": "poder",
    "pudiera": "poder", "pudiese": "poder",
    "quiero": "querer", "quieres": "querer", "quiere": "querer",
    "queremos": "querer", "quieren": "querer",
    "quería": "querer", "querían": "querer",
    "quiso": "querer", "quise": "querer", "quisieron": "querer",
    "quiera": "querer", "quieran": "querer", "quieras": "querer",
    "querría": "querer", "querrían": "querer",
    "quisiera": "querer", "quisiese": "querer", "querido": "querer",
    "digo": "decir", "dices": "decir", "dice": "decir",
    "decimos": "decir", "dicen": "decir",
    "decía": "decir", "decían": "decir",
    "dijo": "decir", "dije": "decir", "dijiste": "decir", "dijeron": "decir",
    "diga": "decir", "digas": "decir", "digan": "decir",
    "diré": "decir", "dirá": "decir", "dirán": "decir",
    "diría": "decir", "dirían": "decir", "dicho": "decir", "diciendo": "decir",
    "sé": "saber", "sabes": "saber", "sabe": "saber",
    "sabemos": "saber", "saben": "saber",
    "sabía": "saber", "sabían": "saber",
    "supo": "saber", "supe": "saber", "supieron": "saber",
    "sepa": "saber", "sepan": "saber",
    "sabré": "saber", "sabrá": "saber",
    "sabría": "saber", "sabrían": "saber", "sabido": "saber",
    "pongo": "poner", "pones": "poner", "pone": "poner",
    "ponemos": "poner", "ponen": "poner",
    "puso": "poner", "puse": "poner", "pusieron": "poner",
    "ponga": "poner", "pongan": "poner",
    "pondré": "poner", "pondrá": "poner",
    "pondría": "poner", "pondrían": "poner", "puesto": "poner",
    "vengo": "venir", "vienes": "venir", "viene": "venir",
    "venimos": "venir", "vienen": "venir",
    "venía": "venir", "venían": "venir",
    "vino": "venir", "vine": "venir", "vinieron": "venir",
    "venga": "venir", "vengan": "venir",
    "vendré": "venir", "vendrá": "venir",
    "vendría": "venir", "vendrían": "venir", "venido": "venir",
    "salgo": "salir", "sales": "salir", "sale": "salir",
    "salimos": "salir", "salen": "salir",
    "salió": "salir", "salí": "salir", "salieron": "salir",
    "salga": "salir", "salgan": "salir",
    "saldré": "salir", "saldrá": "salir",
    "saldría": "salir", "saldrían": "salir", "salido": "salir",
    "doy": "dar", "das": "dar", "da": "dar", "damos": "dar", "dan": "dar",
    "dio": "dar", "di": "dar", "dieron": "dar",
    "dé": "dar", "den": "dar",
    "daré": "dar", "dará": "dar", "daría": "dar", "dado": "dar",
    "veo": "ver", "ves": "ver", "ve": "ver", "vemos": "ver", "ven": "ver",
    "veía": "ver", "veían": "ver",
    "vio": "ver", "vi": "ver", "vieron": "ver",
    "vea": "ver", "vean": "ver",
    "veré": "ver", "verá": "ver",
    "vería": "ver", "verían": "ver", "visto": "ver", "viendo": "ver",
    "conozco": "conocer", "conoce": "conocer", "conocen": "conocer",
    "duermo": "dormir", "duerme": "dormir", "duermen": "dormir",
    "pido": "pedir", "pide": "pedir", "piden": "pedir",
    "pienso": "pensar", "piensa": "pensar", "piensan": "pensar",
    "encuentro": "encontrar", "encuentra": "encontrar", "encuentran": "encontrar",
    "vuelvo": "volver", "vuelve": "volver", "vuelven": "volver",
    "siento": "sentir", "siente": "sentir", "sienten": "sentir",
    "muero": "morir", "muere": "morir", "mueren": "morir",
    "murió": "morir", "muerto": "morir",
    "creo": "creer", "cree": "creer", "creen": "creer",
    "creí": "creer", "creyó": "creer", "creído": "creer",
}


def heuristic_lemma(form: str) -> str:
    lo = form.lower()
    hit = _IRREGULAR.get(lo)
    if hit:
        return hit
    for suf, repl in [
        ("ando", "ar"), ("iendo", "ir"), ("endo", "er"),
        ("ado", "ar"), ("ido", "ir"),
        ("aba", "ar"), ("ía", "er"),
        ("amos", "ar"), ("emos", "er"), ("imos", "ir"),
        ("an", "ar"), ("en", "er"),
        ("ará", "ar"), ("erá", "er"), ("irá", "ir"),
        ("aría", "ar"), ("ería", "er"), ("iría", "ir"),
        ("ó", "ar"),
    ]:
        if lo.endswith(suf) and len(lo) > len(suf) + 1:
            return lo[: -len(suf)] + repl
    if lo.endswith("ces") and len(lo) > 4:
        return lo[:-3] + "z"
    if lo.endswith("iones") and len(lo) > 6:
        return lo[:-5] + "ión"
    if lo.endswith("es") and len(lo) > 3:
        return lo[:-2]
    if lo.endswith("s") and len(lo) > 2:
        return lo[:-1]
    return lo


_FUNC_POS: Dict[str, str] = {}
for _w in ("el", "la", "los", "las", "un", "una", "unos", "unas"):
    _FUNC_POS[_w] = "DET"
for _w in ("de", "en", "a", "por", "para", "con", "sin", "sobre",
           "entre", "hasta", "desde", "hacia", "contra", "bajo",
           "tras", "ante", "según", "durante", "mediante"):
    _FUNC_POS[_w] = "ADP"
for _w in ("y", "o", "pero", "ni", "sino", "e", "u"):
    _FUNC_POS[_w] = "CCONJ"
for _w in ("que", "porque", "cuando", "si", "como", "aunque",
           "mientras", "donde", "pues"):
    _FUNC_POS[_w] = "SCONJ"
for _w in ("yo", "tú", "él", "ella", "nosotros", "ellos", "ellas",
           "me", "te", "se", "nos", "lo", "le", "les",
           "esto", "eso", "ello", "nada", "algo", "alguien", "nadie",
           "usted", "ustedes"):
    _FUNC_POS[_w] = "PRON"
for _w in ("mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra",
           "nuestros", "nuestras", "este", "esta", "estos", "estas",
           "ese", "esa", "esos", "esas", "otro", "otra", "otros",
           "otras", "cada", "todo", "toda", "todos", "todas",
           "mucho", "mucha", "muchos", "muchas",
           "algún", "alguna", "algunos", "algunas",
           "ningún", "ninguna"):
    _FUNC_POS[_w] = "DET"
for _w in ("no", "sí", "muy", "más", "menos", "bien", "mal", "ya",
           "también", "siempre", "nunca", "aquí", "allí", "ahora",
           "hoy", "ayer", "mañana", "después", "antes", "entonces",
           "casi", "solo", "sólo", "todavía", "aún"):
    _FUNC_POS[_w] = "ADV"


def heuristic_pos(form: str) -> str:
    lo = form.lower()
    if lo in _FUNC_POS:
        return _FUNC_POS[lo]
    if not is_word(form):
        return "PUNCT"
    lem = heuristic_lemma(lo)
    if lem.endswith(("ar", "er", "ir")) and len(lem) > 3:
        return "VERB"
    if lem.endswith("mente"):
        return "ADV"
    return "NOUN"


# ===================================================================
# SEMANTIC HINTS
# ===================================================================

SEMANTIC_HINTS: Dict[str, List[str]] = {
    "person": [
        "hombre", "mujer", "persona", "gente", "hijo", "hija",
        "padre", "madre", "hermano", "hermana", "amigo", "amiga",
        "doctor", "profesor", "profesora", "estudiante", "jefe",
        "rey", "reina", "niño", "niña", "señor", "señora",
    ],
    "animal": [
        "perro", "gato", "animal", "caballo", "pez", "vaca",
        "toro", "oso", "lobo", "rata",
    ],
    "food": [
        "comida", "pan", "carne", "fruta", "arroz", "queso",
        "sopa", "pescado", "pollo", "huevo", "sal", "chocolate",
    ],
    "drink": ["agua", "leche", "vino", "cerveza", "jugo", "café", "té"],
    "place": [
        "casa", "ciudad", "calle", "escuela", "pueblo", "mundo",
        "tierra", "mar", "bosque", "iglesia", "hotel", "hospital",
        "oficina", "tienda", "parque", "playa", "campo", "plaza",
    ],
    "time": [
        "hora", "momento", "tiempo", "noche", "semana", "mes",
        "minuto", "segundo", "siglo", "tarde", "día", "año",
        "época", "mañana",
    ],
    "abstract": [
        "vida", "amor", "cosa", "parte", "forma", "idea", "verdad",
        "guerra", "paz", "muerte", "historia", "cultura", "libertad",
        "poder", "justicia", "derecho", "ley", "fe", "esperanza",
        "razón", "problema", "nación", "opinión",
    ],
    "object": [
        "libro", "mesa", "puerta", "coche", "carta", "piedra",
        "llave", "silla", "cama", "reloj", "espejo", "botella",
        "bolsa", "caja", "papel", "foto", "teléfono",
    ],
    "clothing": [
        "ropa", "camisa", "zapato", "sombrero", "vestido",
        "falda", "abrigo", "chaqueta", "pantalón",
    ],
    "text": [
        "libro", "carta", "palabra", "nombre", "historia",
        "mensaje", "noticia", "página", "periódico", "canción",
    ],
}

_NOUN_SEM: Dict[str, str] = {}
for _c, _ns in SEMANTIC_HINTS.items():
    for _n in _ns:
        _NOUN_SEM[norm_key(_n)] = _c


# ===================================================================
# PASS 1: FAST STREAMING
# ===================================================================

def pass1(
    corpus_path: str,
    lexicon_lemmas: Set[str],
    lexicon_forms_reverse: Dict[str, str],
    ctx_size: int = 50,
    min_w: int = 3,
    max_w: int = 25,
) -> Tuple[Dict, Dict, Counter, Counter, Counter, Dict, int]:
    print("=" * 60)
    print("PASS 1: Streaming n-grams + contexts + forms")
    print("=" * 60)
    print("  Counting lines...")
    total = count_lines(corpus_path)
    print(f"  {total:,} lines\n")

    bi: Counter = Counter()
    tri: Counter = Counter()
    pos_tr: Counter = Counter()
    ctx: Dict[str, List[dict]] = defaultdict(list)
    ctx_n: Dict[str, int] = defaultdict(int)
    forms: Dict[str, List[dict]] = defaultdict(list)
    form_seen: Dict[str, Set[str]] = defaultdict(set)
    spacy_samp: Dict[str, List[str]] = defaultdict(list)
    spacy_n: Dict[str, int] = defaultdict(int)
    SPACY_RES = 150
    kept = 0

    with open(corpus_path, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, total=total, desc="  Pass 1", unit=" lines",
                         mininterval=2.0):
            text = line.rstrip("\n\r").strip()
            if not text or BAD_LINE_RE.search(text):
                continue
            # Fast reject before expensive regex tokenization.
            space_count = text.count(" ")
            if space_count < min_w - 1 or space_count > max_w + 5:
                continue
            tokens = tokenize(text)
            words = [t for t in tokens if is_word(t)]
            nw = len(words)
            if nw < min_w or nw > max_w:
                continue
            kept += 1

            lo = [w.lower() for w in words]
            lems = [heuristic_lemma(w) for w in lo]
            poses = [heuristic_pos(w) for w in words]

            wl = ["<START>"] + lo + ["<END>"]
            pl = ["<START>"] + poses + ["<END>"]
            for i in range(len(wl) - 1):
                bi[(wl[i], wl[i + 1])] += 1
                pos_tr[(pl[i], pl[i + 1])] += 1
            for i in range(len(wl) - 2):
                tri[(wl[i], wl[i + 1], wl[i + 2])] += 1

            matched: Set[str] = set()
            for i, (raw, low, lem) in enumerate(zip(words, lo, lems)):
                lx = None
                if lem in lexicon_lemmas:
                    lx = lem
                elif low in lexicon_lemmas:
                    lx = low
                elif low in lexicon_forms_reverse:
                    lx = lexicon_forms_reverse[low]
                if lx is None:
                    continue

                if low not in form_seen[lx]:
                    form_seen[lx].add(low)
                    forms[lx].append({"form": low, "morph": {}})

                ctx_n[lx] += 1
                reservoir_add(ctx[lx], {
                    "sentence": text,
                    "tokens": list(lo),
                    "index": i,
                    "target_form": low,
                    "target_pos": poses[i],
                    "left": lo[i - 1] if i > 0 else "<START>",
                    "right": lo[i + 1] if i < len(lo) - 1 else "<END>",
                    "source": "corpus",
                }, ctx_size, ctx_n[lx])

                matched.add(lx)

            for lx in matched:
                spacy_n[lx] += 1
                reservoir_add(spacy_samp[lx], text, SPACY_RES, spacy_n[lx])

    print(f"\n  Kept:            {kept:,}")
    print(f"  Unique bigrams:  {len(bi):,}")
    print(f"  Unique trigrams: {len(tri):,}")
    print(f"  Lemmas w/ ctx:   {sum(1 for v in ctx.values() if v):,}")
    print(f"  Lemmas w/ forms: {sum(1 for v in forms.values() if v):,}\n")
    return dict(ctx), dict(forms), bi, tri, pos_tr, dict(spacy_samp), kept


# ===================================================================
# PASS 2: SELECTIVE spaCy
# ===================================================================

REFL_PRON = frozenset({"me", "te", "se", "nos", "os"})


def pass2(
    spacy_samp: Dict[str, List[str]],
    lex_rows: Dict[str, dict],
    forms: Dict[str, List[dict]],
    sample_n: int = 100,
) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    print("=" * 60)
    print("PASS 2: Selective spaCy for enriched lexicon")
    print("=" * 60)

    import spacy
    print("  Loading es_core_news_lg...")
    nlp = spacy.load("es_core_news_lg", disable=["ner"])
    nlp.max_length = 2_000_000
    print("  Loaded.\n")

    all_sents: List[str] = []
    sent_set: Set[str] = set()
    for sents in spacy_samp.values():
        for s in sents[:sample_n]:
            if s not in sent_set:
                sent_set.add(s)
                all_sents.append(s)
    print(f"  Unique sentences: {len(all_sents):,}\n")

    lex_set = set(lex_rows.keys())
    ng_votes: Dict[str, Counter] = defaultdict(Counter)
    v_deps: Dict[str, Set[str]] = defaultdict(set)
    v_prep: Dict[str, Counter] = defaultdict(Counter)
    v_refl: Dict[str, int] = defaultdict(int)
    v_total: Dict[str, int] = defaultdict(int)
    adj_heads: Dict[str, Counter] = defaultdict(Counter)
    morph_bf: Dict[str, Dict[str, Dict]] = defaultdict(dict)

    BS = 500
    nb = (len(all_sents) + BS - 1) // BS
    for bi in tqdm(range(nb), desc="  Pass 2"):
        docs = list(nlp.pipe(all_sents[bi * BS:(bi + 1) * BS], batch_size=BS))
        for doc in docs:
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue
                lem = tok.lemma_.lower()
                if lem not in lex_set:
                    continue
                fm = tok.text.lower()
                pos = tok.pos_
                md: Dict[str, str] = {}
                if tok.morph:
                    for feat in tok.morph:
                        p = feat.split("=")
                        if len(p) == 2:
                            md[p[0]] = p[1]
                if fm not in morph_bf[lem]:
                    morph_bf[lem][fm] = md
                if pos == "NOUN":
                    g = md.get("Gender", "")
                    if g == "Masc":
                        ng_votes[lem]["m"] += 1
                    elif g == "Fem":
                        ng_votes[lem]["f"] += 1
                if pos in ("VERB", "AUX"):
                    v_total[lem] += 1
                    for ch in tok.children:
                        v_deps[lem].add(ch.dep_)
                        if ch.dep_ in ("obl", "nmod", "obl:arg"):
                            for gc in ch.children:
                                if gc.dep_ == "case" and gc.pos_ == "ADP":
                                    v_prep[lem][gc.text.lower()] += 1
                        if (ch.dep_ in ("expl", "expl:pv")
                                and ch.text.lower() in REFL_PRON):
                            v_refl[lem] += 1
                if pos == "ADJ" and tok.head and tok.head.pos_ == "NOUN":
                    adj_heads[lem][tok.head.lemma_.lower()] += 1

    # Morph backfill
    bf_n = 0
    for lem, fl in forms.items():
        mm = morph_bf.get(lem, {})
        for e in fl:
            if not e.get("morph") and e["form"] in mm:
                e["morph"] = mm[e["form"]]
                bf_n += 1
    print(f"\n  Morph backfilled: {bf_n:,}")

    def _gg(lemma):
        if lemma.endswith(("idad", "edad", "tad", "tud")):
            return "f", "high"
        if lemma.endswith(("cia", "gia")):
            return "f", "high"
        if re.search(r"[cs]ión$", lemma):
            return "f", "high"
        if lemma.endswith("aje"):
            return "m", "high"
        if lemma.endswith("a") and not lemma.endswith("ma"):
            return "f", "low"
        if lemma.endswith("o"):
            return "m", "low"
        if lemma.endswith("or"):
            return "m", "low"
        return None, "none"

    enriched: Dict[str, dict] = {}
    for lemma, row in lex_rows.items():
        pr = row.get("pos", "").strip().lower()
        try:
            rk = int(row.get("rank", "99999").strip())
        except ValueError:
            rk = 99999
        e: Dict[str, Any] = {
            "lemma": lemma, "rank": rk, "pos": pr,
            "translation": row.get("translation", "").strip(),
            "gender": None, "gender_confidence": "none", "gender_source": None,
            "semantic_class": None, "semantic_class_source": None,
            "verb_type": None, "verb_type_confidence": "none",
            "required_prep": None, "required_prep_confidence": "none",
            "is_reflexive": False, "reflexive_confidence": "none",
        }
        if pr == "n":
            if lemma in ng_votes and ng_votes[lemma]:
                tv = sum(ng_votes[lemma].values())
                tg, tc = ng_votes[lemma].most_common(1)[0]
                if tv >= 3 and tc / tv >= 0.7:
                    e["gender"], e["gender_confidence"], e["gender_source"] = tg, "corpus", "corpus"
                elif tv >= 1:
                    e["gender"], e["gender_confidence"], e["gender_source"] = tg, "low", "corpus"
            if e["gender"] is None:
                g, c = _gg(lemma)
                if g:
                    e["gender"], e["gender_confidence"], e["gender_source"] = g, c, "rule"
            sc = _NOUN_SEM.get(norm_key(lemma))
            if sc:
                e["semantic_class"], e["semantic_class_source"] = sc, "manual"
        if pr == "v":
            ds = v_deps.get(lemma, set())
            tt = v_total.get(lemma, 0)
            if "obj" in ds or "dobj" in ds:
                e["verb_type"] = "transitive"
                e["verb_type_confidence"] = "corpus" if tt >= 5 else "low"
            elif ds:
                e["verb_type"] = "intransitive"
                e["verb_type_confidence"] = "corpus" if tt >= 5 else "low"
            if lemma in v_prep and v_prep[lemma]:
                tp, tpc = v_prep[lemma].most_common(1)[0]
                if tpc >= 3:
                    e["required_prep"] = tp
                    e["required_prep_confidence"] = "corpus" if tpc >= 5 else "low"
            rl = v_refl.get(lemma, 0)
            if tt >= 5 and rl / max(tt, 1) > 0.3:
                e["is_reflexive"], e["reflexive_confidence"] = True, "corpus"
        if pr == "adj" and lemma in adj_heads and adj_heads[lemma]:
            tn = adj_heads[lemma].most_common(1)[0][0]
            sc = _NOUN_SEM.get(norm_key(tn))
            if sc:
                e["semantic_class"], e["semantic_class_source"] = sc, "corpus"
        enriched[lemma] = e

    return enriched, forms


# ===================================================================
# SAVE
# ===================================================================

def save(outdir, ctx, forms, bi, tri, pos_tr, enriched, n_sent):
    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)
    os.makedirs(outdir, exist_ok=True)

    bi_next: Dict[str, Dict[str, int]] = defaultdict(dict)
    bi_tot: Dict[str, int] = defaultdict(int)
    for (w1, w2), c in bi.items():
        bi_next[w1][w2] = c
        bi_tot[w1] += c

    tri_next: Dict[Tuple, Dict[str, int]] = defaultdict(dict)
    tri_tot: Dict[Tuple, int] = defaultdict(int)
    for (w1, w2, w3), c in tri.items():
        k = (w1, w2)
        tri_next[k][w3] = c
        tri_tot[k] += c

    pn: Dict[str, Dict[str, int]] = defaultdict(dict)
    pt: Dict[str, int] = defaultdict(int)
    for (p1, p2), c in pos_tr.items():
        pn[p1][p2] = c
        pt[p1] += c

    arts = {
        "lemma_contexts.pkl": ctx,
        "lemma_forms.pkl": forms,
        "bigrams.pkl": {"next": dict(bi_next), "totals": dict(bi_tot)},
        "trigrams.pkl": {"next": dict(tri_next), "totals": dict(tri_tot)},
        "pos_transitions.pkl": {"next": dict(pn), "totals": dict(pt)},
        "enriched_lexicon.pkl": enriched,
    }
    for name, data in arts.items():
        p = os.path.join(outdir, name)
        with open(p, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = os.path.getsize(p) / (1 << 20)
        ni = len(data.get("next", data)) if isinstance(data, dict) and "next" in data else len(data)
        print(f"  {name:<25} {ni:>12,} entries   {mb:>8.1f} MB")
    print(f"\n  Saved to {outdir}/   ({n_sent:,} sentences)")


# ===================================================================
# MAIN
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Rebuild artifacts from a large plain-text corpus."
    )
    ap.add_argument("--corpus", required=True, help="One sentence per line")
    ap.add_argument("--lexicon", required=True, help="stg_words_spa.csv")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--context-reservoir-size", type=int, default=50)
    ap.add_argument("--spacy-sample-per-lemma", type=int, default=100)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-spacy", action="store_true",
                    help="Reuse existing enriched_lexicon.pkl")
    args = ap.parse_args()
    random.seed(args.seed)

    for p, label in [(args.corpus, "Corpus"), (args.lexicon, "Lexicon")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("Loading lexicon...")
    lex_lemmas: Set[str] = set()
    lex_rows: Dict[str, dict] = {}
    with open(args.lexicon, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lem = row.get("lemma", "").strip().lower()
            if lem:
                lex_lemmas.add(lem)
                lex_rows[lem] = row
                orig = (row.get("original_lemma") or "").strip().lower()
                if orig and orig != lem:
                    lex_lemmas.add(orig)
                    if orig not in lex_rows:
                        lex_rows[orig] = row
    print(f"  {len(lex_lemmas):,} lemmas\n")

    rev: Dict[str, str] = {}
    for lem in lex_lemmas:
        rev[lem] = lem
    for lem in list(lex_lemmas):
        hl = heuristic_lemma(lem)
        if hl != lem and hl in lex_lemmas:
            rev[lem] = hl

    ctx, forms, bi, tri, pos_tr, sp_samp, n_sent = pass1(
        args.corpus, lex_lemmas, rev,
        ctx_size=args.context_reservoir_size,
        min_w=args.min_words, max_w=args.max_words,
    )

    if args.skip_spacy:
        ep = os.path.join(args.outdir, "enriched_lexicon.pkl")
        if os.path.exists(ep):
            with open(ep, "rb") as f:
                enriched = pickle.load(f)
            print(f"  Reused enriched_lexicon.pkl ({len(enriched):,} entries)")
        else:
            enriched = {}
            print("  WARNING: no existing enriched_lexicon.pkl found")
    else:
        enriched, forms = pass2(
            sp_samp, lex_rows, forms,
            sample_n=args.spacy_sample_per_lemma,
        )

    save(args.outdir, ctx, forms, bi, tri, pos_tr, enriched, n_sent)
    print("\nDONE.\n")


if __name__ == "__main__":
    main()
