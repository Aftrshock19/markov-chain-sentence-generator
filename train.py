#!/usr/bin/env python3
"""
train.py - Corpus Training Pipeline (v3)
=========================================
Processes a raw Tatoeba TSV + your stg_words_spa.csv to produce all the
artifacts the sentence generator needs.

REQUIRES: Python 3.9+

SETUP (run once):
    pip install spacy tqdm
    python -m spacy download es_core_news_lg

USAGE:
    python train.py --tatoeba sentences.tsv --lexicon stg_words_spa.csv --outdir models/

OUTPUTS (all in --outdir):
    corpus.txt              Cleaned deduplicated Spanish sentences
    lemma_forms.pkl         lemma -> [{form, morph}, ...]
    lemma_contexts.pkl      lemma -> [{left, right, sentence, ...}, ...]
    bigrams.pkl             counts + next-word index + totals
    trigrams.pkl            counts + continuation index + totals
    pos_transitions.pkl     counts + next-POS index + totals
    enriched_lexicon.pkl    lemma -> {gender, verb_type, confidence fields, ...}

EXPECTED TIME:  ~10-20 min on Isambard Grace, ~30-40 on a laptop.
EXPECTED RAM:   ~3 GB (spaCy model + counts in memory).
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
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


# ===================================================================
# UTILITIES
# ===================================================================

def strip_accents(s: str) -> str:
    """Remove diacritics for fuzzy matching. Keep base characters."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def norm_key(s: str) -> str:
    """Normalize a string for dictionary lookup: lowercase + strip accents."""
    return strip_accents(s.strip().lower())


def morph_signature(morph_dict: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Create a hashable signature from a morph dict."""
    return tuple(sorted(morph_dict.items()))


def reservoir_add(
    reservoir: List, item: object, max_size: int, count_seen: int
) -> None:
    """
    Reservoir sampling: maintain a uniform random sample of max_size
    items from a stream of count_seen items seen so far.
    """
    if len(reservoir) < max_size:
        reservoir.append(item)
    else:
        j = random.randint(0, count_seen - 1)
        if j < max_size:
            reservoir[j] = item


# ===================================================================
# PHASE 1: CORPUS CLEANING
# ===================================================================

# Reject lines that contain clearly non-Spanish content.
# This is a blocklist approach (reject bad) rather than a whitelist
# (insist every char is in a tiny set). Keeps ellipses, slashes,
# percent signs, apostrophes, etc.
BAD_PATTERNS = re.compile(
    r"https?://|www\.|@|\|"   # URLs, emails, pipes
    r"|[\x00-\x08\x0e-\x1f]"  # control characters
)


def clean_corpus(tatoeba_path: str, output_path: str) -> List[str]:
    """
    Read raw Tatoeba TSV, filter to Spanish, clean, deduplicate,
    and write corpus.txt. Returns list of cleaned sentences.
    """
    print("=" * 60)
    print("PHASE 1: Cleaning corpus")
    print("=" * 60)

    raw_sentences: List[str] = []
    skipped: Counter = Counter()

    with open(tatoeba_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading Tatoeba"):
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                skipped["malformed"] += 1
                continue

            lang = parts[1].strip()
            if lang not in ("spa", "es"):
                skipped["not_spanish"] += 1
                continue

            text = parts[2].strip()

            # Length filters (keep up to 30 for richer training data;
            # the generator enforces band-specific limits at retrieval)
            word_count = len(text.split())
            if word_count < 3:
                skipped["too_short"] += 1
                continue
            if word_count > 30:
                skipped["too_long"] += 1
                continue

            # Content filters (blocklist, not whitelist)
            if text.isupper():
                skipped["all_caps"] += 1
                continue
            if BAD_PATTERNS.search(text):
                skipped["bad_content"] += 1
                continue

            # Normalise whitespace
            text = " ".join(text.split())
            raw_sentences.append(text)

    # Deduplicate (case-insensitive, preserving first occurrence)
    seen: Set[str] = set()
    sentences: List[str] = []
    dupes = 0
    for s in raw_sentences:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            sentences.append(s)
        else:
            dupes += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"\n  Kept:       {len(sentences):,} sentences")
    print(f"  Duplicates: {dupes:,} removed")
    for reason, count in skipped.most_common():
        print(f"  Skipped:    {count:,} ({reason})")
    print(f"  Saved:      {output_path}\n")

    return sentences


# ===================================================================
# SEMANTIC CLASS DEFINITIONS
# ===================================================================

# All entries use correct Spanish accents. Lookups use norm_key()
# so accented and unaccented forms both match.

SEMANTIC_HINTS: Dict[str, List[str]] = {
    "person": [
        "hombre", "mujer", "persona", "gente",
        "hijo", "hija", "padre", "madre",
        "hermano", "hermana", "amigo", "amiga",
        "doctor", "profesor", "profesora",
        "estudiante", "jefe", "rey", "reina",
    ],
    "animal": [
        "perro", "gato", "animal", "caballo", "pez",
        "vaca", "toro", "oso", "lobo", "rata",
    ],
    "food": [
        "comida", "pan", "carne", "fruta", "arroz",
        "queso", "sopa", "pescado", "pollo", "huevo",
        "sal", "chocolate", "torta", "pastel", "galleta",
    ],
    "drink": [
        "agua", "leche", "vino", "cerveza", "jugo",
        "refresco",
    ],
    "place": [
        "casa", "ciudad", "calle", "escuela", "pueblo",
        "mundo", "tierra", "mar", "bosque",
        "iglesia", "hotel", "hospital", "oficina", "tienda",
        "parque", "playa", "campo", "plaza",
    ],
    "time": [
        "hora", "momento", "tiempo", "noche",
        "semana", "mes", "minuto", "segundo",
        "siglo", "tarde",
        # accented forms
        "d\u00eda", "a\u00f1o", "\u00e9poca", "ma\u00f1ana",
    ],
    "abstract": [
        "vida", "amor", "cosa", "parte", "forma", "idea",
        "verdad", "guerra", "paz", "muerte",
        "historia", "cultura", "libertad", "poder",
        "justicia", "derecho", "ley", "fe", "esperanza",
        # accented forms
        "raz\u00f3n", "problema", "naci\u00f3n", "opini\u00f3n",
    ],
    "object": [
        "libro", "mesa", "puerta", "coche", "carta", "piedra",
        "llave", "silla", "cama", "reloj", "espejo",
        "botella", "bolsa", "caja", "papel", "foto",
        # accented forms
        "tel\u00e9fono",
    ],
    "body_part": [
        "mano", "ojo", "cabeza", "pie", "cara", "brazo",
        "boca", "nariz", "oreja", "dedo", "pelo",
        "piel", "sangre", "hueso", "espalda", "pierna",
        # accented forms
        "coraz\u00f3n", "est\u00f3mago",
    ],
    "clothing": [
        "ropa", "camisa", "zapato", "sombrero", "vestido",
        "falda", "abrigo", "chaqueta",
        # accented forms
        "pantal\u00f3n",
    ],
    "nature": [
        "sol", "luna", "estrella", "cielo", "nube", "lluvia",
        "viento", "nieve", "flor", "hierba",
        # accented forms
        "\u00e1rbol",
    ],
    "text": [
        "libro", "carta", "palabra", "nombre", "historia",
        "mensaje", "noticia",
        # accented forms
        "p\u00e1gina", "peri\u00f3dico", "canci\u00f3n",
    ],
}

# Build reverse index using normalized keys for reliable matching
_NOUN_TO_SEMANTIC: Dict[str, str] = {}
for _sem_class, _nouns in SEMANTIC_HINTS.items():
    for _n in _nouns:
        _NOUN_TO_SEMANTIC[norm_key(_n)] = _sem_class


def lookup_semantic_class(lemma: str) -> Optional[str]:
    """Look up semantic class for a noun lemma, accent-insensitive."""
    return _NOUN_TO_SEMANTIC.get(norm_key(lemma))


# ===================================================================
# PHASE 2: spaCy PROCESSING + ARTIFACT BUILDING
# ===================================================================

REFLEXIVE_PRONOUNS = frozenset({"me", "te", "se", "nos", "os"})


def process_corpus(
    sentences: List[str], outdir: str, lexicon_path: str
) -> None:
    """
    Run spaCy on all sentences and build every artifact.
    """
    print("=" * 60)
    print("PHASE 2: spaCy processing + artifact building")
    print("=" * 60)

    import spacy
    print("  Loading spaCy model (es_core_news_lg, NER disabled)...")
    nlp = spacy.load("es_core_news_lg", disable=["ner"])
    nlp.max_length = 2_000_000
    print("  Model loaded.\n")

    # -- Load lexicon --
    lexicon_lemmas: Set[str] = set()
    lexicon_lemmas_norm: Dict[str, str] = {}   # norm_key -> original lemma
    lexicon_rows: Dict[str, dict] = {}
    with open(lexicon_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = row.get("lemma", "").strip()
            if lemma:
                lexicon_lemmas.add(lemma)
                lexicon_lemmas_norm[norm_key(lemma)] = lemma
                lexicon_rows[lemma] = row

    print(f"  Lexicon: {len(lexicon_lemmas):,} lemmas loaded.\n")

    # -- Accumulators --
    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()
    pos_trans_counts: Counter = Counter()

    lemma_forms: Dict[str, List[dict]] = defaultdict(list)
    lemma_form_seen: Dict[str, Set[Tuple]] = defaultdict(set)

    lemma_contexts: Dict[str, List[dict]] = defaultdict(list)
    lemma_context_counts: Dict[str, int] = defaultdict(int)
    CONTEXT_RESERVOIR_SIZE = 30

    # Enrichment evidence
    noun_gender_votes: Dict[str, Counter] = defaultdict(Counter)
    verb_dep_evidence: Dict[str, Set[str]] = defaultdict(set)
    verb_prep_evidence: Dict[str, Counter] = defaultdict(Counter)
    verb_reflexive_count: Dict[str, int] = defaultdict(int)
    verb_total_count: Dict[str, int] = defaultdict(int)
    adj_head_nouns: Dict[str, Counter] = defaultdict(Counter)

    # -- Process in batches --
    BATCH_SIZE = 500
    n_batches = (len(sentences) + BATCH_SIZE - 1) // BATCH_SIZE

    print("  Processing sentences through spaCy...")
    for batch_idx in tqdm(range(n_batches), desc="  Batches"):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(sentences))
        batch_texts = sentences[start:end]
        docs = list(nlp.pipe(batch_texts, batch_size=BATCH_SIZE))

        for doc in docs:
            tokens = [tok for tok in doc if not tok.is_space]
            if len(tokens) < 2:
                continue

            # Preserve original sentence text (not reconstructed)
            sentence_text = doc.text.strip()

            words: List[str] = []
            lemmas_list: List[str] = []
            pos_tags: List[str] = []

            for tok in tokens:
                w = tok.text.lower()
                lem = tok.lemma_.lower()
                pos = tok.pos_
                words.append(w)
                lemmas_list.append(lem)
                pos_tags.append(pos)

                # -- Lemma forms (dedup includes morph signature) --
                morph_dict: Dict[str, str] = {}
                if tok.morph:
                    for feat in tok.morph:
                        fp = feat.split("=")
                        if len(fp) == 2:
                            morph_dict[fp[0]] = fp[1]

                msig = morph_signature(morph_dict)
                form_key = (lem, w, msig)
                if form_key not in lemma_form_seen[lem]:
                    lemma_form_seen[lem].add(form_key)
                    lemma_forms[lem].append({
                        "form": w,
                        "morph": morph_dict,
                    })

                # -- Noun gender evidence --
                if pos == "NOUN" and lem in lexicon_lemmas:
                    g_raw = morph_dict.get("Gender", "")
                    if g_raw == "Masc":
                        noun_gender_votes[lem]["m"] += 1
                    elif g_raw == "Fem":
                        noun_gender_votes[lem]["f"] += 1

                # -- Verb occurrence count --
                if pos == "VERB" and lem in lexicon_lemmas:
                    verb_total_count[lem] += 1

                # -- Adjective head nouns --
                if pos == "ADJ" and lem in lexicon_lemmas:
                    if tok.head and tok.head.pos_ == "NOUN":
                        adj_head_nouns[lem][tok.head.lemma_.lower()] += 1

            # -- Verb dependency + preposition + reflexive evidence --
            for tok in tokens:
                if tok.pos_ != "VERB":
                    continue
                lem = tok.lemma_.lower()
                if lem not in lexicon_lemmas:
                    continue

                for child in tok.children:
                    verb_dep_evidence[lem].add(child.dep_)

                    # Prepositions: look for "case" dep under obl/nmod
                    if child.dep_ in ("obl", "nmod", "obl:arg"):
                        for gc in child.children:
                            if gc.dep_ == "case" and gc.pos_ == "ADP":
                                verb_prep_evidence[lem][gc.text.lower()] += 1

                    # Reflexive: check pronoun children
                    if (child.dep_ in ("expl", "expl:pv")
                            and child.text.lower() in REFLEXIVE_PRONOUNS
                            and child.pos_ == "PRON"):
                        verb_reflexive_count[lem] += 1

            # -- N-gram counts --
            wl = ["<START>"] + words + ["<END>"]
            pl = ["<START>"] + pos_tags + ["<END>"]

            for i in range(len(wl) - 1):
                bigram_counts[(wl[i], wl[i + 1])] += 1
                pos_trans_counts[(pl[i], pl[i + 1])] += 1
            for i in range(len(wl) - 2):
                trigram_counts[(wl[i], wl[i + 1], wl[i + 2])] += 1

            # -- Lemma contexts (reservoir sampled, rich metadata) --
            for i, lem in enumerate(lemmas_list):
                if lem not in lexicon_lemmas:
                    continue
                lemma_context_counts[lem] += 1

                # Left and right bigrams
                left_w = words[i - 1] if i > 0 else "<START>"
                right_w = words[i + 1] if i < len(words) - 1 else "<END>"
                left_lem = lemmas_list[i - 1] if i > 0 else "<START>"
                right_lem = lemmas_list[i + 1] if i < len(lemmas_list) - 1 else "<END>"
                left_bigram = (
                    (words[i - 2] if i > 1 else "<START>"),
                    left_w,
                )
                right_bigram = (
                    right_w,
                    (words[i + 2] if i < len(words) - 2 else "<END>"),
                )

                # Lemma window (2 left, 2 right)
                window_start = max(0, i - 2)
                window_end = min(len(lemmas_list), i + 3)
                lemma_window = lemmas_list[window_start:window_end]

                ctx = {
                    "left": left_w,
                    "right": right_w,
                    "left_lemma": left_lem,
                    "right_lemma": right_lem,
                    "left_bigram": left_bigram,
                    "right_bigram": right_bigram,
                    "lemma_window": lemma_window,
                    "target_form": words[i],
                    "target_pos": pos_tags[i],
                    "sentence": sentence_text,
                    "tokens": words,
                    "index": i,
                    "sent_len": len(words),
                }
                reservoir_add(
                    lemma_contexts[lem], ctx,
                    CONTEXT_RESERVOIR_SIZE,
                    lemma_context_counts[lem],
                )

    # ===============================================================
    # PHASE 3: BUILD ENRICHED LEXICON
    # ===============================================================

    print("\n" + "=" * 60)
    print("PHASE 3: Building enriched lexicon")
    print("=" * 60)

    def guess_gender(lemma: str) -> Tuple[Optional[str], str]:
        """
        Rule-based gender guess for NOUNS only.
        Returns (gender_or_None, confidence).
        """
        # Strong feminine
        if lemma.endswith(("idad", "edad", "tad", "tud")):
            return ("f", "high")
        if lemma.endswith(("cia", "gia")):
            return ("f", "high")
        # -cion/-sion (but NOT -mente which is adverbs, not nouns)
        if re.search(r"[cs]i\u00f3n$", lemma) or re.search(r"[cs]ion$", lemma):
            return ("f", "high")
        if lemma.endswith("umbre"):
            return ("f", "high")

        # Strong masculine
        if lemma.endswith("aje"):
            return ("m", "high")

        # Moderate patterns
        if lemma.endswith("a") and not lemma.endswith("ma"):
            return ("f", "low")
        if lemma.endswith("o"):
            return ("m", "low")
        if lemma.endswith("or"):
            return ("m", "low")
        if lemma.endswith("ez"):
            return ("f", "low")

        # No confident guess
        return (None, "none")

    enriched_lexicon: Dict[str, dict] = {}

    for lemma, row in lexicon_rows.items():
        pos_raw = row.get("pos", "").strip().lower()
        rank_str = row.get("rank", "99999").strip()
        try:
            rank = int(rank_str)
        except ValueError:
            rank = 99999

        entry: Dict[str, object] = {
            "lemma": lemma,
            "rank": rank,
            "pos": pos_raw,
            "translation": row.get("translation", "").strip(),
            "gender": None,
            "gender_confidence": "none",
            "gender_source": None,
            "semantic_class": None,
            "semantic_class_source": None,
            "verb_type": None,
            "verb_type_confidence": "none",
            "required_prep": None,
            "required_prep_confidence": "none",
            "is_reflexive": False,
            "reflexive_confidence": "none",
        }

        # -- Noun enrichment --
        if pos_raw == "n":
            # Gender: corpus evidence first
            if lemma in noun_gender_votes and noun_gender_votes[lemma]:
                total_v = sum(noun_gender_votes[lemma].values())
                top_g, top_c = noun_gender_votes[lemma].most_common(1)[0]
                if total_v >= 3 and top_c / total_v >= 0.7:
                    entry["gender"] = top_g
                    entry["gender_confidence"] = "corpus"
                    entry["gender_source"] = "corpus"
                elif total_v >= 1:
                    entry["gender"] = top_g
                    entry["gender_confidence"] = "low"
                    entry["gender_source"] = "corpus"

            # Fallback to rules (only if no corpus evidence)
            if entry["gender"] is None:
                guessed, conf = guess_gender(lemma)
                if guessed is not None:
                    entry["gender"] = guessed
                    entry["gender_confidence"] = conf
                    entry["gender_source"] = "rule"

            # Semantic class (accent-safe lookup)
            sc = lookup_semantic_class(lemma)
            if sc is not None:
                entry["semantic_class"] = sc
                entry["semantic_class_source"] = "manual"

        # -- Verb enrichment --
        if pos_raw == "v":
            deps = verb_dep_evidence.get(lemma, set())
            total = verb_total_count.get(lemma, 0)
            if "obj" in deps or "dobj" in deps:
                entry["verb_type"] = "transitive"
                entry["verb_type_confidence"] = "corpus" if total >= 5 else "low"
            elif deps:
                entry["verb_type"] = "intransitive"
                entry["verb_type_confidence"] = "corpus" if total >= 5 else "low"
            # else: None - not enough evidence

            # Required preposition
            if lemma in verb_prep_evidence and verb_prep_evidence[lemma]:
                top_p, top_pc = verb_prep_evidence[lemma].most_common(1)[0]
                if top_pc >= 5:
                    entry["required_prep"] = top_p
                    entry["required_prep_confidence"] = "corpus"
                elif top_pc >= 3:
                    entry["required_prep"] = top_p
                    entry["required_prep_confidence"] = "low"

            # Reflexive (tentative)
            refl = verb_reflexive_count.get(lemma, 0)
            if total >= 5 and refl / max(total, 1) > 0.3:
                entry["is_reflexive"] = True
                entry["reflexive_confidence"] = "corpus"
            elif total >= 3 and refl / max(total, 1) > 0.5:
                entry["is_reflexive"] = True
                entry["reflexive_confidence"] = "low"

        # -- Adjective enrichment --
        if pos_raw == "adj":
            if lemma in adj_head_nouns and adj_head_nouns[lemma]:
                top_noun = adj_head_nouns[lemma].most_common(1)[0][0]
                sc = lookup_semantic_class(top_noun)
                if sc is not None:
                    entry["semantic_class"] = sc
                    entry["semantic_class_source"] = "corpus"

        enriched_lexicon[lemma] = entry

    # ===============================================================
    # PHASE 4: BUILD INDEXED STRUCTURES + SAVE
    # ===============================================================

    print("\n" + "=" * 60)
    print("PHASE 4: Building indexes and saving artifacts")
    print("=" * 60)

    # Bigram: next-word index + context totals
    print("  Building bigram index...")
    bigram_next: Dict[str, Dict[str, int]] = defaultdict(dict)
    bigram_totals: Dict[str, int] = defaultdict(int)
    for (w1, w2), count in bigram_counts.items():
        bigram_next[w1][w2] = count
        bigram_totals[w1] += count

    # Trigram: continuation index + context totals
    print("  Building trigram index...")
    trigram_next: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)
    trigram_totals: Dict[Tuple[str, str], int] = defaultdict(int)
    for (w1, w2, w3), count in trigram_counts.items():
        key = (w1, w2)
        trigram_next[key][w3] = count
        trigram_totals[key] += count

    # POS: next-POS index + totals
    print("  Building POS transition index...")
    pos_next: Dict[str, Dict[str, int]] = defaultdict(dict)
    pos_totals: Dict[str, int] = defaultdict(int)
    for (p1, p2), count in pos_trans_counts.items():
        pos_next[p1][p2] = count
        pos_totals[p1] += count

    os.makedirs(outdir, exist_ok=True)

    artifacts = {
        "lemma_forms.pkl": dict(lemma_forms),
        "lemma_contexts.pkl": dict(lemma_contexts),
        "bigrams.pkl": {
            "counts": dict(bigram_counts),
            "next": dict(bigram_next),
            "totals": dict(bigram_totals),
        },
        "trigrams.pkl": {
            "counts": dict(trigram_counts),
            "next": dict(trigram_next),
            "totals": dict(trigram_totals),
        },
        "pos_transitions.pkl": {
            "counts": dict(pos_trans_counts),
            "next": dict(pos_next),
            "totals": dict(pos_totals),
        },
        "enriched_lexicon.pkl": enriched_lexicon,
    }

    for filename, data in artifacts.items():
        path = os.path.join(outdir, filename)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if isinstance(data, dict) and "counts" in data:
            item_count = len(data["counts"])
        else:
            item_count = len(data)
        print(f"  {filename:<25} {item_count:>10,} entries   {size_mb:>8.1f} MB")

    # ===============================================================
    # SUMMARY
    # ===============================================================

    print("\n" + "=" * 60)
    print("DONE - Summary")
    print("=" * 60)

    total_nouns = sum(1 for e in enriched_lexicon.values() if e["pos"] == "n")
    total_verbs = sum(1 for e in enriched_lexicon.values() if e["pos"] == "v")

    ng_corpus = sum(
        1 for e in enriched_lexicon.values()
        if e["pos"] == "n" and e["gender_source"] == "corpus"
    )
    ng_rule = sum(
        1 for e in enriched_lexicon.values()
        if e["pos"] == "n" and e["gender_source"] == "rule"
    )
    ng_none = sum(
        1 for e in enriched_lexicon.values()
        if e["pos"] == "n" and e["gender"] is None
    )
    vt_any = sum(
        1 for e in enriched_lexicon.values()
        if e["pos"] == "v" and e["verb_type"] is not None
    )
    vr_any = sum(
        1 for e in enriched_lexicon.values()
        if e["pos"] == "v" and e["is_reflexive"]
    )
    ctx_any = sum(1 for v in lemma_contexts.values() if len(v) > 0)
    forms_any = sum(1 for v in lemma_forms.values() if len(v) > 0)

    print(f"\n  Corpus sentences:       {len(sentences):,}")
    print(f"  Unique bigrams:         {len(bigram_counts):,}")
    print(f"  Unique trigrams:        {len(trigram_counts):,}")
    print(f"  POS transitions:        {len(pos_trans_counts):,}")
    print(f"  Lemmas with forms:      {forms_any:,}")
    print(f"  Lemmas with contexts:   {ctx_any:,} / {len(lexicon_lemmas):,}")
    print()
    print(f"  Noun gender coverage:   {ng_corpus + ng_rule:,} / {total_nouns:,}")
    print(f"    from corpus:          {ng_corpus:,}")
    print(f"    from rules:           {ng_rule:,}")
    print(f"    unknown:              {ng_none:,}")
    print()
    print(f"  Verb type coverage:     {vt_any:,} / {total_verbs:,}")
    print(f"  Verbs flagged reflexive:{vr_any:,}")
    print()
    print(f"  All artifacts saved to: {outdir}/")
    print(f"  Upload these .pkl files back to Claude to continue.\n")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train corpus artifacts for the Spanish sentence generator."
    )
    parser.add_argument(
        "--tatoeba", required=True,
        help="Path to raw Tatoeba TSV file (id, lang, text)"
    )
    parser.add_argument(
        "--lexicon", required=True,
        help="Path to stg_words_spa.csv"
    )
    parser.add_argument(
        "--outdir", default="models",
        help="Directory to save .pkl artifacts (default: models/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible context sampling (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.tatoeba):
        print(f"ERROR: Tatoeba file not found: {args.tatoeba}")
        sys.exit(1)
    if not os.path.exists(args.lexicon):
        print(f"ERROR: Lexicon file not found: {args.lexicon}")
        sys.exit(1)

    corpus_path = os.path.join(args.outdir, "corpus.txt")
    os.makedirs(args.outdir, exist_ok=True)
    sentences = clean_corpus(args.tatoeba, corpus_path)

    if len(sentences) == 0:
        print("ERROR: No sentences survived cleaning.")
        print("Expected TSV with columns: id<tab>lang<tab>text")
        print("Language code should be 'spa' or 'es'.")
        sys.exit(1)

    process_corpus(sentences, args.outdir, args.lexicon)


if __name__ == "__main__":
    main()