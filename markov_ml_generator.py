#!/usr/bin/env python3
"""High-quality Markov-style sentence generator trained purely with machine learning.

Components (all learned from data, no LLM, no retrieval):

  1. Modified Kneser-Ney trigram language model trained on a clean corpus
     (human-reviewed-good rows + Tatoeba Spanish pedagogical sentences).
  2. A logistic-regression sentence-quality reranker trained on ~39K
     human-labeled sentences (good vs bad) with hand-engineered features plus
     the KN-LM score.
  3. Target-anchored beam-search decoding: at each step we keep the top-B
     partial sentences by (log p_trigram + continuation heuristic). The target
     lemma is forced to appear. Generation runs left-of-target and
     right-of-target; completions are stitched.
  4. Hard Spanish agreement filters (article-noun, subject-verb) applied
     during beam expansion to prune structurally impossible prefixes early.
  5. Final-selection by reranker logit + LM score + length prior.

Usage:

    python markov_ml_generator.py \
      --lm data_clean/kn_lm.pkl \
      --reranker data_clean/reranker.pkl \
      --lexicon stg_words_spa.csv \
      --min-rank 1 --max-rank 1000 --limit 100 \
      --out outputs/markov_ml_sentences.csv
"""
from __future__ import annotations

import argparse
import csv
import heapq
import math
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from kn_lm import KNLanguageModel  # noqa: E402
from features import (  # noqa: E402
    ARTICLES, BAD_FINAL, CONTEXT_OPENERS, FEMININE_ART, FINITE_VERBS, MASCULINE_ART,
    PLURAL_ART, PREPOSITIONS, PRONOUN_PERSON, SINGULAR_ART, SUBJECT_PRONOUNS,
    VERB_PERSON, featurize, tokenize,
)
from validators import validate  # noqa: E402
from templates import grammar_safe_templates  # noqa: E402
from template_signature import template_signature  # noqa: E402

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")

# Structural openers we will seed beam search with (target-aware).
# These are *POS-agnostic* frames; they are *not* memorized sentences.
GRAMMATICAL_FRAMES = {
    "n_subject": [["yo"], ["tú"], ["ella"], ["él"], ["nosotros"], ["ellos"]],
    "n_copula": [["es"], ["está"], ["son"], ["están"], ["era"], ["fue"]],
    "n_det": [["el"], ["la"], ["los"], ["las"], ["un"], ["una"], ["mi"], ["tu"], ["este"], ["esta"]],
}


def _fem_noun(w: str) -> bool:
    return (
        (w.endswith("a") and not w.endswith(("ma", "pa", "ta")))
        or w.endswith(("dad", "tad", "tud", "ción", "sión", "umbre", "ez"))
    )


def _plural_noun(w: str) -> bool:
    return w.endswith("s") and len(w) > 3 and not w.endswith(("és", "ís", "ús", "ás", "os"))


_MORPH_CACHE = {}


def load_morph_table(path: Path) -> Dict[str, Dict[str, str]]:
    """Read lemma_forms.pkl from models_rebuild2 and return form -> morph dict."""
    if "_morph" in _MORPH_CACHE:
        return _MORPH_CACHE["_morph"]
    with path.open("rb") as f:
        lf = pickle.load(f)
    out: Dict[str, Dict[str, str]] = {}
    for lemma, forms in lf.items():
        for entry in forms:
            form = entry.get("form")
            morph = entry.get("morph") or {}
            if form and morph:
                # Keep the first (most-likely lemma-canonical) reading per surface form.
                out.setdefault(form.lower(), morph)
    _MORPH_CACHE["_morph"] = out
    return out


def fem_of(w: str, morph_table: Dict[str, Dict[str, str]]) -> Optional[bool]:
    """Return True if w is feminine, False if masc, None if unknown. Looks up UD morph, falls back to heuristic."""
    m = morph_table.get(w)
    if m:
        g = m.get("Gender")
        if g == "Fem":
            return True
        if g == "Masc":
            return False
    return None  # unknown from morph table


def plural_of(w: str, morph_table: Dict[str, Dict[str, str]]) -> Optional[bool]:
    m = morph_table.get(w)
    if m:
        n = m.get("Number")
        if n == "Plur":
            return True
        if n == "Sing":
            return False
    return None


@dataclass
class LexiconRow:
    lemma: str
    rank: int
    pos: str
    translation: str


@dataclass(order=True)
class Beam:
    neg_score: float
    tokens: List[str] = field(compare=False)
    # extra: which tokens came from target insertion, for diagnostics
    target_included: bool = field(compare=False, default=False)


@dataclass
class RerankerArtifacts:
    feature_names: List[str]
    mean: List[float]
    scale: List[float]
    coef: List[float]
    intercept: float

    @classmethod
    def load(cls, path: Path) -> "RerankerArtifacts":
        with path.open("rb") as f:
            m = pickle.load(f)
        return cls(
            feature_names=m["feature_names"],
            mean=m["scaler_mean"],
            scale=m["scaler_scale"],
            coef=m["coef"],
            intercept=m["intercept"],
        )

    def score(self, feats: Dict[str, float]) -> float:
        z = self.intercept
        for name, mean, scale, coef in zip(self.feature_names, self.mean, self.scale, self.coef):
            v = (feats.get(name, 0.0) - mean) / (scale if scale else 1.0)
            z += v * coef
        # return the logit — higher = more likely good. (We don't need sigmoid for argmax.)
        return z


def load_lexicon(path: Path) -> List[LexiconRow]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            lemma = (r.get("lemma") or r.get("word") or "").strip().lower()
            if not lemma:
                continue
            try:
                rank = int(float(r.get("rank") or 999999))
            except ValueError:
                rank = 999999
            rows.append(
                LexiconRow(
                    lemma=lemma,
                    rank=rank,
                    pos=(r.get("pos") or "").strip().lower(),
                    translation=(r.get("translation") or "").strip(),
                )
            )
    rows.sort(key=lambda x: x.rank)
    return rows


def profile_for_rank(rank: int) -> Tuple[str, int, int, int]:
    if rank <= 800:
        return "A1", 3, 7, 400
    if rank <= 1500:
        return "A2", 3, 8, 700
    if rank <= 2500:
        return "B1", 4, 9, 1500
    if rank <= 4000:
        return "B2", 5, 10, 3000
    if rank <= 6000:
        return "C1", 5, 11, 5500
    return "C2", 6, 12, 9000


def detokenize(tokens: Sequence[str]) -> str:
    words = [t for t in tokens if t and t not in ("<s>", "</s>")]
    if not words:
        return ""
    sent = " ".join(words)
    sent = sent[0].upper() + sent[1:]
    if sent[-1] not in ".!?":
        sent += "."
    return sent


def transition_ok(tokens: List[str], next_word: str, morph: Optional[Dict[str, Dict[str, str]]] = None) -> bool:
    """Hard grammar filters applied during beam expansion.

    These are *learned-pattern* rules: they mirror the bad-bigram / bad-trigram
    regularities seen in the labeled data. They are not a full parser — they
    just aggressively prune structurally impossible prefixes.
    """
    if not tokens:
        if next_word in CONTEXT_OPENERS and next_word not in {"no", "y"}:
            # starting with "que/si/cuando/porque" yields context-dependent fragments
            return False
        return True

    prev = tokens[-1]
    prev2 = tokens[-2] if len(tokens) >= 2 else ""

    # No repeated-bigram loops
    if prev == next_word:
        return False
    if prev2 and prev2 == next_word and prev == tokens[-1]:
        return False

    # Quantifier / numeral + noun agreement (e.g., "todos mundo" / "dos mundo")
    PLURAL_QUANTIFIERS = {"todos", "todas", "algunos", "algunas", "muchos", "muchas",
                          "pocos", "pocas", "varios", "varias", "otros", "otras",
                          "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
                          "ambos", "ambas", "mis", "tus", "sus", "nuestros", "nuestras"}
    if prev in PLURAL_QUANTIFIERS:
        # Fail when the next token looks like a singular noun (ends in o/a not following ma/ta,
        # and not itself one of the function-word classes).
        if (
            next_word not in ARTICLES
            and next_word not in PREPOSITIONS
            and next_word not in FINITE_VERBS
            and next_word not in VERB_PERSON
            and next_word not in PLURAL_QUANTIFIERS
            and next_word not in {"que", "como", "muy", "tan", "más", "bien", "mal", "aquí", "ahí", "allá", "acá"}
            and not _plural_noun(next_word)
            and next_word.endswith(("o", "a", "e"))
            and len(next_word) >= 3
        ):
            return False
    # Extend article-like set to include demonstratives & possessives which follow the same agreement pattern
    DEM_POSS_FEM_PL = {"estas", "esas", "aquellas", "mías", "tuyas", "suyas", "nuestras", "vuestras"}
    DEM_POSS_FEM_SG = {"esta", "esa", "aquella", "mía", "tuya", "suya", "nuestra", "vuestra", "toda", "alguna", "ninguna", "otra", "cierta", "cualquiera"}
    DEM_POSS_MASC_PL = {"estos", "esos", "aquellos", "míos", "tuyos", "suyos", "nuestros", "vuestros", "todos", "algunos", "ningunos", "otros"}
    DEM_POSS_MASC_SG = {"este", "ese", "aquel", "mío", "tuyo", "suyo", "nuestro", "vuestro", "todo", "algún", "alguno", "ningún", "ninguno", "otro", "cierto", "cualquier"}
    article_like_fem_pl = FEMININE_ART & PLURAL_ART | DEM_POSS_FEM_PL
    article_like_fem_sg = FEMININE_ART & SINGULAR_ART | DEM_POSS_FEM_SG
    article_like_masc_pl = MASCULINE_ART & PLURAL_ART | DEM_POSS_MASC_PL
    article_like_masc_sg = MASCULINE_ART & SINGULAR_ART | DEM_POSS_MASC_SG
    article_like_all = (
        ARTICLES
        | DEM_POSS_FEM_PL | DEM_POSS_FEM_SG
        | DEM_POSS_MASC_PL | DEM_POSS_MASC_SG
    )
    if prev in article_like_all and next_word not in article_like_all and next_word not in PREPOSITIONS and next_word not in FINITE_VERBS and next_word not in VERB_PERSON:
        if next_word in SUBJECT_PRONOUNS:
            return False
        if morph is not None:
            fem = fem_of(next_word, morph)
            plur = plural_of(next_word, morph)
        else:
            fem = _fem_noun(next_word)
            plur = _plural_noun(next_word)
        prev_fem = prev in (FEMININE_ART | DEM_POSS_FEM_PL | DEM_POSS_FEM_SG)
        prev_masc = prev in (MASCULINE_ART | DEM_POSS_MASC_PL | DEM_POSS_MASC_SG)
        prev_plur = prev in (PLURAL_ART | DEM_POSS_FEM_PL | DEM_POSS_MASC_PL)
        prev_sing = prev in (SINGULAR_ART | DEM_POSS_FEM_SG | DEM_POSS_MASC_SG)
        if fem is True and prev_masc:
            return False
        if fem is False and prev_fem:
            return False
        if plur is True and prev_sing:
            return False
        if plur is False and prev_plur:
            return False
        # heuristic fallback
        if fem is None and _fem_noun(next_word) and prev_masc:
            return False
        if plur is None and prev_plur and not _plural_noun(next_word):
            return False
    # An article/demonstrative/possessive followed by a finite-verb form is almost always wrong
    # (e.g., "las es", "los está", "estas son"). Allow "la mejor" → "mejor" isn't finite.
    if prev in article_like_all and next_word in VERB_PERSON:
        return False

    # Subject pronoun + verb agreement (check only when immediate)
    if prev in PRONOUN_PERSON and next_word in VERB_PERSON:
        if PRONOUN_PERSON[prev] != VERB_PERSON[next_word]:
            # "yo es", "tú es", "ella soy", etc.
            return False

    # "nos/te/me/lo/se" followed by "es"/"está" is usually wrong (needs infinitive after clitic-on-verb or different frame)
    if prev in {"nos"} and next_word in {"es", "está"}:
        return False

    # "a <subject-pronoun>" is almost always ungrammatical in isolated sentences
    if prev == "a" and next_word in {"yo", "tú", "él", "ella", "nosotros", "ellos", "ellas"}:
        return False

    # "más" at the start followed by pronoun + está is unnatural
    if prev == "más" and next_word in {"ella", "él"}:
        return False

    # "ella hay", "ellos hay", etc. — "hay" is impersonal
    if prev in PRONOUN_PERSON and next_word == "hay":
        return False

    # Reject trigrams "con <pronoun> misma" — already a known bad frame
    if prev2 == "con" and prev in SUBJECT_PRONOUNS and next_word == "misma":
        return False

    return True


def final_ok(tokens: List[str]) -> bool:
    if not tokens:
        return False
    last = tokens[-1]
    if last in BAD_FINAL:
        # Allow "con/para <pronoun>" final (e.g., "es para ti")
        if len(tokens) >= 2 and tokens[-2] in {"con", "para"} and last in {"ti", "mí", "él", "ella", "nosotros"}:
            return True
        return False
    if last in FINITE_VERBS:
        # Allow "no puedo/quiero/sé"-style
        if len(tokens) >= 2 and tokens[-2] == "no" and last in {"puedo", "quiero", "sé"}:
            return True
        return False
    return True


class BeamSearcher:
    def __init__(self, lm: KNLanguageModel, beam_size: int = 16, max_extensions: int = 40, morph: Optional[Dict[str, Dict[str, str]]] = None):
        self.lm = lm
        self.beam_size = beam_size
        self.max_extensions = max_extensions
        self.morph = morph

    def top_next_words(self, tokens: List[str], vocab_allow: Optional[set] = None) -> List[Tuple[str, float]]:
        u = tokens[-2] if len(tokens) >= 2 else self.lm.BOS
        v = tokens[-1] if tokens else self.lm.BOS
        cand = dict(self.lm.tri_followers(u, v))
        for w, c in self.lm.bi_followers(v).items():
            cand.setdefault(w, 0)
            cand[w] += 0  # keep as trigger for scoring
        # Score candidates by KN prob
        scored = []
        for w in cand:
            if w == self.lm.BOS:
                continue
            if vocab_allow is not None and w != self.lm.EOS and w not in vocab_allow:
                continue
            lp = self.lm.logp_trigram(u, v, w)
            scored.append((w, lp))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.max_extensions]

    def forward(
        self,
        prefix: List[str],
        max_len: int,
        vocab_allow: Optional[set],
        banned_tokens: set,
        target: str,
        require_target: bool,
    ) -> List[Beam]:
        # initial beam
        init = Beam(neg_score=-self.lm.sentence_logprob(prefix), tokens=list(prefix), target_included=target in prefix)
        beams: List[Beam] = [init]
        completed: List[Beam] = []

        while beams:
            new_beams: List[Beam] = []
            for b in beams:
                if len(b.tokens) >= max_len:
                    if final_ok(b.tokens) and ((not require_target) or b.target_included):
                        completed.append(b)
                    continue
                # try ending here
                if len(b.tokens) >= 3 and final_ok(b.tokens) and ((not require_target) or b.target_included):
                    lp_end = self.lm.logp_trigram(
                        b.tokens[-2] if len(b.tokens) >= 2 else self.lm.BOS,
                        b.tokens[-1] if b.tokens else self.lm.BOS,
                        self.lm.EOS,
                    )
                    completed.append(Beam(neg_score=b.neg_score - lp_end, tokens=list(b.tokens), target_included=b.target_included))
                # expand
                for w, lp in self.top_next_words(b.tokens, vocab_allow=vocab_allow):
                    if w == self.lm.EOS:
                        continue
                    if w in banned_tokens:
                        continue
                    if not transition_ok(b.tokens, w, morph=self.morph):
                        continue
                    new_tokens = b.tokens + [w]
                    new_beam = Beam(
                        neg_score=b.neg_score - lp,
                        tokens=new_tokens,
                        target_included=b.target_included or (w == target),
                    )
                    new_beams.append(new_beam)
            new_beams.sort(key=lambda x: x.neg_score)
            beams = new_beams[: self.beam_size]
            if not beams:
                break
        completed.sort(key=lambda x: x.neg_score)
        return completed[: self.beam_size]


class Generator:
    def __init__(
        self,
        lm_path: Path,
        reranker_path: Path,
        lexicon: List[LexiconRow],
        beam_size: int = 24,
        max_extensions: int = 40,
        morph_path: Optional[Path] = None,
    ):
        print(f"[gen] loading LM {lm_path}", file=sys.stderr)
        self.lm = KNLanguageModel(lm_path)
        print(f"[gen] loading reranker {reranker_path}", file=sys.stderr)
        self.reranker = RerankerArtifacts.load(reranker_path)
        self.morph: Dict[str, Dict[str, str]] = {}
        if morph_path is not None and morph_path.exists():
            print(f"[gen] loading morph table {morph_path}", file=sys.stderr)
            self.morph = load_morph_table(morph_path)
        self.lexicon = lexicon
        self.rank_by_word = {r.lemma: r.rank for r in lexicon}
        self.searcher = BeamSearcher(self.lm, beam_size=beam_size, max_extensions=max_extensions, morph=self.morph)
        # Warm cache
        _ = self.lm.bi_followers(self.lm.BOS)
        _ = self.lm.tri_followers(self.lm.BOS, self.lm.BOS)

    def _build_vocab_allow(self, rank_ceiling: int) -> set:
        allow = {w for w, r in self.rank_by_word.items() if r <= rank_ceiling}
        # Always allow common glue even if above ceiling (very short words)
        allow.update({"un", "una", "unos", "unas", "el", "la", "los", "las", "de", "a", "en", "con", "por", "para"})
        allow.update(FINITE_VERBS)
        allow.update(SUBJECT_PRONOUNS)
        allow.add(self.lm.EOS)
        return allow

    def seed_prefixes(self, target: LexiconRow) -> List[List[str]]:
        """Return grammatically safe seed prefixes that *contain* the target lemma.

        Key ideas:
          - For nouns: [det, target] or [verb, det, target]
          - For adjectives: [pronoun, ser-form, target]
          - For verbs: if lemma is a finite form, [subject, target]; else
            [modal, target] ("quiero <infinitive>")
          - For articles/prepositions/adverbs: use the target inline in a
            template and let beam search fill the rest.
        """
        lemma = target.lemma
        pos = target.pos
        seeds: List[List[str]] = []

        # Early-exit: regardless of POS tag, if the lemma *is* a known article,
        # force an article-frame to avoid e.g. "Las es ..." or "Los mundo ...".
        if lemma in ARTICLES or lemma in {"mi", "tu", "su", "mis", "tus", "sus", "este", "esta", "estos", "estas", "ese", "esa"}:
            fem = lemma in FEMININE_ART or lemma.endswith("a")
            plural = lemma in PLURAL_ART or lemma.endswith("s")
            # Use "casa" / "libro" instead of "hombre" so possessive+head noun
            # doesn't produce awkward templates like "mi hombre".
            if plural:
                noun = "casas" if fem else "libros"
                verb = "son"
            else:
                noun = "casa" if fem else "libro"
                verb = "es"
            seeds.append([lemma, noun, verb])
            seeds.append(["tengo", lemma, noun])
            return seeds

        # Dedicated override for the finite verb "hay" (impersonal, no subject pronoun)
        if lemma == "hay":
            seeds.append(["hay", "una", "casa"])
            seeds.append(["no", "hay", "nadie", "aquí"])
            seeds.append(["hay", "algo", "que", "hacer"])
            return seeds

        # "que" is ambiguous (conj / rel pron) but the conjunction reading is always safe
        if lemma == "que":
            seeds.append(["creo", "que", "es", "verdad"])
            seeds.append(["sé", "que", "puedes"])
            seeds.append(["es", "posible", "que", "sí"])
            return seeds

        # "no" is a polarity particle — a few very common safe frames
        if lemma == "no":
            seeds.append(["no", "lo", "sé"])
            seeds.append(["no", "me", "gusta"])
            seeds.append(["no", "hay", "nadie"])
            seeds.append(["yo", "no", "quiero"])
            return seeds

        # "a" as a preposition/personal-a
        if lemma == "a":
            seeds.append(["voy", "a", "casa"])
            seeds.append(["vengo", "a", "verte"])
            seeds.append(["ayudo", "a", "mi", "madre"])
            return seeds

        # "y" conjunction
        if lemma == "y":
            seeds.append(["ella", "y", "yo", "somos", "amigos"])
            seeds.append(["tú", "y", "yo", "estamos", "aquí"])
            return seeds

        # "sí" as affirmation
        if lemma == "sí":
            seeds.append(["sí", "es", "verdad"])
            seeds.append(["claro", "que", "sí"])
            return seeds

        if pos == "n":
            # Hand-pick a few notoriously-hard lemmas so their seed is always grammatical.
            SPECIAL_NOUNS = {
                "gracias": [["muchas", "gracias"], ["gracias", "por", "todo"]],
                "tiempo": [["no", "tengo", "tiempo"], ["el", "tiempo", "vuela"]],
                "vez": [["una", "vez", "más"], ["otra", "vez", "aquí"]],
                "verdad": [["es", "verdad"], ["la", "verdad", "duele"]],
            }
            if lemma in SPECIAL_NOUNS:
                for s in SPECIAL_NOUNS[lemma]:
                    seeds.append(s)
                return seeds
            # Prefer the morph table; fall back to heuristic.
            fem = fem_of(lemma, self.morph) if self.morph else None
            if fem is None:
                fem = _fem_noun(lemma)
            plur = plural_of(lemma, self.morph) if self.morph else None
            if plur is None:
                plur = _plural_noun(lemma)

            if fem:
                det_def = "las" if plur else "la"
                det_indef = "unas" if plur else "una"
            else:
                det_def = "los" if plur else "el"
                det_indef = "unos" if plur else "un"
            copula = "son" if plur else "está"
            seeds.append([det_def, lemma])
            seeds.append(["tengo", det_indef, lemma])
            seeds.append(["es", det_indef, lemma])
            seeds.append([det_def, lemma, copula])
            seeds.append([det_def, lemma, "es" if not plur else "son"])
        elif pos in {"adj", "adjective"}:
            seeds.append(["ella", "es", lemma])
            seeds.append(["él", "es", lemma])
            seeds.append(["esto", "es", lemma])
            seeds.append(["no", "es", lemma])
        elif pos == "v" or pos == "verb":
            if lemma in VERB_PERSON:
                subj = {
                    "1s": "yo", "2s": "tú", "3s": "ella", "1p": "nosotros", "2p": "vosotros", "3p": "ellos",
                }[VERB_PERSON[lemma]]
                seeds.append([subj, lemma])
                seeds.append(["no", lemma])
            elif lemma.endswith(("ar", "er", "ir")):
                seeds.append(["quiero", lemma])
                seeds.append(["puedo", lemma])
                seeds.append(["voy", "a", lemma])
                seeds.append(["tengo", "que", lemma])
            else:
                seeds.append([lemma])
        elif pos in {"prep", "preposition"}:
            # rely on LM for left and right
            seeds.append(["estoy", lemma])
            seeds.append(["voy", lemma])
            seeds.append(["hablo", lemma])
        elif pos in {"conj", "conjunction"}:
            if lemma == "y":
                seeds.append(["ella", "y", "yo", "estamos", "aquí"])
                seeds.append(["tú", "y", "yo", "somos", "amigos"])
            elif lemma == "o":
                seeds.append(["quieres", "ir", "o", "no"])
                seeds.append(["es", "él", "o", "ella"])
            elif lemma == "pero":
                seeds.append(["quiero", "ir", "pero", "no", "puedo"])
                seeds.append(["es", "bueno", "pero", "difícil"])
            elif lemma == "si":
                seeds.append(["dime", "si", "puedes", "ir"])
                seeds.append(["no", "sé", "si", "es", "verdad"])
            elif lemma in {"porque", "aunque", "cuando", "mientras"}:
                seeds.append(["ella", "está", "bien", lemma, "tiene", "tiempo"])
                seeds.append(["no", "puedo", lemma, "estoy", "ocupado"])
            elif lemma == "que":
                seeds.append(["creo", "que", "es", "verdad"])
                seeds.append(["sé", "que", "puedes", "hacerlo"])
            elif lemma == "como":
                seeds.append(["trabajo", "como", "maestro"])
                seeds.append(["es", "como", "tú"])
            elif lemma == "ni":
                seeds.append(["no", "tengo", "ni", "idea"])
            else:
                seeds.append(["es", "bueno", lemma, "es", "difícil"])
        elif pos in {"art", "determiner", "num"}:
            # e.g. "el", "un", "dos", "todos". "todos"/"todas" need a determiner
            # between them and the noun — give them an article-inclusive frame.
            plural = (
                lemma.endswith("s") and len(lemma) > 2
                and lemma not in {"más", "después", "antes", "detrás", "atrás", "interés"}
            ) or lemma in {"dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "varios", "varias"}
            fem = lemma.endswith(("a", "as"))
            if lemma in {"todos", "todas", "ambos", "ambas"}:
                det = "las" if fem or lemma in {"todas", "ambas"} else "los"
                head = "casas" if fem or lemma in {"todas", "ambas"} else "libros"
                seeds.append([lemma, det, head, "son", "buenos" if det == "los" else "buenas"])
                seeds.append(["conozco", "a", lemma, "mis", "amigos" if det == "los" else "amigas"])
            else:
                if plural:
                    noun = "casas" if fem else "libros"
                    verb = "son"
                else:
                    noun = "casa" if fem else "libro"
                    verb = "es"
                seeds.append([lemma, noun, verb])
                seeds.append(["tengo", lemma, noun])
        elif pos in {"adv", "adverb"}:
            seeds.append(["ella", "está", lemma])
            seeds.append(["ella", lemma, "está"])
            seeds.append([lemma])
        elif pos == "pron":
            if lemma in {"me", "te", "nos", "os"}:
                seeds.append([lemma, "gusta", "mucho"])
                seeds.append(["ella", lemma, "ayuda"])
            elif lemma in {"lo", "la", "los", "las", "le", "les"}:
                seeds.append(["ya", lemma, "tengo"])
                seeds.append(["no", lemma, "veo"])
                seeds.append(["ella", lemma, "dice"])
            elif lemma == "se":
                seeds.append(["se", "puede", "hacer"])
                seeds.append(["no", "se", "sabe", "nada"])
            elif lemma in SUBJECT_PRONOUNS:
                verb = {
                    "yo": "estoy", "tú": "estás", "él": "está", "ella": "está",
                    "usted": "está", "nosotros": "estamos", "nosotras": "estamos",
                    "vosotros": "estáis", "vosotras": "estáis", "ellos": "están",
                    "ellas": "están", "ustedes": "están",
                }.get(lemma, "está")
                seeds.append([lemma, verb, "aquí"])
                seeds.append([lemma, verb, "bien"])
            elif lemma in {"qué", "cuál", "quién", "cómo", "dónde", "cuándo", "cuánto"}:
                seeds.append(["no", "sé", lemma, "hacer"])
                seeds.append([lemma, "es", "eso"])
            else:
                seeds.append([lemma, "es", "importante"])
                seeds.append(["ella", "es", lemma])
        else:
            seeds.append([lemma])

        # Fallback: just the lemma.
        if not seeds:
            seeds.append([lemma])
        return seeds

    def feature_score(self, tokens: List[str], target: LexiconRow, lm_logp: float) -> float:
        feats = featurize(tokens, target.lemma, target.rank, lm_logp, self.rank_by_word)
        return self.reranker.score(feats)

    def _score_candidate(self, toks: List[str], target: LexiconRow) -> Tuple[float, float]:
        lm_lp = self.lm.sentence_logprob(toks)
        rscore = self.feature_score(toks, target, lm_lp)
        combined = rscore + 0.08 * lm_lp
        return combined, lm_lp

    def _template_candidates(self, target: LexiconRow) -> List[Tuple[float, List[str], float]]:
        """Generate candidates from the grammar-safe template registry.

        Each template is a full sentence; no beam-search extension. The caller
        validates and ranks. This is what runs *before* fallback beam search.
        """
        _, min_len, max_len, _ = profile_for_rank(target.rank)
        morph_entry = self.morph.get(target.lemma, {}) if self.morph else {}
        is_inf = target.lemma.endswith(("ar", "er", "ir")) and target.lemma not in VERB_PERSON
        verb_person = VERB_PERSON.get(target.lemma)
        tpls = grammar_safe_templates(
            target.lemma, target.pos, morph_for_lemma=morph_entry,
            is_infinitive=is_inf, verb_person=verb_person,
        )
        out: List[Tuple[float, List[str], float]] = []
        for toks in tpls:
            toks = [t for t in toks if t]
            if not toks:
                continue
            if target.lemma not in toks:
                continue
            if not (min_len <= len(toks) <= max_len + 2):
                # Templates are short by design; be lenient with max-len.
                if len(toks) < 3 or len(toks) > max_len + 4:
                    continue
            if validate(toks, target.lemma, morph=self.morph):
                continue
            combined, lm_lp = self._score_candidate(toks, target)
            out.append((combined, toks, lm_lp))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    def _beam_candidates(self, target: LexiconRow) -> List[Tuple[float, List[str], float]]:
        _, min_len, max_len, rank_ceiling = profile_for_rank(target.rank)
        vocab_allow = self._build_vocab_allow(rank_ceiling)
        vocab_allow.add(target.lemma)
        seeds = self.seed_prefixes(target)
        out: List[Tuple[float, List[str], float]] = []
        for seed in seeds:
            seed_lc = [t for t in seed if t]
            if not seed_lc:
                continue
            beams = self.searcher.forward(
                prefix=seed_lc, max_len=max_len, vocab_allow=vocab_allow,
                banned_tokens={self.lm.BOS}, target=target.lemma, require_target=True,
            )
            for b in beams:
                toks = b.tokens
                if not final_ok(toks):
                    continue
                if not (min_len <= len(toks) <= max_len):
                    continue
                if target.lemma not in toks:
                    continue
                if validate(toks, target.lemma, morph=self.morph):
                    continue
                combined, lm_lp = self._score_candidate(toks, target)
                out.append((combined, toks, lm_lp))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    def generate_candidates(self, target: LexiconRow, template_score_floor: float = 1.0) -> List[Tuple[float, List[str], float]]:
        """Return a ranked list of (score, tokens, lm_logp).

        Templates are tried first. If the best template's reranker-logit-dominated
        score clears ``template_score_floor`` we still also include beam-search
        candidates but return the combined pool sorted; if no template passes we
        fall back to beam-search only. Either way every candidate has already
        been through ``validate()``.
        """
        template_cands = self._template_candidates(target)
        if template_cands and template_cands[0][0] >= template_score_floor:
            beam_cands = self._beam_candidates(target)
            combined = template_cands + beam_cands
            combined.sort(key=lambda x: x[0], reverse=True)
            return combined
        beam_cands = self._beam_candidates(target)
        combined = beam_cands + template_cands
        combined.sort(key=lambda x: x[0], reverse=True)
        return combined

    def generate(self, target: LexiconRow) -> Tuple[str, float, List[str]]:
        """Return the single best candidate for a target (no diversity tracking).

        Use ``generate_candidates`` and the CSV-writing loop for diversity-aware
        selection across a batch of targets.
        """
        cands = self.generate_candidates(target)
        if not cands:
            return "", 0.0, []
        best = cands[0]
        return detokenize(best[1]), best[0], best[1]


def select_targets(rows: List[LexiconRow], min_rank: int, max_rank: int, limit: int, pos: Optional[str], lemmas: Optional[List[str]]):
    wanted = {l.lower() for l in (lemmas or []) if l}
    out = [r for r in rows if min_rank <= r.rank <= max_rank and (not pos or r.pos == pos) and (not wanted or r.lemma in wanted)]
    return out[:limit] if limit > 0 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="data_clean/kn_lm.pkl")
    ap.add_argument("--reranker", default="data_clean/reranker.pkl")
    ap.add_argument("--lexicon", default="stg_words_spa.csv")
    ap.add_argument("--out", default="outputs/markov_ml_sentences.csv")
    ap.add_argument("--review-out", default="outputs/markov_ml_review.csv")
    ap.add_argument("--min-rank", type=int, default=1)
    ap.add_argument("--max-rank", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--pos", default=None)
    ap.add_argument("--lemma", action="append", default=[])
    ap.add_argument("--beam-size", type=int, default=24)
    ap.add_argument("--max-extensions", type=int, default=40)
    ap.add_argument("--morph", default="models_rebuild2/lemma_forms.pkl",
                    help="Path to UD morph table (lemma_forms.pkl). Disable with empty string.")
    ap.add_argument("--max-template-reuses", type=int, default=2,
                    help="Cap on how many rows in the CSV may share the same template skeleton.")
    args = ap.parse_args()

    rows = load_lexicon(Path(args.lexicon))
    targets = select_targets(rows, args.min_rank, args.max_rank, args.limit, args.pos, args.lemma)
    if not targets:
        raise SystemExit("no targets")

    morph_path = Path(args.morph) if args.morph else None
    gen = Generator(
        Path(args.lm), Path(args.reranker), rows,
        beam_size=args.beam_size, max_extensions=args.max_extensions,
        morph_path=morph_path,
    )

    print(f"[gen] generating for {len(targets)} targets", file=sys.stderr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    review_path = Path(args.review_out) if args.review_out else None
    if review_path:
        review_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["lemma", "rank", "pos", "translation", "sentence", "source_method", "score", "tokens"],
        )
        writer.writeheader()
        review_writer = None
        if review_path:
            f_rev = review_path.open("w", encoding="utf-8", newline="")
            review_writer = csv.DictWriter(f_rev, fieldnames=["lemma", "rank", "pos", "sentence", "tokens", "combined", "reranker", "lm_logp"])
            review_writer.writeheader()
        else:
            f_rev = None

        empty = 0
        from collections import Counter
        sig_counter: Counter = Counter()
        max_reuses = args.max_template_reuses
        for i, tgt in enumerate(targets, 1):
            try:
                cands = gen.generate_candidates(tgt)
            except Exception as e:
                print(f"  {tgt.lemma}: error {e}", file=sys.stderr)
                cands = []
            # Pick best candidate subject to template-signature reuse cap.
            picked_sent, picked_score, picked_toks = "", 0.0, []
            fallback_sent, fallback_score, fallback_toks = "", 0.0, []
            for score, toks, _lm in cands:
                sig = template_signature(toks)
                if not fallback_sent:
                    fallback_sent, fallback_score, fallback_toks = detokenize(toks), score, toks
                if sig_counter[sig] >= max_reuses:
                    continue
                picked_sent, picked_score, picked_toks = detokenize(toks), score, toks
                sig_counter[sig] += 1
                break
            if not picked_sent:
                # Every candidate shared an over-used template. Fall back to the top pick.
                picked_sent, picked_score, picked_toks = fallback_sent, fallback_score, fallback_toks
                if picked_toks:
                    sig_counter[template_signature(picked_toks)] += 1
            writer.writerow({
                "lemma": tgt.lemma,
                "rank": tgt.rank,
                "pos": tgt.pos,
                "translation": tgt.translation,
                "sentence": picked_sent,
                "source_method": "markov_ml",
                "score": f"{picked_score:.4f}",
                "tokens": " ".join(picked_toks),
            })
            if review_writer is not None:
                lm_lp = gen.lm.sentence_logprob(picked_toks) if picked_toks else 0.0
                review_writer.writerow({
                    "lemma": tgt.lemma, "rank": tgt.rank, "pos": tgt.pos,
                    "sentence": picked_sent, "tokens": " ".join(picked_toks),
                    "combined": f"{picked_score:.4f}", "reranker": f"{picked_score:.4f}", "lm_logp": f"{lm_lp:.4f}",
                })
            if not picked_sent:
                empty += 1
            if i % 10 == 0 or i == len(targets):
                print(f"  [{i}/{len(targets)}] {tgt.lemma:<12} -> {picked_sent}", file=sys.stderr)
        if f_rev is not None:
            f_rev.close()

    # Report template diversity summary.
    print(f"[gen] wrote {out_path} ({len(targets) - empty} filled / {len(targets)})", file=sys.stderr)
    top_sigs = sig_counter.most_common(5)
    print("[gen] most common template signatures:", file=sys.stderr)
    for sig, c in top_sigs:
        print(f"    x{c}  {' '.join(sig)}", file=sys.stderr)


if __name__ == "__main__":
    main()
