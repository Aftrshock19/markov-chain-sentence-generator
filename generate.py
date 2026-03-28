#!/usr/bin/env python3
import argparse
import csv
import math
import os
import pickle
import random
import re
import statistics
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable


WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
STRIP_RE = re.compile(r'^[¿¡"“”\'\(\[\{]+|[.,;:!?"”\'\)\]\}]+$')
ARTICLE_SET = {"el", "la", "los", "las", "un", "una", "unos", "unas", "este", "esta", "estos", "estas"}
DEMONSTRATIVE_SET = {"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas"}
PRONOUNS = ["yo", "tú", "él", "ella", "nosotros", "ellos"]
COMMON_ADVERBS = ["bien", "mucho", "siempre", "ahora", "aquí"]
VERB_OBJECT_PREFS = {
    "comer": ["food"],
    "beber": ["drink", "food"],
    "leer": ["text", "object"],
    "visitar": ["place"],
    "vivir": ["place"],
    "comprar": ["object", "food", "clothing"],
    "abrir": ["object"],
    "cerrar": ["object"],
    "ver": ["person", "animal", "object"],
    "conocer": ["person", "place"],
    "buscar": ["person", "object"],
    "escribir": ["text"],
    "llevar": ["clothing", "object"],
    "conducir": ["vehicle"],
}
ADJ_SUBJECT_PREFS = {
    "cansado": ["person", "animal"],
    "grande": ["object", "place", "animal"],
    "pequeño": ["object", "animal"],
    "difícil": ["abstract", "activity"],
    "delicioso": ["food", "drink"],
    "rápido": ["person", "animal", "vehicle"],
    "bonito": ["object", "place", "person"],
}
COPULA_FORMS = {"soy", "eres", "es", "somos", "son", "estoy", "estás", "está", "estamos", "están"}
PERSON_NUMBER_TO_CODE = {
    ("1", "Sing"): "1sg",
    ("2", "Sing"): "2sg",
    ("3", "Sing"): "3sg",
    ("1", "Plur"): "1pl",
    ("2", "Plur"): "2pl",
    ("3", "Plur"): "3pl",
}
PERSON_CODE_TO_SUBJECT = {
    "1sg": "yo",
    "2sg": "tú",
    "3sg": "ella",
    "1pl": "nosotros",
    "2pl": "ustedes",
    "3pl": "ellos",
}
SUBJECT_FEATURES = {
    "yo": {"person_code": "1sg", "gender": None, "number": "sg"},
    "tú": {"person_code": "2sg", "gender": None, "number": "sg"},
    "él": {"person_code": "3sg", "gender": "m", "number": "sg"},
    "ella": {"person_code": "3sg", "gender": "f", "number": "sg"},
    "nosotros": {"person_code": "1pl", "gender": "m", "number": "pl"},
    "ustedes": {"person_code": "3pl", "gender": None, "number": "pl"},
    "ellos": {"person_code": "3pl", "gender": "m", "number": "pl"},
}
PUNCT_ATTACH_LEFT = {".", ",", ";", ":", "!", "?", "%", "…"}
PUNCT_ATTACH_RIGHT = {"¿", "¡", "(", "[", "{"}
SPECIAL_VERB_LEMMAS = {"ser", "estar", "haber", "ir", "poder", "saber", "creer"}
PLACE_PREP_VERBS = {"ir", "venir", "vivir", "llegar", "entrar", "salir", "quedar"}


@dataclass
class Lexeme:
    lemma: str
    rank: int
    pos: str
    translation: str = ""
    gender: Optional[str] = None
    semantic_class: Optional[str] = None
    verb_type: Optional[str] = None
    required_prep: Optional[str] = None
    is_reflexive: bool = False
    canonical_lemma: Optional[str] = None


@dataclass
class DifficultyProfile:
    band: str
    min_len: int
    max_len: int
    filler_ceil: int
    avg_ceil: int
    slot_range: Tuple[int, int]
    complexity: int

    @property
    def ideal_length(self) -> int:
        return (self.min_len + self.max_len) // 2


@dataclass
class Candidate:
    lemma: str
    rank: int
    pos: str
    band: str
    translation: str
    sentence: str
    target_form: str
    target_index: int
    support_ranks: List[int]
    avg_support_rank: float
    max_support_rank: int
    template_id: str
    source_method: str
    score: float = 0.0
    english_sentence: str = ""
    english_source_method: str = ""
    parallel_pair_id: Optional[int] = None
    english_score: float = 0.0


BANDS = [
    (1, 100, DifficultyProfile("A1", 3, 5, 150, 100, (0, 1), 1)),
    (101, 500, DifficultyProfile("A2", 3, 6, 500, 300, (0, 2), 2)),
    (501, 2000, DifficultyProfile("B1", 4, 7, 2000, 1200, (1, 3), 3)),
    (2001, 8000, DifficultyProfile("B2", 5, 8, 5000, 3000, (1, 4), 4)),
    (8001, 20000, DifficultyProfile("C1", 5, 9, 10000, 6000, (2, 5), 5)),
    (20001, 10**9, DifficultyProfile("C2", 6, 10, 20000, 12000, (2, 6), 6)),
]


IRREGULAR_PRESENT = {
    "ser": {"1sg": "soy", "2sg": "eres", "3sg": "es", "1pl": "somos", "3pl": "son"},
    "estar": {"1sg": "estoy", "2sg": "estás", "3sg": "está", "1pl": "estamos", "3pl": "están"},
    "ir": {"1sg": "voy", "2sg": "vas", "3sg": "va", "1pl": "vamos", "3pl": "van"},
    "tener": {"1sg": "tengo", "2sg": "tienes", "3sg": "tiene", "1pl": "tenemos", "3pl": "tienen"},
    "haber": {"1sg": "he", "2sg": "has", "3sg": "ha", "1pl": "hemos", "3pl": "han"},
    "poder": {"1sg": "puedo", "2sg": "puedes", "3sg": "puede", "1pl": "podemos", "3pl": "pueden"},
    "querer": {"1sg": "quiero", "2sg": "quieres", "3sg": "quiere", "1pl": "queremos", "3pl": "quieren"},
    "decir": {"1sg": "digo", "2sg": "dices", "3sg": "dice", "1pl": "decimos", "3pl": "dicen"},
    "hacer": {"1sg": "hago", "2sg": "haces", "3sg": "hace", "1pl": "hacemos", "3pl": "hacen"},
    "saber": {"1sg": "sé", "2sg": "sabes", "3sg": "sabe", "1pl": "sabemos", "3pl": "saben"},
    "poner": {"1sg": "pongo", "2sg": "pones", "3sg": "pone", "1pl": "ponemos", "3pl": "ponen"},
    "venir": {"1sg": "vengo", "2sg": "vienes", "3sg": "viene", "1pl": "venimos", "3pl": "vienen"},
    "salir": {"1sg": "salgo", "2sg": "sales", "3sg": "sale", "1pl": "salimos", "3pl": "salen"},
    "dar": {"1sg": "doy", "2sg": "das", "3sg": "da", "1pl": "damos", "3pl": "dan"},
    "ver": {"1sg": "veo", "2sg": "ves", "3sg": "ve", "1pl": "vemos", "3pl": "ven"},
    "conocer": {"1sg": "conozco", "2sg": "conoces", "3sg": "conoce", "1pl": "conocemos", "3pl": "conocen"},
    "dormir": {"1sg": "duermo", "2sg": "duermes", "3sg": "duerme", "1pl": "dormimos", "3pl": "duermen"},
    "pedir": {"1sg": "pido", "2sg": "pides", "3sg": "pide", "1pl": "pedimos", "3pl": "piden"},
    "pensar": {"1sg": "pienso", "2sg": "piensas", "3sg": "piensa", "1pl": "pensamos", "3pl": "piensan"},
    "encontrar": {"1sg": "encuentro", "2sg": "encuentras", "3sg": "encuentra", "1pl": "encontramos", "3pl": "encuentran"},
    "volver": {"1sg": "vuelvo", "2sg": "vuelves", "3sg": "vuelve", "1pl": "volvemos", "3pl": "vuelven"},
}


def get_profile(rank: int) -> DifficultyProfile:
    for lo, hi, profile in BANDS:
        if lo <= rank <= hi:
            return profile
    return BANDS[-1][2]


def allowed_support_rank(target_rank: int, profile: DifficultyProfile) -> int:
    return min(profile.filler_ceil, max(300, int(target_rank * 0.8)))


def is_word_token(token: str) -> bool:
    return bool(WORD_RE.search(token))


def normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = token.strip("¿¡\"“”'()[]{}.,;:!?")
    return token


class SentenceGenerator:
    def __init__(self, lexicon_path: str, models_dir: str, seed: int = 42):
        self.random = random.Random(seed)
        self.lexicon: Dict[str, Lexeme] = self._load_lexicon(lexicon_path, models_dir)
        self.lemma_forms = self._load_pickle(models_dir, "lemma_forms.pkl")
        self.lemma_contexts = self._load_pickle(models_dir, "lemma_contexts.pkl")
        self.bigrams = self._load_pickle(models_dir, "bigrams.pkl")
        self.trigrams = self._load_pickle(models_dir, "trigrams.pkl")
        self.pos_transitions = self._load_pickle(models_dir, "pos_transitions.pkl")
        self.enriched = self._load_pickle(models_dir, "enriched_lexicon.pkl")
        self.form_to_lemmas = self._build_form_index()
        self.generation_lexicon = self._build_generation_lexicon()
        self.pos_buckets = self._build_pos_buckets()

    def _load_pickle(self, models_dir: str, name: str):
        path = os.path.join(models_dir, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_lexicon(self, lexicon_path: str, models_dir: str) -> Dict[str, Lexeme]:
        enriched_path = os.path.join(models_dir, "enriched_lexicon.pkl")
        with open(enriched_path, "rb") as f:
            enriched = pickle.load(f)
        lexicon: Dict[str, Lexeme] = {}
        with open(lexicon_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lemma = row.get("lemma", "").strip().lower()
                if not lemma:
                    continue
                canonical = (row.get("original_lemma") or lemma).strip().lower()
                rank = int(row.get("rank", 99999) or 99999)
                base = enriched.get(canonical) or enriched.get(lemma, {})
                pos = (row.get("pos") or base.get("pos") or "").strip().lower()
                translation = (row.get("translation") or base.get("translation") or "").strip()
                lexicon[lemma] = Lexeme(
                    lemma=lemma,
                    rank=rank,
                    pos=pos,
                    translation=translation,
                    gender=base.get("gender"),
                    semantic_class=base.get("semantic_class"),
                    verb_type=base.get("verb_type"),
                    required_prep=base.get("required_prep"),
                    is_reflexive=bool(base.get("is_reflexive", False)),
                    canonical_lemma=canonical,
                )
        return lexicon

    def _build_generation_lexicon(self) -> Dict[str, Lexeme]:
        generation: Dict[str, Lexeme] = {}
        for lex in self.lexicon.values():
            canonical = lex.canonical_lemma or lex.lemma
            candidate = Lexeme(
                lemma=canonical,
                rank=lex.rank,
                pos=lex.pos,
                translation=lex.translation,
                gender=lex.gender,
                semantic_class=lex.semantic_class,
                verb_type=lex.verb_type,
                required_prep=lex.required_prep,
                is_reflexive=lex.is_reflexive,
                canonical_lemma=canonical,
            )
            existing = generation.get(canonical)
            if existing is None or candidate.rank < existing.rank:
                generation[canonical] = candidate
        return generation

    def _build_form_index(self) -> Dict[str, List[str]]:
        form_to_lemmas: Dict[str, List[str]] = {}
        for lemma, forms in self.lemma_forms.items():
            seen = set()
            for entry in forms:
                form = normalize_token(entry.get("form", ""))
                if not form or form in seen:
                    continue
                seen.add(form)
                form_to_lemmas.setdefault(form, []).append(lemma)
        return form_to_lemmas

    def _build_pos_buckets(self) -> Dict[str, List[Lexeme]]:
        buckets: Dict[str, List[Lexeme]] = {}
        for lex in self.generation_lexicon.values():
            buckets.setdefault(lex.pos, []).append(lex)
        for values in buckets.values():
            values.sort(key=lambda x: x.rank)
        return buckets

    def lookup_lemma(self, surface: str) -> Optional[str]:
        s = normalize_token(surface)
        if not s:
            return None
        if s in self.lexicon:
            return s
        lemmas = self.form_to_lemmas.get(s, [])
        if not lemmas:
            return None
        return min(lemmas, key=lambda l: self.lexicon.get(l, Lexeme(l, 99999, "")).rank)

    def lookup_rank(self, surface: str) -> int:
        lemma = self.lookup_lemma(surface)
        if not lemma:
            return 99999
        return self.lexicon.get(lemma, Lexeme(lemma, 99999, "")).rank

    def lookup_pos(self, surface: str) -> str:
        lemma = self.lookup_lemma(surface)
        if not lemma:
            return ""
        return self.lexicon.get(lemma, Lexeme(lemma, 99999, "")).pos

    def choose_article(self, gender: Optional[str], definite: bool = True, demonstrative: bool = False) -> str:
        gender = gender or "m"
        if demonstrative:
            return "esta" if gender == "f" else "este"
        if definite:
            return "la" if gender == "f" else "el"
        return "una" if gender == "f" else "un"

    def inflect_adj(self, lemma: str, gender: Optional[str], number: str = "sg") -> str:
        gender = gender or "m"
        if lemma.endswith("o"):
            stem = lemma[:-1]
            mapping = {
                ("m", "sg"): "o",
                ("f", "sg"): "a",
                ("m", "pl"): "os",
                ("f", "pl"): "as",
            }
            return stem + mapping[(gender, number)]
        if number == "pl":
            if lemma.endswith(("e", "a", "o")):
                return lemma + "s"
            return lemma + "es"
        return lemma

    def conjugate_present(self, lemma: str, person: str = "3sg") -> str:
        if lemma in IRREGULAR_PRESENT and person in IRREGULAR_PRESENT[lemma]:
            return IRREGULAR_PRESENT[lemma][person]
        if lemma.endswith("ar"):
            endings = {"1sg": "o", "2sg": "as", "3sg": "a", "1pl": "amos", "3pl": "an"}
        elif lemma.endswith("er"):
            endings = {"1sg": "o", "2sg": "es", "3sg": "e", "1pl": "emos", "3pl": "en"}
        elif lemma.endswith("ir"):
            endings = {"1sg": "o", "2sg": "es", "3sg": "e", "1pl": "imos", "3pl": "en"}
        else:
            return lemma
        return lemma[:-2] + endings.get(person, endings["3sg"])

    def apply_contractions(self, tokens: List[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens):
                pair = (tokens[i].lower(), tokens[i + 1].lower())
                if pair == ("a", "el"):
                    out.append("al")
                    i += 2
                    continue
                if pair == ("de", "el"):
                    out.append("del")
                    i += 2
                    continue
            out.append(tokens[i])
            i += 1
        return out

    def canonical_lemma_for(self, lex: Lexeme) -> str:
        canonical = lex.canonical_lemma or lex.lemma
        if normalize_token(canonical) == normalize_token(lex.lemma):
            return canonical
        surface = normalize_token(lex.lemma)
        if any(normalize_token(entry.get("form", "")) == surface for entry in self.lemma_forms.get(canonical, [])):
            return canonical
        return lex.lemma

    def target_form_metadata(self, lex: Lexeme) -> Dict[str, str]:
        canonical = self.canonical_lemma_for(lex)
        surface = normalize_token(lex.lemma)
        for entry in self.lemma_forms.get(canonical, []):
            if normalize_token(entry.get("form", "")) == surface:
                return entry.get("morph") or {}
        return {}

    def get_known_lexeme(self, lemma: Optional[str]) -> Optional[Lexeme]:
        if not lemma:
            return None
        return self.lexicon.get(lemma) or self.generation_lexicon.get(lemma)

    def person_code_from_morph(self, morph: Dict[str, str]) -> Optional[str]:
        person = str(morph.get("Person", ""))
        number = morph.get("Number")
        return PERSON_NUMBER_TO_CODE.get((person, number))

    def subject_for_target(self, target: Lexeme, fallback: str = "ella") -> Tuple[str, str]:
        morph = self.target_form_metadata(target)
        person_code = self.person_code_from_morph(morph)
        if person_code:
            return PERSON_CODE_TO_SUBJECT.get(person_code, fallback), person_code
        fallback_features = SUBJECT_FEATURES.get(fallback, SUBJECT_FEATURES["ella"])
        return fallback, fallback_features["person_code"]

    def inflect_adj_for_subject(self, lemma: str, subject: str) -> str:
        features = SUBJECT_FEATURES.get(subject, SUBJECT_FEATURES["ella"])
        return self.inflect_adj(lemma, features["gender"], features["number"])

    def target_verb_form(self, target: Lexeme, person_code: str) -> str:
        canonical = self.canonical_lemma_for(target)
        if normalize_token(target.lemma) != normalize_token(canonical):
            return target.lemma
        morph = self.target_form_metadata(target)
        if morph.get("VerbForm") and morph.get("VerbForm") != "Inf":
            return target.lemma
        return self.conjugate_present(canonical, person_code)

    def can_template_target(self, target: Lexeme) -> bool:
        if target.pos != "v":
            return True
        canonical = self.canonical_lemma_for(target)
        if normalize_token(canonical) != normalize_token(target.lemma):
            return True
        morph = self.target_form_metadata(target)
        if morph.get("VerbForm") == "Inf":
            return True
        return canonical in IRREGULAR_PRESENT or canonical.endswith(("ar", "er", "ir"))

    def detokenize(self, tokens: List[str]) -> str:
        parts: List[str] = []
        for token in tokens:
            if not token:
                continue
            if not parts:
                parts.append(token)
                continue
            if token in PUNCT_ATTACH_LEFT:
                parts[-1] += token
                continue
            if parts[-1] in PUNCT_ATTACH_RIGHT:
                parts[-1] += token
                continue
            parts.append(token)
        return " ".join(parts)

    def contexts_for_target(self, target: Lexeme) -> List[Dict]:
        seen = set()
        out: List[Dict] = []
        keys = [target.lemma]
        canonical = self.canonical_lemma_for(target)
        if canonical not in keys:
            keys.append(canonical)
        for key in keys:
            for ctx in self.lemma_contexts.get(key, []):
                marker = (tuple(ctx.get("tokens") or []), ctx.get("index", -1))
                if marker in seen:
                    continue
                seen.add(marker)
                out.append(ctx)
        return out

    def participle_form(self, lemma: str) -> Optional[str]:
        for entry in self.lemma_forms.get(lemma, []):
            morph = entry.get("morph") or {}
            if morph.get("VerbForm") == "Part" and morph.get("Number", "Sing") == "Sing":
                gender = morph.get("Gender")
                if gender in {None, "Masc"}:
                    return entry.get("form")
        if lemma.endswith("ar"):
            return lemma[:-2] + "ado"
        if lemma.endswith(("er", "ir")):
            return lemma[:-2] + "ido"
        return None

    def special_verb_candidate(self, target: Lexeme, allowed: int, template_id: str, source_method: str) -> Optional[Candidate]:
        canonical = self.canonical_lemma_for(target)
        subject, person_code = self.subject_for_target(target)
        verb = self.target_verb_form(target, person_code)

        if canonical == "haber":
            if normalize_token(verb) == "hay":
                noun = self.pick_candidate("n", allowed, exclude={target.lemma, canonical})
                if not noun:
                    return None
                article = self.choose_article(noun.gender, definite=False)
                return self.build_candidate(target, [verb, article, noun.lemma], f"{template_id}_haber_existential", source_method, 0)
            aux_target = self.pick_candidate("v", allowed, exclude={target.lemma, canonical, "ser", "haber"})
            if not aux_target:
                return None
            participle = self.participle_form(aux_target.lemma)
            if not participle:
                return None
            obj_classes = VERB_OBJECT_PREFS.get(aux_target.lemma)
            obj = self.pick_candidate("n", allowed, semantic_classes=obj_classes, exclude={target.lemma, canonical, aux_target.lemma})
            if obj:
                article = self.choose_article(obj.gender, definite=False)
                tokens = [subject, verb, participle, article, obj.lemma]
            else:
                tokens = [subject, verb, participle, "aquí"]
            return self.build_candidate(target, tokens, f"{template_id}_haber_perfect", source_method, 1)

        if canonical == "ser":
            adj = self.pick_candidate("adj", allowed, exclude={target.lemma, canonical})
            if adj:
                return self.build_candidate(
                    target,
                    [subject, verb, self.inflect_adj_for_subject(adj.lemma, subject)],
                    f"{template_id}_ser_adj",
                    source_method,
                    1,
                )
            noun = self.pick_candidate("n", allowed, semantic_classes=["person", "object"], exclude={target.lemma, canonical})
            if noun:
                article = self.choose_article(noun.gender, definite=False)
                return self.build_candidate(target, [subject, verb, article, noun.lemma], f"{template_id}_ser_noun", source_method, 1)
            return None

        if canonical == "estar":
            place = self.pick_candidate("n", allowed, semantic_classes=["place"], exclude={target.lemma, canonical})
            if place:
                article = self.choose_article(place.gender, definite=True)
                return self.build_candidate(target, [subject, verb, "en", article, place.lemma], f"{template_id}_estar_place", source_method, 1)
            return self.build_candidate(target, [subject, verb, "aquí"], f"{template_id}_estar_here", source_method, 1)

        if canonical == "ir":
            place = self.pick_candidate("n", allowed, semantic_classes=["place"], exclude={target.lemma, canonical})
            if not place:
                return None
            article = self.choose_article(place.gender, definite=True)
            return self.build_candidate(target, [subject, verb, "a", article, place.lemma], f"{template_id}_ir_place", source_method, 1)

        if canonical == "poder":
            infinitive = self.pick_candidate("v", allowed, exclude={target.lemma, canonical, "ser", "estar", "haber"})
            if not infinitive:
                return None
            obj_classes = VERB_OBJECT_PREFS.get(infinitive.lemma)
            if obj_classes:
                obj = self.pick_candidate("n", allowed, semantic_classes=obj_classes, exclude={target.lemma, canonical, infinitive.lemma})
                if obj:
                    article = self.choose_article(obj.gender, definite=False)
                    return self.build_candidate(target, [subject, verb, infinitive.lemma, article, obj.lemma], f"{template_id}_poder_inf_obj", source_method, 1)
            return self.build_candidate(target, [subject, verb, infinitive.lemma], f"{template_id}_poder_inf", source_method, 1)

        if canonical == "saber":
            return self.build_candidate(target, [subject, verb, "la", "verdad"], f"{template_id}_saber_truth", source_method, 1)

        if canonical == "creer":
            return self.build_candidate(target, [subject, verb, "eso"], f"{template_id}_creer_object", source_method, 1)

        return None

    def pick_candidate(
        self,
        pos: str,
        rank_ceiling: int,
        semantic_classes: Optional[List[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[Lexeme]:
        exclude = {normalize_token(x) for x in (exclude or []) if x}
        bucket = self.pos_buckets.get(pos, [])
        candidates = [
            x
            for x in bucket
            if x.rank <= rank_ceiling
            and normalize_token(x.lemma) not in exclude
            and normalize_token(self.canonical_lemma_for(x)) not in exclude
        ]
        if semantic_classes:
            preferred = [x for x in candidates if x.semantic_class in semantic_classes]
            if preferred:
                candidates = preferred
        if not candidates:
            return None
        weights = [1.0 / max(1, c.rank) for c in candidates[:500]]
        trimmed = candidates[:500]
        return self.random.choices(trimmed, weights=weights, k=1)[0]

    def score_sequence(self, tokens: List[str]) -> float:
        words = ["<START>"] + [normalize_token(t) for t in tokens if is_word_token(t)] + ["<END>"]
        if len(words) <= 2:
            return -10.0
        tri_counts = self.trigrams["counts"]
        bi_counts = self.bigrams["counts"]
        total = 0.0
        for i in range(1, len(words)):
            if i >= 2 and (words[i - 2], words[i - 1], words[i]) in tri_counts:
                total += math.log(tri_counts[(words[i - 2], words[i - 1], words[i])] + 1)
            elif (words[i - 1], words[i]) in bi_counts:
                total += math.log(bi_counts[(words[i - 1], words[i])] + 1)
            else:
                total -= 10.0
        return total / max(1, len(words) - 1)

    def build_candidate(
        self,
        target: Lexeme,
        tokens: List[str],
        template_id: str,
        source_method: str,
        target_index: Optional[int] = None,
    ) -> Optional[Candidate]:
        clean_tokens = [t for t in tokens if t]
        clean_tokens = self.apply_contractions(clean_tokens)
        word_tokens = [t for t in clean_tokens if is_word_token(t)]
        if not word_tokens:
            return None
        if target_index is None:
            target_index = next((i for i, t in enumerate(clean_tokens) if normalize_token(t) == normalize_token(target.lemma)), -1)
        target_form = clean_tokens[target_index] if 0 <= target_index < len(clean_tokens) else target.lemma
        support_ranks: List[int] = []
        for i, tok in enumerate(clean_tokens):
            if not is_word_token(tok):
                continue
            if i == target_index:
                continue
            support_ranks.append(self.lookup_rank(tok))
        avg_support = statistics.mean(support_ranks) if support_ranks else 0.0
        max_support = max(support_ranks) if support_ranks else 0
        sentence = self.detokenize(clean_tokens)
        sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        return Candidate(
            lemma=target.lemma,
            rank=target.rank,
            pos=target.pos,
            band=get_profile(target.rank).band,
            translation=target.translation,
            sentence=sentence,
            target_form=target_form,
            target_index=target_index,
            support_ranks=support_ranks,
            avg_support_rank=avg_support,
            max_support_rank=max_support,
            template_id=template_id,
            source_method=source_method,
        )

    def validate(self, candidate: Candidate) -> Tuple[bool, List[float]]:
        profile = get_profile(candidate.rank)
        allowed = allowed_support_rank(candidate.rank, profile)
        penalties: List[float] = []
        words = [normalize_token(t) for t in candidate.sentence.split() if is_word_token(t)]
        words = [w for w in words if w]
        if not (profile.min_len <= len(words) <= profile.max_len):
            return False, penalties
        target_forms = {candidate.lemma, normalize_token(candidate.target_form)}
        target_forms.update({normalize_token(f.get("form", "")) for f in self.lemma_forms.get(candidate.lemma, [])})
        if not any(w in target_forms for w in words):
            return False, penalties
        if candidate.max_support_rank > allowed:
            return False, penalties
        if candidate.avg_support_rank > profile.avg_ceil:
            return False, penalties
        for a, b in zip(words, words[1:]):
            if a == b:
                return False, penalties
        if not any(self.lookup_pos(w) == "v" or w in COPULA_FORMS for w in words):
            return False, penalties
        if candidate.pos == "v" and candidate.source_method != "retrieved_corpus":
            target = self.lexicon.get(candidate.lemma)
            if target:
                person_code = self.person_code_from_morph(self.target_form_metadata(target))
                if person_code and words:
                    subject = words[0]
                    expected = SUBJECT_FEATURES.get(subject, {}).get("person_code")
                    if expected and expected != person_code:
                        return False, penalties
        if len(words) == profile.min_len:
            penalties.append(0.2)
        if len(words) == profile.max_len:
            penalties.append(0.2)
        return True, penalties

    def score(self, candidate: Candidate, penalties: List[float]) -> float:
        profile = get_profile(candidate.rank)
        value = 3.0
        if candidate.source_method == "retrieved_corpus":
            value += 2.0
        elif candidate.source_method == "seeded_template":
            value += 1.0
        if candidate.avg_support_rank < candidate.rank:
            value += 1.5
        if candidate.max_support_rank < candidate.rank:
            value += 1.0
        value += self.score_sequence(candidate.sentence.split())
        value -= sum(penalties)
        value -= 0.3 * abs(len([t for t in candidate.sentence.split() if is_word_token(t)]) - profile.ideal_length)
        candidate.score = value
        return value

    def retrieve_candidates(self, target: Lexeme) -> List[Candidate]:
        contexts = self.contexts_for_target(target)
        out: List[Candidate] = []
        for ctx in contexts:
            tokens = ctx.get("tokens") or []
            idx = ctx.get("index", -1)
            if not tokens or idx < 0 or idx >= len(tokens):
                continue
            cand = self.build_candidate(target, tokens, template_id="retrieved", source_method="retrieved_corpus", target_index=idx)
            if not cand:
                continue
            ok, penalties = self.validate(cand)
            if ok:
                self.score(cand, penalties)
                out.append(cand)
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:10]

    def seeded_template_candidate(self, target: Lexeme) -> Optional[Candidate]:
        contexts = self.contexts_for_target(target)
        if not contexts:
            return None
        ctx = self.random.choice(contexts)
        left = normalize_token(ctx.get("left", ""))
        right = normalize_token(ctx.get("right", ""))
        allowed = allowed_support_rank(target.rank, get_profile(target.rank))
        canonical = self.canonical_lemma_for(target)

        if target.pos == "n":
            article = left if left in ARTICLE_SET else self.choose_article(target.gender, definite=True)
            right_lemma = self.lookup_lemma(right)
            right_lex = self.get_known_lexeme(right_lemma)
            if right_lex and right_lex.pos == "adj" and right_lex.rank <= allowed:
                adj = self.inflect_adj(right_lex.lemma, target.gender)
                tokens = [article, target.lemma, "es", adj]
                return self.build_candidate(target, tokens, "seeded_noun_adj", "seeded_template", 1)
            tokens = [article, target.lemma, "está", "aquí"]
            return self.build_candidate(target, tokens, "seeded_noun_here", "seeded_template", 1)

        if target.pos == "v":
            if canonical in SPECIAL_VERB_LEMMAS:
                special = self.special_verb_candidate(target, allowed, "seeded_special", "seeded_template")
                if special:
                    return special
            subject, person = self.subject_for_target(target, fallback=left if left in SUBJECT_FEATURES else "ella")
            verb = self.target_verb_form(target, person)
            right_lemma = self.lookup_lemma(right)
            right_lex = self.get_known_lexeme(right_lemma)
            if right_lex and right_lex.pos == "n" and right_lex.rank <= allowed:
                obj = right_lex
                article = self.choose_article(obj.gender, definite=True)
                tokens = [subject, verb, article, obj.lemma]
                return self.build_candidate(target, tokens, "seeded_verb_obj", "seeded_template", 1)
            tokens = [subject, verb, "mucho"]
            return self.build_candidate(target, tokens, "seeded_verb_adv", "seeded_template", 1)

        if target.pos == "adj":
            left_lemma = self.lookup_lemma(left)
            left_lex = self.get_known_lexeme(left_lemma)
            if left_lex and left_lex.pos == "n" and left_lex.rank <= allowed:
                noun = left_lex
                article = self.choose_article(noun.gender, definite=True)
                tokens = [article, noun.lemma, "es", self.inflect_adj(target.lemma, noun.gender)]
                return self.build_candidate(target, tokens, "seeded_adj_noun", "seeded_template", 3)
            noun = self.pick_candidate("n", allowed, exclude={target.lemma})
            if noun:
                article = self.choose_article(noun.gender, definite=True)
                tokens = [article, noun.lemma, "es", self.inflect_adj(target.lemma, noun.gender)]
                return self.build_candidate(target, tokens, "seeded_adj_fallback", "seeded_template", 3)
        return None

    def pure_template_candidate(self, target: Lexeme) -> Optional[Candidate]:
        profile = get_profile(target.rank)
        allowed = allowed_support_rank(target.rank, profile)
        canonical = self.canonical_lemma_for(target)

        if target.pos == "n":
            choices = ["n_a1_1", "n_a1_2", "n_a1_3"] if profile.band in {"A1", "A2"} else ["n_b1_1", "n_b1_2"]
            template = self.random.choice(choices)
            if template == "n_a1_1":
                adj_classes = None
                if target.semantic_class:
                    adj_classes = ADJ_SUBJECT_PREFS.get(target.lemma)
                adj = self.pick_candidate("adj", allowed, semantic_classes=adj_classes, exclude={target.lemma})
                if not adj:
                    return None
                article = self.choose_article(target.gender, definite=True)
                tokens = [article, target.lemma, "es", self.inflect_adj(adj.lemma, target.gender)]
                return self.build_candidate(target, tokens, template, "template_generated", 1)
            if template == "n_a1_2":
                article = self.choose_article(target.gender, definite=True)
                tokens = [article, target.lemma, "está", "aquí"]
                return self.build_candidate(target, tokens, template, "template_generated", 1)
            if template == "n_a1_3":
                subject = self.random.choice(["yo", "él", "ella"])
                verb = self.conjugate_present("tener", "1sg" if subject == "yo" else "3sg")
                article = self.choose_article(target.gender, definite=False)
                tokens = [subject, verb, article, target.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 3)
            if template == "n_b1_1":
                verb = self.pick_candidate("v", allowed, exclude={target.lemma, "ser", "estar", "tener"})
                place = self.pick_candidate("n", allowed, semantic_classes=["place"], exclude={target.lemma})
                if not verb or not place:
                    return None
                article = self.choose_article(target.gender, definite=True)
                place_article = self.choose_article(place.gender, definite=True)
                tokens = ["ella", self.conjugate_present(verb.lemma, "3sg"), article, target.lemma, "en", place_article, place.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 3)
            if template == "n_b1_2":
                adj = self.pick_candidate("adj", allowed, exclude={target.lemma})
                verb = self.pick_candidate("v", allowed, exclude={target.lemma, "ser", "estar", "tener"})
                if not adj or not verb:
                    return None
                article = self.choose_article(target.gender, definite=True)
                tokens = [article, self.inflect_adj(adj.lemma, target.gender), target.lemma, self.conjugate_present(verb.lemma, "3sg")]
                return self.build_candidate(target, tokens, template, "template_generated", 2)

        if target.pos == "v":
            if canonical in SPECIAL_VERB_LEMMAS:
                special = self.special_verb_candidate(target, allowed, "template_special", "template_generated")
                if special:
                    return special
            choices = ["v_a1_1", "v_a1_2", "v_a1_4"] if profile.band in {"A1", "A2"} else ["v_b1_1"]
            template = self.random.choice(choices)
            subject, person = self.subject_for_target(target)
            verb = self.target_verb_form(target, person)
            if template == "v_a1_1":
                obj_classes = VERB_OBJECT_PREFS.get(canonical)
                obj = self.pick_candidate("n", allowed, semantic_classes=obj_classes, exclude={target.lemma})
                if not obj:
                    return None
                article = self.choose_article(obj.gender, definite=False)
                tokens = [subject, verb, article, obj.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 1)
            if template == "v_a1_2":
                tokens = [subject, self.random.choice(["siempre", "ahora"]), verb]
                return self.build_candidate(target, tokens, template, "template_generated", 2)
            if template == "v_a1_4":
                if canonical not in PLACE_PREP_VERBS:
                    return None
                prep = target.required_prep
                if not prep:
                    return None
                place = self.pick_candidate("n", allowed, semantic_classes=["place"], exclude={target.lemma})
                if not place:
                    return None
                article = self.choose_article(place.gender, definite=True)
                tokens = [subject, verb, prep, article, place.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 1)
            if template == "v_b1_1":
                obj_classes = VERB_OBJECT_PREFS.get(canonical)
                obj = self.pick_candidate("n", allowed, semantic_classes=obj_classes, exclude={target.lemma})
                extra = self.pick_candidate("n", allowed, semantic_classes=["person", "place"], exclude={target.lemma, obj.lemma if obj else ""})
                if not obj or not extra:
                    return None
                art1 = self.choose_article(obj.gender, definite=True)
                art2 = self.choose_article(extra.gender, definite=True)
                prep = target.required_prep or self.random.choice(["para", "en", "con"])
                tokens = [subject, verb, art1, obj.lemma, prep, art2, extra.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 1)

        if target.pos == "adj":
            choices = ["adj_a1_1", "adj_a1_3"] if profile.band in {"A1", "A2"} else ["adj_b1_1"]
            template = self.random.choice(choices)
            if template == "adj_a1_1":
                pref_classes = ADJ_SUBJECT_PREFS.get(target.lemma)
                noun = self.pick_candidate("n", allowed, semantic_classes=pref_classes, exclude={target.lemma})
                if not noun:
                    noun = self.pick_candidate("n", allowed, exclude={target.lemma})
                if not noun:
                    return None
                article = self.choose_article(noun.gender, definite=True)
                tokens = [article, noun.lemma, "es", self.inflect_adj(target.lemma, noun.gender)]
                return self.build_candidate(target, tokens, template, "template_generated", 3)
            if template == "adj_a1_3":
                subject = self.random.choice(["ella", "él"])
                tokens = [subject, "es", target.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 2)
            if template == "adj_b1_1":
                noun = self.pick_candidate("n", allowed, exclude={target.lemma})
                extra = self.pick_candidate("n", allowed, semantic_classes=["place"], exclude={target.lemma, noun.lemma if noun else ""})
                if not noun or not extra:
                    return None
                article1 = self.choose_article(noun.gender, definite=True)
                article2 = self.choose_article(extra.gender, definite=True)
                tokens = [article1, noun.lemma, "es", self.inflect_adj(target.lemma, noun.gender), "en", article2, extra.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 3)
        return None

    def generate_for_lemma(self, lemma: str) -> Candidate:
        lemma = lemma.strip().lower()
        if lemma not in self.lexicon:
            raise KeyError(f"Lemma not in lexicon: {lemma}")
        target = self.lexicon[lemma]
        candidates: List[Candidate] = []
        candidates.extend(self.retrieve_candidates(target))

        if target.pos in {"n", "v", "adj"} and self.can_template_target(target):
            for _ in range(12):
                cand = self.seeded_template_candidate(target)
                if cand:
                    ok, penalties = self.validate(cand)
                    if ok:
                        self.score(cand, penalties)
                        candidates.append(cand)

            for _ in range(20):
                cand = self.pure_template_candidate(target)
                if cand:
                    ok, penalties = self.validate(cand)
                    if ok:
                        self.score(cand, penalties)
                        candidates.append(cand)

        if candidates:
            return max(candidates, key=lambda c: c.score)

        return Candidate(
            lemma=target.lemma,
            rank=target.rank,
            pos=target.pos,
            band=get_profile(target.rank).band,
            translation=target.translation,
            sentence="",
            target_form=target.lemma,
            target_index=-1,
            support_ranks=[],
            avg_support_rank=0.0,
            max_support_rank=0,
            template_id="",
            source_method="manual_review_needed",
        )

    def generate_batch(
        self,
        limit: int,
        out_csv: str,
        min_rank: int = 1,
        max_rank: int = 10**9,
        pos_filter: Optional[str] = None,
        lemma_filter: Optional[List[str]] = None,
    ) -> List[Candidate]:
        rows = list(self.lexicon.values())
        rows = [x for x in rows if min_rank <= x.rank <= max_rank]
        if pos_filter:
            rows = [x for x in rows if x.pos == pos_filter]
        if lemma_filter:
            wanted = {x.lower() for x in lemma_filter}
            rows = [x for x in rows if x.lemma in wanted]
        rows.sort(key=lambda x: x.rank)
        rows = rows[:limit]
        generated: List[Candidate] = []
        for lex in rows:
            try:
                generated.append(self.generate_for_lemma(lex.lemma))
            except Exception as exc:
                print(f"[warn] failed to generate for {lex.lemma}: {exc}", file=sys.stderr)
                generated.append(
                    Candidate(
                        lemma=lex.lemma,
                        rank=lex.rank,
                        pos=lex.pos,
                        band=get_profile(lex.rank).band,
                        translation=lex.translation,
                        sentence="",
                        target_form=lex.lemma,
                        target_index=-1,
                        support_ranks=[],
                        avg_support_rank=0.0,
                        max_support_rank=0,
                        template_id="",
                        source_method="manual_review_needed",
                    )
                )
        self.write_csv(generated, out_csv)
        return generated

    def write_csv(self, rows: List[Candidate], out_csv: str) -> None:
        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "band",
            "translation",
            "sentence",
            "english_sentence",
            "target_form",
            "target_index",
            "support_ranks",
            "avg_support_rank",
            "max_support_rank",
            "template_id",
            "source_method",
            "english_source_method",
            "parallel_pair_id",
            "score",
            "english_score",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                d = asdict(row)
                d["support_ranks"] = " ".join(str(x) for x in row.support_ranks)
                writer.writerow(d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spanish sentence generator using monolingual corpus artifacts.")
    parser.add_argument("--lexicon", required=True, help="Path to stg_words_spa.csv")
    parser.add_argument("--models-dir", required=True, help="Directory containing .pkl artifacts")
    parser.add_argument("--out", default="generated_sentences.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=100, help="Number of lexicon rows to generate")
    parser.add_argument("--min-rank", type=int, default=1, help="Minimum rank to include")
    parser.add_argument("--max-rank", type=int, default=10**9, help="Maximum rank to include")
    parser.add_argument("--pos", default=None, help="Optional POS filter: n, v, adj")
    parser.add_argument("--lemma", action="append", help="Generate only these lemma(s). Can be repeated.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    gen = SentenceGenerator(args.lexicon, args.models_dir, seed=args.seed)
    rows = gen.generate_batch(
        limit=args.limit,
        out_csv=args.out,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        pos_filter=args.pos,
        lemma_filter=args.lemma,
    )

    reviewed = sum(1 for r in rows if r.source_method == "manual_review_needed")
    retrieved = sum(1 for r in rows if r.source_method == "retrieved_corpus")
    seeded = sum(1 for r in rows if r.source_method == "seeded_template")
    templated = sum(1 for r in rows if r.source_method == "template_generated")

    print(f"Generated: {len(rows):,}")
    print(f"  retrieved_corpus:     {retrieved:,}")
    print(f"  seeded_template:      {seeded:,}")
    print(f"  template_generated:   {templated:,}")
    print(f"  manual_review_needed: {reviewed:,}")
    print(f"Saved: {args.out}")

    preview = [r for r in rows if r.sentence][:5]
    if preview:
        print("\nPreview:")
        for row in preview:
            print(f"  {row.lemma:<15} [{row.source_method:<18}] {row.sentence}")


if __name__ == "__main__":
    main()
