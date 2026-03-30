#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from complete_generate import Candidate, Lexeme, allowed_support_rank, get_profile, normalize_token

POS_TO_FAMILY = {
    "n": "noun",
    "prop": "noun",
    "v": "verb",
    "adj": "adjective",
    "adv": "adverb",
    "prep": "function",
    "pron": "function",
    "determiner": "function",
    "art": "function",
    "conj": "function",
    "interj": "function",
    "num": "function",
    "contraction": "function",
    "letter": "function",
    "prefix": "function",
    "phrase": "function",
    "particle": "function",
    "": "function",
    "none": "function",
}

DEFAULT_INF_LEMMAS = ["leer", "comer", "beber", "ver", "comprar", "buscar", "ir", "hablar"]
DEFAULT_ADJ_LEMMAS = ["grande", "bonito", "nuevo", "seguro", "claro"]
DEFAULT_ADV_LEMMAS = ["bien", "hoy", "ya", "aquí"]
DEFAULT_LOC = ["aquí", "en casa"]
COMMON_FUNCTION_FILLERS = {
    "{PREP_PHRASE}": ["de casa", "en casa", "con él"],
}


class LearnedFrameRouter:
    def __init__(
        self,
        generator,
        frames_path: str,
        lemma_pref_path: Optional[str] = None,
    ):
        self.generator = generator
        self.frames_path = frames_path
        self.frames = self._load_frames(frames_path)
        self.lemma_preferences = self._load_lemma_preferences(lemma_pref_path)

    def _load_frames(self, path: str) -> Dict[str, List[Dict[str, Any]]]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for family, items in (data or {}).items():
            valid_items = []
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                pattern_tokens = item.get("pattern_tokens") or []
                if "{TARGET}" not in pattern_tokens:
                    continue
                valid_items.append(item)
            out[family] = valid_items
        return out

    def _load_lemma_preferences(self, path: Optional[str]) -> Dict[str, Dict[str, int]]:
        if not path or not Path(path).exists():
            return {}
        out: Dict[str, Dict[str, int]] = {}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                lemma = (row.get("lemma") or "").strip().lower()
                frame_id = (row.get("frame_id") or "").strip()
                count = int(float(row.get("count") or 0))
                if lemma and frame_id and count > 0:
                    out.setdefault(lemma, {})[frame_id] = count
        return out

    def has_frames(self) -> bool:
        return any(self.frames.values())

    def pos_family(self, entry: Lexeme) -> str:
        return POS_TO_FAMILY.get((entry.pos or "").strip().lower(), "function")

    def generate_from_learned_frames(self, entry: Lexeme, max_candidates: int = 10) -> List[Candidate]:
        family = self.pos_family(entry)
        frame_candidates = self._select_frames(entry, family, limit=max_candidates * 2)
        out: List[Candidate] = []
        for frame in frame_candidates:
            tokens = self._render_frame(entry, frame)
            if not tokens:
                continue
            target_index = next((i for i, tok in enumerate(tokens) if normalize_token(tok) == normalize_token(entry.lemma)), -1)
            if target_index < 0:
                continue
            candidate = self.generator.build_candidate(
                target=entry,
                tokens=tokens,
                template_id=frame.get("frame_id", "learned_frame"),
                source_method="learned_frame_generated",
                target_index=target_index,
            )
            if not candidate:
                continue
            ok, penalties = self.generator.validate(candidate)
            self.generator.score(candidate, penalties)
            if not ok:
                continue
            setattr(candidate, "_learned_frame_id", frame.get("frame_id", "learned_frame"))
            setattr(candidate, "_learned_frame_weight", frame.get("weight", 0.0))
            out.append(candidate)
            if len(out) >= max_candidates:
                break
        return self.generator.dedupe_candidates(out)[:max_candidates]

    def _select_frames(self, entry: Lexeme, family: str, limit: int) -> List[Dict[str, Any]]:
        candidates = list(self.frames.get(family, []))
        if not candidates:
            return []
        lemma_pref = self.lemma_preferences.get(normalize_token(entry.lemma), {})
        support_cap = allowed_support_rank(entry.rank, get_profile(entry.rank)) + 800
        filtered = [
            frame for frame in candidates
            if int(frame.get("support_rank_max", 0) or 0) <= max(1200, support_cap)
            and int(frame.get("max_rank", entry.rank) or entry.rank) >= entry.rank
        ]
        ranked = sorted(
            filtered,
            key=lambda frame: (
                -lemma_pref.get(frame.get("frame_id", ""), 0),
                -float(frame.get("weight", 0.0) or 0.0),
                -int(frame.get("count", 0) or 0),
            ),
        )
        return ranked[:limit]

    def _render_frame(self, entry: Lexeme, frame: Dict[str, Any]) -> Optional[List[str]]:
        tokens: List[str] = []
        exclude = {normalize_token(entry.lemma), normalize_token(self.generator.canonical_lemma_for(entry))}
        rank_ceiling = allowed_support_rank(entry.rank, get_profile(entry.rank))
        for token in frame.get("pattern_tokens", []):
            if token == "{TARGET}":
                tokens.append(entry.lemma)
                continue
            if not token.startswith("{"):
                tokens.append(token)
                continue
            filled = self._fill_slot(token, entry, rank_ceiling=rank_ceiling, exclude=exclude)
            if not filled:
                return None
            tokens.extend(filled)
        return tokens

    def _fill_slot(self, slot: str, entry: Lexeme, rank_ceiling: int, exclude: Iterable[str]) -> Optional[List[str]]:
        if slot == "{NOUN}":
            noun = None
            if entry.pos == "v":
                noun = self.generator.pick_safe_object_noun_for_verb(self.generator.canonical_lemma_for(entry), rank_ceiling, exclude=exclude)
            if noun is None:
                noun = self.generator.pick_template_friendly_noun(rank_ceiling, exclude=exclude)
            return [noun.lemma] if noun else None
        if slot == "{ADJ}":
            if entry.pos == "n":
                gender = self.generator.safe_noun_gender(entry.lemma, entry.gender)
                return [self.generator.inflect_adj(self._pick_known(DEFAULT_ADJ_LEMMAS), gender)]
            noun = self.generator.pick_template_friendly_noun(rank_ceiling, exclude=exclude)
            gender = self.generator.safe_noun_gender(noun.lemma if noun else "casa", noun.gender if noun else None)
            return [self.generator.inflect_adj(self._pick_known(DEFAULT_ADJ_LEMMAS), gender)]
        if slot == "{INF}":
            lemma = self._pick_known(DEFAULT_INF_LEMMAS)
            return [lemma] if lemma else None
        if slot == "{ADV}":
            lemma = self._pick_known(DEFAULT_ADV_LEMMAS)
            return [lemma] if lemma else None
        if slot == "{LOC}":
            loc = DEFAULT_LOC[0]
            return loc.split()
        if slot == "{PREP_PHRASE}":
            phrase = COMMON_FUNCTION_FILLERS[slot][0]
            return phrase.split()
        if slot == "{VERB}":
            return [self.generator.conjugate_present(self._pick_known(DEFAULT_INF_LEMMAS), "3sg")]
        return None

    def _pick_known(self, options: Sequence[str]) -> str:
        for lemma in options:
            if lemma in self.generator.generation_lexicon or lemma in self.generator.lexicon:
                return lemma
        return options[0]
