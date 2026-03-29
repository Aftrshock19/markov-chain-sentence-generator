#!/usr/bin/env python3
"""Hybrid Spanish sentence generator.

Combines retrieved corpus candidates, stochastic decoding, and template search
on top of the complete_generate SentenceGenerator shell.
"""
import argparse
import csv
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

import complete_generate as cg
from complete_generate import (
    ADV_TIME_LEMMAS,
    COORDINATING_CONJUNCTIONS,
    CONTEXT_DEPENDENT_OPENERS,
    DifficultyProfile,
    FUNCTION_WORD_FAMILIES,
    SPECIAL_VERB_LEMMAS,
    STARTER_ADJ_ALLOWED_NOUN_CLASSES,
    SUBJECT_FEATURES,
    SIMPLE_PREPOSITIONS,
    Candidate,
    Lexeme,
    SentenceGenerator,
    allowed_support_rank,
    get_profile,
    is_word_token,
    normalize_token,
)
from reranker import predict_candidate_scores

_BANNED_CONTINUATIONS = CONTEXT_DEPENDENT_OPENERS | {
    "pues", "sino", "ni", "oh", "ay", "eh", "bueno",
}
_SAFE_TEMPLATE_SOURCES = {
    "seeded_template",
    "template_generated",
    "emergency_template",
    "emergency_starter",
}
_FRAGILE_STOCHASTIC_SCORE_FLOORS = {
    "adv": 8.9,
    "prep": 9.4,
    "contraction": 9.4,
    "pron": 9.5,
    "determiner": 9.6,
    "art": 9.6,
    "conj": 9.6,
    "interj": 9.0,
    "num": 9.1,
    "letter": 9.8,
    "prefix": 9.8,
    "phrase": 9.8,
    "particle": 9.8,
    "residual": 9.8,
}

HYBRID_BANDS = [
    (1, 800, DifficultyProfile("A1", 3, 7, 400, 250, (0, 1), 1)),
    (801, 1500, DifficultyProfile("A2", 3, 8, 700, 400, (0, 2), 2)),
    (1501, 2500, DifficultyProfile("B1", 4, 9, 1500, 900, (1, 3), 3)),
    (2501, 4000, DifficultyProfile("B2", 5, 10, 3000, 1800, (1, 4), 4)),
    (4001, 6000, DifficultyProfile("C1", 5, 11, 5500, 3000, (2, 5), 5)),
    (6001, 10**9, DifficultyProfile("C2", 6, 12, 9000, 5000, (2, 6), 6)),
]

LOCKED_SUCCESS_METRICS = {
    "rank_window": [1, 1000],
    "min_nonempty_rate": 0.95,
    "max_bad_candidate_rate": 0.01,
}


def hybrid_get_profile(rank: int) -> DifficultyProfile:
    for lo, hi, profile in HYBRID_BANDS:
        if lo <= rank <= hi:
            return profile
    return HYBRID_BANDS[-1][2]


def hybrid_allowed_support_rank(target_rank: int, profile: DifficultyProfile) -> int:
    return min(profile.filler_ceil, max(300, int(target_rank * 0.8)))


get_profile = hybrid_get_profile
allowed_support_rank = hybrid_allowed_support_rank
cg.get_profile = hybrid_get_profile
cg.allowed_support_rank = hybrid_allowed_support_rank


class StochasticSentenceGenerator(SentenceGenerator):
    """Sentence generator whose primary extra source is stochastic decoding."""

    def __init__(self, lexicon_path: str, models_dir: str, seed: int = 42):
        super().__init__(lexicon_path, models_dir, seed=seed)
        self._tri_next: Dict[Tuple[str, str], Dict[str, int]] = self.trigrams.get("next", {})
        self._tri_totals: Dict[Tuple[str, str], int] = self.trigrams.get("totals", {})
        self._bi_next: Dict[str, Dict[str, int]] = self.bigrams.get("next", {})
        self._bi_totals: Dict[str, int] = self.bigrams.get("totals", {})

    def initial_seeds_for_target(self, target: Lexeme) -> List[Tuple[List[str], int]]:
        seeds: List[Tuple[List[str], int]] = []
        canonical = self.canonical_lemma_for(target)
        family = self.normalized_pos_family(target)
        surface = target.lemma
        normalized = normalize_token(surface)

        if family == "n":
            if not self.noun_is_template_friendly(target):
                return seeds
            gender = self.safe_noun_gender(target.lemma, target.gender)
            for definite in (True, False):
                art = self.choose_article(gender, definite=definite)
                seeds.append(([art, target.lemma], 1))
            return seeds

        if family == "v":
            if canonical in SPECIAL_VERB_LEMMAS:
                special = self._verb_special_seeds(target, canonical)
                if special:
                    return special
            subjects_tried: set = set()
            for fallback in ("ella", "yo", "él"):
                subj, pc = self.subject_for_target(target, fallback=fallback)
                if subj in subjects_tried:
                    continue
                subjects_tried.add(subj)
                verb_form = self.target_verb_form(target, pc)
                seeds.append(([subj, verb_form], 1))
            return seeds

        if family == "adj":
            profile = get_profile(target.rank)
            allowed = allowed_support_rank(target.rank, profile)
            allowed_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(normalize_token(target.lemma))
            tried_nouns: List[str] = []
            for _ in range(12):
                noun = self.pick_template_friendly_noun(
                    allowed,
                    semantic_classes=list(allowed_classes) if allowed_classes else None,
                    exclude={target.lemma, *tried_nouns},
                )
                if not noun:
                    break
                tried_nouns.append(noun.lemma)
                noun_gender = self.safe_noun_gender(noun.lemma, noun.gender)
                if self.inflect_adj(target.lemma, noun_gender) != target.lemma:
                    continue
                art = self.choose_article(noun_gender, definite=True)
                seeds.append(([art, noun.lemma, "es", target.lemma], 3))
                if len(seeds) >= 3:
                    break
            return seeds

        if family == "prep":
            seeds.append((["ella", "está", surface], 2))
            seeds.append((["ella", "va", surface], 2))
            return seeds

        if family == "conj":
            if normalized in COORDINATING_CONJUNCTIONS:
                seeds.append((["ella", "va", surface], 2))
            else:
                seeds.append((["ella", "sale", surface], 2))
            return seeds

        if family == "pron":
            if normalized in SUBJECT_FEATURES:
                seeds.append(([surface], 0))
            elif normalized in {"lo", "la", "los", "las", "le", "les", "me", "te", "se", "nos"}:
                seeds.append((["ella", surface], 1))
            elif normalized in {"eso", "esto", "nada", "algo", "aquello"}:
                seeds.append(([surface, "es"], 0))
            elif normalized in {"qué", "quién", "cómo"}:
                seeds.append(([surface], 0))
            else:
                seeds.append(([surface, "es"], 0))
            return seeds

        if family in {"determiner", "art"}:
            seeds.append(([surface], 0))
            return seeds

        if family == "adv":
            seeds.append(([surface, "ella"], 0))
            seeds.append((["ella", surface], 1))
            non_manner_adverbs = ADV_TIME_LEMMAS | {
                "no", "aquí", "ahi", "ahí", "allí", "alli", "allá", "alla",
                "ya", "también", "tambien", "nunca",
            }
            if normalized not in non_manner_adverbs:
                seeds.append((["ella", "habla", surface], 2))
            return seeds

        if family == "contraction":
            if normalized == "al":
                seeds.append((["ella", "va", surface], 2))
            elif normalized == "del":
                seeds.append((["ella", "habla", surface], 2))
            return seeds

        if family in {"interj", "num"}:
            seeds.append(([surface], 0))
            return seeds

        seeds.append(([surface, "es"], 0))
        return seeds

    def _verb_special_seeds(self, target: Lexeme, canonical: str) -> List[Tuple[List[str], int]]:
        subj, pc = self.subject_for_target(target)
        verb = self.target_verb_form(target, pc)
        morph = self.target_form_metadata(target)
        verb_form = morph.get("VerbForm", "")
        mood = morph.get("Mood", "")

        if canonical == "haber" and normalize_token(verb) == "hay":
            return [([verb], 0)]

        # Gerund forms: "Ella está [hablando]."
        if verb_form == "Ger":
            return [
                (["ella", "está", target.lemma], 2),
                (["él", "está", target.lemma], 2),
            ]

        # Past participle forms: "Ella ha [hecho] eso."
        if verb_form == "Part":
            if canonical == "haber":
                return [([subj, verb], 1)]
            return [
                (["ella", "ha", target.lemma], 2),
                (["él", "ha", target.lemma], 2),
            ]

        # Subjunctive forms: "Quiero que [sea] bueno."
        if mood == "Sub":
            return [
                (["quiero", "que", target.lemma], 2),
                (["espero", "que", target.lemma], 2),
                (["no", "creo", "que", target.lemma], 3),
            ]

        # Conditional forms: "Ella [podría] ir."
        if mood == "Cnd":
            return [
                ([subj, target.lemma], 1),
                ([subj, target.lemma, "ir"], 1),
            ]

        return [([subj, verb], 1)]

    def get_next_word_candidates(
        self,
        context_tokens: List[str],
        target: Lexeme,
        limit: int = 50,
    ) -> List[Tuple[str, float]]:
        profile = get_profile(target.rank)
        rank_ceiling = allowed_support_rank(target.rank, profile)

        ctx = [normalize_token(t) for t in context_tokens if is_word_token(t)]
        if not ctx:
            ctx = ["<START>"]

        raw: Dict[str, int] = {}
        total = 0
        if len(ctx) >= 2:
            key = (ctx[-2], ctx[-1])
            tri_cands = self._tri_next.get(key)
            if tri_cands:
                total = self._tri_totals.get(key, 0)
                if total > 0:
                    raw = tri_cands

        if not raw:
            bi_cands = self._bi_next.get(ctx[-1])
            if bi_cands:
                total = self._bi_totals.get(ctx[-1], 0)
                if total > 0:
                    raw = bi_cands

        if not raw or total <= 0:
            return []

        tail = set(ctx[-2:])
        results: List[Tuple[str, float]] = []
        for word, count in raw.items():
            if word == "<START>":
                continue
            if not is_word_token(word) and word != "<END>":
                continue
            if word in tail:
                continue
            if word in _BANNED_CONTINUATIONS:
                continue
            if word != "<END>" and normalize_token(word) != normalize_token(target.lemma):
                wr = self.lookup_rank(word)
                if wr > rank_ceiling and wr > target.rank:
                    continue
            prob = count / total
            results.append((word, prob))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def sample_next_word(
        self,
        candidates: List[Tuple[str, float]],
        top_k: int = 10,
        temperature: float = 1.0,
    ) -> Optional[str]:
        if not candidates:
            return None
        top = candidates[:top_k]
        if temperature <= 0.01:
            return top[0][0]

        weights = [p ** (1.0 / temperature) if p > 0 else 0.0 for _, p in top]
        total_w = sum(weights)
        if total_w <= 0:
            return top[0][0]

        return self.random.choices([w for w, _ in top], weights=weights, k=1)[0]

    def decode_from_seed(
        self,
        seed_tokens: List[str],
        target: Lexeme,
        target_index: int,
        max_steps: int = 6,
        attempts: int = 100,
    ) -> List[List[str]]:
        profile = get_profile(target.rank)
        decoded: List[List[str]] = []

        for _ in range(attempts):
            tokens = list(seed_tokens)
            ctx = ["<START>"] + [normalize_token(t) for t in tokens if is_word_token(t)]

            for _step in range(max_steps):
                wc = sum(1 for t in tokens if is_word_token(t))
                if wc >= profile.max_len:
                    break

                cands = self.get_next_word_candidates(ctx, target, limit=50)
                if not cands:
                    break

                if wc < profile.min_len:
                    cands = [(w, p) for w, p in cands if w != "<END>"]
                    if not cands:
                        break

                word = self.sample_next_word(cands, top_k=10, temperature=0.9)
                if not word or word == "<END>":
                    break

                tokens.append(word)
                ctx.append(normalize_token(word))

            wc = sum(1 for t in tokens if is_word_token(t))
            if profile.min_len <= wc <= profile.max_len:
                decoded.append(tokens)

        return decoded

    def generate_stochastic_candidates(self, target: Lexeme, attempts: int = 150) -> List[Candidate]:
        seeds = self.initial_seeds_for_target(target)
        if not seeds:
            return []

        per_seed = max(10, attempts // max(1, len(seeds)))
        candidates: List[Candidate] = []
        for seed_tokens, tidx in seeds:
            sequences = self.decode_from_seed(seed_tokens, target, tidx, max_steps=6, attempts=per_seed)
            for seq in sequences:
                cand = self.build_candidate(
                    target,
                    seq,
                    template_id="stochastic",
                    source_method="stochastic_decoder",
                    target_index=tidx,
                )
                if not cand:
                    continue
                ok, penalties = self.validate(cand)
                if not ok:
                    continue
                self.score(cand, penalties)
                candidates.append(cand)

        return self.dedupe_candidates(candidates)[:20]


class HybridSentenceGenerator(StochasticSentenceGenerator):
    def __init__(
        self,
        lexicon_path: str,
        models_dir: str,
        seed: int = 42,
        max_total_attempts: int = 400,
        max_candidates_to_keep: int = 20,
    ):
        super().__init__(lexicon_path, models_dir, seed=seed)
        self.max_total_attempts = max(1, max_total_attempts)
        self.max_candidates_to_keep = max(1, max_candidates_to_keep)
        self._hybrid_pool_cache: Dict[str, List[Candidate]] = {}
        self._hybrid_search_cache: Dict[str, Dict[str, Any]] = {}

    def load_and_apply_overrides(self, path: str) -> None:
        super().load_and_apply_overrides(path)
        self._hybrid_pool_cache.clear()
        self._hybrid_search_cache.clear()

    def _candidate_is_valid(self, candidate: Optional[Candidate]) -> bool:
        if not candidate or not candidate.sentence:
            return False
        ok, _ = self.validate(candidate)
        return ok

    def _candidate_family(self, candidate: Candidate) -> str:
        return self.normalized_pos_family(self.lexicon.get(candidate.lemma))

    def _family_is_fragile(self, family: str) -> bool:
        return family in FUNCTION_WORD_FAMILIES

    def _is_safe_template_source(self, candidate: Candidate) -> bool:
        return candidate.source_method in _SAFE_TEMPLATE_SOURCES

    def _stochastic_fragile_reasons(self, candidate: Candidate, family: str) -> List[str]:
        words = self.word_tokens(candidate.sentence)
        lowered = [normalize_token(word) for word in words]
        target_form = normalize_token(candidate.target_form or candidate.lemma)
        finite_count = self.finite_verb_count(words)
        verb_count = self.verb_token_count(words)
        reasons: List[str] = []

        if family in {"letter", "prefix", "phrase", "particle", "residual"}:
            reasons.append("fragile_stochastic_unsupported_family")
            return reasons

        if candidate.score < _FRAGILE_STOCHASTIC_SCORE_FLOORS.get(family, 9.5):
            reasons.append("fragile_stochastic_low_score")

        if family == "adv":
            if len(words) > 5:
                reasons.append("fragile_stochastic_long_adv")
            if verb_count > 2:
                reasons.append("fragile_stochastic_adv_chain")
        elif family in {"prep", "contraction"}:
            prepish_count = sum(1 for token in lowered if token in SIMPLE_PREPOSITIONS | {"a", "al", "del"})
            if len(words) > 5:
                reasons.append("fragile_stochastic_long_prep")
            if prepish_count > 1:
                reasons.append("fragile_stochastic_nested_prep")
            if finite_count != 1:
                reasons.append("fragile_stochastic_prep_clause")
        elif family == "pron":
            if target_form in {"que", "qué", "quien", "quién", "como", "cómo"} and finite_count < 2:
                reasons.append("fragile_stochastic_orphan_pron")
            if target_form in {"me", "te", "se", "nos", "os", "lo", "la", "los", "las", "le", "les"} and len(words) > 4:
                reasons.append("fragile_stochastic_clitic_chain")
            if target_form in SUBJECT_FEATURES and verb_count > 1:
                reasons.append("fragile_stochastic_subject_chain")
        elif family in {"determiner", "art"}:
            if len(words) > 5:
                reasons.append("fragile_stochastic_long_np")
            if verb_count > 1:
                reasons.append("fragile_stochastic_np_chain")
            if candidate.target_index >= 0 and candidate.target_index < len(words) - 1:
                if self.lookup_pos(words[candidate.target_index + 1]) != "n":
                    reasons.append("fragile_stochastic_det_frame")
        elif family == "conj":
            if len(words) > 6:
                reasons.append("fragile_stochastic_long_conj")
            if finite_count < 2:
                reasons.append("fragile_stochastic_thin_conjunction")
            if verb_count > 2:
                reasons.append("fragile_stochastic_verb_chain")
        elif family == "interj":
            if len(words) > 4:
                reasons.append("fragile_stochastic_interjection_tail")
        elif family == "num":
            if len(words) > 4:
                reasons.append("fragile_stochastic_long_number")
            if candidate.target_index < len(words) - 1 and self.lookup_pos(words[candidate.target_index + 1]) != "n":
                reasons.append("fragile_stochastic_number_frame")

        return list(dict.fromkeys(reasons))

    def _hybrid_policy_reasons(self, candidate: Optional[Candidate]) -> List[str]:
        if not candidate or not candidate.sentence:
            return ["manual_review_needed"]
        family = self._candidate_family(candidate)
        words = self.word_tokens(candidate.sentence)
        lowered = [normalize_token(word) for word in words]
        target_form = normalize_token(candidate.target_form or candidate.lemma)
        reasons: List[str] = []
        if not self._family_is_fragile(family):
            return reasons
        if family == "adv" and target_form == "no" and lowered and lowered[-1] == "no":
            reasons.append("terminal_negation")
        if candidate.source_method == "stochastic_decoder":
            reasons.extend(self._stochastic_fragile_reasons(candidate, family))
        return list(dict.fromkeys(reason for reason in reasons if reason))

    def _safe_template_publishable_override(self, candidate: Candidate) -> bool:
        family = self._candidate_family(candidate)
        if not self._family_is_fragile(family):
            return False
        if not self._is_safe_template_source(candidate):
            return False
        if not self._candidate_is_valid(candidate):
            return False
        if self._hybrid_policy_reasons(candidate):
            return False

        grammatical_ok, natural_ok, learner_clear_ok, _ = self.review_flags(candidate)
        if grammatical_ok != "1":
            return False
        if natural_ok == "1" and learner_clear_ok == "1":
            return True

        words = self.word_tokens(candidate.sentence)
        lowered = [normalize_token(word) for word in words]
        target_form = normalize_token(candidate.target_form or candidate.lemma)
        override_score_floors = {
            "prep": 4.0,
            "contraction": 4.0,
            "adv": 6.6,
            "art": 6.8,
            "determiner": 6.8,
            "num": 6.5,
            "pron": 7.0,
        }
        if candidate.score < override_score_floors.get(family, 7.2) or len(words) > 5:
            return False
        if family in {"conj", "interj", "letter", "prefix", "phrase", "particle", "residual"}:
            return False
        if family in {"prep", "contraction"}:
            prepish_count = sum(1 for token in lowered if token in SIMPLE_PREPOSITIONS | {"a", "al", "del"})
            return prepish_count == 1 and self.verb_token_count(words) <= 1
        if family in {"adv", "art", "determiner", "num"}:
            return self.verb_token_count(words) <= 1
        if family == "pron":
            if target_form in SUBJECT_FEATURES:
                return self.verb_token_count(words) <= 1
            if target_form in {"me", "te", "se", "nos", "os", "lo", "la", "los", "las", "le", "les"}:
                return len(words) <= 3
            if target_form in {"esto", "eso", "aquello", "algo", "nada", "nadie", "alguien"}:
                return self.verb_token_count(words) <= 1
            return False
        return False

    def candidate_is_hybrid_publishable(self, candidate: Optional[Candidate]) -> bool:
        if not candidate or not candidate.sentence:
            return False
        if not self.candidate_is_general_publishable(candidate):
            return False
        return not self._hybrid_policy_reasons(candidate)

    def _candidate_is_hybrid_strong(self, candidate: Optional[Candidate]) -> bool:
        if not candidate:
            return False
        if not self.candidate_is_hybrid_publishable(candidate):
            return False
        family = self._candidate_family(candidate)
        fragile = self._family_is_fragile(family)
        if candidate.source_method == "retrieved_corpus":
            return candidate.score >= (9.0 if fragile else 8.8)
        if self._is_safe_template_source(candidate):
            return candidate.score >= (9.7 if fragile else 9.1)
        if candidate.source_method == "stochastic_decoder":
            if fragile:
                return candidate.score >= 11.5 and len(self.word_tokens(candidate.sentence)) <= 4
            return candidate.score >= 9.5
        return False

    def _should_early_exit(self, candidate: Optional[Candidate]) -> bool:
        if not self._candidate_is_hybrid_strong(candidate):
            return False
        if not candidate:
            return False
        family = self._candidate_family(candidate)
        if self._family_is_fragile(family) and candidate.source_method == "stochastic_decoder":
            return False
        return True

    def _source_priority(self, candidate: Candidate, family: str) -> int:
        if self._family_is_fragile(family):
            order = {
                "retrieved_corpus": 0,
                "seeded_template": 1,
                "template_generated": 2,
                "emergency_template": 3,
                "emergency_starter": 4,
                "stochastic_decoder": 5,
                "manual_review_needed": 6,
                "no_candidate_found": 7,
            }
        else:
            order = {
                "retrieved_corpus": 0,
                "stochastic_decoder": 1,
                "seeded_template": 2,
                "template_generated": 3,
                "emergency_template": 4,
                "emergency_starter": 5,
                "manual_review_needed": 6,
                "no_candidate_found": 7,
            }
        return order.get(candidate.source_method, 8)

    def _candidate_bucket(self, candidate: Candidate, family: str) -> Tuple[int, int]:
        publishable = self.candidate_is_hybrid_publishable(candidate)
        valid = self._candidate_is_valid(candidate)
        if publishable:
            trust = 0
        elif valid:
            trust = 1
        elif candidate.sentence:
            trust = 2
        else:
            trust = 3
        return trust, self._source_priority(candidate, family)

    def _choose_best_in_bucket(self, candidates: List[Candidate]) -> Candidate:
        if len(candidates) == 1:
            return candidates[0]
        if self.reranker:
            try:
                predictions = predict_candidate_scores(self.reranker, candidates)
            except Exception:
                predictions = []
            if len(predictions) == len(candidates):
                return max(zip(candidates, predictions), key=lambda item: (item[1], item[0].score))[0]
        return max(candidates, key=lambda candidate: candidate.score)

    def _hybrid_failure_reasons(self, candidate: Candidate) -> List[str]:
        reasons: List[str] = []
        grammatical_ok, natural_ok, learner_clear_ok, notes = self.review_flags(candidate)
        if grammatical_ok != "1":
            reasons.append("not_grammatical")
        if natural_ok != "1":
            reasons.append("not_natural")
        if learner_clear_ok != "1":
            reasons.append("not_learner_clear")
        if notes and notes != "manual review needed":
            reasons.extend([part.strip() for part in notes.split(";") if part.strip()])
        reasons.extend(self._hybrid_policy_reasons(candidate))
        return list(dict.fromkeys(reason for reason in reasons if reason))

    def _has_preferred_publishable_candidate(self, candidates: List[Candidate], family: str) -> bool:
        for candidate in candidates:
            if candidate.source_method == "stochastic_decoder":
                continue
            if self._family_is_fragile(family) and self.candidate_is_hybrid_publishable(candidate):
                return True
        return False

    def _absorb_prescored_candidate(
        self,
        candidate: Optional[Candidate],
        best_valid: Optional[Candidate],
        best_any: Optional[Candidate],
    ) -> Tuple[Optional[Candidate], Optional[Candidate], Optional[Candidate], int]:
        if not candidate or not candidate.sentence:
            return None, best_valid, best_any, 0
        if best_valid is None or candidate.score > best_valid.score:
            best_valid = candidate
        if best_any is None or candidate.score > best_any.score:
            best_any = candidate
        return candidate, best_valid, best_any, 1

    def _evaluate_raw_candidate(
        self,
        candidate: Optional[Candidate],
        best_valid: Optional[Candidate],
        best_any: Optional[Candidate],
    ) -> Tuple[Optional[Candidate], Optional[Candidate], Optional[Candidate], int]:
        if not candidate or not candidate.sentence:
            return None, best_valid, best_any, 0
        ok, penalties = self.validate(candidate)
        self.score(candidate, penalties)
        if ok:
            if best_valid is None or candidate.score > best_valid.score:
                best_valid = candidate
            if best_any is None or candidate.score > best_any.score:
                best_any = candidate
            return candidate, best_valid, best_any, 1
        if best_any is None or candidate.score > best_any.score:
            best_any = candidate
        return candidate, best_valid, best_any, 0

    def _no_candidate_result(self, target: Lexeme) -> Candidate:
        return Candidate(
            lemma=target.lemma,
            rank=target.rank,
            pos=target.pos,
            band=get_profile(target.rank).band,
            translation=target.translation,
            sentence="",
            target_form="",
            target_index=-1,
            support_ranks=[],
            avg_support_rank=0.0,
            max_support_rank=0,
            template_id="no_candidate_found",
            source_method="no_candidate_found",
            canonical_lemma=self.canonical_lemma_for(target),
            target_morph="",
        )

    def hybrid_output_metadata(
        self,
        candidate: Candidate,
        search_attempts_used: int,
        candidates_found: int,
    ) -> Dict[str, Any]:
        publishable = self.candidate_is_hybrid_publishable(candidate)
        valid = self._candidate_is_valid(candidate)
        if candidate.source_method == "no_candidate_found":
            quality_tier = "no_candidate_found"
        elif self._candidate_is_hybrid_strong(candidate):
            quality_tier = "strong"
        elif publishable:
            quality_tier = "acceptable"
        elif candidate.sentence and valid:
            quality_tier = "weak"
        elif candidate.sentence:
            quality_tier = "bad"
        else:
            quality_tier = "no_candidate_found"

        bad_candidate = quality_tier in {"bad", "no_candidate_found"}
        failure_reason = ""
        if candidate.source_method == "no_candidate_found":
            failure_reason = "no_publishable_candidate"
        elif not publishable:
            failure_reason = "; ".join(self._hybrid_failure_reasons(candidate))

        return {
            "publishable": publishable,
            "quality_tier": quality_tier,
            "bad_candidate": bad_candidate,
            "failure_reason": failure_reason,
            "search_attempts_used": int(search_attempts_used),
            "candidates_found": int(candidates_found),
            "best_source_method": candidate.source_method,
        }

    def _choose_from_pool(
        self,
        pool: List[Candidate],
        best_valid: Optional[Candidate],
        best_any: Optional[Candidate],
        target: Lexeme,
    ) -> Candidate:
        trimmed = self.dedupe_candidates(pool)[: self.max_candidates_to_keep]
        publishable_candidates = [
            candidate for candidate in trimmed
            if self.candidate_is_hybrid_publishable(candidate)
        ]
        if publishable_candidates:
            return self._choose_best_in_bucket(publishable_candidates)
        return self._no_candidate_result(target)

    def collect_candidates_for_lemma(
        self,
        lemma: str,
        max_candidates_per_lemma: Optional[int] = None,
    ) -> List[Candidate]:
        lemma = lemma.strip().lower()
        if lemma not in self.lexicon:
            raise KeyError(f"Lemma not in lexicon: {lemma}")

        if lemma not in self._hybrid_pool_cache:
            target = self.lexicon[lemma]
            family = self.normalized_pos_family(target)
            fragile_family = self._family_is_fragile(family)
            attempts_used = 0
            raw_pool: List[Candidate] = []
            best_valid: Optional[Candidate] = None
            best_any: Optional[Candidate] = None
            valid_candidates_found = 0

            retrieved = self.retrieve_candidates(target)
            attempts_used += 1
            for cand in retrieved:
                raw_pool.append(cand)
                _, best_valid, best_any, valid_inc = self._absorb_prescored_candidate(
                    cand, best_valid, best_any
                )
                valid_candidates_found += valid_inc
            if self._should_early_exit(best_valid):
                deduped = self.dedupe_candidates(raw_pool)[: self.max_candidates_to_keep]
                selected = self._choose_from_pool(deduped, best_valid, best_any, target)
                self._hybrid_pool_cache[lemma] = deduped
                self._hybrid_search_cache[lemma] = {
                    "attempts_used": attempts_used,
                    "best_valid": best_valid,
                    "best_any": best_any,
                    "selected": selected,
                    "valid_count": valid_candidates_found,
                }
            else:
                use_content_templates = family in {"n", "v", "adj"} and self.can_template_target(target)
                use_pos_templates = family not in {"n", "v", "adj"}

                if not fragile_family:
                    stochastic_budget = min(
                        self.max_total_attempts - attempts_used,
                        max(80, int(self.max_total_attempts * 0.65)),
                    )
                    if stochastic_budget > 0 and attempts_used < self.max_total_attempts:
                        stochastic_candidates = self.generate_stochastic_candidates(target, attempts=stochastic_budget)
                        attempts_used += 1
                        for cand in stochastic_candidates:
                            raw_pool.append(cand)
                            _, best_valid, best_any, valid_inc = self._absorb_prescored_candidate(
                                cand, best_valid, best_any
                            )
                            valid_candidates_found += valid_inc

                if not self._should_early_exit(best_valid) and use_content_templates:
                    template_attempts = 0
                    use_seeded = True
                    while attempts_used < self.max_total_attempts:
                        builder = self.seeded_template_candidate if use_seeded else self.pure_template_candidate
                        use_seeded = not use_seeded
                        attempts_used += 1
                        template_attempts += 1
                        cand = builder(target)
                        ranked, best_valid, best_any, valid_inc = self._evaluate_raw_candidate(
                            cand, best_valid, best_any
                        )
                        if ranked:
                            raw_pool.append(ranked)
                        valid_candidates_found += valid_inc
                        if template_attempts % 25 == 0 and self._should_early_exit(best_valid):
                            break

                if not self._should_early_exit(best_valid) and use_pos_templates:
                    template_attempts = 0
                    use_seeded = True
                    while attempts_used < self.max_total_attempts:
                        builder = self.seeded_pos_template_candidate if use_seeded else self.pure_pos_template_candidate
                        use_seeded = not use_seeded
                        attempts_used += 1
                        template_attempts += 1
                        cand = builder(target)
                        ranked, best_valid, best_any, valid_inc = self._evaluate_raw_candidate(
                            cand, best_valid, best_any
                        )
                        if ranked:
                            raw_pool.append(ranked)
                        valid_candidates_found += valid_inc
                        if template_attempts % (40 if fragile_family else 25) == 0 and self._should_early_exit(best_valid):
                            break

                if best_valid is None and attempts_used < self.max_total_attempts:
                    attempts_used += 1
                    cand = self.emergency_pos_template_candidate(target)
                    if cand:
                        cand.source_method = "emergency_template"
                        cand.template_id = cand.template_id or "emergency"
                    ranked, best_valid, best_any, valid_inc = self._evaluate_raw_candidate(
                        cand, best_valid, best_any
                    )
                    if ranked:
                        raw_pool.append(ranked)
                    valid_candidates_found += valid_inc

                if best_valid is None and attempts_used < self.max_total_attempts:
                    attempts_used += 1
                    cand = self.emergency_starter_candidate(target)
                    if cand:
                        cand.source_method = "emergency_starter"
                        cand.template_id = cand.template_id or "emergency"
                    ranked, best_valid, best_any, valid_inc = self._evaluate_raw_candidate(
                        cand, best_valid, best_any
                    )
                    if ranked:
                        raw_pool.append(ranked)
                    valid_candidates_found += valid_inc

                if fragile_family and attempts_used < self.max_total_attempts and not self._has_preferred_publishable_candidate(raw_pool, family):
                    stochastic_budget = min(
                        self.max_total_attempts - attempts_used,
                        max(40, int(self.max_total_attempts * 0.25)),
                    )
                    if stochastic_budget > 0:
                        stochastic_candidates = self.generate_stochastic_candidates(target, attempts=stochastic_budget)
                        attempts_used += 1
                        for cand in stochastic_candidates:
                            raw_pool.append(cand)
                            _, best_valid, best_any, valid_inc = self._absorb_prescored_candidate(
                                cand, best_valid, best_any
                            )
                            valid_candidates_found += valid_inc

                deduped = self.dedupe_candidates(raw_pool)[: self.max_candidates_to_keep]
                selected = self._choose_from_pool(deduped, best_valid, best_any, target)
                self._hybrid_pool_cache[lemma] = deduped
                self._hybrid_search_cache[lemma] = {
                    "attempts_used": attempts_used,
                    "best_valid": best_valid,
                    "best_any": best_any,
                    "selected": selected,
                    "valid_count": valid_candidates_found,
                }

        pool = list(self._hybrid_pool_cache[lemma])
        if max_candidates_per_lemma is not None:
            return pool[: max(0, max_candidates_per_lemma)]
        return pool

    def generate_for_lemma(self, lemma: str) -> Candidate:
        lemma = lemma.strip().lower()
        if lemma not in self.lexicon:
            raise KeyError(f"Lemma not in lexicon: {lemma}")
        self.collect_candidates_for_lemma(lemma)
        selected = self._hybrid_search_cache.get(lemma, {}).get("selected")
        if selected is not None:
            meta = self.hybrid_output_metadata(
                selected,
                self._hybrid_search_cache[lemma].get("attempts_used", 0),
                self._hybrid_search_cache[lemma].get("valid_count", 0),
            )
            setattr(selected, "_hybrid_meta", meta)
            return selected
        candidate = self._no_candidate_result(self.lexicon[lemma])
        setattr(candidate, "_hybrid_meta", self.hybrid_output_metadata(candidate, 0, 0))
        return candidate

    def generate_sentence_for_target(self, target_lemma: str, target_rank: int) -> Dict[str, Any]:
        lemma = (target_lemma or "").strip().lower()
        band = get_profile(target_rank).band
        if not lemma:
            return {
                "lemma": lemma,
                "rank": target_rank,
                "band": band,
                "pos": "",
                "sentence": "",
                "target_form": "",
                "canonical_lemma": "",
                "source_method": "invalid_request",
                "template_id": "",
                "score": 0.0,
                "publishable": False,
                "quality_tier": "no_candidate_found",
                "bad_candidate": True,
                "failure_reason": "missing_target_lemma",
                "search_attempts_used": 0,
                "candidates_found": 0,
                "best_source_method": "invalid_request",
            }
        if lemma not in self.lexicon:
            return {
                "lemma": lemma,
                "rank": target_rank,
                "band": band,
                "pos": "",
                "sentence": "",
                "target_form": "",
                "canonical_lemma": "",
                "source_method": "missing_lemma",
                "template_id": "",
                "score": 0.0,
                "publishable": False,
                "quality_tier": "no_candidate_found",
                "bad_candidate": True,
                "failure_reason": "lemma_not_in_lexicon",
                "search_attempts_used": 0,
                "candidates_found": 0,
                "best_source_method": "missing_lemma",
            }

        candidate = self.generate_for_lemma(lemma)
        target = self.lexicon[lemma]
        meta = getattr(candidate, "_hybrid_meta", self.hybrid_output_metadata(candidate, 0, 0))
        return {
            "lemma": lemma,
            "rank": target_rank,
            "band": band,
            "pos": target.pos,
            "sentence": candidate.sentence,
            "target_form": candidate.target_form,
            "canonical_lemma": candidate.canonical_lemma or self.canonical_lemma_for(target),
            "source_method": candidate.source_method,
            "template_id": candidate.template_id,
            "score": candidate.score,
            "publishable": meta["publishable"],
            "quality_tier": meta["quality_tier"],
            "bad_candidate": meta["bad_candidate"],
            "failure_reason": meta["failure_reason"],
            "search_attempts_used": meta["search_attempts_used"],
            "candidates_found": meta["candidates_found"],
            "best_source_method": meta["best_source_method"],
        }

    def _output_row_dict(self, row: Candidate) -> Dict[str, Any]:
        meta = getattr(row, "_hybrid_meta", self.hybrid_output_metadata(row, 0, 0))
        return {
            "lemma": row.lemma,
            "rank": row.rank,
            "pos": row.pos,
            "band": row.band,
            "translation": row.translation,
            "sentence": row.sentence,
            "target_form": row.target_form,
            "canonical_lemma": row.canonical_lemma,
            "target_morph": row.target_morph,
            "target_index": row.target_index,
            "support_ranks": " ".join(str(x) for x in row.support_ranks),
            "avg_support_rank": row.avg_support_rank,
            "max_support_rank": row.max_support_rank,
            "template_id": row.template_id,
            "source_method": row.source_method,
            "score": row.score,
            "publishable": meta["publishable"],
            "quality_tier": meta["quality_tier"],
            "bad_candidate": meta["bad_candidate"],
            "failure_reason": meta["failure_reason"],
            "search_attempts_used": meta["search_attempts_used"],
            "candidates_found": meta["candidates_found"],
            "best_source_method": meta["best_source_method"],
        }

    def generate_batch(
        self,
        limit: int,
        out_csv: str,
        min_rank: int = 1,
        max_rank: int = 10**9,
        pos_filter: Optional[str] = None,
        lemma_filter: Optional[List[str]] = None,
        mvp_only: bool = False,
        candidates_out: Optional[str] = None,
        max_candidates_per_lemma: int = 10,
        progress_every: int = 25,
    ) -> List[Candidate]:
        self.last_candidate_export_stats = None
        if lemma_filter:
            rows = []
            seen = set()
            for lemma in lemma_filter:
                key = lemma.strip().lower()
                if not key or key in seen or key not in self.lexicon:
                    continue
                seen.add(key)
                rows.append(self.lexicon[key])
        else:
            rows = list(self.lexicon.values())
            rows.sort(key=lambda x: x.rank)
        rows = [x for x in rows if min_rank <= x.rank <= max_rank]
        if mvp_only:
            rows = [x for x in rows if x.pos in {"n", "v", "adj"} and get_profile(x.rank).band in {"A1", "A2", "B1"}]
        if pos_filter:
            rows = [x for x in rows if x.pos == pos_filter]
        rows = rows[:limit]

        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "band",
            "translation",
            "sentence",
            "target_form",
            "canonical_lemma",
            "target_morph",
            "target_index",
            "support_ranks",
            "avg_support_rank",
            "max_support_rank",
            "template_id",
            "source_method",
            "score",
            "publishable",
            "quality_tier",
            "bad_candidate",
            "failure_reason",
            "search_attempts_used",
            "candidates_found",
            "best_source_method",
        ]

        generated: List[Candidate] = []
        total = len(rows)

        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()

            for idx, lex in enumerate(rows, start=1):
                try:
                    row = self.generate_for_lemma(lex.lemma)
                except Exception as exc:
                    print(f"[warn] hybrid failed for {lex.lemma}: {exc}", file=sys.stderr, flush=True)
                    row = self._no_candidate_result(lex)
                    setattr(row, "_hybrid_meta", self.hybrid_output_metadata(row, 0, 0))

                generated.append(row)
                writer.writerow(self._output_row_dict(row))
                f.flush()

                if progress_every > 0 and (idx % progress_every == 0 or idx == total):
                    meta = getattr(row, "_hybrid_meta", {})
                    print(
                        f"[progress] {idx}/{total} "
                        f"lemma={lex.lemma} "
                        f"source={row.source_method} "
                        f"tier={meta.get('quality_tier', '')} "
                        f"publishable={meta.get('publishable', False)}",
                        flush=True,
                    )

        if candidates_out:
            candidate_rows: List[Candidate] = []
            lemmas_with_candidates = 0
            for lex in rows:
                pool = self.collect_candidates_for_lemma(
                    lex.lemma,
                    max_candidates_per_lemma=max_candidates_per_lemma,
                )
                if pool:
                    lemmas_with_candidates += 1
                    candidate_rows.extend(pool)
            self.write_candidates_csv(candidate_rows, candidates_out)
            avg_candidates = float(len(candidate_rows)) / lemmas_with_candidates if lemmas_with_candidates else 0.0
            self.last_candidate_export_stats = {
                "lemmas_processed": float(len(rows)),
                "candidate_rows_written": float(len(candidate_rows)),
                "lemmas_with_candidates": float(lemmas_with_candidates),
                "avg_candidates_per_lemma_with_candidates": avg_candidates,
            }
        return generated

    def write_csv(self, rows: List[Candidate], out_csv: str) -> None:
        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "band",
            "translation",
            "sentence",
            "target_form",
            "canonical_lemma",
            "target_morph",
            "target_index",
            "support_ranks",
            "avg_support_rank",
            "max_support_rank",
            "template_id",
            "source_method",
            "score",
            "publishable",
            "quality_tier",
            "bad_candidate",
            "failure_reason",
            "search_attempts_used",
            "candidates_found",
            "best_source_method",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(self._output_row_dict(row))


def generate_sentence_for_target(
    generator: HybridSentenceGenerator,
    target_lemma: str,
    target_rank: int,
) -> Dict[str, Any]:
    return generator.generate_sentence_for_target(target_lemma, target_rank)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Spanish sentence generator using retrieval, stochastic decoding, and template search.",
        epilog=(
            "Example: python hybrid_generator.py "
            "--lexicon stg_words_spa.csv --models-dir models "
            "--target-lemma casa --target-rank 500"
        ),
    )
    parser.add_argument("--lexicon", required=True, help="Path to stg_words_spa.csv")
    parser.add_argument("--models-dir", required=True, help="Directory containing .pkl artifacts")
    parser.add_argument("--out", default="hybrid_generated.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=100, help="Number of lexicon rows to generate")
    parser.add_argument("--min-rank", type=int, default=1, help="Minimum rank to include")
    parser.add_argument("--max-rank", type=int, default=10**9, help="Maximum rank to include")
    parser.add_argument("--pos", default=None, help="Optional POS filter")
    parser.add_argument("--lemma", action="append", help="Generate only these lemma(s). Can be repeated.")
    parser.add_argument("--gold-set", default=None, help="Path to a newline-delimited lemma list.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mvp-only", action="store_true", help="Limit to A1-B1 nouns, verbs, adjectives.")
    parser.add_argument("--candidates-out", default=None, help="Optional multi-candidate CSV export path.")
    parser.add_argument("--max-candidates-per-lemma", type=int, default=10, help="Max candidate rows per lemma.")
    parser.add_argument("--lexicon-overrides", action="append", default=[], help="Overrides CSV/JSON. Repeatable.")
    parser.add_argument("--target-lemma", default=None, help="Generate one structured result for this lemma.")
    parser.add_argument("--target-rank", type=int, default=None, help="Rank for --target-lemma band derivation.")
    parser.add_argument("--review-out", default=None, help="Optional review CSV export path.")
    parser.add_argument("--max-total-attempts", type=int, default=400, help="Hard ceiling on generation attempts per lemma.")
    parser.add_argument("--max-candidates-to-keep", type=int, default=20, help="Maximum scored candidates retained per lemma.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N lemmas.")
    args = parser.parse_args()

    gen = HybridSentenceGenerator(
        args.lexicon,
        args.models_dir,
        seed=args.seed,
        max_total_attempts=args.max_total_attempts,
        max_candidates_to_keep=args.max_candidates_to_keep,
    )
    for override_path in args.lexicon_overrides:
        gen.load_and_apply_overrides(override_path)
        print(f"Loaded {len(gen.overrides)} total lexicon overrides after {override_path}")

    if args.target_lemma is not None:
        target_rank = args.target_rank if args.target_rank is not None else gen.lexicon.get(args.target_lemma.strip().lower(), Lexeme("", 99999, "")).rank
        result = generate_sentence_for_target(gen, args.target_lemma, target_rank)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    lemma_filter: Optional[List[str]] = list(args.lemma or [])
    if args.gold_set:
        lemma_filter.extend(gen.load_gold_set(args.gold_set))
    if not lemma_filter:
        lemma_filter = None

    rows = gen.generate_batch(
        limit=args.limit,
        out_csv=args.out,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        pos_filter=args.pos,
        lemma_filter=lemma_filter,
        mvp_only=args.mvp_only,
        candidates_out=args.candidates_out,
        max_candidates_per_lemma=args.max_candidates_per_lemma,
        progress_every=args.progress_every,
    )
    if args.review_out:
        gen.write_review_csv(rows, args.review_out)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.source_method] = counts.get(row.source_method, 0) + 1
    print(f"Generated: {len(rows):,}")
    for key in sorted(counts):
        print(f"  {key:<20} {counts[key]:,}")
    print(f"Saved: {args.out}")
    if args.review_out:
        print(f"Review: {args.review_out}")
    if args.candidates_out and gen.last_candidate_export_stats:
        s = gen.last_candidate_export_stats
        print(f"Candidates: {args.candidates_out}")
        print(f"  candidate_rows_written: {int(s['candidate_rows_written']):,}")
        print(f"  avg_candidates_per_lemma: {s['avg_candidates_per_lemma_with_candidates']:.2f}")

    preview = [r for r in rows if r.sentence][:5]
    if preview:
        print("\nPreview:")
        for row in preview:
            meta = getattr(row, "_hybrid_meta", {})
            print(f"  {row.lemma:<15} [{row.source_method:<20}] {meta.get('quality_tier', ''):<18} {row.sentence}")


if __name__ == "__main__":
    main()
