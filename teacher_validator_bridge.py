#!/usr/bin/env python3
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import complete_generate as cg
from complete_generate import Candidate, Lexeme, SentenceGenerator, normalize_token

try:
    from hybrid_generator import HybridSentenceGenerator
except Exception:
    HybridSentenceGenerator = None

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_META_RE = re.compile(r"\b(?:explicación|explanation|translation|traducción|note|nota)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


@dataclass
class ValidationResult:
    accepted: bool
    candidate: Optional[Candidate]
    rejection_reasons: List[str]
    quality_flags: Dict[str, str]
    support_words: List[str]
    support_rank_max: int
    support_rank_avg: float
    token_count: int
    pattern_ready: bool


class TeacherValidatorBridge:
    def __init__(self, lexicon_path: str, models_dir: str, seed: int = 42):
        if HybridSentenceGenerator is not None:
            self.generator = HybridSentenceGenerator(lexicon_path, models_dir, seed=seed)
        else:
            self.generator = SentenceGenerator(lexicon_path, models_dir, seed=seed)

    def validate_teacher_sentence(
        self,
        lemma: str,
        rank: int,
        pos: str,
        sentence: str,
        translation: str = "",
        min_tokens: int = 2,
        max_tokens: int = 10,
    ) -> ValidationResult:
        reasons = self._hard_filter_reasons(lemma=lemma, sentence=sentence, min_tokens=min_tokens, max_tokens=max_tokens)
        token_count = len(_word_tokens(sentence))
        if reasons:
            return ValidationResult(
                accepted=False,
                candidate=None,
                rejection_reasons=reasons,
                quality_flags={"grammatical_ok": "0", "natural_ok": "0", "learner_clear_ok": "0", "notes": "; ".join(reasons)},
                support_words=[],
                support_rank_max=0,
                support_rank_avg=0.0,
                token_count=token_count,
                pattern_ready=False,
            )

        target = self.generator.lexicon.get(normalize_token(lemma))
        if target is None:
            target = Lexeme(lemma=normalize_token(lemma), rank=rank, pos=pos, translation=translation)
        else:
            target = Lexeme(
                lemma=target.lemma,
                rank=rank or target.rank,
                pos=pos or target.pos,
                translation=translation or target.translation,
                gender=target.gender,
                semantic_class=target.semantic_class,
                verb_type=target.verb_type,
                required_prep=target.required_prep,
                is_reflexive=target.is_reflexive,
                canonical_lemma=target.canonical_lemma,
            )

        tokens = self.generator.sentence_tokens(sentence)
        target_index = next((i for i, tok in enumerate(tokens) if normalize_token(tok) == normalize_token(lemma)), -1)
        if target_index < 0:
            return ValidationResult(
                accepted=False,
                candidate=None,
                rejection_reasons=["missing_exact_target"],
                quality_flags={"grammatical_ok": "0", "natural_ok": "0", "learner_clear_ok": "0", "notes": "missing_exact_target"},
                support_words=[],
                support_rank_max=0,
                support_rank_avg=0.0,
                token_count=token_count,
                pattern_ready=False,
            )

        candidate = self.generator.build_candidate(
            target=target,
            tokens=tokens,
            template_id="teacher_candidate",
            source_method="teacher_generated",
            target_index=target_index,
        )
        if candidate is None:
            return ValidationResult(
                accepted=False,
                candidate=None,
                rejection_reasons=["build_candidate_failed"],
                quality_flags={"grammatical_ok": "0", "natural_ok": "0", "learner_clear_ok": "0", "notes": "build_candidate_failed"},
                support_words=[],
                support_rank_max=0,
                support_rank_avg=0.0,
                token_count=token_count,
                pattern_ready=False,
            )

        valid, penalties = self.generator.validate(candidate)
        self.generator.score(candidate, penalties)
        grammatical_ok, natural_ok, learner_clear_ok, notes = self.generator.review_flags(candidate)
        support_words = [
            normalize_token(tok)
            for i, tok in enumerate(self.generator.sentence_tokens(candidate.sentence))
            if self.generator.lookup_rank(tok) < 99999 and i != candidate.target_index and self.generator.lookup_pos(tok) != ""
        ]
        support_ranks = [self.generator.lookup_rank(word) for word in support_words]
        support_rank_max = max(support_ranks) if support_ranks else 0
        support_rank_avg = sum(support_ranks) / len(support_ranks) if support_ranks else 0.0

        extra_reasons: List[str] = []
        if not valid:
            extra_reasons.append("failed_generator_validation")
        if grammatical_ok != "1":
            extra_reasons.append("not_grammatical")
        if natural_ok != "1":
            extra_reasons.append("not_natural")
        if learner_clear_ok != "1":
            extra_reasons.append("not_learner_clear")
        if notes and notes != "manual review needed":
            extra_reasons.extend([part.strip() for part in notes.split(";") if part.strip()])
        extra_reasons.extend(self._learner_suitability_reasons(candidate))
        extra_reasons = list(dict.fromkeys(reason for reason in extra_reasons if reason))

        accepted = not extra_reasons
        return ValidationResult(
            accepted=accepted,
            candidate=candidate,
            rejection_reasons=extra_reasons,
            quality_flags={
                "grammatical_ok": grammatical_ok,
                "natural_ok": natural_ok,
                "learner_clear_ok": learner_clear_ok,
                "notes": notes,
            },
            support_words=support_words,
            support_rank_max=support_rank_max,
            support_rank_avg=support_rank_avg,
            token_count=token_count,
            pattern_ready=accepted and candidate.sentence != "",
        )

    def _hard_filter_reasons(self, lemma: str, sentence: str, min_tokens: int, max_tokens: int) -> List[str]:
        reasons: List[str] = []
        stripped = (sentence or "").strip()
        tokens = _word_tokens(stripped)
        if not stripped:
            reasons.append("empty_sentence")
            return reasons
        if len(tokens) < min_tokens:
            reasons.append("too_short")
        if len(tokens) > max_tokens:
            reasons.append("too_long")
        if _META_RE.search(stripped):
            reasons.append("meta_text")
        if len([chunk for chunk in _SENTENCE_SPLIT_RE.split(stripped) if chunk.strip()]) > 1:
            reasons.append("multiple_sentences")
        if normalize_token(lemma) not in {normalize_token(tok) for tok in tokens}:
            reasons.append("missing_exact_target")
        if stripped.count("(") != stripped.count(")"):
            reasons.append("unbalanced_punctuation")
        if stripped.count('"') % 2 == 1:
            reasons.append("unbalanced_quotes")
        return reasons

    def _learner_suitability_reasons(self, candidate: Candidate) -> List[str]:
        reasons: List[str] = []
        words = self.generator.word_tokens(candidate.sentence)
        profile = cg.get_profile(candidate.rank)
        max_reasonable_support = max(profile.filler_ceil, int(candidate.rank * 1.6))
        support_ranks = []
        for idx, token in enumerate(words):
            if idx == candidate.target_index:
                continue
            rank = self.generator.lookup_rank(token)
            if rank < 99999:
                support_ranks.append(rank)
            if self.generator.lookup_pos(token) == "prop":
                reasons.append("contains_name")
        if support_ranks and max(support_ranks) > max_reasonable_support:
            reasons.append("support_vocabulary_too_rare")
        if self.generator.finite_verb_count(words) > 2:
            reasons.append("too_many_verbs")
        return list(dict.fromkeys(reasons))



def _word_tokens(sentence: str) -> List[str]:
    return _WORD_RE.findall(sentence or "")
