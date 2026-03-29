#!/usr/bin/env python3
"""Hybrid Spanish sentence generator.

Combines retrieved corpus candidates, stochastic decoding, and template search
on top of the complete_generate SentenceGenerator shell.
"""
import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from complete_generate import (
    CONTEXT_DEPENDENT_OPENERS,
    SPECIAL_VERB_LEMMAS,
    STARTER_ADJ_ALLOWED_NOUN_CLASSES,
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

        if target.pos == "n":
            if not self.noun_is_template_friendly(target):
                return seeds
            gender = self.safe_noun_gender(target.lemma, target.gender)
            for definite in (True, False):
                art = self.choose_article(gender, definite=definite)
                seeds.append(([art, target.lemma], 1))
            return seeds

        if target.pos == "v":
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

        if target.pos == "adj":
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

        return seeds

    def _verb_special_seeds(self, target: Lexeme, canonical: str) -> List[Tuple[List[str], int]]:
        subj, pc = self.subject_for_target(target)
        verb = self.target_verb_form(target, pc)

        if canonical == "haber" and normalize_token(verb) == "hay":
            return [([verb], 0)]

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
        if candidate.source_method == "hardcoded_fallback":
            return False
        ok, _ = self.validate(candidate)
        return ok

    def _should_early_exit(self, candidate: Optional[Candidate]) -> bool:
        if not candidate:
            return False
        if not self.candidate_is_general_publishable(candidate):
            return False
        if candidate.source_method == "retrieved_corpus":
            return candidate.score >= 8.5
        return candidate.score >= 8.0

    def _rank_candidate(
        self,
        candidate: Optional[Candidate],
        best_valid: Optional[Candidate],
        best_any: Optional[Candidate],
    ) -> Tuple[Optional[Candidate], Optional[Candidate], Optional[Candidate]]:
        if not candidate or not candidate.sentence:
            return None, best_valid, best_any
        if candidate.source_method != "hardcoded_fallback":
            ok, penalties = self.validate(candidate)
            if ok:
                self.score(candidate, penalties)
                if best_valid is None or candidate.score > best_valid.score:
                    best_valid = candidate
                if best_any is None or candidate.score > best_any.score:
                    best_any = candidate
                return candidate, best_valid, best_any
            self.score(candidate, penalties)
            if best_any is None or candidate.score > best_any.score:
                best_any = candidate
            return candidate, best_valid, best_any
        if best_any is None or candidate.score > best_any.score:
            best_any = candidate
        return candidate, best_valid, best_any

    def _manual_candidate(self, target: Lexeme, sentence: str, source_method: str, template_id: str) -> Candidate:
        tokens = self.sentence_tokens(sentence)
        target_index = next((i for i, tok in enumerate(tokens) if normalize_token(tok) == normalize_token(target.lemma)), -1)
        target_form = tokens[target_index] if 0 <= target_index < len(tokens) else target.lemma
        support_ranks = [
            self.lookup_rank(tok)
            for i, tok in enumerate(tokens)
            if is_word_token(tok) and i != target_index
        ]
        avg_support_rank = (sum(support_ranks) / len(support_ranks)) if support_ranks else 0.0
        max_support_rank = max(support_ranks) if support_ranks else 0
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
            avg_support_rank=avg_support_rank,
            max_support_rank=max_support_rank,
            template_id=template_id,
            source_method=source_method,
            canonical_lemma=self.canonical_lemma_for(target),
            target_morph=self.target_morph_for_request(target, target_form, sentence_tokens=tokens, target_idx=target_index) if target_index >= 0 else "",
        )

    def hardcoded_fallback_candidate(self, target: Lexeme) -> Candidate:
        canonical = self.canonical_lemma_for(target)
        if target.pos == "n":
            article = self.choose_article(self.safe_noun_gender(target.lemma, target.gender), definite=True)
            tokens = [article, target.lemma, "está", "aquí"]
            candidate = self.build_candidate(target, tokens, "hardcoded_fallback", "hardcoded_fallback", 1)
        elif target.pos == "v":
            morph = self.target_form_metadata(target)
            if normalize_token(target.lemma) == normalize_token(canonical) and morph.get("VerbForm", "Inf") == "Inf":
                tokens = ["ella", "quiere", target.lemma]
                candidate = self.build_candidate(target, tokens, "hardcoded_fallback", "hardcoded_fallback", 2)
            else:
                verb_form = self.conjugate_present(canonical, "3sg")
                tokens = ["ella", verb_form]
                candidate = self.build_candidate(target, tokens, "hardcoded_fallback", "hardcoded_fallback", 1)
        elif target.pos == "adj":
            tokens = ["es", target.lemma]
            candidate = self.build_candidate(target, tokens, "hardcoded_fallback", "hardcoded_fallback", 1)
        else:
            tokens = [target.lemma.capitalize(), "es", "una", "palabra"]
            candidate = self.build_candidate(target, tokens, "hardcoded_fallback", "hardcoded_fallback", 0)

        if candidate:
            candidate.source_method = "hardcoded_fallback"
            candidate.template_id = "hardcoded_fallback"
            return candidate

        sentence = ""
        if target.pos == "n":
            article = self.choose_article(self.safe_noun_gender(target.lemma, target.gender), definite=True)
            sentence = f"{article.capitalize()} {target.lemma} está aquí."
        elif target.pos == "v":
            morph = self.target_form_metadata(target)
            if normalize_token(target.lemma) == normalize_token(canonical) and morph.get("VerbForm", "Inf") == "Inf":
                sentence = f"Ella quiere {target.lemma}."
            else:
                sentence = f"Ella {self.conjugate_present(canonical, '3sg')}."
        elif target.pos == "adj":
            sentence = f"Es {target.lemma}."
        else:
            sentence = f"{target.lemma.capitalize()} es una palabra."
        candidate = self._manual_candidate(target, sentence, "hardcoded_fallback", "hardcoded_fallback")
        candidate.score = -1.0
        return candidate

    def hybrid_output_metadata(
        self,
        candidate: Candidate,
        search_attempts_used: int,
        candidates_found: int,
    ) -> Dict[str, Any]:
        publishable = self.candidate_is_general_publishable(candidate)
        valid = self._candidate_is_valid(candidate)
        if candidate.source_method == "hardcoded_fallback":
            quality_tier = "hardcoded_fallback"
        elif candidate.source_method in {"emergency_template", "emergency_starter"}:
            quality_tier = "emergency"
        elif publishable and candidate.score >= 8.0:
            quality_tier = "strong"
        elif publishable:
            quality_tier = "acceptable"
        elif candidate.sentence and valid:
            quality_tier = "weak"
        else:
            quality_tier = "weak"

        bad_candidate = quality_tier in {"emergency", "hardcoded_fallback"}
        failure_reason = ""
        if not publishable:
            grammatical_ok, natural_ok, learner_clear_ok, notes = self.review_flags(candidate)
            reasons: List[str] = []
            if grammatical_ok != "1":
                reasons.append("not_grammatical")
            if natural_ok != "1":
                reasons.append("not_natural")
            if learner_clear_ok != "1":
                reasons.append("not_learner_clear")
            if notes and notes != "manual review needed":
                reasons.append(notes)
            failure_reason = "; ".join(dict.fromkeys(r for r in reasons if r))

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
        fallback: Candidate,
    ) -> Candidate:
        trimmed = self.dedupe_candidates(pool)[: self.max_candidates_to_keep]
        winner: Optional[Candidate] = None
        if trimmed:
            if self.reranker:
                try:
                    predictions = predict_candidate_scores(self.reranker, trimmed)
                except Exception:
                    predictions = []
                if len(predictions) == len(trimmed):
                    winner = max(zip(trimmed, predictions), key=lambda item: (item[1], item[0].score))[0]
            if winner is None:
                winner = self.select_best_candidate(trimmed)

        if winner is not None and (winner.source_method == "hardcoded_fallback" or self._candidate_is_valid(winner)):
            return winner
        if best_valid is not None:
            return best_valid
        if best_any is not None and best_any.sentence:
            return best_any
        return fallback

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
            attempts_used = 0
            raw_pool: List[Candidate] = []
            best_valid: Optional[Candidate] = None
            best_any: Optional[Candidate] = None

            retrieved = self.retrieve_candidates(target)
            attempts_used += 1
            for cand in retrieved:
                raw_pool.append(cand)
                _, best_valid, best_any = self._rank_candidate(cand, best_valid, best_any)
            if self._should_early_exit(best_valid):
                deduped = self.dedupe_candidates(raw_pool)[: self.max_candidates_to_keep]
                fallback = self.hardcoded_fallback_candidate(target)
                selected = self._choose_from_pool(deduped, best_valid, best_any, fallback)
                valid_count = sum(1 for cand in deduped if self._candidate_is_valid(cand))
                self._hybrid_pool_cache[lemma] = deduped
                self._hybrid_search_cache[lemma] = {
                    "attempts_used": attempts_used,
                    "best_valid": best_valid,
                    "best_any": best_any,
                    "selected": selected,
                    "valid_count": valid_count,
                }
            else:
                stochastic_budget = min(
                    self.max_total_attempts - attempts_used,
                    max(50, int(self.max_total_attempts * 0.4)),
                )
                if stochastic_budget > 0 and attempts_used < self.max_total_attempts:
                    stochastic_candidates = self.generate_stochastic_candidates(target, attempts=stochastic_budget)
                    attempts_used += 1
                    for cand in stochastic_candidates:
                        raw_pool.append(cand)
                        _, best_valid, best_any = self._rank_candidate(cand, best_valid, best_any)

                if not self._should_early_exit(best_valid) and self.can_template_target(target):
                    template_attempts = 0
                    use_seeded = True
                    while attempts_used < self.max_total_attempts:
                        builder = self.seeded_template_candidate if use_seeded else self.pure_template_candidate
                        use_seeded = not use_seeded
                        attempts_used += 1
                        template_attempts += 1
                        cand = builder(target)
                        ranked, best_valid, best_any = self._rank_candidate(cand, best_valid, best_any)
                        if ranked:
                            raw_pool.append(ranked)
                        if template_attempts % 25 == 0 and self._should_early_exit(best_valid):
                            break

                if best_valid is None and attempts_used < self.max_total_attempts:
                    attempts_used += 1
                    cand = self.emergency_pos_template_candidate(target)
                    ranked, best_valid, best_any = self._rank_candidate(cand, best_valid, best_any)
                    if ranked:
                        ranked.source_method = "emergency_template"
                        raw_pool.append(ranked)
                        if best_valid is ranked:
                            best_valid.source_method = "emergency_template"
                        if best_any is ranked:
                            best_any.source_method = "emergency_template"

                if best_valid is None and attempts_used < self.max_total_attempts:
                    attempts_used += 1
                    cand = self.emergency_starter_candidate(target)
                    ranked, best_valid, best_any = self._rank_candidate(cand, best_valid, best_any)
                    if ranked:
                        ranked.source_method = "emergency_starter"
                        raw_pool.append(ranked)
                        if best_valid is ranked:
                            best_valid.source_method = "emergency_starter"
                        if best_any is ranked:
                            best_any.source_method = "emergency_starter"

                fallback = self.hardcoded_fallback_candidate(target)
                if best_valid is None and (best_any is None or not best_any.sentence):
                    best_any = fallback

                deduped = self.dedupe_candidates(raw_pool)[: self.max_candidates_to_keep]
                selected = self._choose_from_pool(deduped, best_valid, best_any, fallback)
                valid_count = sum(1 for cand in deduped if self._candidate_is_valid(cand))
                self._hybrid_pool_cache[lemma] = deduped
                self._hybrid_search_cache[lemma] = {
                    "attempts_used": attempts_used,
                    "best_valid": best_valid,
                    "best_any": best_any,
                    "selected": selected,
                    "valid_count": valid_count,
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
        if selected and selected.sentence:
            meta = self.hybrid_output_metadata(
                selected,
                self._hybrid_search_cache[lemma].get("attempts_used", 0),
                self._hybrid_search_cache[lemma].get("valid_count", 0),
            )
            setattr(selected, "_hybrid_meta", meta)
            return selected
        fallback = self.hardcoded_fallback_candidate(self.lexicon[lemma])
        setattr(fallback, "_hybrid_meta", self.hybrid_output_metadata(fallback, 0, 0))
        return fallback

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
                "quality_tier": "hardcoded_fallback",
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
                "quality_tier": "hardcoded_fallback",
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

        generated: List[Candidate] = []
        candidate_rows: List[Candidate] = []
        lemmas_with_candidates = 0
        for lex in rows:
            try:
                row = self.generate_for_lemma(lex.lemma)
            except Exception as exc:
                print(f"[warn] hybrid failed for {lex.lemma}: {exc}", file=sys.stderr)
                row = self.hardcoded_fallback_candidate(lex)
                setattr(row, "_hybrid_meta", self.hybrid_output_metadata(row, 0, 0))
            generated.append(row)
            if candidates_out:
                pool = self.collect_candidates_for_lemma(lex.lemma, max_candidates_per_lemma=max_candidates_per_lemma)
                if pool:
                    lemmas_with_candidates += 1
                    candidate_rows.extend(pool)

        self.write_csv(generated, out_csv)
        if candidates_out:
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
                meta = getattr(row, "_hybrid_meta", self.hybrid_output_metadata(row, 0, 0))
                writer.writerow(
                    {
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
                )


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
