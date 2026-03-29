#!/usr/bin/env python3
"""Stochastic Spanish sentence generator.

Uses trigram/bigram corpus artifacts for probabilistic left-to-right decoding,
then validates and scores candidates through the existing SentenceGenerator shell.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from generate import (
    CONTEXT_DEPENDENT_OPENERS,
    SPECIAL_VERB_LEMMAS,
    STARTER_ADJ_ALLOWED_NOUN_CLASSES,
    STARTER_SAFE_SUPPORT_NOUNS_FOR_ADJ,
    Candidate,
    Lexeme,
    SentenceGenerator,
    allowed_support_rank,
    get_profile,
    is_word_token,
    normalize_token,
)

_BANNED_CONTINUATIONS = CONTEXT_DEPENDENT_OPENERS | {
    "pues", "sino", "ni", "oh", "ay", "eh", "bueno",
}


class StochasticSentenceGenerator(SentenceGenerator):
    """Sentence generator whose primary candidate source is a constrained
    trigram/bigram stochastic decoder, with templates as fallback only."""

    def __init__(self, lexicon_path: str, models_dir: str, seed: int = 42):
        super().__init__(lexicon_path, models_dir, seed=seed)
        self._stochastic_pool_cache: Dict[str, List[Candidate]] = {}
        self._tri_next: Dict[Tuple[str, str], Dict[str, int]] = self.trigrams.get("next", {})
        self._tri_totals: Dict[Tuple[str, str], int] = self.trigrams.get("totals", {})
        self._bi_next: Dict[str, Dict[str, int]] = self.bigrams.get("next", {})
        self._bi_totals: Dict[str, int] = self.bigrams.get("totals", {})

    def load_and_apply_overrides(self, path: str) -> None:
        super().load_and_apply_overrides(path)
        self._stochastic_pool_cache.clear()

    # ------------------------------------------------------------------
    # A.  Seed generation — thin POS-specific logic
    # ------------------------------------------------------------------

    def initial_seeds_for_target(
        self, target: Lexeme
    ) -> List[Tuple[List[str], int]]:
        """Return ``(seed_tokens, target_index)`` pairs.

        Each seed is a short token prefix that already contains the target
        word so the decoder only needs to extend it to the right.
        """
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
            allowed_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(
                normalize_token(target.lemma)
            )
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

    def _verb_special_seeds(
        self, target: Lexeme, canonical: str
    ) -> List[Tuple[List[str], int]]:
        subj, pc = self.subject_for_target(target)
        verb = self.target_verb_form(target, pc)

        if canonical == "haber" and normalize_token(verb) == "hay":
            return [([verb], 0)]

        return [([subj, verb], 1)]

    # ------------------------------------------------------------------
    # B.  Next-word candidate retrieval (trigram → bigram)
    # ------------------------------------------------------------------

    def get_next_word_candidates(
        self,
        context_tokens: List[str],
        target: Lexeme,
        limit: int = 50,
    ) -> List[Tuple[str, float]]:
        """Return ``(word, probability)`` pairs sorted by descending probability."""
        profile = get_profile(target.rank)
        rank_ceiling = allowed_support_rank(target.rank, profile)

        ctx = [normalize_token(t) for t in context_tokens if is_word_token(t)]
        if not ctx:
            ctx = ["<START>"]

        raw: Dict[str, int] = {}
        total = 0

        # Trigram first
        if len(ctx) >= 2:
            key = (ctx[-2], ctx[-1])
            tri_cands = self._tri_next.get(key)
            if tri_cands:
                total = self._tri_totals.get(key, 0)
                if total > 0:
                    raw = tri_cands

        # Bigram fallback
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
            # Repetition loop guard
            if word in tail:
                continue
            if word in _BANNED_CONTINUATIONS:
                continue
            # Rank ceiling (skip for <END> and the target itself)
            if word != "<END>" and normalize_token(word) != normalize_token(target.lemma):
                wr = self.lookup_rank(word)
                if wr > rank_ceiling and wr > target.rank:
                    continue
            prob = count / total
            results.append((word, prob))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # C.  Stochastic sampling
    # ------------------------------------------------------------------

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

        return self.random.choices(
            [w for w, _ in top], weights=weights, k=1
        )[0]

    # ------------------------------------------------------------------
    # D.  Left-to-right constrained decoder
    # ------------------------------------------------------------------

    def decode_from_seed(
        self,
        seed_tokens: List[str],
        target: Lexeme,
        target_index: int,
        max_steps: int = 6,
        attempts: int = 100,
    ) -> List[List[str]]:
        """Run the decoder ``attempts`` times from ``seed_tokens``.

        Returns token lists whose word-count falls inside the profile's
        ``[min_len, max_len]`` window.
        """
        profile = get_profile(target.rank)
        max_len = profile.max_len
        min_len = profile.min_len
        decoded: List[List[str]] = []

        for _ in range(attempts):
            tokens = list(seed_tokens)
            ctx = ["<START>"] + [
                normalize_token(t) for t in tokens if is_word_token(t)
            ]

            for _step in range(max_steps):
                wc = sum(1 for t in tokens if is_word_token(t))
                if wc >= max_len:
                    break

                cands = self.get_next_word_candidates(ctx, target, limit=50)
                if not cands:
                    break

                # Don't end too early
                if wc < min_len:
                    cands = [(w, p) for w, p in cands if w != "<END>"]
                    if not cands:
                        break

                word = self.sample_next_word(cands, top_k=10, temperature=0.9)
                if not word or word == "<END>":
                    break

                tokens.append(word)
                ctx.append(normalize_token(word))

            wc = sum(1 for t in tokens if is_word_token(t))
            if min_len <= wc <= max_len:
                decoded.append(tokens)

        return decoded

    # ------------------------------------------------------------------
    # E.  Stochastic candidate pipeline
    # ------------------------------------------------------------------

    def generate_stochastic_candidates(
        self, target: Lexeme, attempts: int = 150
    ) -> List[Candidate]:
        seeds = self.initial_seeds_for_target(target)
        if not seeds:
            return []

        per_seed = max(10, attempts // max(1, len(seeds)))
        candidates: List[Candidate] = []

        for seed_tokens, tidx in seeds:
            sequences = self.decode_from_seed(
                seed_tokens, target, tidx,
                max_steps=6, attempts=per_seed,
            )
            for seq in sequences:
                cand = self.build_candidate(
                    target, seq,
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

    # ------------------------------------------------------------------
    # F.  Candidate collection override
    # ------------------------------------------------------------------

    def collect_candidates_for_lemma(
        self,
        lemma: str,
        max_candidates_per_lemma: Optional[int] = None,
    ) -> List[Candidate]:
        lemma = lemma.strip().lower()
        if lemma not in self.lexicon:
            raise KeyError(f"Lemma not in lexicon: {lemma}")

        if lemma not in self._stochastic_pool_cache:
            target = self.lexicon[lemma]
            candidates: List[Candidate] = []

            # 1. Retrieved corpus candidates (unchanged)
            candidates.extend(self.retrieve_candidates(target))

            # 2. Stochastic decoder candidates (core POS only)
            if target.pos in {"n", "v", "adj"}:
                candidates.extend(
                    self.generate_stochastic_candidates(target)
                )

            # 3. Small template fallback (not the main engine)
            if target.pos in {"n", "v", "adj"} and self.can_template_target(target):
                for _ in range(6):
                    cand = self.seeded_template_candidate(target)
                    if cand:
                        ok, penalties = self.validate(cand)
                        if ok:
                            self.score(cand, penalties)
                            candidates.append(cand)
                for _ in range(10):
                    cand = self.pure_template_candidate(target)
                    if cand:
                        ok, penalties = self.validate(cand)
                        if ok:
                            self.score(cand, penalties)
                            candidates.append(cand)

            self._stochastic_pool_cache[lemma] = self.dedupe_candidates(
                candidates
            )

        pool = list(self._stochastic_pool_cache[lemma])
        if max_candidates_per_lemma is not None:
            return pool[: max(0, max_candidates_per_lemma)]
        return pool

    # ------------------------------------------------------------------
    # G.  Public API  (generate_for_lemma / generate_sentence_for_target
    #     are inherited and will use our collect_candidates_for_lemma)
    # ------------------------------------------------------------------


def generate_sentence_for_target(
    generator: StochasticSentenceGenerator,
    target_lemma: str,
    target_rank: int,
) -> Dict[str, Any]:
    """Public API matching the contract in generate.py."""
    return generator.generate_sentence_for_target(target_lemma, target_rank)


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stochastic Spanish sentence generator using trigram/bigram decoding.",
        epilog=(
            "Example: python stochastic_generator.py "
            "--lexicon stg_words_spa.csv --models-dir models "
            "--target-lemma casa --target-rank 500"
        ),
    )
    parser.add_argument("--lexicon", required=True, help="Path to stg_words_spa.csv")
    parser.add_argument("--models-dir", required=True, help="Directory containing .pkl artifacts")
    parser.add_argument("--out", default="stochastic_generated.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=100, help="Number of lexicon rows to generate")
    parser.add_argument("--min-rank", type=int, default=1, help="Minimum rank to include")
    parser.add_argument("--max-rank", type=int, default=10**9, help="Maximum rank to include")
    parser.add_argument("--pos", default=None, help="Optional POS filter: n, v, adj")
    parser.add_argument("--lemma", action="append", help="Generate only these lemma(s). Can be repeated.")
    parser.add_argument("--gold-set", default=None, help="Path to a newline-delimited lemma list.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mvp-only", action="store_true", help="Limit to A1-B1 nouns, verbs, adjectives.")
    parser.add_argument("--candidates-out", default=None, help="Optional multi-candidate CSV export path.")
    parser.add_argument("--max-candidates-per-lemma", type=int, default=10, help="Max candidate rows per lemma.")
    parser.add_argument("--lexicon-overrides", action="append", default=[], help="Overrides CSV/JSON. Repeatable.")
    parser.add_argument("--target-lemma", default=None, help="Generate one structured result for this lemma.")
    parser.add_argument("--target-rank", type=int, default=None, help="Rank for --target-lemma band derivation.")
    args = parser.parse_args()

    gen = StochasticSentenceGenerator(args.lexicon, args.models_dir, seed=args.seed)
    for override_path in args.lexicon_overrides:
        gen.load_and_apply_overrides(override_path)
        print(f"Loaded {len(gen.overrides)} total lexicon overrides after {override_path}")

    # --- Single-target mode ---
    if args.target_lemma is not None:
        target_rank = (
            args.target_rank
            if args.target_rank is not None
            else gen.lexicon.get(
                args.target_lemma.strip().lower(), Lexeme("", 99999, "")
            ).rank
        )
        result = generate_sentence_for_target(gen, args.target_lemma, target_rank)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # --- Batch mode ---
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

    stochastic = sum(1 for r in rows if r.source_method == "stochastic_decoder")
    retrieved = sum(1 for r in rows if r.source_method == "retrieved_corpus")
    seeded = sum(1 for r in rows if r.source_method == "seeded_template")
    templated = sum(1 for r in rows if r.source_method == "template_generated")
    reviewed = sum(1 for r in rows if r.source_method == "manual_review_needed")
    print(f"Generated: {len(rows):,}")
    print(f"  stochastic_decoder:   {stochastic:,}")
    print(f"  retrieved_corpus:     {retrieved:,}")
    print(f"  seeded_template:      {seeded:,}")
    print(f"  template_generated:   {templated:,}")
    print(f"  manual_review_needed: {reviewed:,}")
    print(f"Saved: {args.out}")
    if args.candidates_out and gen.last_candidate_export_stats:
        s = gen.last_candidate_export_stats
        print(f"Candidates: {args.candidates_out}")
        print(f"  candidate_rows_written: {int(s['candidate_rows_written']):,}")
        print(f"  avg_candidates_per_lemma: {s['avg_candidates_per_lemma_with_candidates']:.2f}")

    preview = [r for r in rows if r.sentence][:5]
    if preview:
        print("\nPreview:")
        for row in preview:
            print(f"  {row.lemma:<15} [{row.source_method:<20}] {row.sentence}")


if __name__ == "__main__":
    main()
