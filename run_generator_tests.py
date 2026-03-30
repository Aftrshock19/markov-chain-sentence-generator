#!/usr/bin/env python3
import sys
import types
import unittest

# Allow importing complete_generate without the training helper present.
reranker_stub = types.ModuleType("reranker")
reranker_stub.load_reranker_model = lambda *args, **kwargs: None
reranker_stub.predict_candidate_scores = lambda *args, **kwargs: []
sys.modules.setdefault("reranker", reranker_stub)

from complete_generate import Candidate, Lexeme, SentenceGenerator
from coverage_utils import policy_exclusion_reason
from evaluate_generator import compare_summaries, row_pos_family
from hybrid_generator import HybridSentenceGenerator


class SentenceGeneratorContractTests(unittest.TestCase):
    def make_generator(self):
        gen = SentenceGenerator.__new__(SentenceGenerator)
        gen.normalized_pos_family = lambda target: {
            "pron": "pron",
            "v": "v",
            "adj": "adj",
            "none": "residual",
            "": "residual",
        }.get(target.pos, target.pos)
        gen.canonical_lemma_for = lambda target: target.canonical_lemma or target.lemma
        gen.requested_lemma_allows_inflected_target = lambda target: target.pos in {"v", "n", "adj"}
        gen.exact_surface_template_candidate = lambda target, source: object() if target.lemma == "gran" else None
        gen.candidate_is_publishable = lambda candidate: candidate.source_method != "manual_review_needed"
        gen.review_flags = lambda candidate: ("1", "1", "1", "")
        return gen

    def test_target_generation_policy_marks_pronoun_exact_surface(self):
        gen = self.make_generator()
        policy = gen.target_generation_policy(Lexeme("que", 2, "pron"))
        self.assertTrue(policy["exact_surface_required"])
        self.assertFalse(policy["allows_inflected_target"])

    def test_target_generation_policy_allows_inflected_verbs(self):
        gen = self.make_generator()
        policy = gen.target_generation_policy(Lexeme("dar", 410, "v", canonical_lemma="dar"))
        self.assertFalse(policy["exact_surface_required"])
        self.assertTrue(policy["allows_inflected_target"])


    def test_target_generation_policy_marks_letter_as_policy_excluded(self):
        gen = self.make_generator()
        gen.policy_exclusion_reason = lambda target: "policy_excluded_single_letter" if target.lemma == "s" else None
        policy = gen.target_generation_policy(Lexeme("s", 737, "letter"))
        self.assertTrue(policy["excluded_by_policy"])
        self.assertEqual(policy["policy_reason"], "policy_excluded_single_letter")

    def test_policy_exclusion_distinguishes_letters_from_ordinary_single_letter_function_words(self):
        self.assertIsNone(policy_exclusion_reason("a", "none"))
        self.assertIsNone(policy_exclusion_reason("y", "conj"))
        self.assertEqual(policy_exclusion_reason("s", "letter"), "policy_excluded_single_letter")

    def test_policy_exclusion_keeps_residual_abbreviations_available_for_metalinguistic_routes(self):
        self.assertIsNone(policy_exclusion_reason("sr.", "none"))
        self.assertIsNone(policy_exclusion_reason("dr.", "none"))

    def test_failure_result_uses_structured_contract(self):
        gen = self.make_generator()
        result = gen.failure_result("", 500, "invalid_request", "missing_target_lemma")
        self.assertEqual(result["quality_tier"], "no_candidate_found")
        self.assertEqual(result["failure_reason"], "missing_target_lemma")
        self.assertFalse(result["publishable"])

    def test_result_meta_for_candidate_marks_manual_review(self):
        gen = self.make_generator()
        cand = Candidate(
            lemma="de",
            rank=1,
            pos="prep",
            band="A1",
            translation="of",
            sentence="",
            target_form="de",
            target_index=-1,
            support_ranks=[],
            avg_support_rank=0.0,
            max_support_rank=0,
            template_id="",
            source_method="manual_review_needed",
        )
        meta = gen._result_meta_for_candidate(cand)
        self.assertEqual(meta["quality_tier"], "no_candidate_found")
        self.assertEqual(meta["failure_reason"], "manual_review_needed")

    def test_has_exact_surface_template_detects_exact_route(self):
        gen = self.make_generator()
        self.assertTrue(gen.has_exact_surface_template(Lexeme("gran", 171, "adj")))
        self.assertFalse(gen.has_exact_surface_template(Lexeme("casa", 40, "n")))

    def test_review_override_template_accepts_exact_template_prefixes(self):
        gen = self.make_generator()
        cand = Candidate(
            lemma="de",
            rank=1,
            pos="prep",
            band="A1",
            translation="of",
            sentence="Hablo de mi trabajo.",
            target_form="de",
            target_index=1,
            support_ranks=[],
            avg_support_rank=0.0,
            max_support_rank=0,
            template_id="prep_exact_de_work",
            source_method="template_generated",
        )
        self.assertTrue(gen.review_override_template(cand))

    def test_exact_surface_template_specs_include_safe_hasta_route(self):
        gen = SentenceGenerator.__new__(SentenceGenerator)
        gen.normalized_pos_family = lambda target: target.pos
        specs = gen.exact_surface_template_specs(Lexeme("hasta", 98, "prep"))
        self.assertEqual(specs[0], ("prep_exact_hasta_home", ["voy", "hasta", "casa"], 1))


class HybridSelectorTests(unittest.TestCase):
    def test_choose_from_pool_prefers_exact_surface_candidate(self):
        gen = HybridSentenceGenerator.__new__(HybridSentenceGenerator)
        gen.max_candidates_to_keep = 20
        gen.dedupe_candidates = lambda candidates: candidates
        gen.candidate_is_hybrid_publishable = lambda candidate: True
        gen.has_exact_surface_template = lambda target: True
        gen.normalized_pos_family = lambda target: "prep"
        gen._candidate_bucket = lambda candidate, family: (0, 0 if candidate.template_id.startswith("prep_exact_") else 1)
        gen._choose_best_in_bucket = lambda candidates: max(candidates, key=lambda candidate: candidate.score)

        exact = Candidate(
            lemma="según",
            rank=377,
            pos="prep",
            band="A1",
            translation="according to",
            sentence="Según él es verdad.",
            target_form="según",
            target_index=0,
            support_ranks=[],
            avg_support_rank=0.0,
            max_support_rank=0,
            template_id="prep_exact_segun_el",
            source_method="template_generated",
            score=8.5,
        )
        retrieved = Candidate(
            lemma="según",
            rank=377,
            pos="prep",
            band="A1",
            translation="according to",
            sentence="Según dice ella.",
            target_form="según",
            target_index=0,
            support_ranks=[],
            avg_support_rank=0.0,
            max_support_rank=0,
            template_id="retrieved",
            source_method="retrieved_corpus",
            score=9.5,
        )

        chosen = gen._choose_from_pool([retrieved, exact], best_valid=exact, best_any=retrieved, target=Lexeme("según", 377, "prep"))
        self.assertEqual(chosen.template_id, "prep_exact_segun_el")


class EvaluatorTests(unittest.TestCase):
    def test_compare_summaries_reports_delta(self):
        base = {"rows_in_scope": 10, "nonempty_rows": 7, "good_rows": 7, "bad_shipped_rows": 1, "no_candidate_rows": 3, "nonempty_rate": 0.7, "good_rate": 0.7, "bad_shipped_rate": 0.142857}
        new = {"rows_in_scope": 10, "nonempty_rows": 9, "good_rows": 9, "bad_shipped_rows": 0, "no_candidate_rows": 1, "nonempty_rate": 0.9, "good_rate": 0.9, "bad_shipped_rate": 0.0}
        comparison = compare_summaries(base, new)
        self.assertEqual(comparison["nonempty_rows"]["delta"], 2)
        self.assertEqual(comparison["bad_shipped_rows"]["delta"], -1)

    def test_row_pos_family_handles_contractions_and_residuals(self):
        self.assertEqual(row_pos_family({"lemma": "del", "pos": "none"}), "contraction")
        self.assertEqual(row_pos_family({"lemma": "al", "pos": ""}), "contraction")
        self.assertEqual(row_pos_family({"lemma": "eh", "pos": ""}), "residual")



    def test_row_pos_family_uses_shared_mapping_for_none_and_single_letters(self):
        self.assertEqual(row_pos_family({"lemma": "de", "pos": "none"}), "prep")
        self.assertEqual(row_pos_family({"lemma": "s", "pos": "letter"}), "letter")

if __name__ == "__main__":
    unittest.main()
