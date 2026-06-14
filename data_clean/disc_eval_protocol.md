# Coherence-discriminator evaluation protocol (PRE-REGISTERED)

Locked **before** the discriminator is built/trained, to prevent fitting the
design to the test (the 6 hand pairs were reused 3× and are now leaked — not
used here).

## Frozen eval set
- Source: `outputs/markov_ml_regen2.csv` (current generator, gender fix on),
  labeled per-row by an independent judge into {good, nonsense, other}.
- Build `data_clean/disc_eval_frozen.json`: take the labeled `good` and
  `nonsense` rows; sample a balanced, FROZEN set (target ~100 each, capped by
  availability). Store verbatim sentences + label. Once written, do not edit.
- The eval set is HELD OUT: none of these exact sentences may appear in
  discriminator training data (corruptions are built from the corpus, generator
  negatives are filtered to exclude any sentence in the frozen set).

## Metric
- For every (good, nonsense) cross pair in the frozen set, score both with the
  discriminator; count the fraction where score(good) > score(nonsense).
- Report this pairwise accuracy (AUC-equivalent).

## PASS CONDITION (pre-registered)
- **PASS iff pairwise accuracy ≥ 0.70 on the frozen held-out set.**
- Decided now, before any results are seen.

## Decision gate
- PASS → wire the discriminator in as a coherence energy and build the CGMH
  (Metropolis-Hastings) decoder; re-run the full regen + census A/B.
- FAIL → the corpus does not contain a learnable token-level coherence signal
  for this failure mode; recommend offline LLM-judge distillation (label
  generator output good/bad with an LLM offline, retrain the reranker). This
  still preserves the zero-LLM-at-generation product claim.

## Training design (amendments 2 & 3)
- Two negative classes:
  1. Minimal-edit corruptions of real corpus sentences (anchor features so the
     discriminator can't cheat on length / template / vocab rank).
  2. Generator-output negatives (sentence-level pipeline-junk signature),
     filtered to exclude frozen-eval sentences and ideally to low-quality rows.
- Corruptor: substitutes drawn from the trigram conditioned on local context
  (locally fluent negatives); CONTENT WORDS ONLY; exclude same-fine-class swaps
  (e.g. madre/padre) that yield valid sentences and would mislabel positives as
  negatives. Noisy negatives → false rejection of good flashcards = the
  expensive product error.
