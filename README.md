# markov-chain-sentence-generator
A Markov chain based sentence generator that creates random sentences containing a target input word, with adjustable difficulty and sentence length for language learning and experimentation.

## Making a good sentence file

The old `outputs/sentences.csv` contains many raw corpus hits. Use the review file to avoid shipping rows already flagged as awkward or bad:

```bash
source sentgen_env/bin/activate
python make_good_sentences.py \
  --replacement missing_rank_word_filled1.csv \
  --out outputs/good_sentences.csv \
  --queue-out outputs/needs_good_sentence.csv
```

This writes:

- `outputs/good_sentences.csv`: keeps reviewed-good rows, applies replacement rows, blanks known-bad rows.
- `outputs/needs_good_sentence.csv`: ranks that still need a better sentence.
- `outputs/needs_good_sentence.warnings.csv`: replacements skipped because they did not contain the target word.

To fill the queue with Qwen/Hugging Face Router:

```bash
export HF_TOKEN=your_huggingface_token
python teacher_dataset_builder.py \
  --input-batch outputs/needs_good_sentence.csv \
  --output outputs/teacher_sentences_raw.csv \
  --lexicon stg_words_spa.csv \
  --failure-filter retrieved retrieved_raw missing_or_not_reviewed \
  --min-rank 1 \
  --max-rank 1000 \
  --limit 200 \
  --n 8

python teacher_filter.py \
  --input outputs/teacher_sentences_raw.csv \
  --accepted-out outputs/teacher_sentences_accepted.csv \
  --rejected-out outputs/teacher_sentences_rejected.csv \
  --lexicon stg_words_spa.csv \
  --models-dir models_rebuild2

python make_good_sentences.py \
  --replacement missing_rank_word_filled1.csv \
  --replacement outputs/teacher_sentences_accepted.csv \
  --out outputs/good_sentences.csv \
  --queue-out outputs/needs_good_sentence.csv
```

Use `models_rebuild2` for validation/generation unless you rebuild `models/`; the current `models/` directory is missing required artifacts such as `lemma_forms.pkl`.

## Markov-only generation

To generate fresh sentences from the learned Markov chain without returning retrieved corpus rows:

```bash
source sentgen_env/bin/activate
python markov_only_generator.py \
  --models-dir models_rebuild2 \
  --lexicon stg_words_spa.csv \
  --min-rank 1 \
  --max-rank 1000 \
  --limit 100 \
  --attempts 700 \
  --out outputs/markov_only_sentences.csv \
  --review-out outputs/markov_only_review.csv
```

This uses `models_rebuild2/bigrams.pkl` as the Markov transition model. It does not call `retrieve_candidates`, does not use the old sentence CSV, and does not call an LLM.

## Machine-learned high-quality generation (no LLM, no retrieval)

The original `markov_only_generator.py` samples from raw bigram counts and produces a lot of ungrammatical output ("Tú es posible.", "Sus amigos es un poco.", etc.). `markov_ml_generator.py` replaces the pure sampling with a trained pipeline:

1. **Modified Kneser-Ney trigram LM** (`data_clean/kn_lm.pkl`, ~27 MB) trained on a clean corpus of 12,603 human-reviewed-good rows + 400,000 Tatoeba Spanish sentences (`data_clean/good_corpus.txt`). Kneser-Ney smoothing gives well-calibrated probabilities for unseen trigrams by backing off to continuation counts.
2. **Logistic-regression sentence-quality reranker** (`data_clean/reranker.pkl`) trained on 39,154 human-labeled sentences (12,603 labeled good; 27,158 labeled bad). Features include the KN-LM per-sentence log-prob, per-token fluency, length buckets, subject-verb agreement, article-noun agreement, bad-bigram indicators, OOV ratios, etc. Test-set AUC ≈ 0.81.
3. **Target-anchored beam search** with configurable beam size and per-step top-k extensions, where each beam tracks whether the target lemma is present. Hard Spanish agreement filters (article-noun gender/number, subject-pronoun + verb-person, quantifier + plural-noun, disallowed prepositional objects) prune structurally impossible prefixes.
4. **Grammatical seed prefixes** are chosen per lemma / POS so the beam starts from a valid partial sentence (e.g. `[la, casa, está]` for a feminine noun, `[quiero, <inf>]` for an infinitive, `[ella, y, yo, somos, amigos]` for the conjunction `y`).
5. **Final selection** combines the reranker logit + LM score + hard end-of-sentence validity (`final_ok`) to pick the best completion across seeds.

### Training the pipeline

```bash
source sentgen_env/bin/activate
pip install scikit-learn  # one-time

# 1. Assemble the clean corpus
python scripts/build_clean_corpus.py

# 2. Train the KN trigram LM
python scripts/train_kn_lm.py

# 3. Train the logistic-regression reranker
python scripts/train_reranker.py
```

### Running the ML generator

```bash
python markov_ml_generator.py \
  --lm data_clean/kn_lm.pkl \
  --reranker data_clean/reranker.pkl \
  --lexicon stg_words_spa.csv \
  --min-rank 1 --max-rank 1000 --limit 100 \
  --beam-size 18 --max-extensions 30 \
  --out outputs/markov_ml_sentences.csv \
  --review-out outputs/markov_ml_review.csv
```

### Evaluation

Score any generator CSV against the trained reranker + LM:

```bash
python scripts/eval_quality.py \
  outputs/markov_only_sentences.csv \
  outputs/markov_ml_sentences.csv \
  --diff-out outputs/markov_eval_diff.tsv
```

Representative comparison on ranks 1–1000 (limit 100):

| Metric                     | `markov_only` (old) | `markov_ml` (new) |
| -------------------------- | ------------------- | ----------------- |
| empty rows                 | 10                  | 0                 |
| mean reranker logit (z)    | −0.71               | **+2.38**         |
| mean P(good)               | 0.35                | **0.89**          |
| % scored good (z > 0)      | 12 %                | **98 %**          |
| mean LM log-prob / token   | −7.61               | **−4.34**         |

All components are learned from data — no LLM calls, no retrieval from the old sentence CSV.

### Sample output (ranks 1-30)

```
de     Estoy de acuerdo contigo en ese momento.
que    Creo que es verdad que estás haciendo.
la     La casa está a punto de salir.
no     Yo no quiero ir a la gente.
a      Voy a casa de mi padre.
el     El hombre está a punto de salir.
y      Tú y yo estamos aquí para ver.
en     Estoy en casa todo el mundo.
es     Ella es la mejor forma de hablar.
un     Tengo un hombre con quien hablar.
lo     Ya lo tengo que ir a ver.
los    Los libros están en el mundo.
una    Tengo una casa de mi padre.
se     No se sabe nada de esto.
con    Estoy con un poco más de tres.
qué    No sé qué hacer con los ojos.
me     Me gusta mucho el uno al otro.
las    Las casas son un poco de agua.
su     Su hombre está a punto de salir.
pero   Quiero ir pero no puedo ver nada.
está   Ella está en contra de la ciudad.
```

vs. the old `markov_only_generator.py` on the same ranks:

```
de     Soy de aquí.
que    Creo que está aquí.
la     La casa está bien.
no     No hay nada.
a      Voy a casa.
el     El mundo es posible.
y      Tú y yo estamos aquí.
es     Tú es posible.              ← agreement error
un     Un hombre tiene razón.
sus    Sus amigos es un poco.       ← agreement error
más    Más ella está bien.          ← word order error
cuando Ella va cuando el mundo.    ← dangling fragment
hay    Ella hay algo.               ← hay is impersonal
nos    Nos está bien.               ← wrong clitic frame
```
