# Offline teacher to learned frame pipeline

This implementation keeps the existing generator as the primary system.
The teacher model is only used offline to collect candidate sentences, filter them, induce reusable frames, and feed those frames back into the normal generator as a native route.

## New files

- `teacher_llm.py`
- `teacher_dataset_builder.py`
- `teacher_validator_bridge.py`
- `teacher_filter.py`
- `pattern_induction.py`
- `learned_frame_router.py`
- `teacher_repair.py`
- `hybrid_generator.py` (patched)
- `hybrid_generator.patch`

## Integration notes

Place the new files next to your existing `complete_generate.py`, `hybrid_generator.py`, and `reranker.py`.
Then replace your current `hybrid_generator.py` with the patched one, or apply `hybrid_generator.patch`.

## CLI flow

Build raw teacher data:

```bash
python teacher_dataset_builder.py \
  --input-batch outputs/hybrid_batch.csv \
  --output teacher_sentences_raw.csv \
  --failure-filter no_candidate_found bad_candidate \
  --min-rank 1 \
  --max-rank 1000 \
  --limit 200 \
  --n 20 \
  --lexicon stg_words_spa.csv
```

Filter teacher data:

```bash
python teacher_filter.py \
  --input teacher_sentences_raw.csv \
  --accepted-out teacher_sentences_accepted.csv \
  --rejected-out teacher_sentences_rejected.csv \
  --lexicon stg_words_spa.csv \
  --models-dir models
```

Induce learned frames:

```bash
python pattern_induction.py \
  --input teacher_sentences_accepted.csv \
  --frames-out learned_frames.json \
  --lemma-pref-out lemma_frame_preferences.csv \
  --pos-stats-out pos_family_frame_stats.csv \
  --lexicon stg_words_spa.csv
```

Run the generator with learned frames enabled:

```bash
python hybrid_generator.py \
  --lexicon stg_words_spa.csv \
  --models-dir models \
  --out outputs/with_learned_frames.csv \
  --learned-frames learned_frames.json \
  --lemma-frame-preferences lemma_frame_preferences.csv \
  --min-rank 1 \
  --max-rank 1000 \
  --limit 1000
```

## Smoke checks

Teacher backend:

```bash
python teacher_llm.py --lemma casa --rank 500 --pos n --translation house --n 5
```

Compile check:

```bash
python -m py_compile teacher_llm.py teacher_dataset_builder.py teacher_validator_bridge.py teacher_filter.py pattern_induction.py learned_frame_router.py teacher_repair.py hybrid_generator.py
```
