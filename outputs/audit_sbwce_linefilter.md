# Corpus audit — SBWCE (line-filter → segment → per-sentence)

Source: `/projects/b35cg/corpora/sbwce_sample1000.txt`  —  read-only; input unmodified, no cleaned corpus written.
Normalisation (eval only): NFC + soft-hyphen/zero-width strip + whitespace collapse + single leading dash strip. Case preserved.

## Stage A — line-level filter (BEFORE segmentation)

Drop any input LINE firing `has_space_token` OR `legal_marker`.

| metric | lines | % of input |
|---|---|---|
| input lines | 1,000 | 100.00% |
| firing has_space_token | 18 | 1.80% |
| firing legal_marker | 58 | 5.80% |
| dropped (either) | 61 | 6.10% |
| **surviving** | **939** | **93.90%** |

_has_space_token and legal_marker overlap heavily (legal blocks carry both), so 'dropped' < their sum._

## Stage B — segmentation of survivors

- **Method:** spaCy es_core_news_sm v3.8.14, senter only (pipes=['tok2vec', 'senter']), nlp.pipe(batch_size=200)
- **Surviving lines in:** 939
- **Sentences out:** 948
- **Sentences per line (weld factor on non-legal lines):** 1.01

## Stage C — per-sentence predicates (over Stage-B sentences)

| predicate | sentences firing | % |
|---|---|---|
| subtitle_furniture | 10 | 1.05% |
| ocr_iforl | 0 | 0.00% |
| digit_in_alpha | 18 | 1.90% |
| high_digit_ratio | 35 | 3.69% |
| length_out_of_window | 395 | 41.67% |
| english_content_token | 15 | 1.58% |
| missing_space_weld | 0 | 0.00% |

_`has_space_token` and `legal_marker` are not in Stage C — they are the Stage-A line filter._
_`missing_space_weld` = clause 1 only (token >20 chars, no internal hyphen/digit). Clause 2 (spaCy-vocab miss rate) is INERT on es_core_news_sm — no vectors / no lexeme_prob table, so is_oov is True and `tok in nlp.vocab` is True for every token; it is excluded, not computed._

## Co-occurrence (Stage-C predicates)

- firing ≥1 predicate (would-drop): 446 (47.05%)
- firing ≥2 predicates: 25 (2.64%)
- firing 0 predicates (estimated survival): 502 (52.95%)

## Token-count histogram (over Stage-B sentences)

| tokens | sentences | % |
|---|---|---|
| 1 | 2 | 0.21% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |
| 4-7 | 147 | 15.51% |
| 8-15 | 206 | 21.73% |
| 16-30 | 200 | 21.10% |
| 31+ | 393 | 41.46% |

## english_content_token — which tokens fired

| token | sentence hits |
|---|---|
| the | 8 |
| of | 8 |
| and | 6 |
| for | 2 |
| that | 1 |
| you | 1 |
| with | 1 |

## missing_space_weld — 0 firing (up to 15 samples)

_none_

## Zero-predicate would-keep set (up to 15 samples)

- `Esta es una cuestión que se aborda en la parte final del séptimo informe`
- `Si la fuerza de esa radiación cambia estacionalmente se puede lograr muy poco o demasiado de una dosis`
- `Revista de la facultad Ciencias de la Salud`
- `Ha sido internacional con la Selección de fútbol de Francia Sub-21 Fue convocado en la lista preliminar para la Eurocopa 2012 aunque finalmente no fue seleccionado`
- `Supongo que usted encuentre esta todo muy mal gusto`
- `Las dificultades no pueden padecerlas siempre las mismas personas es decir los desempleados los trabajadores temporales y los pensionistas con reducciones de las pensiones`
- `Pero basándome en nuestra conversación de anoche usted me debe creer un poco loco`
- `Yo fui creciendo y seguí creciendo buscando abrigo para no morir sólo las aves con sus cantares me han dado aliento para vivir`
- `Cuota fija de servicio 12,02 EUR abonado trimestre`
- `Es una cultura donde el topless es común`
- `Tornó a empujar el timbre y fue el silencio la sola contestación que obtuvo`
- `Mike tomate un segundo si`
- `Una fosa de plaga y aproximadamente 1000 tumbas de los siglos V y IV a.`
- `No se debe confundir con la episiotomía que es una incisión en el periné para facilitar el parto La cesárea se hace por encima de la pelvis`
- `No sabes lo que día de la semana es y aunque lo supieras no importa`

## Combined note

Stage A dropped 6.10% of input lines. Of the 939 survivors, senter produced 948 sentences (weld 1.01). Estimated final survival = Stage-B sentences firing zero Stage-C predicates: **52.95%** (502 of 948).
