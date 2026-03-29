#!/usr/bin/env python3
"""
Diagnose starter dataset gaps.

Run from the project root:
  python3 diagnose_gaps.py \
    --lexicon stg_words_spa.csv \
    --models-dir models \
    --lexicon-overrides starter_overrides.csv

Produces a categorized report of why eligible starter lemmas fail.
"""
import sys

sys.path.insert(0, ".")

from generate import (
    APOCOPATED_ADJECTIVE_FEATURES,
    BARE_INFINITIVE_OK,
    PLACE_PREP_VERBS,
    SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB,
    SPECIAL_VERB_LEMMAS,
    STARTER_ADJ_ALLOWED_NOUN_CLASSES,
    STARTER_ELIGIBLE_POS,
    STARTER_EXCLUDED_TARGET_LEMMAS,
    STARTER_INFINITIVE_COMPLEMENTS,
    STARTER_INFINITIVE_REJECT,
    STARTER_LOW_VALUE_TARGET_LEMMAS,
    STARTER_MAX_BAND,
    STARTER_SAFE_NOUN_CLASSES,
    STARTER_SAFE_SUPPORT_NOUNS_FOR_ADJ,
    SentenceGenerator,
    get_profile,
    normalize_token,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--lexicon-overrides", action="append", default=[])
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    gen = SentenceGenerator(args.lexicon, args.models_dir, seed=42)
    for path in args.lexicon_overrides:
        gen.load_and_apply_overrides(path)

    all_rows = sorted(gen.lexicon.values(), key=lambda x: x.rank)
    eligible = []
    for lex in all_rows:
        if lex.pos not in STARTER_ELIGIBLE_POS:
            continue
        if get_profile(lex.rank).band not in STARTER_MAX_BAND:
            continue
        if not lex.translation:
            continue
        if not gen.starter_target_translation_ok(lex):
            continue
        if not gen.is_clean_starter_target(lex):
            continue
        if normalize_token(lex.lemma) in STARTER_EXCLUDED_TARGET_LEMMAS:
            continue
        if normalize_token(lex.lemma) in STARTER_LOW_VALUE_TARGET_LEMMAS:
            continue
        eligible.append(lex)
    eligible = eligible[: args.limit]

    categories = {
        "verb_no_template_path": [],
        "verb_special_no_object": [],
        "verb_not_in_bare_ok": [],
        "adj_no_noun_class_mapping": [],
        "adj_no_support_noun": [],
        "noun_no_template": [],
        "noun_abstract_class": [],
        "other_manual_review": [],
    }

    for lex in eligible:
        try:
            best = gen.generate_starter_for_lemma(lex.lemma)
        except Exception:
            best = gen.manual_review_candidate(lex)

        if gen.candidate_is_starter_publishable(best):
            continue

        canonical = normalize_token(gen.canonical_lemma_for(lex))
        surface = normalize_token(lex.lemma)

        if lex.pos == "v":
            has_special = canonical in SPECIAL_VERB_LEMMAS
            has_object_map = canonical in SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB
            has_inf_complement = canonical in STARTER_INFINITIVE_COMPLEMENTS
            in_bare_ok = canonical in BARE_INFINITIVE_OK
            in_inf_reject = canonical in STARTER_INFINITIVE_REJECT
            is_place_verb = canonical in PLACE_PREP_VERBS

            obj_nouns_reachable = False
            if has_object_map:
                for noun_lemma in SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB[canonical]:
                    if noun_lemma in gen.generation_lexicon:
                        noun_lex = gen.generation_lexicon[noun_lemma]
                        if noun_lex.rank <= 200:
                            obj_nouns_reachable = True
                            break

            info = (
                f"  {lex.lemma:<15} rank={lex.rank:<5} canonical={canonical}"
                f"  special={has_special} obj_map={has_object_map} obj_reachable={obj_nouns_reachable}"
                f"  inf_comp={has_inf_complement} bare_ok={in_bare_ok} inf_reject={in_inf_reject}"
                f"  place_verb={is_place_verb} prep={lex.required_prep}"
                f"  best_method={best.source_method} best_sentence={best.sentence[:50] if best.sentence else 'NONE'}"
            )

            if in_inf_reject:
                categories["verb_no_template_path"].append(
                    info + "  REASON: in STARTER_INFINITIVE_REJECT"
                )
            elif not has_special and not has_object_map and not in_bare_ok and not has_inf_complement:
                categories["verb_no_template_path"].append(
                    info + "  REASON: no template path at all"
                )
            elif has_object_map and not obj_nouns_reachable:
                categories["verb_special_no_object"].append(
                    info + "  REASON: object nouns not reachable at rank ceiling"
                )
            elif not in_bare_ok and not has_inf_complement and not has_object_map:
                categories["verb_not_in_bare_ok"].append(
                    info + "  REASON: not in BARE_INFINITIVE_OK"
                )
            else:
                categories["other_manual_review"].append(info)

        elif lex.pos == "adj":
            allowed_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(surface)
            info = (
                f"  {lex.lemma:<15} rank={lex.rank:<5}"
                f"  allowed_classes={allowed_classes}"
                f"  best_method={best.source_method} best_sentence={best.sentence[:50] if best.sentence else 'NONE'}"
            )

            if not allowed_classes:
                inflected = gen.inflect_adj(lex.lemma, "m", "sg")
                if inflected != lex.lemma:
                    info += f"  NOTE: inflect_adj('{lex.lemma}', 'm') = '{inflected}'"
                categories["adj_no_noun_class_mapping"].append(info)
            else:
                support_nouns_available = [
                    n
                    for n in gen.pos_buckets.get("n", [])
                    if n.rank <= 200
                    and gen.noun_is_template_friendly(n)
                    and normalize_token(n.lemma) in STARTER_SAFE_SUPPORT_NOUNS_FOR_ADJ
                    and n.semantic_class in allowed_classes
                ]
                info += f"  support_nouns={len(support_nouns_available)}"
                if not support_nouns_available:
                    categories["adj_no_support_noun"].append(info)
                else:
                    reasons = gen.starter_rejection_reasons(best) if best.sentence else ["no_sentence"]
                    info += f"  reasons={reasons}"
                    categories["other_manual_review"].append(info)

        elif lex.pos == "n":
            info = (
                f"  {lex.lemma:<15} rank={lex.rank:<5} class={lex.semantic_class} gender={lex.gender}"
                f"  template_friendly={gen.noun_is_template_friendly(lex)}"
                f"  ser_adj_ok={gen.noun_supports_ser_adjective_template(lex)}"
                f"  possess_ok={gen.noun_supports_possession_template(lex)}"
                f"  best_method={best.source_method} best_sentence={best.sentence[:50] if best.sentence else 'NONE'}"
            )
            if lex.semantic_class not in STARTER_SAFE_NOUN_CLASSES:
                categories["noun_abstract_class"].append(info)
            elif not gen.noun_is_template_friendly(lex):
                categories["noun_no_template"].append(info)
            else:
                reasons = gen.starter_rejection_reasons(best) if best.sentence else ["no_sentence"]
                info += f"  reasons={reasons}"
                categories["other_manual_review"].append(info)

    print("\n" + "=" * 80)
    print("STARTER GAP DIAGNOSIS")
    print("=" * 80)

    for cat_name, items in categories.items():
        if not items:
            continue
        print(f"\n--- {cat_name} ({len(items)} lemmas) ---")
        for item in items[:30]:
            print(item)
        if len(items) > 30:
            print(f"  ... and {len(items) - 30} more")

    print("\n--- SUMMARY ---")
    total_failing = sum(len(v) for v in categories.values())
    print(f"Total failing eligible lemmas: {total_failing}")
    for cat_name, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        if items:
            print(f"  {cat_name:<35} {len(items):>4}")

    print("\n--- SUGGESTED FIXES ---")

    bare_candidates = [
        item for item in categories["verb_no_template_path"] if "no template path" in item
    ]
    if bare_candidates:
        print(f"\nAdd to BARE_INFINITIVE_OK ({len(bare_candidates)} verbs):")
        for item in bare_candidates:
            lemma = item.split()[0].strip()
            print(f'    "{lemma}",')

    if categories["adj_no_noun_class_mapping"]:
        count = len(categories["adj_no_noun_class_mapping"])
        print(f"\nAdd to STARTER_SAFE_ADJECTIVES_BY_CLASS ({count} adjectives):")
        for item in categories["adj_no_noun_class_mapping"]:
            lemma = item.split()[0].strip()
            print(f'    "{lemma}": suggest adding to relevant class sets')

    if categories["verb_special_no_object"]:
        count = len(categories["verb_special_no_object"])
        print(f"\nCheck object noun ranks ({count} verbs):")
        for item in categories["verb_special_no_object"]:
            canonical = item.split("canonical=")[1].split()[0]
            if canonical in SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB:
                nouns = SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB[canonical]
                for noun in nouns:
                    glex = gen.generation_lexicon.get(noun)
                    rank = glex.rank if glex else "NOT_FOUND"
                    print(f"    {canonical} -> {noun}: rank={rank}")


if __name__ == "__main__":
    main()
