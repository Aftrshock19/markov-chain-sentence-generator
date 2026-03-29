#!/usr/bin/env python3
"""
Focused diagnostic for starter dataset gaps.
Checks exactly why each clean starter lemma is failing.

Usage:
  python3 diagnose_starter_gaps.py \
    --lexicon stg_words_spa.csv \
    --models-dir models \
    --lexicon-overrides starter_overrides.csv \
    --lexicon-overrides starter_exclusions.csv
"""
import sys

sys.path.insert(0, ".")

from generate import (
    BARE_INFINITIVE_OK,
    PLACE_PREP_VERBS,
    SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB,
    SAFE_SER_ADJECTIVES,
    SPECIAL_VERB_LEMMAS,
    STARTER_ADJ_ALLOWED_NOUN_CLASSES,
    STARTER_ELIGIBLE_POS,
    STARTER_INFINITIVE_COMPLEMENTS,
    STARTER_INFINITIVE_REJECT,
    STARTER_MAX_BAND,
    STARTER_SAFE_ADJECTIVES_BY_CLASS,
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
        if not lex.translation or not gen.starter_target_translation_ok(lex):
            continue
        if not gen.is_clean_starter_target(lex):
            continue
        eligible.append(lex)
    eligible = eligible[: args.limit]

    adj_fails = []
    verb_fails = []
    noun_fails = []

    for lex in eligible:
        try:
            best = gen.generate_starter_for_lemma(lex.lemma)
        except Exception:
            best = gen.manual_review_candidate(lex)

        if gen.candidate_is_starter_publishable(best):
            continue

        canonical = normalize_token(gen.canonical_lemma_for(lex))

        if lex.pos == "adj":
            allowed_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(normalize_token(lex.lemma))
            inflects_to_self_m = gen.inflect_adj(lex.lemma, "m", "sg") == lex.lemma
            inflects_to_self_f = gen.inflect_adj(lex.lemma, "f", "sg") == lex.lemma

            avail_nouns_m = []
            avail_nouns_f = []
            if allowed_classes:
                for noun in gen.pos_buckets.get("n", []):
                    if noun.rank > 350 or not gen.noun_is_template_friendly(noun):
                        continue
                    if normalize_token(noun.lemma) not in STARTER_SAFE_SUPPORT_NOUNS_FOR_ADJ:
                        continue
                    if noun.semantic_class not in allowed_classes:
                        continue
                    noun_gender = gen.safe_noun_gender(noun.lemma, noun.gender)
                    if noun_gender == "m":
                        avail_nouns_m.append(noun.lemma)
                    else:
                        avail_nouns_f.append(noun.lemma)

            reasons = gen.starter_rejection_reasons(best) if best.sentence else ["no_sentence"]
            adj_fails.append(
                {
                    "lemma": lex.lemma,
                    "rank": lex.rank,
                    "allowed_classes": allowed_classes,
                    "inflects_m": inflects_to_self_m,
                    "inflects_f": inflects_to_self_f,
                    "can_template": gen.can_template_target(lex),
                    "nouns_m": avail_nouns_m[:5],
                    "nouns_f": avail_nouns_f[:5],
                    "best_sentence": best.sentence[:60] if best.sentence else "NONE",
                    "best_method": best.source_method,
                    "reasons": reasons,
                }
            )

        elif lex.pos == "v":
            has_special = canonical in SPECIAL_VERB_LEMMAS
            has_obj_map = canonical in SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB
            has_inf_comp = canonical in STARTER_INFINITIVE_COMPLEMENTS
            in_bare = canonical in BARE_INFINITIVE_OK
            in_reject = canonical in STARTER_INFINITIVE_REJECT
            is_place = canonical in PLACE_PREP_VERBS
            can_template = gen.can_template_target(lex)
            template_reason = gen.template_support_reason(lex)

            obj_reachable = []
            if has_obj_map:
                for noun_lemma in SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB[canonical]:
                    glex = gen.generation_lexicon.get(noun_lemma)
                    if glex:
                        obj_reachable.append(f"{noun_lemma}(r={glex.rank})")
                    else:
                        obj_reachable.append(f"{noun_lemma}(NOT_FOUND)")

            inf_comp_status = []
            if has_inf_comp:
                for noun_lemma, _article in STARTER_INFINITIVE_COMPLEMENTS[canonical]:
                    glex = gen.generation_lexicon.get(noun_lemma)
                    if glex:
                        inf_comp_status.append(f"{noun_lemma}(r={glex.rank})")
                    else:
                        inf_comp_status.append(f"{noun_lemma}(NOT_FOUND)")

            reasons = gen.starter_rejection_reasons(best) if best.sentence else ["no_sentence"]
            verb_fails.append(
                {
                    "lemma": lex.lemma,
                    "rank": lex.rank,
                    "canonical": canonical,
                    "can_template": can_template,
                    "template_reason": template_reason,
                    "special": has_special,
                    "obj_map": has_obj_map,
                    "obj_reachable": obj_reachable,
                    "inf_comp": has_inf_comp,
                    "inf_comp_status": inf_comp_status,
                    "bare_ok": in_bare,
                    "inf_reject": in_reject,
                    "place_verb": is_place,
                    "prep": lex.required_prep,
                    "best_sentence": best.sentence[:60] if best.sentence else "NONE",
                    "best_method": best.source_method,
                    "reasons": reasons,
                }
            )

        elif lex.pos == "n":
            reasons = gen.starter_rejection_reasons(best) if best.sentence else ["no_sentence"]
            noun_fails.append(
                {
                    "lemma": lex.lemma,
                    "rank": lex.rank,
                    "class": lex.semantic_class,
                    "gender": lex.gender,
                    "inferred_gender": gen.infer_noun_gender(lex.lemma),
                    "template_friendly": gen.noun_is_template_friendly(lex),
                    "ser_adj_ok": gen.noun_supports_ser_adjective_template(lex),
                    "possess_ok": gen.noun_supports_possession_template(lex),
                    "best_sentence": best.sentence[:60] if best.sentence else "NONE",
                    "best_method": best.source_method,
                    "reasons": reasons,
                }
            )

    print("=" * 80)
    print(f"ADJECTIVE FAILURES ({len(adj_fails)})")
    print("=" * 80)
    no_class = [a for a in adj_fails if not a["allowed_classes"]]
    has_class_no_noun = [a for a in adj_fails if a["allowed_classes"] and not a["nouns_m"]]
    has_noun_still_fails = [a for a in adj_fails if a["allowed_classes"] and a["nouns_m"]]

    if no_class:
        print(f"\n  -- No STARTER_ADJ_ALLOWED_NOUN_CLASSES mapping ({len(no_class)}) --")
        print("  These adjectives need class mappings:")
        for adj in no_class:
            print(
                f"    {adj['lemma']:<15} inflects_m={adj['inflects_m']} "
                f"inflects_f={adj['inflects_f']} can_template={adj['can_template']}"
            )

    if has_class_no_noun:
        print(f"\n  -- Has class mapping but no masc support nouns ({len(has_class_no_noun)}) --")
        for adj in has_class_no_noun:
            print(f"    {adj['lemma']:<15} classes={adj['allowed_classes']} fem_nouns={adj['nouns_f'][:3]}")

    if has_noun_still_fails:
        print(f"\n  -- Has class + nouns but still fails ({len(has_noun_still_fails)}) --")
        for adj in has_noun_still_fails:
            print(f"    {adj['lemma']:<15} nouns_m={adj['nouns_m'][:3]} reasons={adj['reasons'][:3]}")
            print(f"      best: [{adj['best_method']}] {adj['best_sentence']}")

    print(f"\n{'=' * 80}")
    print(f"VERB FAILURES ({len(verb_fails)})")
    print("=" * 80)
    no_path = [v for v in verb_fails if not v["can_template"]]
    has_path_fails = [v for v in verb_fails if v["can_template"]]

    if no_path:
        print(f"\n  -- can_template_target=False ({len(no_path)}) --")
        for verb in no_path:
            print(f"    {verb['lemma']:<15} canonical={verb['canonical']} reason={verb['template_reason']}")

    if has_path_fails:
        print(f"\n  -- can_template=True but still fails ({len(has_path_fails)}) --")
        for verb in has_path_fails:
            print(
                f"    {verb['lemma']:<15} canonical={verb['canonical']} "
                f"special={verb['special']} obj_map={verb['obj_map']} "
                f"bare={verb['bare_ok']} reject={verb['inf_reject']}"
            )
            if verb["obj_reachable"]:
                print(f"      obj_nouns: {verb['obj_reachable']}")
            if verb["inf_comp_status"]:
                print(f"      inf_comp: {verb['inf_comp_status']}")
            print(f"      best: [{verb['best_method']}] {verb['best_sentence']}")
            print(f"      reasons: {verb['reasons'][:4]}")

    print(f"\n{'=' * 80}")
    print(f"NOUN FAILURES ({len(noun_fails)})")
    print("=" * 80)
    for noun in noun_fails:
        print(
            f"    {noun['lemma']:<15} class={noun['class']} gender={noun['gender']} "
            f"inferred={noun['inferred_gender']} template_friendly={noun['template_friendly']} "
            f"ser_adj_ok={noun['ser_adj_ok']} possess_ok={noun['possess_ok']}"
        )
        print(f"      best: [{noun['best_method']}] {noun['best_sentence']}")
        print(f"      reasons: {noun['reasons'][:4]}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"  Adjective failures: {len(adj_fails)}")
    print(f"    no class mapping: {len(no_class)}")
    print(f"    mapped but no masc noun: {len(has_class_no_noun)}")
    print(f"    mapped with noun but still fails: {len(has_noun_still_fails)}")
    print(f"  Verb failures: {len(verb_fails)}")
    print(f"    can_template_target=False: {len(no_path)}")
    print(f"    can_template_target=True but fails later: {len(has_path_fails)}")
    print(f"  Noun failures: {len(noun_fails)}")

    print("\nReference starter adjective classes loaded:")
    for noun_class, adjectives in STARTER_SAFE_ADJECTIVES_BY_CLASS.items():
        print(f"  {noun_class:<10} {len(adjectives):>2} adjectives")

    print(f"\nReference safe ser adjectives: {len(SAFE_SER_ADJECTIVES)}")


if __name__ == "__main__":
    main()
