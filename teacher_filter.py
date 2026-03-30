#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List

from teacher_validator_bridge import TeacherValidatorBridge

ACCEPTED_FIELDS = [
    "lemma",
    "rank",
    "pos",
    "translation",
    "sentence",
    "token_count",
    "quality_flags",
    "support_words",
    "support_rank_max",
    "support_rank_avg",
    "pattern_ready",
]

REJECTED_FIELDS = [
    "lemma",
    "rank",
    "pos",
    "sentence",
    "rejection_reasons",
]


def _has_fatal_teacher_reason(reasons: List[str]) -> bool:
    fatal = {
        "empty_sentence",
        "missing_exact_target",
        "multiple_sentences",
        "meta_text",
        "build_candidate_failed",
    }
    return any(reason in fatal for reason in reasons)


def filter_teacher_rows(
    input_csv: str,
    accepted_out: str,
    rejected_out: str,
    lexicon_path: str,
    models_dir: str,
    min_tokens: int = 2,
    max_tokens: int = 10,
) -> Dict[str, int]:
    bridge = TeacherValidatorBridge(lexicon_path=lexicon_path, models_dir=models_dir)
    Path(accepted_out).parent.mkdir(parents=True, exist_ok=True)
    Path(rejected_out).parent.mkdir(parents=True, exist_ok=True)

    accepted_count = 0
    rejected_count = 0

    with open(input_csv, encoding="utf-8", newline="") as src, \
        open(accepted_out, "w", encoding="utf-8", newline="") as acc, \
        open(rejected_out, "w", encoding="utf-8", newline="") as rej:

        reader = csv.DictReader(src)
        acc_writer = csv.DictWriter(acc, fieldnames=ACCEPTED_FIELDS)
        rej_writer = csv.DictWriter(rej, fieldnames=REJECTED_FIELDS)
        acc_writer.writeheader()
        rej_writer.writeheader()

        for row in reader:
            lemma = (row.get("lemma") or "").strip().lower()
            if not lemma:
                continue

            result = bridge.validate_teacher_sentence(
                lemma=lemma,
                rank=int(float(row.get("rank") or 999999)),
                pos=(row.get("pos") or "").strip().lower(),
                sentence=(row.get("sentence") or "").strip(),
                translation=(row.get("translation") or "").strip(),
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )

            reasons = list(result.rejection_reasons or [])

            accept_for_induction = (
                result.candidate is not None
                and not _has_fatal_teacher_reason(reasons)
            )

            if accept_for_induction:
                quality_flags = "|".join(
                    [
                        f"grammatical_ok={result.quality_flags.get('grammatical_ok', 0)}",
                        f"natural_ok={result.quality_flags.get('natural_ok', 0)}",
                        f"learner_clear_ok={result.quality_flags.get('learner_clear_ok', 0)}",
                        f"notes={result.quality_flags.get('notes', '')}",
                        f"rejection_reasons={'; '.join(reasons)}",
                    ]
                )
                acc_writer.writerow(
                    {
                        "lemma": lemma,
                        "rank": row.get("rank", ""),
                        "pos": row.get("pos", ""),
                        "translation": row.get("translation", ""),
                        "sentence": result.candidate.sentence,
                        "token_count": result.token_count,
                        "quality_flags": quality_flags,
                        "support_words": " ".join(result.support_words),
                        "support_rank_max": result.support_rank_max,
                        "support_rank_avg": f"{result.support_rank_avg:.2f}",
                        "pattern_ready": str(result.pattern_ready).lower(),
                    }
                )
                accepted_count += 1
            else:
                rej_writer.writerow(
                    {
                        "lemma": lemma,
                        "rank": row.get("rank", ""),
                        "pos": row.get("pos", ""),
                        "sentence": row.get("sentence", ""),
                        "rejection_reasons": "; ".join(reasons),
                    }
                )
                rejected_count += 1

    return {"accepted": accepted_count, "rejected": rejected_count}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter raw teacher outputs through hard rules and generator validation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--accepted-out", required=True)
    parser.add_argument("--rejected-out", required=True)
    parser.add_argument("--lexicon", default="stg_words_spa.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    stats = filter_teacher_rows(
        input_csv=args.input,
        accepted_out=args.accepted_out,
        rejected_out=args.rejected_out,
        lexicon_path=args.lexicon,
        models_dir=args.models_dir,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    print(f"Accepted: {stats['accepted']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Saved: {args.accepted_out}")
    print(f"Saved: {args.rejected_out}")


if __name__ == "__main__":
    main()