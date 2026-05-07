#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ.]+")
RANK_CANDIDATES = ("rank", "rango", "id", "idx", "index")
WORD_CANDIDATES = ("word", "lemma", "palabra", "spanish", "target", "lexeme", "lemma_lower")


def normalize_header(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def find_column(fieldnames: Optional[List[str]], candidates: Iterable[str]) -> Optional[str]:
    if not fieldnames:
        return None
    normalized = {normalize_header(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for original in fieldnames:
        n = normalize_header(original)
        for candidate in candidates:
            if candidate in n:
                return original
    return None


def normalize_token(text: str) -> str:
    return (text or "").strip().lower().strip('.,;:!?¡¿"“”\'()[]{}')


def sentence_contains_word(sentence: str, word: str) -> bool:
    target = normalize_token(word)
    if not target:
        return False
    if "." in target:
        return target in (sentence or "").lower()
    return target in {normalize_token(token) for token in WORD_RE.findall(sentence or "")}


def load_source_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} must have a header")
        if "rank" not in reader.fieldnames or "sentence" not in reader.fieldnames:
            raise ValueError(f"{path} must contain rank and sentence columns")
        return reader.fieldnames, list(reader)


def load_rank_to_word(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rank_col = find_column(reader.fieldnames, RANK_CANDIDATES)
        word_col = find_column(reader.fieldnames, WORD_CANDIDATES)
        if rank_col is None or word_col is None:
            raise ValueError(f"{path} must contain rank and word/lemma columns")
        out: Dict[str, str] = {}
        for row in reader:
            rank = (row.get(rank_col) or "").strip()
            word = (row.get(word_col) or "").strip()
            if rank and word and rank not in out:
                out[rank] = word
        return out


def review_is_good(row: Dict[str, str]) -> bool:
    flags = (
        (row.get("grammatical_ok") or "").strip(),
        (row.get("natural_ok") or "").strip(),
        (row.get("learner_clear_ok") or "").strip(),
    )
    if all(flag in {"0", "1"} for flag in flags):
        return flags == ("1", "1", "1")

    publishable = (row.get("publishable") or "").strip().lower()
    quality_tier = (row.get("quality_tier") or "").strip().lower()
    bad_candidate = (row.get("bad_candidate") or "").strip().lower()
    return publishable == "true" and quality_tier in {"strong", "acceptable"} and bad_candidate != "true"


def load_good_review(path: Optional[Path]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    if path is None:
        return {}, {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "rank" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a rank column")
        good: Dict[str, bool] = {}
        reason: Dict[str, str] = {}
        for row in reader:
            rank = (row.get("rank") or "").strip()
            if not rank:
                continue
            is_good = review_is_good(row)
            good[rank] = is_good
            if not is_good:
                notes = (
                    row.get("failure_reason")
                    or row.get("starter_notes")
                    or row.get("notes")
                    or row.get("template_id")
                    or "review_flags_not_all_good"
                )
                reason[rank] = notes
        return good, reason


def load_replacements(paths: List[Path], rank_to_word: Dict[str, str]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    replacements: Dict[str, str] = {}
    warnings: List[Dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "rank" not in reader.fieldnames or "sentence" not in reader.fieldnames:
                raise ValueError(f"{path} must contain rank and sentence columns")
            for row in reader:
                rank = (row.get("rank") or "").strip()
                sentence = (row.get("sentence") or "").strip()
                if not rank or not sentence:
                    continue
                word = (row.get("word") or row.get("lemma") or rank_to_word.get(rank, "")).strip()
                if word and not sentence_contains_word(sentence, word):
                    warnings.append(
                        {
                            "rank": rank,
                            "word": word,
                            "sentence": sentence,
                            "warning": "replacement_does_not_contain_target_word",
                        }
                    )
                    continue
                replacements[rank] = sentence
    return replacements, warnings


def make_good_sentences(
    source: Path,
    out: Path,
    queue_out: Path,
    lexicon: Path,
    review: Optional[Path],
    replacement_paths: List[Path],
    keep_unreviewed: bool,
) -> Dict[str, int]:
    fieldnames, rows = load_source_rows(source)
    rank_to_word = load_rank_to_word(lexicon)
    reviewed_good, review_reasons = load_good_review(review)
    replacements, warnings = load_replacements(replacement_paths, rank_to_word)

    out.parent.mkdir(parents=True, exist_ok=True)
    queue_out.parent.mkdir(parents=True, exist_ok=True)
    warning_out = queue_out.with_suffix(".warnings.csv")

    kept_good = 0
    replaced = 0
    blanked = 0
    queue_rows: List[Dict[str, str]] = []
    output_rows: List[Dict[str, str]] = []

    for row in rows:
        rank = (row.get("rank") or "").strip()
        old_sentence = (row.get("sentence") or "").strip()
        new_row = dict(row)
        word = rank_to_word.get(rank, "")
        reviewed = rank in reviewed_good
        is_good = reviewed_good.get(rank, keep_unreviewed)

        if rank in replacements:
            new_row["sentence"] = replacements[rank]
            replaced += 1
        elif old_sentence and is_good:
            kept_good += 1
        else:
            new_row["sentence"] = ""
            blanked += 1
            queue_rows.append(
                {
                    "rank": rank,
                    "lemma": word,
                    "word": word,
                    "old_sentence": old_sentence,
                    "failure_reason": review_reasons.get(rank, "missing_or_not_reviewed"),
                }
            )

        output_rows.append(new_row)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    with queue_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "lemma", "word", "old_sentence", "failure_reason"])
        writer.writeheader()
        writer.writerows(queue_rows)

    with warning_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "word", "sentence", "warning"])
        writer.writeheader()
        writer.writerows(warnings)

    return {
        "rows": len(rows),
        "kept_good": kept_good,
        "replaced": replaced,
        "blanked": blanked,
        "queued": len(queue_rows),
        "replacement_warnings": len(warnings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a rank,sentence CSV that does not ship rows already flagged as bad."
    )
    parser.add_argument("--source", default="outputs/sentences.csv", help="Existing rank,sentence CSV")
    parser.add_argument("--review", default="outputs/sentances_review.csv", help="Review CSV with quality flags")
    parser.add_argument("--lexicon", default="stg_words_spa.csv", help="Lexicon CSV with rank and lemma columns")
    parser.add_argument(
        "--replacement",
        action="append",
        default=[],
        help="CSV with rank,sentence replacements. Can be repeated.",
    )
    parser.add_argument("--out", default="outputs/good_sentences.csv", help="Filtered output CSV")
    parser.add_argument("--queue-out", default="outputs/needs_good_sentence.csv", help="Rows still needing replacements")
    parser.add_argument(
        "--keep-unreviewed",
        action="store_true",
        help="Keep source sentences when no review row exists. Default blanks them for safety.",
    )
    args = parser.parse_args()

    stats = make_good_sentences(
        source=Path(args.source),
        out=Path(args.out),
        queue_out=Path(args.queue_out),
        lexicon=Path(args.lexicon),
        review=Path(args.review) if args.review else None,
        replacement_paths=[Path(path) for path in args.replacement],
        keep_unreviewed=args.keep_unreviewed,
    )
    print(
        "rows={rows} kept_good={kept_good} replaced={replaced} "
        "blanked={blanked} queued={queued} replacement_warnings={replacement_warnings}".format(**stats)
    )


if __name__ == "__main__":
    main()
