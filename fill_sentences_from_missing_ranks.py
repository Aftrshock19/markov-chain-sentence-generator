#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load_rank_to_sentence(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "rank" not in reader.fieldnames or "sentence" not in reader.fieldnames:
            raise ValueError(f"{path} must contain 'rank' and 'sentence' columns")

        rank_to_sentence: dict[str, str] = {}
        for row in reader:
            rank = (row.get("rank") or "").strip()
            sentence = row.get("sentence") or ""
            if rank and sentence:
                rank_to_sentence[rank] = sentence
        return rank_to_sentence


def fill_sentences(source_csv: Path, target_csv: Path, output_csv: Path) -> tuple[int, int, int]:
    rank_to_sentence = load_rank_to_sentence(source_csv)

    with target_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"{target_csv} is empty")
    if rows[0][:2] != ["rank", "sentence"]:
        raise ValueError(f"{target_csv} must start with header: rank,sentence")

    filled = 0
    already_present = 0
    missing_in_source = 0

    updated_rows = [rows[0]]
    for row in rows[1:]:
        if not row:
            updated_rows.append(row)
            continue

        rank = row[0].strip()
        sentence = row[1] if len(row) > 1 else ""

        if sentence.strip():
            already_present += 1
            updated_rows.append(row)
            continue

        replacement = rank_to_sentence.get(rank, "")
        if replacement:
            new_row = [rank, replacement]
            if len(row) > 2:
                new_row.extend(row[2:])
            updated_rows.append(new_row)
            filled += 1
        else:
            updated_rows.append(row)
            if rank:
                missing_in_source += 1

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)

    return filled, already_present, missing_in_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill blank rows in outputs/sentences.csv using rank-matched sentences from missing_rank_word_filled.csv."
    )
    parser.add_argument("--source", default="missing_rank_word_filled.csv", help="CSV with rank and sentence columns")
    parser.add_argument("--target", default="outputs/sentences.csv", help="CSV with rank,sentence rows to fill")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the filled CSV. Defaults to overwriting --target.",
    )
    args = parser.parse_args()

    source_csv = Path(args.source)
    target_csv = Path(args.target)
    output_csv = Path(args.output) if args.output else target_csv

    filled, already_present, missing_in_source = fill_sentences(source_csv, target_csv, output_csv)
    print(f"filled={filled} already_present={already_present} missing_in_source={missing_in_source} output={output_csv}")


if __name__ == "__main__":
    main()
