#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from coverage_utils import pos_family_from_values, rank_block, policy_exclusion_reason, FUNCTION_WORD_FAMILIES

RANK_EXPORTS = {
    "no_candidate_top_1000.csv": lambda r: 1 <= r <= 1000,
    "no_candidate_1001_2000.csv": lambda r: 1001 <= r <= 2000,
    "no_candidate_2001_3000.csv": lambda r: 2001 <= r <= 3000,
    "no_candidate_3001_4000.csv": lambda r: 3001 <= r <= 4000,
    "no_candidate_4001_5000.csv": lambda r: 4001 <= r <= 5000,
}

FRAGILE_OR_EXACT_SURFACE_FAMILIES = FUNCTION_WORD_FAMILIES | {"residual"}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def enrich(row: Dict[str, str]) -> Dict[str, object]:
    rank = to_int(row.get("rank"), 0)
    lemma = (row.get("lemma") or "").strip()
    pos = (row.get("pos") or "").strip()
    family = pos_family_from_values(pos, lemma)
    return {
        **row,
        "rank": rank,
        "family": family,
        "rank_block": rank_block(rank),
        "policy_exclusion_reason": policy_exclusion_reason(lemma, pos) or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn an old hybrid batch into a reusable coverage backlog.")
    parser.add_argument("--batch", required=True, help="Path to the old hybrid batch CSV.")
    parser.add_argument("--out-dir", default="coverage_backlog", help="Directory for backlog exports.")
    args = parser.parse_args()

    batch_path = Path(args.batch)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [enrich(row) for row in load_rows(batch_path)]
    no_candidate_rows = [row for row in rows if (row.get("source_method") or "").strip() == "no_candidate_found"]
    no_candidate_rows.sort(key=lambda row: (row["rank"], str(row.get("lemma") or "")))

    base_fields = [
        "lemma", "rank", "pos", "family", "rank_block", "translation", "canonical_lemma",
        "source_method", "template_id", "quality_tier", "failure_reason", "policy_exclusion_reason",
    ]
    write_csv(out_dir / "no_candidate_all.csv", no_candidate_rows, base_fields)

    for filename, predicate in RANK_EXPORTS.items():
        subset = [row for row in no_candidate_rows if predicate(row["rank"])]
        write_csv(out_dir / filename, subset, base_fields)

    by_family = Counter(row["family"] for row in no_candidate_rows)
    family_rows = [
        {"family": family, "no_candidate_count": count}
        for family, count in sorted(by_family.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_csv(out_dir / "no_candidate_by_family.csv", family_rows, ["family", "no_candidate_count"])

    by_family_block = defaultdict(int)
    for row in no_candidate_rows:
        by_family_block[(row["family"], row["rank_block"])] += 1
    family_block_rows = [
        {"family": family, "rank_block": block, "no_candidate_count": count}
        for (family, block), count in sorted(by_family_block.items(), key=lambda item: (item[0][1], -item[1], item[0][0]))
    ]
    write_csv(
        out_dir / "no_candidate_by_family_and_rank_block.csv",
        family_block_rows,
        ["family", "rank_block", "no_candidate_count"],
    )

    exact_surface_backlog = [
        row for row in no_candidate_rows
        if row["family"] in FRAGILE_OR_EXACT_SURFACE_FAMILIES and 1 <= row["rank"] <= 5000
    ]
    write_csv(out_dir / "exact_surface_candidate_backlog.csv", exact_surface_backlog, base_fields)

    summary = {
        "input_file": str(batch_path),
        "total_rows": len(rows),
        "no_candidate_rows": len(no_candidate_rows),
        "no_candidate_by_family": dict(sorted(by_family.items(), key=lambda item: (-item[1], item[0]))),
        "no_candidate_by_rank_block": {
            block: sum(1 for row in no_candidate_rows if row["rank_block"] == block)
            for block in ["1_1000", "1001_2000", "2001_3000", "3001_4000", "4001_5000", "outside_1_5000"]
        },
        "fragile_family_under_1000": sum(
            1 for row in no_candidate_rows if row["family"] in FUNCTION_WORD_FAMILIES and 1 <= row["rank"] <= 1000
        ),
        "top_missing_surfaces": [
            {"lemma": row["lemma"], "rank": row["rank"], "family": row["family"]}
            for row in no_candidate_rows[:100]
        ],
    }
    (out_dir / "coverage_backlog_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
