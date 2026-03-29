#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOCKED_SUCCESS_METRICS = {
    "rank_window": (1, 1000),
    "min_nonempty_rate": 0.95,
    "max_bad_shipped_rate": 0.01,
}

BAD_SUBJECT_PATTERNS = [
    re.compile(r"^(mí|ti|sí)\s+", re.IGNORECASE),
    re.compile(r"^(vosotros|vosotras)\s+(está|es|va)\b", re.IGNORECASE),
]
DETERMINER_MISMATCH_PATTERNS = [
    re.compile(r"^(todos|todas|muchos|muchas)\s+\w+\s+está\b", re.IGNORECASE),
    re.compile(r"^(todo|toda|mucho|mucha)\s+\w+s\s+está\b", re.IGNORECASE),
]
TEMPORAL_MISMATCH_PATTERNS = [
    re.compile(r"\b(voy|va|vamos|van|llega|llego|llegan)\b.*\bayer\b", re.IGNORECASE),
    re.compile(r"\bayer\b.*\b(voy|va|vamos|van|llega|llego|llegan)\b", re.IGNORECASE),
]
SEMANTIC_NONSENSE_PATTERNS = [
    re.compile(r"\bva a ser nada\b", re.IGNORECASE),
    re.compile(r"\bvoy a ser nada\b", re.IGNORECASE),
    re.compile(r"\bes un poco de\b", re.IGNORECASE),
]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes"}


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def shipped_sentence(row: Dict[str, str]) -> bool:
    return bool((row.get("sentence") or "").strip())


def row_is_good(row: Dict[str, str]) -> bool:
    if "publishable" in row:
        return _to_bool(row.get("publishable"))
    return (
        str(row.get("grammatical_ok", "0")) == "1"
        and str(row.get("natural_ok", "0")) == "1"
        and str(row.get("learner_clear_ok", "0")) == "1"
    )


def row_is_bad_shipped(row: Dict[str, str]) -> bool:
    return shipped_sentence(row) and not row_is_good(row)


def failure_cluster(row: Dict[str, str]) -> str:
    sentence = (row.get("sentence") or "").strip()
    source_method = (row.get("source_method") or "").strip()
    template_id = (row.get("template_id") or "").strip()
    notes = (row.get("notes") or row.get("failure_reason") or "").strip().lower()

    if not sentence or source_method == "no_candidate_found":
        return "no_candidate"
    if template_id == "det_anchor_clear":
        return "determiner_template"
    if template_id == "pron_fallback_here":
        return "pronoun_fallback"
    if template_id == "adv_time_home":
        return "temporal_template"
    if source_method == "stochastic_decoder":
        for pattern in SEMANTIC_NONSENSE_PATTERNS:
            if pattern.search(sentence):
                return "stochastic_semantic_nonsense"
    for pattern in BAD_SUBJECT_PATTERNS:
        if pattern.search(sentence):
            return "invalid_subject_pronoun"
    for pattern in DETERMINER_MISMATCH_PATTERNS:
        if pattern.search(sentence):
            return "determiner_agreement"
    for pattern in TEMPORAL_MISMATCH_PATTERNS:
        if pattern.search(sentence):
            return "temporal_mismatch"
    if "manual review needed" in notes:
        return "manual_review"
    if notes:
        first = notes.split(";")[0].strip().replace(" ", "_")
        if first:
            return first
    if template_id:
        return template_id
    if source_method:
        return source_method
    return "unclassified"


def sample_rows(rows: List[Dict[str, str]], count: int, seed: int) -> List[Dict[str, str]]:
    if len(rows) <= count:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, count)


def metrics_summary(rows: List[Dict[str, str]]) -> Dict[str, object]:
    lo, hi = LOCKED_SUCCESS_METRICS["rank_window"]
    scoped = [row for row in rows if lo <= _to_int(row.get("rank"), 0) <= hi]
    total = len(scoped)
    shipped = [row for row in scoped if shipped_sentence(row)]
    good = [row for row in scoped if row_is_good(row)]
    bad_shipped = [row for row in scoped if row_is_bad_shipped(row)]
    no_candidate = [row for row in scoped if not shipped_sentence(row)]

    nonempty_rate = (len(shipped) / total) if total else 0.0
    bad_shipped_rate = (len(bad_shipped) / len(shipped)) if shipped else 0.0
    good_rate = (len(good) / total) if total else 0.0

    return {
        "rank_window": [lo, hi],
        "rows_in_scope": total,
        "nonempty_rows": len(shipped),
        "good_rows": len(good),
        "bad_shipped_rows": len(bad_shipped),
        "no_candidate_rows": len(no_candidate),
        "nonempty_rate": nonempty_rate,
        "good_rate": good_rate,
        "bad_shipped_rate": bad_shipped_rate,
        "passes_locked_nonempty_rate": nonempty_rate >= LOCKED_SUCCESS_METRICS["min_nonempty_rate"],
        "passes_locked_bad_shipped_rate": bad_shipped_rate <= LOCKED_SUCCESS_METRICS["max_bad_shipped_rate"],
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rank 1 to 1000 sentence-generator output and cluster failures.")
    parser.add_argument("--generated", required=True, help="CSV output from hybrid_generator.py or a review CSV.")
    parser.add_argument("--out-dir", default="eval_report", help="Directory for metrics and cluster exports.")
    parser.add_argument("--sample-size", type=int, default=50, help="How many accepted and failed rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    generated_path = Path(args.generated)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(generated_path)
    summary = metrics_summary(rows)

    failed_rows = [row for row in rows if row_is_bad_shipped(row) or not shipped_sentence(row)]
    cluster_counts: Dict[str, int] = {}
    cluster_rows: List[Dict[str, object]] = []
    for row in failed_rows:
        cluster = failure_cluster(row)
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        cluster_rows.append({
            "cluster": cluster,
            "lemma": row.get("lemma", ""),
            "rank": row.get("rank", ""),
            "pos": row.get("pos", ""),
            "source_method": row.get("source_method", ""),
            "template_id": row.get("template_id", ""),
            "sentence": row.get("sentence", ""),
            "notes": row.get("notes", row.get("failure_reason", "")),
        })

    cluster_table = [
        {"cluster": cluster, "count": count}
        for cluster, count in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    accepted_rows = [row for row in rows if row_is_good(row)]
    accepted_samples = sample_rows(accepted_rows, args.sample_size, args.seed)
    failed_samples = sample_rows(failed_rows, args.sample_size, args.seed)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(out_dir / "failure_clusters.csv", cluster_table, ["cluster", "count"])
    write_csv(out_dir / "failure_samples.csv", [
        {
            "cluster": failure_cluster(row),
            "lemma": row.get("lemma", ""),
            "rank": row.get("rank", ""),
            "pos": row.get("pos", ""),
            "source_method": row.get("source_method", ""),
            "template_id": row.get("template_id", ""),
            "sentence": row.get("sentence", ""),
            "notes": row.get("notes", row.get("failure_reason", "")),
        }
        for row in failed_samples
    ], ["cluster", "lemma", "rank", "pos", "source_method", "template_id", "sentence", "notes"])
    write_csv(out_dir / "accepted_samples.csv", [
        {
            "lemma": row.get("lemma", ""),
            "rank": row.get("rank", ""),
            "pos": row.get("pos", ""),
            "source_method": row.get("source_method", ""),
            "template_id": row.get("template_id", ""),
            "sentence": row.get("sentence", ""),
        }
        for row in accepted_samples
    ], ["lemma", "rank", "pos", "source_method", "template_id", "sentence"])
    write_csv(out_dir / "failure_rows_full.csv", cluster_rows, ["cluster", "lemma", "rank", "pos", "source_method", "template_id", "sentence", "notes"])

    summary_txt = out_dir / "summary.txt"
    summary_txt.write_text(
        "\n".join([
            f"rows_in_scope: {summary['rows_in_scope']}",
            f"nonempty_rows: {summary['nonempty_rows']}",
            f"good_rows: {summary['good_rows']}",
            f"bad_shipped_rows: {summary['bad_shipped_rows']}",
            f"no_candidate_rows: {summary['no_candidate_rows']}",
            f"nonempty_rate: {summary['nonempty_rate']:.4f}",
            f"good_rate: {summary['good_rate']:.4f}",
            f"bad_shipped_rate: {summary['bad_shipped_rate']:.4f}",
            f"passes_locked_nonempty_rate: {summary['passes_locked_nonempty_rate']}",
            f"passes_locked_bad_shipped_rate: {summary['passes_locked_bad_shipped_rate']}",
        ]),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if cluster_table:
        print("Top failure clusters:")
        for row in cluster_table[:10]:
            print(f"  {row['cluster']}: {row['count']}")
    print(f"Saved report to {out_dir}")


if __name__ == "__main__":
    main()
