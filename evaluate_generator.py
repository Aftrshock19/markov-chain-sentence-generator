#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from coverage_utils import pos_family_from_values

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
    re.compile(r"\bestá haciendo tarde\b", re.IGNORECASE),
    re.compile(r"\bserá una cosa así\b", re.IGNORECASE),
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


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def row_is_excluded_by_policy(row: Dict[str, str]) -> bool:
    return (row.get("quality_tier") or "").strip().lower() == "excluded_by_policy" or (row.get("source_method") or "").strip().lower() == "excluded_by_policy"


def row_is_bad_shipped(row: Dict[str, str]) -> bool:
    return shipped_sentence(row) and not row_is_good(row) and not row_is_excluded_by_policy(row)


def row_pos_family(row: Dict[str, str]) -> str:
    return pos_family_from_values(row.get("pos"), row.get("lemma"))


def suspicious_reason(row: Dict[str, str]) -> str:
    sentence = (row.get("sentence") or "").strip()
    source_method = (row.get("source_method") or "").strip()
    template_id = (row.get("template_id") or "").strip()
    notes = (row.get("notes") or row.get("failure_reason") or "").strip().lower()

    if row_is_excluded_by_policy(row):
        return "excluded_by_policy"
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


def scoped_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    lo, hi = LOCKED_SUCCESS_METRICS["rank_window"]
    return [row for row in rows if lo <= _to_int(row.get("rank"), 0) <= hi]


def metrics_summary(rows: List[Dict[str, str]]) -> Dict[str, object]:
    scoped = scoped_rows(rows)
    total = len(scoped)
    excluded = [row for row in scoped if row_is_excluded_by_policy(row)]
    effective_scope = [row for row in scoped if not row_is_excluded_by_policy(row)]
    shipped = [row for row in effective_scope if shipped_sentence(row)]
    good = [row for row in effective_scope if row_is_good(row)]
    bad_shipped = [row for row in effective_scope if row_is_bad_shipped(row)]
    no_candidate = [row for row in effective_scope if not shipped_sentence(row)]

    effective_total = len(effective_scope)
    nonempty_rate = (len(shipped) / effective_total) if effective_total else 0.0
    bad_shipped_rate = (len(bad_shipped) / len(shipped)) if shipped else 0.0
    good_rate = (len(good) / effective_total) if effective_total else 0.0

    return {
        "rank_window": list(LOCKED_SUCCESS_METRICS["rank_window"]),
        "rows_in_scope": total,
        "effective_rows_in_scope": effective_total,
        "excluded_by_policy_rows": len(excluded),
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


def aggregate_by_family(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    stats = defaultdict(lambda: {"total": 0, "nonempty": 0, "good": 0, "bad_shipped": 0, "missing": 0})
    for row in scoped_rows(rows):
        fam = row_pos_family(row)
        stats[fam]["total"] += 1
        if shipped_sentence(row):
            stats[fam]["nonempty"] += 1
        else:
            stats[fam]["missing"] += 1
        if row_is_good(row):
            stats[fam]["good"] += 1
        if row_is_bad_shipped(row):
            stats[fam]["bad_shipped"] += 1
    out = []
    for fam, s in sorted(stats.items()):
        total = s["total"] or 1
        shipped = s["nonempty"] or 1
        out.append({
            "family": fam,
            **s,
            "nonempty_rate": round(s["nonempty"] / total, 4),
            "good_rate": round(s["good"] / total, 4),
            "bad_shipped_rate": round(s["bad_shipped"] / shipped, 4) if s["nonempty"] else 0.0,
        })
    return out


def aggregate_source_method(rows: List[Dict[str, str]], only_good: bool = False, only_bad_shipped: bool = False) -> List[Dict[str, object]]:
    ctr = Counter()
    for row in scoped_rows(rows):
        if only_good and not row_is_good(row):
            continue
        if only_bad_shipped and not row_is_bad_shipped(row):
            continue
        if not shipped_sentence(row) and (only_good or only_bad_shipped):
            continue
        ctr[(row.get("source_method") or "", row.get("template_id") or "")] += 1
    return [
        {"source_method": sm, "template_id": tid, "count": count}
        for (sm, tid), count in sorted(ctr.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]


def top_missing_lemmas_by_family(rows: List[Dict[str, str]], limit: int = 15) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in scoped_rows(rows):
        if not shipped_sentence(row):
            grouped[row_pos_family(row)].append(row)
    out = []
    for fam, items in sorted(grouped.items()):
        for row in items[:limit]:
            out.append({"family": fam, "lemma": row.get("lemma", ""), "rank": row.get("rank", ""), "pos": row.get("pos", "")})
    return out


def compare_summaries(base: Dict[str, object], new: Dict[str, object]) -> Dict[str, object]:
    keys = ["rows_in_scope", "effective_rows_in_scope", "excluded_by_policy_rows", "nonempty_rows", "good_rows", "bad_shipped_rows", "no_candidate_rows", "nonempty_rate", "good_rate", "bad_shipped_rate"]
    out = {}
    for k in keys:
        bv = base.get(k, 0)
        nv = new.get(k, 0)
        out[k] = {"before": bv, "after": nv, "delta": round(nv - bv, 6) if isinstance(bv, float) or isinstance(nv, float) else nv - bv}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rank 1 to 1000 sentence-generator output and cluster failures.")
    parser.add_argument("--generated", required=True, help="CSV output from hybrid_generator.py or a review CSV.")
    parser.add_argument("--out-dir", default="eval_report", help="Directory for metrics and cluster exports.")
    parser.add_argument("--sample-size", type=int, default=50, help="How many accepted and failed rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--compare-to", default=None, help="Optional previous CSV output to compare against.")
    parser.add_argument("--fail-on-metric-miss", action="store_true", help="Exit non-zero if locked metrics are not met.")
    args = parser.parse_args()

    generated_path = Path(args.generated)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(generated_path)
    summary = metrics_summary(rows)
    scoped = scoped_rows(rows)
    failed_rows = [row for row in scoped if row_is_bad_shipped(row) or (not shipped_sentence(row) and not row_is_excluded_by_policy(row))]
    bad_shipped = [row for row in scoped if row_is_bad_shipped(row)]
    accepted_rows = [row for row in scoped if row_is_good(row) and not row_is_excluded_by_policy(row)]

    cluster_counts: Dict[str, int] = {}
    cluster_rows: List[Dict[str, object]] = []
    for row in failed_rows:
        cluster = suspicious_reason(row)
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        cluster_rows.append({
            "cluster": cluster,
            "family": row_pos_family(row),
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

    family_table = aggregate_by_family(rows)
    accepted_by_source = aggregate_source_method(rows, only_good=True)
    suspicious_by_source = aggregate_source_method(rows, only_bad_shipped=True)
    missing_by_family = top_missing_lemmas_by_family(rows)
    accepted_samples = sample_rows(accepted_rows, args.sample_size, args.seed)
    failed_samples = sample_rows(failed_rows, args.sample_size, args.seed)

    report = {
        "summary": summary,
        "family_breakdown": family_table,
        "accepted_by_source_method": accepted_by_source,
        "suspicious_by_source_method": suspicious_by_source,
        "top_missing_lemmas_by_family": missing_by_family,
        "top_failure_clusters": cluster_table,
    }

    if args.compare_to:
        prev_rows = load_csv(Path(args.compare_to))
        report["comparison"] = compare_summaries(metrics_summary(prev_rows), summary)

    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "family_breakdown.csv", family_table, ["family", "total", "nonempty", "good", "bad_shipped", "missing", "nonempty_rate", "good_rate", "bad_shipped_rate"])
    write_csv(out_dir / "accepted_by_source_method.csv", accepted_by_source, ["source_method", "template_id", "count"])
    write_csv(out_dir / "suspicious_by_source_method.csv", suspicious_by_source, ["source_method", "template_id", "count"])
    write_csv(out_dir / "top_missing_lemmas_by_family.csv", missing_by_family, ["family", "lemma", "rank", "pos"])
    write_csv(out_dir / "failure_clusters.csv", cluster_table, ["cluster", "count"])
    write_csv(out_dir / "failure_rows_full.csv", cluster_rows, ["cluster", "family", "lemma", "rank", "pos", "source_method", "template_id", "sentence", "notes"])
    write_csv(out_dir / "failure_samples.csv", [
        {
            "cluster": suspicious_reason(row),
            "family": row_pos_family(row),
            "lemma": row.get("lemma", ""),
            "rank": row.get("rank", ""),
            "pos": row.get("pos", ""),
            "source_method": row.get("source_method", ""),
            "template_id": row.get("template_id", ""),
            "sentence": row.get("sentence", ""),
            "notes": row.get("notes", row.get("failure_reason", "")),
        }
        for row in failed_samples
    ], ["cluster", "family", "lemma", "rank", "pos", "source_method", "template_id", "sentence", "notes"])
    write_csv(out_dir / "accepted_samples.csv", [
        {
            "family": row_pos_family(row),
            "lemma": row.get("lemma", ""),
            "rank": row.get("rank", ""),
            "pos": row.get("pos", ""),
            "source_method": row.get("source_method", ""),
            "template_id": row.get("template_id", ""),
            "sentence": row.get("sentence", ""),
        }
        for row in accepted_samples
    ], ["family", "lemma", "rank", "pos", "source_method", "template_id", "sentence"])

    summary_lines = [
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
    ]
    if args.compare_to and "comparison" in report:
        summary_lines.append("comparison:")
        for key, vals in report["comparison"].items():
            summary_lines.append(f"  {key}: before={vals['before']} after={vals['after']} delta={vals['delta']}")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_metric_miss and not (
        summary["passes_locked_nonempty_rate"] and summary["passes_locked_bad_shipped_rate"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
