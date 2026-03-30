#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from teacher_llm import TeacherLLM, TeacherConfig


RAW_FIELDS = [
    "lemma",
    "rank",
    "pos",
    "translation",
    "teacher_model",
    "teacher_base_url",
    "prompt_style",
    "candidate_idx",
    "sentence",
    "contains_exact_target",
    "token_count",
    "char_count",
    "source_failure_reason",
    "source_template_id",
    "source_quality_tier",
]


class LexiconLookup:
    def __init__(self, path: Optional[str]):
        self.rows: Dict[str, Dict[str, str]] = {}
        if not path:
            return
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                lemma = (row.get("lemma") or "").strip().lower()
                if lemma:
                    self.rows[lemma] = row

    def enrich(self, lemma: str, row: Dict[str, str]) -> Dict[str, str]:
        base = dict(self.rows.get(lemma.lower(), {}))
        merged = dict(base)
        merged.update({k: v for k, v in row.items() if v not in (None, "")})
        return merged



def read_explicit_lemmas(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lemma = line.strip().lower()
            if not lemma:
                continue
            items.append({"lemma": lemma})
    return items



def select_weak_rows(
    input_batch: Optional[str],
    explicit_lemma_list: Optional[str],
    lexicon_path: Optional[str],
    min_rank: int,
    max_rank: int,
    limit: int,
    failure_filters: Iterable[str],
) -> List[Dict[str, str]]:
    lexicon = LexiconLookup(lexicon_path)
    filters = {item.strip() for item in failure_filters if item.strip()}
    rows: List[Dict[str, str]] = []
    seen = set()

    if explicit_lemma_list:
        base_rows = read_explicit_lemmas(explicit_lemma_list)
    elif input_batch:
        with open(input_batch, encoding="utf-8", newline="") as f:
            base_rows = list(csv.DictReader(f))
    else:
        raise ValueError("Provide either --input-batch or --lemma-list")

    for raw in base_rows:
        lemma = (raw.get("lemma") or "").strip().lower()
        if not lemma or lemma in seen:
            continue
        merged = lexicon.enrich(lemma, raw)
        try:
            rank = int(float(merged.get("rank") or 999999))
        except ValueError:
            rank = 999999
        if rank < min_rank or rank > max_rank:
            continue

        if filters:
            failure_reason = (merged.get("failure_reason") or "").strip()
            quality_tier = (merged.get("quality_tier") or "").strip()
            bad_candidate = str(merged.get("bad_candidate") or "").strip().lower() == "true"
            source_method = (merged.get("source_method") or "").strip()
            matched = (
                failure_reason in filters
                or quality_tier in filters
                or source_method in filters
                or (bad_candidate and "bad_candidate" in filters)
                or (quality_tier == "weak" and "awkward" in filters)
                or (quality_tier == "bad" and "low_naturalness" in filters)
            )
            if not matched:
                continue

        rows.append(merged)
        seen.add(lemma)
        if limit > 0 and len(rows) >= limit:
            break
    return rows



def build_raw_teacher_dataset(
    rows: List[Dict[str, str]],
    output_path: str,
    n: int,
    teacher: TeacherLLM,
) -> int:
    count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in rows:
            lemma = (row.get("lemma") or "").strip().lower()
            rank = int(float(row.get("rank") or 999999))
            pos = (row.get("pos") or "").strip().lower()
            translation = (row.get("translation") or "").strip()
            band = (row.get("band") or "").strip()
            sentences = teacher.generate_teacher_candidates(
                lemma=lemma,
                rank=rank,
                pos=pos,
                translation=translation,
                n=n,
                band=band or None,
            )
            for idx, sentence in enumerate(sentences, start=1):
                tokens = sentence.split()
                writer.writerow(
                    {
                        "lemma": lemma,
                        "rank": rank,
                        "pos": pos,
                        "translation": translation,
                        "teacher_model": teacher.config.model,
                        "teacher_base_url": teacher.config.base_url,
                        "prompt_style": "strict_short_spain_spanish_v1",
                        "candidate_idx": idx,
                        "sentence": sentence,
                        "contains_exact_target": str(lemma in {token.strip('.,;:!?¡¿\"\'').lower() for token in tokens}).lower(),
                        "token_count": len(tokens),
                        "char_count": len(sentence),
                        "source_failure_reason": row.get("failure_reason", ""),
                        "source_template_id": row.get("template_id", ""),
                        "source_quality_tier": row.get("quality_tier", ""),
                    }
                )
                count += 1
    return count



def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw offline teacher sentences for weak lemmas.")
    parser.add_argument("--input-batch", default=None)
    parser.add_argument("--lemma-list", default=None)
    parser.add_argument("--lexicon", default="stg_words_spa.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=10**9)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--failure-filter", nargs="*", default=["no_candidate_found", "bad_candidate"])
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-base-url", default=None)
    parser.add_argument("--hf-timeout", type=float, default=None)
    parser.add_argument("--hf-temperature", type=float, default=None)
    parser.add_argument("--hf-max-tokens", type=int, default=None)
    args = parser.parse_args()

    config = TeacherConfig.from_env()
    if args.hf_model:
        config.model = args.hf_model
    if args.hf_base_url:
        config.base_url = args.hf_base_url
    if args.hf_timeout is not None:
        config.timeout = args.hf_timeout
    if args.hf_temperature is not None:
        config.temperature = args.hf_temperature
    if args.hf_max_tokens is not None:
        config.max_tokens = args.hf_max_tokens

    rows = select_weak_rows(
        input_batch=args.input_batch,
        explicit_lemma_list=args.lemma_list,
        lexicon_path=args.lexicon,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        limit=args.limit,
        failure_filters=args.failure_filter,
    )
    teacher = TeacherLLM(config=config)
    written = build_raw_teacher_dataset(rows=rows, output_path=args.output, n=args.n, teacher=teacher)
    print(f"Selected lemmas: {len(rows)}")
    print(f"Raw rows written: {written}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
