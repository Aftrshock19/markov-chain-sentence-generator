#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+|[¿¡.,;:!?\"“”'()\[\]{}]")
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
COMMON_FIXED = {
    "no", "quiero", "puedo", "voy", "va", "vas", "vamos", "van", "tengo", "tiene", "tienen",
    "hay", "es", "está", "esta", "estoy", "estás", "estamos", "están", "la", "el", "una", "un",
    "aquí", "aqui", "allí", "alli", "hoy", "ya", "mi", "tu", "su", "de", "a", "en", "con", "para",
    "sin", "por", "del", "al", "lo", "me", "te", "se", "nos", "si", "que", "eso", "esto",
}
POS_FAMILY_MAP = {
    "n": "noun",
    "v": "verb",
    "adj": "adjective",
    "adv": "adverb",
    "prep": "function",
    "pron": "function",
    "determiner": "function",
    "art": "function",
    "conj": "function",
    "interj": "function",
    "num": "function",
    "contraction": "function",
    "letter": "function",
    "prefix": "function",
    "phrase": "function",
    "particle": "function",
    "prop": "noun",
    "": "function",
    "none": "function",
}


@dataclass
class LexiconInfo:
    pos: str = ""
    rank: int = 999999


@dataclass
class FrameAggregate:
    family: str
    pattern_tokens: List[str]
    slot_types: List[str]
    count: int = 0
    lemmas: set = field(default_factory=set)
    ranks: List[int] = field(default_factory=list)
    support_rank_max_values: List[int] = field(default_factory=list)
    support_rank_avg_values: List[float] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def add(self, lemma: str, rank: int, sentence: str, support_rank_max: int, support_rank_avg: float) -> None:
        self.count += 1
        self.lemmas.add(lemma)
        self.ranks.append(rank)
        self.support_rank_max_values.append(support_rank_max)
        self.support_rank_avg_values.append(support_rank_avg)
        if sentence not in self.examples and len(self.examples) < 5:
            self.examples.append(sentence)

    def to_payload(self) -> Dict[str, object]:
        unique_lemma_count = len(self.lemmas)
        support_rank_max = max(self.support_rank_max_values) if self.support_rank_max_values else 0
        support_rank_avg = sum(self.support_rank_avg_values) / len(self.support_rank_avg_values) if self.support_rank_avg_values else 0.0
        avg_target_rank = sum(self.ranks) / len(self.ranks) if self.ranks else 0.0
        weight = round((math.log1p(self.count) * math.log1p(unique_lemma_count + 1)) / (1.0 + (support_rank_avg / 2000.0)), 4)
        return {
            "frame_id": stable_frame_id(self.family, self.pattern_tokens),
            "pattern_tokens": self.pattern_tokens,
            "slot_types": self.slot_types,
            "weight": weight,
            "count": self.count,
            "unique_lemma_count": unique_lemma_count,
            "average_target_rank": round(avg_target_rank, 2),
            "min_rank": min(self.ranks) if self.ranks else 0,
            "max_rank": max(self.ranks) if self.ranks else 0,
            "support_rank_max": support_rank_max,
            "support_rank_avg": round(support_rank_avg, 2),
            "examples": self.examples,
        }



def load_lexicon(path: Optional[str]) -> Dict[str, LexiconInfo]:
    if not path:
        return {}
    out: Dict[str, LexiconInfo] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lemma = (row.get("lemma") or "").strip().lower()
            if not lemma:
                continue
            try:
                rank = int(float(row.get("rank") or 999999))
            except ValueError:
                rank = 999999
            out[lemma] = LexiconInfo(pos=(row.get("pos") or "").strip().lower(), rank=rank)
    return out



def pos_family(pos: str) -> str:
    return POS_FAMILY_MAP.get((pos or "").strip().lower(), "function")



def tokenize(sentence: str) -> List[str]:
    return TOKEN_RE.findall(sentence or "")



def norm(token: str) -> str:
    return (token or "").strip().lower()



def find_target_index(tokens: List[str], lemma: str) -> int:
    target = norm(lemma)
    for idx, token in enumerate(tokens):
        if norm(token) == target:
            return idx
    return -1



def stable_frame_id(family: str, pattern_tokens: Iterable[str]) -> str:
    body = "_".join(token.strip("{} ").lower() for token in pattern_tokens)
    body = re.sub(r"[^a-z0-9_]+", "_", body)
    body = re.sub(r"_+", "_", body).strip("_")
    return f"learned_{family}_{body[:72] or 'frame'}"



def infer_slot(token: str, lexicon: Dict[str, LexiconInfo]) -> Optional[str]:
    low = norm(token)
    if low in {"aquí", "aqui", "allí", "alli", "allá", "alla"}:
        return "{LOC}"
    if low in COMMON_FIXED:
        return None
    info = lexicon.get(low)
    pos = info.pos if info else ""
    if pos == "n":
        return "{NOUN}"
    if pos == "adj":
        return "{ADJ}"
    if pos == "adv":
        return "{ADV}"
    if pos == "prep":
        return None
    if pos == "v":
        if low.endswith(("ar", "er", "ir")):
            return "{INF}"
        return "{VERB}"
    return None



def collapse_slots(pattern_tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(pattern_tokens):
        token = pattern_tokens[i]
        if token in {"de", "a", "en", "con", "para", "por", "sin", "sobre", "del", "al"} and i + 1 < len(pattern_tokens):
            nxt = pattern_tokens[i + 1]
            if nxt in {"{NOUN}", "{LOC}"}:
                out.append("{PREP_PHRASE}")
                i += 2
                continue
        if out and token == out[-1] and token.startswith("{"):
            i += 1
            continue
        out.append(token)
        i += 1
    return out



def induce_pattern(sentence: str, lemma: str, pos: str, lexicon: Dict[str, LexiconInfo]) -> Optional[List[str]]:
    tokens = tokenize(sentence)
    target_idx = find_target_index(tokens, lemma)
    if target_idx < 0:
        return None
    family = pos_family(pos)
    pattern: List[str] = []
    for idx, token in enumerate(tokens):
        low = norm(token)
        if not WORD_RE.search(token):
            continue
        if idx == target_idx:
            pattern.append("{TARGET}")
            continue
        slot = infer_slot(token, lexicon)
        if family == "verb" and idx == target_idx + 1 and slot == "{VERB}":
            slot = "{INF}"
        if family == "noun" and idx == target_idx + 1 and slot == "{ADJ}":
            pattern.append(slot)
            continue
        if family == "adjective" and slot == "{NOUN}" and idx < target_idx:
            pattern.append(slot)
            continue
        if slot and low not in COMMON_FIXED:
            pattern.append(slot)
        else:
            pattern.append(low)
    pattern = collapse_slots(pattern)
    if pattern.count("{TARGET}") != 1:
        return None
    return pattern



def extract_slot_types(pattern_tokens: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for token in pattern_tokens:
        if token.startswith("{") and token.endswith("}"):
            name = token.strip("{}")
            if name not in seen:
                seen.append(name)
    return seen



def induce_frames(
    input_csv: str,
    frames_out: str,
    lemma_pref_out: str,
    pos_stats_out: str,
    lexicon_path: Optional[str] = None,
) -> Dict[str, int]:
    lexicon = load_lexicon(lexicon_path)
    aggregates: Dict[Tuple[str, Tuple[str, ...]], FrameAggregate] = {}
    lemma_preferences: Dict[Tuple[str, str], int] = defaultdict(int)

    with open(input_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        lemma = (row.get("lemma") or "").strip().lower()
        pos = (row.get("pos") or "").strip().lower()
        sentence = (row.get("sentence") or "").strip()
        if not lemma or not sentence:
            continue
        pattern_tokens = induce_pattern(sentence=sentence, lemma=lemma, pos=pos, lexicon=lexicon)
        if not pattern_tokens:
            continue
        family = pos_family(pos)
        key = (family, tuple(pattern_tokens))
        agg = aggregates.get(key)
        if agg is None:
            agg = FrameAggregate(family=family, pattern_tokens=pattern_tokens, slot_types=extract_slot_types(pattern_tokens))
            aggregates[key] = agg
        agg.add(
            lemma=lemma,
            rank=int(float(row.get("rank") or 999999)),
            sentence=sentence,
            support_rank_max=int(float(row.get("support_rank_max") or 0)),
            support_rank_avg=float(row.get("support_rank_avg") or 0.0),
        )
        lemma_preferences[(lemma, stable_frame_id(family, pattern_tokens))] += 1

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    pos_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"frame_count": 0, "instance_count": 0, "avg_weight_sum": 0.0})
    for agg in sorted(aggregates.values(), key=lambda item: (item.family, -item.count, item.pattern_tokens)):
        if agg.count < 1:
            continue
        payload = agg.to_payload()
        grouped[agg.family].append(payload)
        pos_stats[agg.family]["frame_count"] += 1
        pos_stats[agg.family]["instance_count"] += agg.count
        pos_stats[agg.family]["avg_weight_sum"] += float(payload["weight"])

    Path(frames_out).parent.mkdir(parents=True, exist_ok=True)
    with open(frames_out, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    with open(lemma_pref_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lemma", "frame_id", "count"])
        writer.writeheader()
        for (lemma, frame_id), count in sorted(lemma_preferences.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
            writer.writerow({"lemma": lemma, "frame_id": frame_id, "count": count})

    with open(pos_stats_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pos_family", "frame_count", "instance_count", "avg_weight"])
        writer.writeheader()
        for family, stats in sorted(pos_stats.items()):
            frame_count = int(stats["frame_count"])
            avg_weight = (stats["avg_weight_sum"] / frame_count) if frame_count else 0.0
            writer.writerow(
                {
                    "pos_family": family,
                    "frame_count": frame_count,
                    "instance_count": int(stats["instance_count"]),
                    "avg_weight": f"{avg_weight:.4f}",
                }
            )

    return {"accepted_rows": len(rows), "frames": sum(len(v) for v in grouped.values())}



def main() -> None:
    parser = argparse.ArgumentParser(description="Induce reusable generation frames from accepted teacher outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--frames-out", required=True)
    parser.add_argument("--lemma-pref-out", required=True)
    parser.add_argument("--pos-stats-out", required=True)
    parser.add_argument("--lexicon", default="stg_words_spa.csv")
    args = parser.parse_args()

    stats = induce_frames(
        input_csv=args.input,
        frames_out=args.frames_out,
        lemma_pref_out=args.lemma_pref_out,
        pos_stats_out=args.pos_stats_out,
        lexicon_path=args.lexicon,
    )
    print(f"Accepted rows read: {stats['accepted_rows']}")
    print(f"Frames induced: {stats['frames']}")
    print(f"Saved: {args.frames_out}")
    print(f"Saved: {args.lemma_pref_out}")
    print(f"Saved: {args.pos_stats_out}")


if __name__ == "__main__":
    main()
