#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

POS_FAMILY_MAP = {
    "n": "n",
    "prop": "prop",
    "v": "v",
    "aux": "v",
    "adj": "adj",
    "adv": "adv",
    "determiner": "determiner",
    "det": "determiner",
    "art": "art",
    "pron": "pron",
    "pronoun": "pron",
    "prep": "prep",
    "adp": "prep",
    "conj": "conj",
    "cconj": "conj",
    "sconj": "conj",
    "interj": "interj",
    "num": "num",
    "contraction": "contraction",
    "letter": "letter",
    "prefix": "prefix",
    "phrase": "phrase",
    "particle": "particle",
    "none": "residual",
    "": "residual",
}

FUNCTION_WORD_FAMILIES = {
    "adv",
    "determiner",
    "art",
    "pron",
    "prep",
    "conj",
    "interj",
    "num",
    "contraction",
    "letter",
    "prefix",
    "phrase",
    "particle",
    "residual",
}

_SIMPLE_PREPOSITIONS = {
    "a", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante",
    "en", "entre", "hacia", "hasta", "para", "por", "segun", "según", "sin",
    "sobre", "tras", "al",
}

_POLICY_EXCLUDED_FAMILIES = {"letter", "prefix", "phrase", "particle"}
_METALINGUISTIC_SURFACES = {
    "sr", "sr.", "sra", "sra.", "dr", "dr.", "ud", "ud.", "uds", "uds.",
}


def normalize_surface(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def pos_family_from_values(raw_pos: Optional[str], lemma: Optional[str]) -> str:
    raw = normalize_surface(raw_pos)
    surface = normalize_surface(lemma)
    if raw in {"", "none"}:
        if surface in _SIMPLE_PREPOSITIONS:
            return "prep" if surface not in {"del", "al"} else "contraction"
        if surface == "no":
            return "adv"
    return POS_FAMILY_MAP.get(raw, raw or "residual")


def rank_block(rank_value: object) -> str:
    try:
        rank = int(rank_value)
    except Exception:
        return "unknown"
    if 1 <= rank <= 1000:
        return "1_1000"
    if 1001 <= rank <= 2000:
        return "1001_2000"
    if 2001 <= rank <= 3000:
        return "2001_3000"
    if 3001 <= rank <= 4000:
        return "3001_4000"
    if 4001 <= rank <= 5000:
        return "4001_5000"
    return "outside_1_5000"


def policy_exclusion_reason(lemma: Optional[str], raw_pos: Optional[str]) -> Optional[str]:
    surface = normalize_surface(lemma)
    family = pos_family_from_values(raw_pos, surface)
    if family in _POLICY_EXCLUDED_FAMILIES:
        return f"policy_excluded_{family}"
    if len(surface) == 1 and surface.isalpha():
        return "policy_excluded_single_letter"
    if surface in _METALINGUISTIC_SURFACES and family in {"letter", "residual"}:
        return "policy_excluded_metalinguistic_surface"
    return None
