#!/usr/bin/env python3
import argparse
import csv
import gzip
import orjson
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

BANNED_SENSE_TAGS = {
    "obsolete",
    "archaic",
    "dated",
    "historical",
    "rare",
    "uncommon",
    "superseded",
    "misspelling",
    "typo",
}

LOW_PRIORITY_SENSE_TAGS = {
    "slang",
    "colloquial",
    "regional",
    "dialectal",
    "figurative",
    "literary",
    "formal",
}

SKIP_POS = {
    "abbrev",
    "character",
    "diacritic",
    "glyph",
    "han character",
    "ideophone",
    "symbol",
    "punctuation",
}

FUNCTION_WORD_POS = {
    "article",
    "determiner",
    "pronoun",
    "preposition",
    "conjunction",
    "particle",
    "adverb",
    "verb",
    "contraction",
}

POS_PRIORITY = {
    "article": 100,
    "determiner": 98,
    "pronoun": 96,
    "preposition": 94,
    "conjunction": 92,
    "particle": 90,
    "adverb": 88,
    "verb": 86,
    "noun": 84,
    "adjective": 82,
    "interjection": 80,
    "phrase": 70,
    "abbrev": 10,
}


CLOSED_CLASS_POS = {
    "article",
    "determiner",
    "pronoun",
    "preposition",
    "conjunction",
    "particle",
    "adverb",
    "contraction",
    "interjection",
}

FORM_LIKE_POS = {
    "article",
    "determiner",
    "pronoun",
    "adjective",
    "num",
}

FLASHCARD_OVERRIDES = {
    "de": ("of; from", "preposition"),
    "que": ("that", "conjunction"),
    "la": ("the", "article"),
    "el": ("the", "article"),
    "los": ("the", "article"),
    "las": ("the", "article"),
    "no": ("not", "adverb"),
    "a": ("to; at", "preposition"),
    "y": ("and", "conjunction"),
    "en": ("in; on; at", "preposition"),
    "un": ("a; an", "article"),
    "una": ("a; an", "article"),
    "unos": ("some", "article"),
    "unas": ("some", "article"),
    "lo": ("the", "article"),
    "por": ("by; for", "preposition"),
    "para": ("for; to", "preposition"),
    "del": ("of the; from the", "contraction"),
    "al": ("to the; at the", "contraction"),
    "si": ("if", "conjunction"),
    "sí": ("yes", "particle"),
    "como": ("like; as", "adverb"),
    "mi": ("my", "determiner"),
    "mis": ("my", "determiner"),
    "tu": ("your", "determiner"),
    "tus": ("your", "determiner"),
    "su": ("his; her; its; your; their", "determiner"),
    "sus": ("his; her; its; your; their", "determiner"),
    "me": ("me", "pronoun"),
    "te": ("you", "pronoun"),
    "se": ("oneself; himself; herself; itself; themselves; each other", "pronoun"),
    "le": ("him; her; it", "pronoun"),
    "les": ("them", "pronoun"),
    "yo": ("I", "pronoun"),
    "tú": ("you", "pronoun"),
    "él": ("he; him", "pronoun"),
    "ella": ("she; her", "pronoun"),
    "ellos": ("they; them", "pronoun"),
    "ellas": ("they; them", "pronoun"),
    "este": ("this", "determiner"),
    "esta": ("this", "determiner"),
    "estos": ("these", "determiner"),
    "estas": ("these", "determiner"),
    "ese": ("that", "determiner"),
    "esa": ("that", "determiner"),
    "esos": ("those", "determiner"),
    "esas": ("those", "determiner"),
    "esto": ("this", "pronoun"),
    "eso": ("that", "pronoun"),
    "todo": ("all; every", "determiner"),
    "toda": ("all; every", "determiner"),
    "todos": ("all", "determiner"),
    "todas": ("all", "determiner"),
    "muy": ("very", "adverb"),
    "bien": ("well", "adverb"),
    "hasta": ("until; up to", "preposition"),
    "solo": ("only", "adverb"),
    "sólo": ("only", "adverb"),
    "qué": ("what", "pronoun"),
    "quién": ("who", "pronoun"),
    "cómo": ("how", "adverb"),
    "cuándo": ("when", "adverb"),
    "dónde": ("where", "adverb"),
    "porque": ("because", "conjunction"),
    "con": ("with", "preposition"),
    "sin": ("without", "preposition"),
}




def norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_gloss(text: str) -> str:
    return norm_space(text).strip(" ;,.")


def get_tags(sense: dict) -> set[str]:
    tags = set()
    for key in ("tags", "raw_tags"):
        value = sense.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    tags.add(item.strip().lower())
    return tags


def get_glosses(sense: dict) -> list[str]:
    out = []
    seen = set()
    for key in ("glosses", "raw_glosses"):
        value = sense.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    item = clean_gloss(item)
                    if item:
                        lowered = item.lower()
                        if lowered not in seen:
                            seen.add(lowered)
                            out.append(item)
    return out


def extract_linked_word(items) -> str:
    if not isinstance(items, list):
        return ""
    for item in items:
        if isinstance(item, dict):
            word = norm_space(item.get("word", ""))
            if word:
                return word
        elif isinstance(item, str):
            word = norm_space(item)
            if word:
                return word
    return ""


def get_original_lemma_from_sense(sense: dict) -> str:
    linked = extract_linked_word(sense.get("form_of"))
    if linked:
        return linked
    linked = extract_linked_word(sense.get("alt_of"))
    if linked:
        return linked
    return ""


def extract_english_translations(sense: dict, max_items: int = 5) -> list[str]:
    out = []
    seen = set()
    for tr in sense.get("translations", []):
        if not isinstance(tr, dict):
            continue
        code = norm_space(tr.get("code", "")).lower()
        lang_code = norm_space(tr.get("lang_code", "")).lower()
        lang = norm_space(tr.get("lang", "")).lower()
        if code != "en" and lang_code != "en" and lang not in {"english", "inglés", "ingles"}:
            continue
        word = clean_gloss(tr.get("word", ""))
        if not word:
            continue
        lowered = word.lower()
        if lowered not in seen:
            seen.add(lowered)
            out.append(word)
            if len(out) >= max_items:
                break
    return out


def dedupe_keep_order(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        item = norm_space(item)
        if not item:
            continue
        lowered = item.casefold()
        if lowered not in seen:
            seen.add(lowered)
            out.append(item)
    return out


def contains_spanish_echo(text: str, lemma: str, original_lemma: str = "") -> bool:
    lowered = norm_space(text).casefold()
    blocked = {norm_space(lemma).casefold()}
    if original_lemma:
        blocked.add(norm_space(original_lemma).casefold())
    if lowered in blocked:
        return True
    pieces = {p for p in re.split(r"[^\wáéíóúüñ]+", lowered) if p}
    return bool(pieces & blocked)


def normalize_translation_item(text: str) -> str:
    text = norm_space(text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ;,.")
    text = re.sub(r"\bliterally\b", "", text, flags=re.I)
    text = norm_space(text)
    return text


META_WORD_RE = re.compile(
    r"\b(?:masculine|feminine|singular|plural|definite|indefinite|article|pronoun|determiner|adjective|adverb|noun|verb|contraction|apocopic|accusative|dative|nominative|prepositional|subjective|disjunctive|comparative|superlative|neuter|first-person|second-person|third-person|present|preterite|imperfect|future|conditional|subjunctive|indicative|inflection|form|alternative spelling|alternative form)\b",
    re.I,
)


def is_meta_descriptor(text: str) -> bool:
    t = normalize_translation_item(text).lower()
    if not t:
        return True
    if t.startswith(("plural of ", "feminine singular of ", "masculine singular of ", "feminine plural of ", "masculine plural of ", "inflection of ", "form of ", "apocopic form of ", "prepositional of ", "accusative of ", "dative of ", "nominative of ")):
        return True
    if META_WORD_RE.search(t) and len(t.split()) <= 8:
        return True
    return False


def short_phrase_score(text: str, lemma: str, original_lemma: str = "") -> float:
    t = normalize_translation_item(text)
    if not t:
        return -1e9
    score = 0.0
    if contains_spanish_echo(t, lemma, original_lemma):
        score -= 100.0
    if META_WORD_RE.search(t):
        score -= 25.0
    if ":" in t:
        score -= 8.0
    if "(" in t or ")" in t:
        score -= 6.0
    words = t.split()
    score -= max(len(words) - 3, 0) * 2.5
    score -= max(len(t) - 24, 0) * 0.2
    if ";" in t:
        score += 2.0
    if re.fullmatch(r"[A-Za-z][A-Za-z '\-/]*", t):
        score += 3.0
    return score


def split_simple_list(text: str) -> list[str]:
    text = normalize_translation_item(text)
    if not text:
        return []
    if ";" in text:
        parts = [normalize_translation_item(p) for p in text.split(";")]
        return [p for p in parts if p]
    if "," in text and len(text) <= 40:
        parts = [normalize_translation_item(p) for p in text.split(",")]
        return [p for p in parts if p]
    return [text]


def compact_gloss_candidates(gloss: str, lemma: str, original_lemma: str, pos: str) -> list[str]:
    gloss = normalize_translation_item(gloss)
    if not gloss:
        return []

    raw = []
    raw.extend(re.findall(r'"([^"]+)"', gloss))
    raw.extend(re.findall(r"'([^']+)'", gloss))

    if ":" in gloss:
        raw.append(gloss.split(":", 1)[1])

    raw.extend(re.split(r"\s*;\s*", gloss))

    if not raw:
        raw.append(gloss)

    candidates = []
    for piece in raw:
        piece = normalize_translation_item(re.sub(r"\([^)]*\)", "", piece))
        if not piece:
            continue
        if piece.lower().startswith(("used ", "common ", "commonly ", "only ", "especially ")):
            continue
        for frag in split_simple_list(piece):
            frag = normalize_translation_item(frag)
            if not frag:
                continue
            if contains_spanish_echo(frag, lemma, original_lemma):
                continue
            if is_meta_descriptor(frag):
                continue
            candidates.append(frag)

    if candidates:
        candidates = dedupe_keep_order(candidates)
        candidates.sort(key=lambda x: (-short_phrase_score(x, lemma, original_lemma), len(x)))
        top = []
        for cand in candidates:
            top.append(cand)
            if len(top) >= 4:
                break
        return top

    quoted = dedupe_keep_order(normalize_translation_item(x) for x in re.findall(r'"([^"]+)"', gloss))
    return [x for x in quoted if x and not contains_spanish_echo(x, lemma, original_lemma)]


def filtered_direct_translations(sense: dict, lemma: str, original_lemma: str, max_items: int = 5) -> list[str]:
    out = []
    for item in extract_english_translations(sense, max_items=max_items * 2):
        item = normalize_translation_item(item)
        if not item:
            continue
        if contains_spanish_echo(item, lemma, original_lemma):
            continue
        out.extend(split_simple_list(item))
    out = dedupe_keep_order(out)
    out.sort(key=lambda x: (-short_phrase_score(x, lemma, original_lemma), len(x)))
    return out[:max_items]


def compact_nonverb_glosses(glosses: list[str], lemma: str, original_lemma: str, pos: str, max_items: int = 5) -> list[str]:
    out = []
    for gloss in glosses:
        out.extend(compact_gloss_candidates(gloss, lemma, original_lemma, pos))
    out = dedupe_keep_order(out)
    out.sort(key=lambda x: (-short_phrase_score(x, lemma, original_lemma), len(x)))
    return out[:max_items]


def parse_nonverb_features(sense: dict) -> dict:
    tags = get_tags(sense)
    glosses = get_glosses(sense)
    text = " ".join(sorted(tags)).lower() + " " + " ".join(glosses).lower()
    return {
        "plural": "plural" in text,
    }


def adjust_base_translation_for_nonverb(base: str, sense: dict) -> str:
    feat = parse_nonverb_features(sense)
    items = [normalize_translation_item(x) for x in base.split(";")]
    out = []
    for item in items:
        low = item.lower()
        if feat["plural"]:
            if low == "this":
                item = "these"
            elif low == "that":
                item = "those"
            elif low in {"a", "an"}:
                item = "some"
            elif low == "one":
                item = "some"
        out.append(item)
    return "; ".join(dedupe_keep_order(out))


def compact_join(items: list[str], max_items: int | None = None) -> str:
    items = dedupe_keep_order([normalize_translation_item(x) for x in items])
    if max_items is not None:
        items = items[:max_items]
    return "; ".join(items)


def compact_base_verb_translation(text: str, lemma: str, original_lemma: str = "") -> str:
    parts = []
    for frag in split_simple_list(text):
        frag = normalize_translation_item(frag)
        if not frag:
            continue
        if contains_spanish_echo(frag, lemma, original_lemma):
            continue
        if frag.lower().startswith("to "):
            parts.append(frag)
        else:
            parts.append("to " + frag)
    return compact_join(parts, max_items=3)


def looks_like_form_of(gloss: str) -> bool:
    g = gloss.lower()
    patterns = [
        "inflection of ",
        "form of ",
        "past participle of ",
        "present participle of ",
        "gerund of ",
        "alternative form of ",
        "alternative spelling of ",
        "plural of ",
        "feminine singular of ",
        "masculine singular of ",
        "feminine plural of ",
        "masculine plural of ",
        "third-person singular",
        "third-person plural",
        "second-person singular",
        "second-person plural",
        "first-person singular",
        "first-person plural",
    ]
    return any(p in g for p in patterns)


def has_form_link(sense: dict) -> bool:
    return bool(get_original_lemma_from_sense(sense))


def is_meta_gloss(gloss: str) -> bool:
    g = gloss.lower()
    bad_patterns = [
        "name of the latin script letter",
        "name of the letter",
        "name of letter",
        "freud",
        "greek letter",
        "latin script",
        "used to answer the telephone",
        "letter ",
    ]
    return any(p in g for p in bad_patterns)


def sense_rank(sense: dict) -> tuple[int, int]:
    tags = get_tags(sense)
    banned = 1 if tags & BANNED_SENSE_TAGS else 0
    low_priority_count = len(tags & LOW_PRIORITY_SENSE_TAGS)
    return (banned, low_priority_count)



def sense_score(lemma: str, pos: str, rank: int | None, sense: dict) -> float:
    tags = get_tags(sense)
    banned, low_priority_count = sense_rank(sense)
    if banned:
        return -1e9

    original_lemma = get_original_lemma_from_sense(sense) or lemma
    glosses = get_glosses(sense)
    translations = filtered_direct_translations(sense, lemma, original_lemma, max_items=3)
    compact_glosses = compact_nonverb_glosses(glosses, lemma, original_lemma, pos, max_items=3)

    score = 0.0

    if translations:
        score += 50.0
    if compact_glosses:
        score += 18.0
    elif glosses:
        score += 4.0
    else:
        score -= 20.0

    score -= 6.0 * low_priority_count

    if glosses and all(is_meta_gloss(g) for g in glosses):
        score -= 40.0

    form_link = has_form_link(sense)
    formy_gloss = bool(glosses and any(looks_like_form_of(g) for g in glosses))

    if pos == "verb":
        if form_link or formy_gloss:
            score += 8.0
    else:
        if form_link:
            score -= 6.0
        if formy_gloss and not compact_glosses:
            score -= 8.0

    if rank is not None:
        if rank <= 120:
            if pos in CLOSED_CLASS_POS:
                score += 90.0
            elif pos == "verb" and form_link:
                score -= 22.0
            elif pos in {"noun", "proper noun", "adjective"} and lemma.islower():
                score -= 30.0
        elif rank <= 400:
            if pos in CLOSED_CLASS_POS:
                score += 55.0
            elif pos == "verb" and form_link:
                score -= 15.0
            elif pos in {"noun", "proper noun"} and lemma.islower():
                score -= 18.0
        elif rank <= 1000:
            if pos in CLOSED_CLASS_POS:
                score += 25.0
            elif pos == "verb" and form_link:
                score -= 8.0
            elif pos in {"noun", "proper noun"} and lemma.islower():
                score -= 8.0

    return score



def choose_best_candidate(lemma: str, entries: list[dict], rank: int | None):
    best = None

    for entry in entries:
        pos = norm_space(entry.get("pos", "")).lower()
        if not pos or pos in SKIP_POS:
            continue

        senses = entry.get("senses") or []
        if not isinstance(senses, list):
            continue

        for sense in senses:
            if not isinstance(sense, dict):
                continue

            score = POS_PRIORITY.get(pos, 50) + sense_score(lemma, pos, rank, sense)
            if score <= -1e8:
                continue

            glosses = get_glosses(sense)
            translations = extract_english_translations(sense, max_items=5)
            original_lemma = get_original_lemma_from_sense(sense) or lemma

            candidate = {
                "lemma": lemma,
                "original_lemma": original_lemma,
                "pos": pos,
                "translations": translations,
                "glosses": glosses,
                "score": score,
                "sense": sense,
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return best


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def load_input_rows(path: Path, limit: int | None) -> list[dict]:
    total_rows = count_csv_rows(path)
    if limit is not None:
        total_rows = min(total_rows, limit)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "lemma" not in reader.fieldnames or "rank" not in reader.fieldnames:
            raise ValueError("Input CSV must contain headers: lemma,rank")

        rows = []
        for i, row in enumerate(tqdm(reader, total=total_rows, desc="Loading target lemmas", unit="row")):
            if limit is not None and i >= limit:
                break

            lemma = norm_space(row.get("lemma"))
            rank_raw = norm_space(row.get("rank"))
            rank = int(rank_raw) if rank_raw.isdigit() else None

            if lemma:
                rows.append({"lemma": lemma, "rank": rank})

        return rows

def open_bytes(path: Path):
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, "rb")
    return path.open("rb")


def index_entries(
    wiktextract_path: Path,
    wanted_words: set[str],
    lang_code: str,
    desc: str,
    expand_refs: bool = False,
) -> dict[str, list[dict]]:
    index = defaultdict(list)
    total_bytes = None if wiktextract_path.suffix == ".gz" else wiktextract_path.stat().st_size
    matched = 0
    lang_code = lang_code.lower()

    with open_bytes(wiktextract_path) as f:
        with tqdm(
            total=total_bytes,
            desc=desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for line in f:
                if total_bytes is not None:
                    pbar.update(len(line))

                line = line.strip()
                if not line:
                    continue

                try:
                    obj = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                if not isinstance(obj, dict):
                    continue

                obj_lang = obj.get("lang_code")
                if not isinstance(obj_lang, str) or obj_lang.lower() != lang_code:
                    continue

                word = obj.get("word")
                if not isinstance(word, str) or word not in wanted_words:
                    continue

                index[word].append(obj)
                matched += 1

                if expand_refs:
                    senses = obj.get("senses")
                    if isinstance(senses, list):
                        for sense in senses:
                            if not isinstance(sense, dict):
                                continue
                            linked = get_original_lemma_from_sense(sense)
                            if linked:
                                wanted_words.add(linked)

                if matched % 1000 == 0:
                    pbar.set_postfix(matches=matched, lemmas=len(index))

    return index

def collect_referenced_lemmas(grouped: dict[str, list[dict]]) -> set[str]:
    refs = set()
    for entries in grouped.values():
        for entry in entries:
            for sense in entry.get("senses") or []:
                linked = get_original_lemma_from_sense(sense)
                if linked:
                    refs.add(linked)
    return refs


def normalize_english_verb_base(text: str) -> str:
    text = norm_space(text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.split(r"[;,/]", text)[0]
    text = norm_space(text)
    if text.lower().startswith("to "):
        text = text[3:]
    return norm_space(text).lower()


def split_head_tail(base: str) -> tuple[str, str]:
    parts = base.split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def join_head_tail(head: str, tail: str) -> str:
    return f"{head} {tail}".strip()


def third_person_s(base: str) -> str:
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ies"
    if base.endswith(("o", "ch", "sh", "s", "x", "z")):
        return base + "es"
    return base + "s"


def regular_past(base: str) -> str:
    if base.endswith("e"):
        return base + "d"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ied"
    return base + "ed"


def regular_ing(base: str) -> str:
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
        return base[:-1] + "ing"
    return base + "ing"


IRREGULAR_ENGLISH_FORMS = {
    "be": {
        ("present", "1", "sg"): "am",
        ("present", "2", "sg"): "are",
        ("present", "3", "sg"): "is",
        ("present", "1", "pl"): "are",
        ("present", "2", "pl"): "are",
        ("present", "3", "pl"): "are",
        ("preterite", "1", "sg"): "was",
        ("preterite", "2", "sg"): "were",
        ("preterite", "3", "sg"): "was",
        ("preterite", "1", "pl"): "were",
        ("preterite", "2", "pl"): "were",
        ("preterite", "3", "pl"): "were",
        ("imperfect", "1", "sg"): "was",
        ("imperfect", "2", "sg"): "were",
        ("imperfect", "3", "sg"): "was",
        ("imperfect", "1", "pl"): "were",
        ("imperfect", "2", "pl"): "were",
        ("imperfect", "3", "pl"): "were",
        ("participle", None, None): "been",
        ("gerund", None, None): "being",
    },
    "have": {
        ("present", "3", "sg"): "has",
        ("preterite", None, None): "had",
        ("imperfect", None, None): "had",
        ("participle", None, None): "had",
        ("gerund", None, None): "having",
    },
    "do": {
        ("present", "3", "sg"): "does",
        ("preterite", None, None): "did",
        ("imperfect", None, None): "did",
        ("participle", None, None): "done",
        ("gerund", None, None): "doing",
    },
    "go": {
        ("present", "3", "sg"): "goes",
        ("preterite", None, None): "went",
        ("imperfect", None, None): "went",
        ("participle", None, None): "gone",
        ("gerund", None, None): "going",
    },
    "say": {
        ("present", "3", "sg"): "says",
        ("preterite", None, None): "said",
        ("imperfect", None, None): "said",
        ("participle", None, None): "said",
        ("gerund", None, None): "saying",
    },
    "see": {
        ("present", "3", "sg"): "sees",
        ("preterite", None, None): "saw",
        ("imperfect", None, None): "saw",
        ("participle", None, None): "seen",
        ("gerund", None, None): "seeing",
    },
    "know": {
        ("present", "3", "sg"): "knows",
        ("preterite", None, None): "knew",
        ("imperfect", None, None): "knew",
        ("participle", None, None): "known",
        ("gerund", None, None): "knowing",
    },
    "give": {
        ("present", "3", "sg"): "gives",
        ("preterite", None, None): "gave",
        ("imperfect", None, None): "gave",
        ("participle", None, None): "given",
        ("gerund", None, None): "giving",
    },
    "come": {
        ("present", "3", "sg"): "comes",
        ("preterite", None, None): "came",
        ("imperfect", None, None): "came",
        ("participle", None, None): "come",
        ("gerund", None, None): "coming",
    },
    "put": {
        ("present", "3", "sg"): "puts",
        ("preterite", None, None): "put",
        ("imperfect", None, None): "put",
        ("participle", None, None): "put",
        ("gerund", None, None): "putting",
    },
    "feel": {
        ("present", "3", "sg"): "feels",
        ("preterite", None, None): "felt",
        ("imperfect", None, None): "felt",
        ("participle", None, None): "felt",
        ("gerund", None, None): "feeling",
    },
    "leave": {
        ("present", "3", "sg"): "leaves",
        ("preterite", None, None): "left",
        ("imperfect", None, None): "left",
        ("participle", None, None): "left",
        ("gerund", None, None): "leaving",
    },
}


def pronouns_for(person: str | None, number: str | None) -> list[str]:
    if person == "1" and number == "sg":
        return ["I"]
    if person == "2" and number == "sg":
        return ["you"]
    if person == "3" and number == "sg":
        return ["he", "she", "it"]
    if person == "1" and number == "pl":
        return ["we"]
    if person == "2" and number == "pl":
        return ["you"]
    if person == "3" and number == "pl":
        return ["they"]
    return []


def parse_verb_features(sense: dict) -> dict:
    tags = get_tags(sense)
    glosses = get_glosses(sense)
    text = " ".join(sorted(tags)).lower() + " " + " ".join(glosses).lower()

    person = None
    if "first-person" in text:
        person = "1"
    elif "second-person" in text:
        person = "2"
    elif "third-person" in text:
        person = "3"

    number = None
    if "singular" in text:
        number = "sg"
    elif "plural" in text:
        number = "pl"

    form = "finite"
    if "past participle" in text:
        form = "participle"
    elif "gerund" in text or "present participle" in text:
        form = "gerund"
    elif "infinitive" in text:
        form = "infinitive"

    tense = None
    if "preterite" in text:
        tense = "preterite"
    elif "imperfect" in text:
        tense = "imperfect"
    elif "future" in text:
        tense = "future"
    elif "conditional" in text:
        tense = "conditional"
    elif "present" in text:
        tense = "present"

    return {
        "person": person,
        "number": number,
        "form": form,
        "tense": tense,
    }


def english_head_nonfinite(head: str, form: str) -> str:
    irregular = IRREGULAR_ENGLISH_FORMS.get(head, {})
    if form == "participle":
        return irregular.get(("participle", None, None), regular_past(head))
    if form == "gerund":
        return irregular.get(("gerund", None, None), regular_ing(head))
    return f"to {head}"


def english_head_finite(head: str, tense: str | None, person: str | None, number: str | None) -> str:
    tense = tense or "present"

    if tense == "future":
        return f"will {head}"
    if tense == "conditional":
        return f"would {head}"

    irregular = IRREGULAR_ENGLISH_FORMS.get(head, {})
    exact = irregular.get((tense, person, number))
    if exact:
        return exact

    generic = irregular.get((tense, None, None))
    if generic:
        return generic

    if tense in {"preterite", "imperfect"}:
        return regular_past(head)

    if tense == "present" and person == "3" and number == "sg":
        return third_person_s(head)

    return head


def choose_base_verb_candidate(entries: list[dict]):
    best = None

    for entry in entries:
        pos = norm_space(entry.get("pos", "")).lower()
        if pos != "verb":
            continue

        for sense in entry.get("senses") or []:
            if not isinstance(sense, dict):
                continue
            if get_tags(sense) & BANNED_SENSE_TAGS:
                continue
            if has_form_link(sense):
                continue

            translations = extract_english_translations(sense, max_items=5)
            glosses = get_glosses(sense)

            score = 0.0
            if translations:
                score += 100.0
            if glosses:
                score += 20.0
            if get_tags(sense) & LOW_PRIORITY_SENSE_TAGS:
                score -= 15.0

            candidate = {
                "translations": translations,
                "glosses": glosses,
                "score": score,
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return best


def infer_base_english_verb(original_lemma: str, grouped: dict[str, list[dict]]) -> str:
    base_candidate = choose_base_verb_candidate(grouped.get(original_lemma, []))
    if base_candidate is None:
        return ""

    raw = ""
    if base_candidate["translations"]:
        raw = base_candidate["translations"][0]
    elif base_candidate["glosses"]:
        raw = base_candidate["glosses"][0]

    return normalize_english_verb_base(raw)



def build_literal_verb_translation(candidate: dict, grouped: dict[str, list[dict]]) -> str:
    sense = candidate.get("sense") or {}
    lemma = candidate.get("lemma") or ""
    original_lemma = candidate.get("original_lemma") or lemma

    features = parse_verb_features(sense)
    base = infer_base_english_verb(original_lemma, grouped)

    if features["form"] == "infinitive":
        if base:
            return compact_base_verb_translation(base, lemma, original_lemma)
        direct = filtered_direct_translations(sense, lemma, original_lemma, max_items=3)
        if direct:
            return compact_base_verb_translation(direct[0], lemma, original_lemma)
        return ""

    if features["form"] in {"participle", "gerund"}:
        if base:
            head, tail = split_head_tail(base)
            return join_head_tail(english_head_nonfinite(head, features["form"]), tail)
        direct = filtered_direct_translations(sense, lemma, original_lemma, max_items=3)
        if direct:
            return direct[0]
        return ""

    if base and features["person"] and features["number"]:
        head, tail = split_head_tail(base)
        finite = join_head_tail(english_head_finite(head, features["tense"], features["person"], features["number"]), tail)
        pronouns = pronouns_for(features["person"], features["number"])
        if pronouns:
            return "; ".join(f"{p} {finite}" for p in pronouns)
        return finite

    if base:
        return compact_base_verb_translation(base, lemma, original_lemma)

    direct = filtered_direct_translations(sense, lemma, original_lemma, max_items=3)
    if direct:
        return compact_base_verb_translation(direct[0], lemma, original_lemma)

    return ""


def build_translation(candidate: dict, grouped: dict[str, list[dict]], _seen: set[str] | None = None) -> str:
    if candidate is None:
        return ""

    lemma = candidate.get("lemma") or ""
    original_lemma = candidate.get("original_lemma") or lemma
    pos = candidate.get("pos") or ""
    sense = candidate.get("sense") or {}

    if lemma in FLASHCARD_OVERRIDES:
        return FLASHCARD_OVERRIDES[lemma][0]

    if _seen is None:
        _seen = set()
    guard_key = f"{lemma}|{pos}"
    if guard_key in _seen:
        return ""
    _seen = set(_seen)
    _seen.add(guard_key)

    if pos == "verb":
        translation = build_literal_verb_translation(candidate, grouped)
        if translation and not contains_spanish_echo(translation, lemma, original_lemma):
            return translation

    direct = filtered_direct_translations(sense, lemma, original_lemma, max_items=5)
    if direct:
        if pos == "verb":
            joined = compact_base_verb_translation(direct[0], lemma, original_lemma)
        else:
            joined = compact_join(direct, max_items=4)
        if joined and not contains_spanish_echo(joined, lemma, original_lemma):
            return joined

    glosses = compact_nonverb_glosses(candidate.get("glosses") or [], lemma, original_lemma, pos, max_items=5)
    if glosses:
        joined = compact_join(glosses, max_items=4)
        if joined and not contains_spanish_echo(joined, lemma, original_lemma):
            return joined

    if pos in FORM_LIKE_POS and original_lemma and original_lemma != lemma:
        base_candidate = choose_best_candidate(original_lemma, grouped.get(original_lemma, []), None)
        base_translation = build_translation(base_candidate, grouped, _seen)
        if base_translation:
            adjusted = adjust_base_translation_for_nonverb(base_translation, sense)
            if adjusted and not contains_spanish_echo(adjusted, lemma, original_lemma):
                return adjusted

    return ""



def write_output(path: Path, rows: list[dict], grouped: dict[str, list[dict]]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lemma", "rank", "translation", "pos"],
        )
        writer.writeheader()

        for row in tqdm(rows, total=len(rows), desc="Building rows", unit="lemma"):
            lemma = row["lemma"]
            rank = row["rank"]
            candidate = choose_best_candidate(lemma, grouped.get(lemma, []), rank)

            translation = ""
            pos = ""

            if candidate is not None:
                pos = candidate["pos"]
                if pos == "verb":
                    translation = build_literal_verb_translation(candidate, grouped)
                if not translation:
                    if candidate["translations"]:
                        translation = "; ".join(candidate["translations"])
                    elif candidate["glosses"]:
                        translation = candidate["glosses"][0]

            writer.writerow({
                "lemma": lemma,
                "rank": "" if rank is None else rank,
                "translation": translation,
                "pos": pos,
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--wiktextract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lang-code", default="es")
    parser.add_argument("--limit", type=int, default=39761)
    args = parser.parse_args()

    rows = load_input_rows(args.input, args.limit)
    wanted_words = {row["lemma"] for row in rows}

    print(f"Loaded {len(rows)} input rows")
    print(f"Looking up {len(wanted_words)} unique lemmas")

    grouped = index_entries(
        wiktextract_path=args.wiktextract,
        wanted_words=wanted_words,
        lang_code=args.lang_code,
        desc="Scanning target and referenced lemmas",
        expand_refs=True,
    )

    referenced_lemmas = collect_referenced_lemmas(grouped)
    missing_refs = referenced_lemmas - set(grouped.keys())

    if missing_refs:
        extra_grouped = index_entries(
            wiktextract_path=args.wiktextract,
            wanted_words=missing_refs,
            lang_code=args.lang_code,
            desc="Scanning missing referenced lemmas",
            expand_refs=False,
        )
        for lemma, entries in extra_grouped.items():
            grouped[lemma].extend(entries)

    write_output(args.output, rows, grouped)

    found = sum(1 for row in rows if row["lemma"] in grouped)
    total = len(rows)
    print(f"Done: found {found}/{total} lemmas")


if __name__ == "__main__":
    main()