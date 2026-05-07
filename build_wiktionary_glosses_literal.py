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

    glosses = get_glosses(sense)
    translations = extract_english_translations(sense, max_items=3)

    score = 0.0

    if translations:
        score += 40.0
    if glosses:
        score += 12.0
    else:
        score -= 20.0

    score -= 6.0 * low_priority_count

    if pos != "verb" and has_form_link(sense):
        score -= 10.0

    if glosses and all(is_meta_gloss(g) for g in glosses):
        score -= 40.0

    if glosses and any(looks_like_form_of(g) for g in glosses):
        if pos == "verb":
            score += 5.0
        else:
            score -= 12.0

    if rank is not None and rank <= 250:
        if pos in FUNCTION_WORD_POS:
            score += 10.0
        if pos in {"noun", "proper noun"} and lemma.islower():
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
    original_lemma = candidate.get("original_lemma") or candidate.get("lemma") or ""

    base = infer_base_english_verb(original_lemma, grouped)
    features = parse_verb_features(sense)

    can_build_literal = bool(base) and (
        features["form"] in {"participle", "gerund", "infinitive"}
        or (
            features["form"] == "finite"
            and features["person"] is not None
            and features["number"] is not None
        )
    )

    if can_build_literal:
        head, tail = split_head_tail(base)

        if features["form"] != "finite":
            return join_head_tail(english_head_nonfinite(head, features["form"]), tail)

        finite = join_head_tail(
            english_head_finite(head, features["tense"], features["person"], features["number"]),
            tail,
        )
        pronouns = pronouns_for(features["person"], features["number"])

        if pronouns:
            return "; ".join(f"{p} {finite}" for p in pronouns)
        return finite

    direct = extract_english_translations(sense, max_items=5)
    if direct:
        return "; ".join(direct)

    glosses = get_glosses(sense)
    if glosses:
        return glosses[0]

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
    parser.add_argument("--limit", type=int, default=39760)
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