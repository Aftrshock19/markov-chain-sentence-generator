#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

LEMMA_OVERRIDES = {
    "el": "the (masc.)",
    "la": "the (fem.)",
    "los": "the (pl., masc.)",
    "las": "the (pl., fem.)",
    "un": "a/an (masc.)",
    "una": "a/an (fem.)",
    "unos": "some (masc.)",
    "unas": "some (fem.)",
    "y": "and",
    "e": "and",
    "o": "or",
    "u": "or",
    "pero": "but",
    "aunque": "although",
    "si": "if",
    "sí": "yes",
    "no": "no; not",
    "muy": "very",
    "más": "more",
    "menos": "less",
    "ya": "already",
    "aún": "still; yet",
    "también": "also",
    "solo": "only",
    "sólo": "only",
    "a": "to; at",
    "de": "of; from",
    "del": "of the; from the",
    "al": "to the",
    "en": "in; on; at",
    "con": "with",
    "sin": "without",
    "sobre": "about; on",
    "entre": "between; among",
    "para": "for; to",
    "por": "by; for",
    "como": "like; as",
    "cuando": "when",
    "donde": "where",
    "qué": "what",
    "que": "that; which",
    "quien": "who",
    "cuál": "which",
    "cual": "which",
    "porque": "because",
    "porqué": "reason; motive",
    "desde": "from; since",
    "hasta": "until; up to",
    "ser": "to be",
    "estar": "to be",
    "haber": "to have; there is; there are",
    "tener": "to have",
    "hacer": "to do; to make",
    "decir": "to say; to tell",
    "ir": "to go",
    "ver": "to see",
    "dar": "to give",
    "saber": "to know",
    "conocer": "to know; to be familiar with",
    "poder": "can; to be able to",
    "querer": "to want",
    "venir": "to come",
    "poner": "to put",
    "salir": "to leave; to go out",
    "llegar": "to arrive; to get",
    "sentir": "to feel",
    "parecer": "to seem",
    "creer": "to believe; to think",
    "lo": "it; him",
    "se": "oneself; himself; herself; itself; themselves",
    "me": "me; myself",
    "te": "you; yourself",
    "su": "his; her; its; their; your (formal)",
    "mi": "my",
    "yo": "I",
    "le": "him; her; you (formal)",
    "aquí": "here",
    "tu": "your",
    "todo": "everything; all",
    "esto": "this",
    "esta": "this",
    "ahora": "now",
    "así": "like this; so",
    "hay": "there is; there are",
    "este": "this",
    "algo": "something",
    "él": "he; him",
    "bueno": "good; well",
    "nos": "us; ourselves",
    "nosotros": "we; us",
    "vosotros": "you all",
    "vosotras": "you all",
    "usted": "you (formal)",
    "ustedes": "you all",
    "conmigo": "with me",
    "contigo": "with you",
    "sus": "his; her; its; their; your (formal)",
    "nada": "nothing",
    "tú": "you",
    "vez": "time; occasion",
    "ella": "she; her",
    "todos": "everyone; all",
    "gracias": "thanks",
    "dos": "two",
    "tan": "so; as",
    "entonces": "then; so",
    "tiempo": "time",
    "bien": "well",
    "eso": "that",
    "vale": "okay; fine",
}

SURFACE_OVERRIDES = {
    "es": "he/she/it is",
    "está": "he/she/it is",
    "estoy": "I am",
    "estás": "you are",
    "estamos": "we are",
    "están": "they are",
    "ha": "has",
    "he": "I have",
    "has": "you have",
    "han": "they have",
    "hemos": "we have",
    "hay": "there is; there are",
    "soy": "I am",
    "eres": "you are",
    "somos": "we are",
    "son": "they are",
    "fue": "he/she/it was; he/she/it went",
    "fui": "I was; I went",
    "fuiste": "you were; you went",
    "fuimos": "we were; we went",
    "fueron": "they were; they went",
    "era": "he/she/it was; it used to be",
    "eran": "they were; they used to be",
    "será": "he/she/it will be",
    "tengo": "I have",
    "tiene": "he/she/it has",
    "tienes": "you have",
    "tenemos": "we have",
    "tenía": "he/she/it had; he/she/it used to have",
    "tuvo": "he/she/it had",
    "tuve": "I had",
    "puedo": "I can",
    "puede": "he/she/it can; you can",
    "puedes": "you can",
    "podemos": "we can",
    "quiero": "I want",
    "quiere": "he/she/it wants; you want",
    "quieres": "you want",
    "queremos": "we want",
    "hizo": "he/she/it did; he/she/it made",
    "hicieron": "they did; they made",
    "hiciste": "you did; you made",
    "hice": "I did; I made",
    "vamos": "we go; let's go",
    "voy": "I go",
    "va": "he/she/it goes",
    "vas": "you go",
    "van": "they go",
    "sé": "I know",
    "sabes": "you know",
    "sabe": "he/she/it knows",
    "sabemos": "we know",
    "dijo": "he/she/it said",
    "dijeron": "they said",
    "dije": "I said",
    "dice": "he/she/it says",
    "dices": "you say",
    "hacen": "they do; they make",
    "hago": "I do; I make",
    "haces": "you do; you make",
    "hacemos": "we do; we make",
    "estaba": "he/she/it was",
    "estaban": "they were",
    "estuve": "I was",
    "estuvo": "he/she/it was",
    "siendo": "being",
    "hecho": "done; made",
    "hecha": "done; made",
    "hechos": "done; made",
    "hechas": "done; made",
    "haciendo": "doing; making",
    "vale": "okay; fine",
}

EDGE_PUNCT = " \t\r\n\"'“”‘’`´¡!¿?.,;:()[]{}<>…—–-_/\\|@#$%^&*+=~"
MULTI_TRANSLATIONS = True
DEFAULT_APERTIUM_ANALYZER_CMD = ["apertium", "-d", os.path.expanduser("~/apertium-spa"), "spa-morph"]
PRONOUNS = {("1", "sg"): "I", ("2", "sg"): "you", ("3", "sg"): "he/she/it", ("1", "pl"): "we", ("2", "pl"): "you all", ("3", "pl"): "they"}

EN_IRREGULAR = {
    "be": {"3sg": "is", "past": "was", "past_pl": "were", "pp": "been", "ger": "being"},
    "have": {"3sg": "has", "past": "had", "pp": "had", "ger": "having"},
    "do": {"3sg": "does", "past": "did", "pp": "done", "ger": "doing"},
    "make": {"3sg": "makes", "past": "made", "pp": "made", "ger": "making"},
    "say": {"3sg": "says", "past": "said", "pp": "said", "ger": "saying"},
    "tell": {"3sg": "tells", "past": "told", "pp": "told", "ger": "telling"},
    "go": {"3sg": "goes", "past": "went", "pp": "gone", "ger": "going"},
    "see": {"3sg": "sees", "past": "saw", "pp": "seen", "ger": "seeing"},
    "give": {"3sg": "gives", "past": "gave", "pp": "given", "ger": "giving"},
    "know": {"3sg": "knows", "past": "knew", "pp": "known", "ger": "knowing"},
    "come": {"3sg": "comes", "past": "came", "pp": "come", "ger": "coming"},
    "put": {"3sg": "puts", "past": "put", "pp": "put", "ger": "putting"},
    "feel": {"3sg": "feels", "past": "felt", "pp": "felt", "ger": "feeling"},
    "leave": {"3sg": "leaves", "past": "left", "pp": "left", "ger": "leaving"},
    "arrive": {"3sg": "arrives", "past": "arrived", "pp": "arrived", "ger": "arriving"},
    "get": {"3sg": "gets", "past": "got", "pp": "gotten", "ger": "getting"},
    "want": {"3sg": "wants", "past": "wanted", "pp": "wanted", "ger": "wanting"},
    "think": {"3sg": "thinks", "past": "thought", "pp": "thought", "ger": "thinking"},
    "can": {"3sg": "can", "past": "could", "pp": "been able to", "ger": "being able to"},
}

SPANISH_SPECIAL_VERBS = {"ser", "estar", "haber", "tener", "hacer", "decir", "ir", "ver", "dar", "saber", "conocer", "poder", "querer", "venir", "poner", "salir", "llegar", "sentir", "parecer", "creer"}

POS_MAP = {
    "v": "verb", "n": "noun", "adj": "adj", "adv": "adv", "prop": "noun",
    "pron": "pron", "prn": "pron", "det": "det", "art": "det", "prep": "prep",
    "conj": "conj", "interj": "interj", "intj": "interj",
}


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_token(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = s.strip(EDGE_PUNCT)
    s = s.lower()
    return normalize_spaces(s)


def normalize_gloss(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    return normalize_spaces(s).lower()


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def safe_int(x) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 0


def local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def dix_node_text(node: ET.Element) -> str:
    parts: List[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        tag = local_tag(child.tag)
        if tag == "b":
            parts.append(" ")
        elif tag in {"s", "a", "j", "par", "re"}:
            pass
        else:
            parts.append(dix_node_text(child))
        if child.tail:
            parts.append(child.tail)
    return normalize_spaces("".join(parts))


def parse_analysis_candidates(token_analysis: str, fallback: str) -> List[Dict[str, object]]:
    if not token_analysis or "<sent>" in token_analysis:
        return [{"lemma": normalize_token(fallback), "tags": [], "raw": token_analysis}]
    body = token_analysis.strip()
    if body.startswith("^"):
        body = body[1:]
    if body.endswith("$"):
        body = body[:-1]
    parts = body.split("/")
    if len(parts) < 2:
        return [{"lemma": normalize_token(fallback), "tags": [], "raw": token_analysis}]
    analyses = []
    for ana in parts[1:]:
        if not ana:
            continue
        m = re.match(r"([^<$/]+)(.*)", ana)
        if not m:
            continue
        lemma = normalize_token(m.group(1) or fallback)
        tags = [t.lower() for t in re.findall(r"<([^>]+)>", ana)]
        analyses.append({"lemma": lemma or normalize_token(fallback), "tags": tags, "raw": ana})
    if not analyses:
        analyses.append({"lemma": normalize_token(fallback), "tags": [], "raw": token_analysis})
    return analyses


def choose_best_analysis(surface: str, analyses: List[Dict[str, object]]) -> Dict[str, object]:
    normalized_surface = normalize_token(surface)

    def score(item: Dict[str, object]) -> float:
        lemma = str(item.get("lemma", ""))
        tags = set(item.get("tags", []))
        value = 0.0
        if lemma in LEMMA_OVERRIDES:
            value += 4.0
        if lemma in SPANISH_SPECIAL_VERBS:
            value += 5.0
        if any(t.startswith("vb") or t == "v" for t in tags):
            value += 3.0
        if lemma == normalized_surface:
            value += 0.5
        if lemma in {"ser", "ir"} and normalized_surface in SURFACE_OVERRIDES:
            value += 3.0
        if "n" in tags and normalized_surface in {"a", "o", "s", "i", "m"}:
            value -= 3.0
        return value

    return max(analyses, key=score)


def analyze_surface_forms(surface_forms: Sequence[str], analyzer_cmd: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, object]]:
    cleaned = [normalize_token(x) for x in surface_forms if normalize_token(x)]
    if not cleaned:
        return {}
    cmd = list(analyzer_cmd or DEFAULT_APERTIUM_ANALYZER_CMD)
    if shutil.which(cmd[0]) is None:
        raise FileNotFoundError(f"Apertium analyzer command not found: {cmd[0]}")
    text = " ".join(cleaned) + "\n"
    proc = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError("spa-morph failed:\n" + proc.stderr.decode("utf-8", errors="replace"))
    output = proc.stdout.decode("utf-8", errors="replace")
    raw_chunks = re.findall(r"\^.*?\$", output)
    chunks = [chunk for chunk in raw_chunks if "<sent>" not in chunk]
    if len(chunks) < len(cleaned):
        raise RuntimeError(f"spa-morph token alignment mismatch: got {len(chunks)} analyses for {len(cleaned)} inputs")
    mapping: Dict[str, Dict[str, object]] = {}
    for surf, chunk in zip(cleaned, chunks):
        candidates = parse_analysis_candidates(chunk, surf)
        best = choose_best_analysis(surf, candidates)
        mapping[surf] = {"surface": surf, "lemma": best.get("lemma", surf), "tags": list(best.get("tags", [])), "raw": chunk, "candidates": candidates}
    for surf in cleaned:
        mapping.setdefault(surf, {"surface": surf, "lemma": surf, "tags": [], "raw": "", "candidates": []})
    return mapping


def infer_column(fieldnames: Sequence[str]) -> str:
    preferred = ["lemma", "spanish", "word", "token", "surface"]
    lowered = {name.lower(): name for name in fieldnames}
    for cand in preferred:
        if cand in lowered:
            return lowered[cand]
    raise ValueError(f"Could not find a source column. Available columns: {list(fieldnames)}")


def infer_reverse_from_filename(dict_path: str) -> bool:
    name = os.path.basename(dict_path).lower()
    if "eng-spa" in name or "en-es" in name:
        return True
    if "spa-eng" in name or "es-en" in name:
        return False
    return False


def allowed_entry(direction_attr: str, reverse: bool) -> bool:
    direction_attr = (direction_attr or "").strip().upper()
    if not direction_attr:
        return True
    return direction_attr == ("RL" if reverse else "LR")


def add_mapping(mapping: DefaultDict[str, List[str]], src: str, tgt: str, max_alts_per_word: int) -> None:
    if src and tgt and tgt not in mapping[src] and len(mapping[src]) < max_alts_per_word:
        mapping[src].append(tgt)


def load_dictionary(dict_path: str, max_alts_per_word: int = 3, reverse: Optional[bool] = None) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    bilingual: DefaultDict[str, List[str]] = defaultdict(list)
    accent_index: DefaultDict[str, List[str]] = defaultdict(list)
    if reverse is None:
        reverse = infer_reverse_from_filename(dict_path)
    if dict_path.lower().endswith(".dix"):
        root = ET.parse(dict_path).getroot()
        for entry in root.findall(".//e"):
            direction = entry.attrib.get("r", "")
            for pair in entry.findall("p"):
                left_node = pair.find("l")
                right_node = pair.find("r")
                if left_node is None or right_node is None:
                    continue
                if not allowed_entry(direction, reverse):
                    continue
                if reverse:
                    src = normalize_token(dix_node_text(right_node))
                    tgt = normalize_gloss(dix_node_text(left_node))
                else:
                    src = normalize_token(dix_node_text(left_node))
                    tgt = normalize_gloss(dix_node_text(right_node))
                add_mapping(bilingual, src, tgt, max_alts_per_word)
                add_mapping(accent_index, strip_accents(src), tgt, max_alts_per_word)
            identity = entry.find("i")
            if identity is not None:
                word = normalize_token(dix_node_text(identity))
                add_mapping(bilingual, word, word, 1)
                add_mapping(accent_index, strip_accents(word), word, 1)
        return dict(bilingual), dict(accent_index)
    with open(dict_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                src_raw, tgt_raw = line.split("\t", 1)
                src = normalize_token(src_raw)
                tgt = normalize_gloss(tgt_raw)
            else:
                parts = re.split(r"\s+", line)
                if len(parts) < 2:
                    continue
                src = normalize_token(parts[0])
                tgt = normalize_gloss(" ".join(parts[1:]))
            add_mapping(bilingual, src, tgt, max_alts_per_word)
            add_mapping(accent_index, strip_accents(src), tgt, max_alts_per_word)
    return dict(bilingual), dict(accent_index)


def parse_data_dictionary(data_path: str) -> Dict[str, List[Dict[str, object]]]:
    entries: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    current_word = None
    current_pos = None
    current_glosses: List[str] = []

    def flush_entry() -> None:
        nonlocal current_word, current_pos, current_glosses
        if current_word and current_pos and current_glosses:
            entries[current_word].append({"pos": POS_MAP.get(current_pos, current_pos), "glosses": list(current_glosses)})
        current_pos = None
        current_glosses = []

    with open(data_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "_____":
                flush_entry()
                current_word = None
                continue
            if current_word is None:
                current_word = normalize_token(line)
                continue
            if line.startswith("pos: "):
                flush_entry()
                current_pos = normalize_token(line[5:])
                continue
            if line.startswith("  gloss: "):
                gloss = normalize_spaces(line[9:])
                if gloss:
                    current_glosses.append(gloss)
        flush_entry()
    return dict(entries)


def heuristic_candidates(word: str) -> List[str]:
    candidates: List[str] = []
    def add(value: str) -> None:
        value = normalize_token(value)
        if value and value not in candidates:
            candidates.append(value)
    add(word)
    if word.endswith("se"):
        add(word[:-2])
    else:
        add(word + "se")
    if word.endswith("es") and len(word) > 3:
        add(word[:-2])
    if word.endswith("s") and len(word) > 2:
        add(word[:-1])
    derivations = {
        "ado": ["ar"], "ada": ["ar"], "ados": ["ar"], "adas": ["ar"],
        "ido": ["er", "ir"], "ida": ["er", "ir"], "idos": ["er", "ir"], "idas": ["er", "ir"],
        "ando": ["ar"], "iendo": ["er", "ir"], "yendo": ["er", "ir"],
    }
    for suffix, infinitives in derivations.items():
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            stem = word[:-len(suffix)]
            for infinitive in infinitives:
                add(stem + infinitive)
                add(stem + infinitive + "se")
    return candidates


def join_alts(alts: Sequence[str]) -> str:
    if MULTI_TRANSLATIONS and len(alts) > 1:
        return "; ".join(alts)
    return alts[0]


def split_english_options(gloss: str) -> List[str]:
    pieces = re.split(r"\s*[;/]\s*", gloss)
    out: List[str] = []
    for piece in pieces:
        p = piece.strip()
        if not p:
            continue
        p = re.sub(r"\([^)]*\)", "", p).strip()
        if p:
            out.append(p)
    return out


def english_base_options(lemma_translation: str) -> List[str]:
    opts = []
    for part in split_english_options(lemma_translation):
        p = part.lower().strip()
        if p.startswith("to "):
            p = p[3:].strip()
        if p in {"there is", "there are"}:
            opts.append(p)
            continue
        if p:
            opts.append(p)
    seen = []
    for opt in opts:
        if opt not in seen:
            seen.append(opt)
    return seen


def data_gloss_is_meta(gloss: str) -> bool:
    g = normalize_gloss(gloss)
    patterns = [
        r'^abbreviation of ', r'^initialism of ', r'^letter:?\b', r'^obsolete form of ',
        r'^alternative form of ', r'^misspelling of ', r'^pronunciation spelling of ',
        r'^suffix indicating ', r'^forms ', r'^female equivalent of ', r'^male equivalent of ',
        r'^surname$', r'^given name$',
    ]
    return any(re.search(p, g) for p in patterns)


def clean_data_gloss(gloss: str) -> List[str]:
    g = normalize_spaces(gloss)
    if not g:
        return []
    if ';' in g and ('inflection of "' in g.lower() or 'form of "' in g.lower() or 'spelling of "' in g.lower()):
        tail = g.split(';', 1)[1]
        return clean_data_gloss(tail)
    if data_gloss_is_meta(g):
        return []
    parts = [normalize_spaces(p) for p in re.split(r'\s*;\s*', g) if normalize_spaces(p)]
    cleaned = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(part)
    return cleaned


def choose_data_translation(word: str, expected_pos: str, allow_low_confidence: bool, data_entries: Dict[str, List[Dict[str, object]]]) -> Tuple[str, str, bool]:
    token = normalize_token(word)
    entries = data_entries.get(token, [])
    if not entries:
        return "", "", False
    best_score = -10**9
    best_glosses: List[str] = []
    best_pos = ""
    for entry in entries:
        pos = str(entry.get("pos", ""))
        glosses_raw = entry.get("glosses", [])
        glosses: List[str] = []
        for g in glosses_raw:
            glosses.extend(clean_data_gloss(str(g)))
        if not glosses:
            continue
        score = 0.0
        if expected_pos and pos == expected_pos:
            score += 10.0
        elif expected_pos and pos in {"det", "prep", "conj", "pron", "interj"} and expected_pos in {"det", "prep", "conj", "pron", "interj"}:
            score += 2.0
        if pos == "verb":
            score += 2.0
        if pos == "interj":
            score += 1.0
        if any(x.lower() in {"ok", "okay", "fine"} for x in glosses):
            score += 2.0
        score -= len(glosses) * 0.15
        if score > best_score:
            best_score = score
            best_glosses = glosses
            best_pos = pos
    if not best_glosses:
        return "", "", False
    if best_score < 1 and not allow_low_confidence:
        return "", "", False
    return join_alts(best_glosses[:4]), f"data:{best_pos or 'unknown'}", True


def translate_lemma(raw_lemma: str, dictionary: Dict[str, List[str]], accent_dictionary: Dict[str, List[str]], data_entries: Dict[str, List[Dict[str, object]]]) -> Tuple[str, str, bool, str]:
    normalized = normalize_token(raw_lemma)
    if not normalized:
        return "[UNTRANSLATED]", normalized, False, "untranslated"
    if normalized in LEMMA_OVERRIDES:
        return LEMMA_OVERRIDES[normalized], normalized, True, "override"
    data_translation, data_source, data_ok = choose_data_translation(normalized, "", True, data_entries)
    if data_ok:
        return data_translation, normalized, True, data_source
    if normalized in dictionary:
        return join_alts(dictionary[normalized]), normalized, True, "dict"
    plain = strip_accents(normalized)
    if plain in accent_dictionary:
        return join_alts(accent_dictionary[plain]), normalized, True, "dict_unaccented"
    for candidate in heuristic_candidates(normalized)[1:]:
        if candidate in LEMMA_OVERRIDES:
            return LEMMA_OVERRIDES[candidate], candidate, True, "heuristic_override"
        data_translation, data_source, data_ok = choose_data_translation(candidate, "", False, data_entries)
        if data_ok:
            return data_translation, candidate, True, f"heuristic_{data_source}"
        if candidate in dictionary:
            return join_alts(dictionary[candidate]), candidate, True, "heuristic_dict"
        plain_candidate = strip_accents(candidate)
        if plain_candidate in accent_dictionary:
            return join_alts(accent_dictionary[plain_candidate]), candidate, True, "heuristic_dict_unaccented"
    if " " in normalized:
        out = []
        ok_any = False
        for token in normalized.split():
            translated, _, ok, _ = translate_lemma(token, dictionary, accent_dictionary, data_entries)
            out.append(translated if ok else f"[UNTRANSLATED:{token}]")
            ok_any = ok_any or ok
        return " ".join(out), normalized, ok_any, ("token_fallback" if ok_any else "untranslated")
    return f"[UNTRANSLATED] {normalized}", normalized, False, "untranslated"


def pluralize_noun(noun: str) -> str:
    irregular = {"man": "men", "woman": "women", "person": "people", "child": "children", "foot": "feet", "tooth": "teeth", "mouse": "mice"}
    if noun in irregular:
        return irregular[noun]
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    return noun + "s"


def english_3sg(base: str) -> str:
    if base in EN_IRREGULAR:
        return EN_IRREGULAR[base]["3sg"]
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ies"
    if base.endswith(("s", "sh", "ch", "x", "z", "o")):
        return base + "es"
    return base + "s"


def english_past(base: str, pronoun: str) -> str:
    if base in EN_IRREGULAR:
        if base == "be":
            return EN_IRREGULAR[base]["past_pl"] if pronoun in {"you", "we", "they", "you all"} else EN_IRREGULAR[base]["past"]
        return EN_IRREGULAR[base]["past"]
    if base.endswith("e"):
        return base + "d"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ied"
    return base + "ed"


def english_pp(base: str) -> str:
    if base in EN_IRREGULAR:
        return EN_IRREGULAR[base]["pp"]
    if base.endswith("e"):
        return base + "d"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ied"
    return base + "ed"


def english_ger(base: str) -> str:
    if base in EN_IRREGULAR:
        return EN_IRREGULAR[base]["ger"]
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("e") and base not in {"be", "see"}:
        return base[:-1] + "ing"
    return base + "ing"


def detect_pos(tags: Sequence[str]) -> str:
    tagset = set(tags)
    if any(t.startswith("vb") or t == "v" for t in tagset):
        return "verb"
    if "n" in tagset:
        return "noun"
    if "adj" in tagset:
        return "adj"
    if "adv" in tagset:
        return "adv"
    if "prn" in tagset or "pron" in tagset:
        return "pron"
    if "det" in tagset or "art" in tagset:
        return "det"
    return ""


def tag_value(tags: Sequence[str], choices: Sequence[str]) -> bool:
    tagset = set(tags)
    return any(choice in tagset for choice in choices)


def person_number(tags: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    tagset = set(tags)
    person = None
    number = None
    for p in ("1", "2", "3"):
        if f"p{p}" in tagset or p in tagset:
            person = p
            break
    if "sg" in tagset:
        number = "sg"
    elif "pl" in tagset:
        number = "pl"
    return person, number


def tense_bucket(tags: Sequence[str]) -> str:
    tagset = set(tags)
    if "inf" in tagset:
        return "inf"
    if "ger" in tagset:
        return "ger"
    if "pp" in tagset or "part" in tagset or "pastpart" in tagset:
        return "pp"
    if tag_value(tags, ["pret", "past", "pst"]):
        return "past"
    if tag_value(tags, ["impf", "imperf", "pii", "ifi"]):
        return "imperfect"
    if tag_value(tags, ["fut"]):
        return "future"
    if tag_value(tags, ["cni", "cond", "conditional"]):
        return "conditional"
    if tag_value(tags, ["imp"]) and not tag_value(tags, ["impf", "imperf", "pii", "ifi"]):
        return "imperative"
    if tag_value(tags, ["subj", "sub", "prs", "pres", "pri"]):
        return "present_subj" if tag_value(tags, ["subj", "sub"]) else "present"
    if tag_value(tags, ["prs", "pres", "pri"]):
        return "present"
    return "unknown"


def format_with_subject(pronoun: str, forms: List[str]) -> str:
    return "; ".join(f"{pronoun} {form}" for form in forms if form)


def render_generic_verb(lemma: str, lemma_translation: str, tags: Sequence[str]) -> Tuple[str, str]:
    bases = english_base_options(lemma_translation)
    if not bases:
        return lemma_translation, "low"
    person, number = person_number(tags)
    pronoun = PRONOUNS.get((person, number), "") if person and number else ""
    bucket = tense_bucket(tags)
    if bucket == "inf":
        return lemma_translation, "medium"
    if bucket == "ger":
        return "; ".join(english_ger(base) for base in bases[:3]), "medium"
    if bucket == "pp":
        return "; ".join(english_pp(base) for base in bases[:3]), "medium"
    if bucket == "present" and pronoun:
        forms = [english_3sg(base) if pronoun == "he/she/it" else base for base in bases[:3]]
        return format_with_subject(pronoun, forms), "medium"
    if bucket in {"past", "imperfect"} and pronoun:
        forms = [english_past(base, pronoun) for base in bases[:3]]
        return format_with_subject(pronoun, forms), "medium"
    if bucket == "future" and pronoun:
        forms = [f"will {base}" for base in bases[:3]]
        return format_with_subject(pronoun, forms), "medium"
    if bucket == "conditional" and pronoun:
        forms = [f"would {base}" for base in bases[:3]]
        return format_with_subject(pronoun, forms), "medium"
    if bucket == "present_subj":
        return "; ".join(f"{base} (subj.)" for base in bases[:3]), "low"
    if bucket == "imperative":
        return "; ".join(f"{base}!" for base in bases[:3]), "low"
    return lemma_translation, "low"


def render_special_verb(lemma: str, tags: Sequence[str]) -> Optional[Tuple[str, str]]:
    person, number = person_number(tags)
    pronoun = PRONOUNS.get((person, number), "") if person and number else ""
    bucket = tense_bucket(tags)
    if lemma in {"ser", "estar"}:
        if bucket == "inf": return "to be", "high"
        if bucket == "ger": return "being", "high"
        if bucket == "pp": return "been", "high"
        if bucket == "present" and pronoun:
            forms = {"I": "am", "you": "are", "he/she/it": "is", "we": "are", "you all": "are", "they": "are"}
            return f"{pronoun} {forms[pronoun]}", "high"
        if bucket in {"past", "imperfect"} and pronoun:
            forms = {"I": "was", "you": "were", "he/she/it": "was", "we": "were", "you all": "were", "they": "were"}
            return f"{pronoun} {forms[pronoun]}", "high"
        if bucket == "future" and pronoun: return f"{pronoun} will be", "high"
        if bucket == "conditional" and pronoun: return f"{pronoun} would be", "high"
    if lemma == "haber":
        if bucket == "inf": return "to have; there is; there are", "high"
        if bucket == "ger": return "having", "high"
        if bucket == "pp": return "had", "high"
        if bucket == "present" and pronoun:
            form = "has" if pronoun == "he/she/it" else "have"
            return f"{pronoun} {form}", "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} had", "high"
        if bucket == "future" and pronoun: return f"{pronoun} will have", "high"
        if bucket == "conditional" and pronoun: return f"{pronoun} would have", "high"
    if lemma == "tener":
        if bucket == "inf": return "to have", "high"
        if bucket == "ger": return "having", "high"
        if bucket == "pp": return "had", "high"
        if bucket == "present" and pronoun:
            form = "has" if pronoun == "he/she/it" else "have"
            return f"{pronoun} {form}", "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} had", "high"
        if bucket == "future" and pronoun: return f"{pronoun} will have", "high"
        if bucket == "conditional" and pronoun: return f"{pronoun} would have", "high"
    if lemma == "hacer":
        if bucket == "inf": return "to do; to make", "high"
        if bucket == "ger": return "doing; making", "high"
        if bucket == "pp": return "done; made", "high"
        if bucket == "present" and pronoun:
            if pronoun == "he/she/it": return "he/she/it does; he/she/it makes", "high"
            return f"{pronoun} do; {pronoun} make", "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} did; {pronoun} made", "high"
        if bucket == "future" and pronoun: return f"{pronoun} will do; {pronoun} will make", "high"
        if bucket == "conditional" and pronoun: return f"{pronoun} would do; {pronoun} would make", "high"
    if lemma == "decir":
        if bucket == "inf": return "to say; to tell", "high"
        if bucket == "ger": return "saying; telling", "high"
        if bucket == "pp": return "said", "high"
        if bucket == "present" and pronoun:
            if pronoun == "he/she/it": return "he/she/it says; he/she/it tells", "high"
            return f"{pronoun} say; {pronoun} tell", "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} said", "high"
        if bucket == "future" and pronoun: return f"{pronoun} will say; {pronoun} will tell", "high"
        if bucket == "conditional" and pronoun: return f"{pronoun} would say; {pronoun} would tell", "high"
    if lemma == "ir":
        if bucket == "inf": return "to go", "high"
        if bucket == "ger": return "going", "high"
        if bucket == "pp": return "gone", "high"
        if bucket == "present" and pronoun:
            if pronoun == "he/she/it": return "he/she/it goes", "high"
            if pronoun == "we": return "we go", "high"
            return f"{pronoun} go", "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} went", "high"
        if bucket == "future" and pronoun: return f"{pronoun} will go", "high"
        if bucket == "conditional" and pronoun: return f"{pronoun} would go", "high"
    if lemma == "ver":
        if bucket == "present" and pronoun: return (f"{pronoun} sees" if pronoun == "he/she/it" else f"{pronoun} see"), "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} saw", "high"
        if bucket == "inf": return "to see", "high"
        if bucket == "pp": return "seen", "high"
        if bucket == "ger": return "seeing", "high"
    if lemma == "dar":
        if bucket == "present" and pronoun: return (f"{pronoun} gives" if pronoun == "he/she/it" else f"{pronoun} give"), "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} gave", "high"
        if bucket == "inf": return "to give", "high"
        if bucket == "pp": return "given", "high"
        if bucket == "ger": return "giving", "high"
    if lemma in {"saber", "conocer"}:
        if bucket == "present" and pronoun: return (f"{pronoun} knows" if pronoun == "he/she/it" else f"{pronoun} know"), "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} knew", "high"
        if bucket == "inf": return "to know", "high"
        if bucket == "pp": return "known", "high"
        if bucket == "ger": return "knowing", "high"
    if lemma == "poder":
        if bucket == "inf": return "to be able to; can", "high"
        if bucket == "present" and pronoun: return f"{pronoun} can", "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} could", "high"
        if bucket == "future" and pronoun: return f"{pronoun} will be able to", "high"
        if bucket == "conditional" and pronoun: return f"{pronoun} could; {pronoun} would be able to", "high"
    if lemma == "querer":
        if bucket == "inf": return "to want", "high"
        if bucket == "present" and pronoun: return (f"{pronoun} wants" if pronoun == "he/she/it" else f"{pronoun} want"), "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} wanted", "high"
        if bucket == "future" and pronoun: return f"{pronoun} will want", "high"
        if bucket == "conditional" and pronoun: return f"{pronoun} would want", "high"
    if lemma == "venir":
        if bucket == "inf": return "to come", "high"
        if bucket == "present" and pronoun: return (f"{pronoun} comes" if pronoun == "he/she/it" else f"{pronoun} come"), "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} came", "high"
        if bucket == "pp": return "come", "high"
        if bucket == "ger": return "coming", "high"
    if lemma == "poner":
        if bucket == "inf": return "to put", "high"
        if bucket == "present" and pronoun: return (f"{pronoun} puts" if pronoun == "he/she/it" else f"{pronoun} put"), "high"
        if bucket in {"past", "imperfect"} and pronoun: return f"{pronoun} put", "high"
        if bucket == "pp": return "put", "high"
        if bucket == "ger": return "putting", "high"
    return None


def render_surface_translation(surface: str, lemma: str, tags: Sequence[str], lemma_translation: str, lemma_ok: bool, data_entries: Dict[str, List[Dict[str, object]]]) -> Tuple[str, str, str]:
    token = normalize_token(surface)
    lemma_norm = normalize_token(lemma)
    if token in SURFACE_OVERRIDES:
        return SURFACE_OVERRIDES[token], "surface_override", "high"
    pos = detect_pos(tags)
    data_translation, data_source, data_ok = choose_data_translation(token, pos, False, data_entries)
    if data_ok:
        return data_translation, data_source, "high" if pos and pos in data_source else "medium"
    if not lemma_ok:
        return f"[UNTRANSLATED] {token}", "untranslated", "low"
    if token == lemma_norm:
        return lemma_translation, "lemma_direct", "high"
    if pos == "verb":
        special = render_special_verb(lemma_norm, tags)
        if special is not None:
            return special[0], "verb_special", special[1]
        generic, confidence = render_generic_verb(lemma_norm, lemma_translation, tags)
        return generic, "verb_generic", confidence
    if pos == "noun" and "pl" in set(tags):
        options = split_english_options(lemma_translation)
        if options:
            pluralized = [pluralize_noun(opt) for opt in options[:3]]
            return "; ".join(pluralized), "noun_plural", "medium"
    return lemma_translation, "lemma_fallback", "medium"


def translate_csv(input_csv: str, dict_path: str, data_path: str, out_main: str, out_detail: str, out_untranslated: str, top_n: int = 100, assume_input_is_surface: bool = False, source_column: Optional[str] = None, reverse_dict: Optional[bool] = None, analyzer_cmd: Optional[Sequence[str]] = None) -> None:
    dictionary, accent_dictionary = load_dictionary(dict_path, reverse=reverse_dict)
    data_entries = parse_data_dictionary(data_path)
    rows_cache: List[Dict[str, str]] = []
    all_input_values: List[str] = []
    uncertain_rows: List[Dict[str, str]] = []
    with open(input_csv, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no headers.")
        source_column = source_column or infer_column(reader.fieldnames)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows_cache.append(row)
            all_input_values.append(row.get(source_column, ""))
    analysis_map: Dict[str, Dict[str, object]] = {}
    if assume_input_is_surface:
        unique_surface = sorted({normalize_token(x) for x in all_input_values if normalize_token(x)})
        try:
            analysis_map = analyze_surface_forms(unique_surface, analyzer_cmd=analyzer_cmd)
        except Exception as exc:
            warnings.warn(f"Surface analysis disabled: {exc}")
            analysis_map = {token: {"surface": token, "lemma": token, "tags": [], "raw": "", "candidates": []} for token in unique_surface}
    main_fieldnames = fieldnames + ["english_lemma", "english_surface", "spanish_lemma", "lemma_lookup_used", "lemma_source", "surface_source", "surface_confidence", "analysis_tags"]
    detail_fieldnames = fieldnames + ["source_column", "surface_original", "surface_normalized", "analysis_raw", "analysis_tags", "lemma_normalized", "lemma_lookup_used", "english_lemma", "english_surface", "lemma_translated_flag", "lemma_source", "surface_source", "surface_confidence"]
    with open(out_main, "w", encoding="utf-8", newline="") as main_out, open(out_detail, "w", encoding="utf-8", newline="") as detail_out:
        writer_main = csv.DictWriter(main_out, fieldnames=main_fieldnames)
        writer_main.writeheader()
        writer_detail = csv.DictWriter(detail_out, fieldnames=detail_fieldnames)
        writer_detail.writeheader()
        for row in rows_cache:
            surface_original = row.get(source_column, "")
            surface_normalized = normalize_token(surface_original)
            analysis = analysis_map.get(surface_normalized, {"surface": surface_normalized, "lemma": surface_normalized, "tags": [], "raw": "", "candidates": []})
            lemma_normalized = str(analysis.get("lemma", surface_normalized)) if assume_input_is_surface else surface_normalized
            tags = list(analysis.get("tags", [])) if assume_input_is_surface else []
            analysis_raw = str(analysis.get("raw", "")) if assume_input_is_surface else ""
            english_lemma, lemma_used, lemma_ok, lemma_source = translate_lemma(lemma_normalized, dictionary, accent_dictionary, data_entries)
            english_surface, surface_source, surface_confidence = render_surface_translation(surface_normalized, lemma_normalized, tags, english_lemma, lemma_ok, data_entries)
            main_row = dict(row)
            main_row.update({
                "english_lemma": english_lemma,
                "english_surface": english_surface,
                "spanish_lemma": lemma_normalized,
                "lemma_lookup_used": lemma_used,
                "lemma_source": lemma_source,
                "surface_source": surface_source,
                "surface_confidence": surface_confidence,
                "analysis_tags": " ".join(tags),
            })
            writer_main.writerow(main_row)
            detail_row = dict(row)
            detail_row.update({
                "source_column": source_column,
                "surface_original": surface_original,
                "surface_normalized": surface_normalized,
                "analysis_raw": analysis_raw,
                "analysis_tags": " ".join(tags),
                "lemma_normalized": lemma_normalized,
                "lemma_lookup_used": lemma_used,
                "english_lemma": english_lemma,
                "english_surface": english_surface,
                "lemma_translated_flag": "TRUE" if lemma_ok else "FALSE",
                "lemma_source": lemma_source,
                "surface_source": surface_source,
                "surface_confidence": surface_confidence,
            })
            writer_detail.writerow(detail_row)
            if not lemma_ok or surface_confidence == "low":
                uncertain_rows.append(detail_row)
    uncertain_rows.sort(key=lambda row: safe_int(row.get("count", 0)), reverse=True)
    with open(out_untranslated, "w", encoding="utf-8", newline="") as outfile:
        fieldnames_out = list(uncertain_rows[0].keys()) if uncertain_rows else [source_column or "lemma", "english_surface"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames_out)
        writer.writeheader()
        for row in uncertain_rows[:top_n]:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Spanish tokens into English lemma glosses and flashcard-safe surface glosses using es-en.data + Apertium .dix.")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--dict", required=True, help="Apertium bilingual dictionary (.dix)")
    parser.add_argument("--data", required=True, help="es-en.data path")
    parser.add_argument("--column", default=None, help="Column to translate")
    parser.add_argument("--out_main", default=None, help="Main translated CSV output")
    parser.add_argument("--out_detail", default=None, help="Detailed CSV output")
    parser.add_argument("--out_untranslated", default=None, help="Top uncertain CSV output")
    parser.add_argument("--top_n", type=int, default=100, help="How many uncertain rows to report")
    parser.add_argument("--input_is_surface", action="store_true", help="Treat source column as surface forms and analyze before lookup")
    parser.add_argument("--reverse_dict", action="store_true", help="Read bilingual .dix in reverse")
    parser.add_argument("--analyzer_cmd", nargs="+", default=None, help="Override spa-morph command")
    args = parser.parse_args()
    base, ext = os.path.splitext(args.input)
    out_main = args.out_main or f"{base}_flashcards{ext}"
    out_detail = args.out_detail or f"{base}_flashcards_detail{ext}"
    out_untranslated = args.out_untranslated or f"{base}_flashcards_uncertain_top{args.top_n}{ext}"
    reverse_dict = args.reverse_dict or infer_reverse_from_filename(args.dict)
    translate_csv(
        input_csv=args.input,
        dict_path=args.dict,
        data_path=args.data,
        out_main=out_main,
        out_detail=out_detail,
        out_untranslated=out_untranslated,
        top_n=args.top_n,
        assume_input_is_surface=args.input_is_surface,
        source_column=args.column,
        reverse_dict=reverse_dict,
        analyzer_cmd=args.analyzer_cmd,
    )
    print("Done.")
    print("Main:      ", out_main)
    print("Detail:    ", out_detail)
    print("Uncertain: ", out_untranslated)


if __name__ == "__main__":
    main()
