#!/usr/bin/env python3
"""Markov-only Spanish sentence generator.

This samples from learned bigram transitions and never returns retrieved corpus
rows. The corpus is used only offline to train the transition counts.
"""
import argparse
import csv
import math
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9.-]+")
PUNCT_ATTACH_LEFT = {".", ",", ";", ":", "!", "?", "%"}
PUNCT_ATTACH_RIGHT = {"¿", "¡", "(", "[", "{"}
CONTEXT_OPENERS = {"si", "que", "cuando", "aunque", "porque", "como", "pero", "y", "o"}
BAD_INTERNAL_TOKENS = {"<START>"}
PREPOSITIONS = {"a", "de", "en", "con", "por", "para", "sin", "sobre", "hasta", "desde", "entre"}
ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
SUBJECT_PRONOUNS = {"yo", "tú", "él", "ella", "nosotros", "nosotras", "vosotros", "vosotras", "ellos", "ellas", "usted", "ustedes"}
BAD_FINAL_TOKENS = PREPOSITIONS | {"y", "o", "que", "pero", *ARTICLES, *SUBJECT_PRONOUNS}
COMMON_FINITE_VERBS = {
    "soy", "eres", "es", "somos", "son",
    "estoy", "estás", "está", "estamos", "están",
    "tengo", "tienes", "tiene", "tenemos", "tienen",
    "voy", "vas", "va", "vamos", "van",
    "puedo", "puedes", "puede", "podemos", "pueden",
    "quiero", "quieres", "quiere", "queremos", "quieren",
    "hago", "haces", "hace", "hacemos", "hacen",
    "veo", "ves", "ve", "vemos", "ven",
    "sé", "sabes", "sabe", "sabemos", "saben",
    "digo", "dices", "dice", "decimos", "dicen",
    "dime",
    "creo", "crees", "cree", "creemos", "creen",
    "gusta", "gustan",
    "vivo", "vives", "vive", "vivimos", "viven",
    "vengo", "vienes", "viene", "venimos", "vienen",
    "hablo", "hablas", "habla", "hablamos", "hablan",
    "trabajo", "trabajas", "trabaja", "trabajamos", "trabajan",
    "hay",
}
BEGINNER_COMPLEX_VERB_FORMS = {
    "era", "eras", "éramos", "eran",
    "fue", "fui", "fuiste", "fuimos", "fueron",
    "estaba", "estabas", "estábamos", "estaban",
    "sería", "serías", "seríamos", "serían",
    "había", "habías", "habíamos", "habían",
    "haya", "hayas", "hayamos", "hayan",
}


@dataclass
class LexiconRow:
    lemma: str
    rank: int
    pos: str
    translation: str


@dataclass
class MarkovCandidate:
    lemma: str
    rank: int
    pos: str
    translation: str
    sentence: str
    tokens: List[str]
    score: float
    rejection_reasons: List[str]


def normalize_token(token: str) -> str:
    return (token or "").strip().lower().strip('.,;:!?¡¿"“”\'()[]{}')


def word_tokens(sentence: str) -> List[str]:
    return [normalize_token(t) for t in WORD_RE.findall(sentence or "") if normalize_token(t)]


def detokenize(tokens: Sequence[str]) -> str:
    parts: List[str] = []
    for token in tokens:
        if not token or token in {"<START>", "<END>"}:
            continue
        if not parts:
            parts.append(token)
            continue
        if token in PUNCT_ATTACH_LEFT:
            parts[-1] += token
        elif parts[-1] in PUNCT_ATTACH_RIGHT:
            parts[-1] += token
        else:
            parts.append(token)
    sentence = " ".join(parts).strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."
    return sentence


def profile_for_rank(rank: int) -> Tuple[str, int, int, int, float]:
    if rank <= 800:
        return "A1", 3, 7, 400, 250.0
    if rank <= 1500:
        return "A2", 3, 8, 700, 400.0
    if rank <= 2500:
        return "B1", 4, 9, 1500, 900.0
    if rank <= 4000:
        return "B2", 5, 10, 3000, 1800.0
    if rank <= 6000:
        return "C1", 5, 11, 5500, 3000.0
    return "C2", 6, 12, 9000, 5000.0


def subject_for_verb_form(lemma: str) -> str:
    if lemma in {"soy", "estoy", "voy", "tengo", "puedo", "quiero", "creo", "sé", "hago"}:
        return "yo"
    if lemma in {"eres", "estás", "vas", "tienes", "puedes", "quieres", "sabes", "haces"}:
        return "tú"
    if lemma in {"somos", "estamos", "vamos", "tenemos", "podemos", "queremos", "sabemos", "hacemos"}:
        return "nosotros"
    if lemma in {"son", "están", "van", "tienen", "pueden", "quieren", "saben", "hacen"}:
        return "ellos"
    if lemma.endswith("mos"):
        return "nosotros"
    if lemma.endswith("n") and len(lemma) > 3:
        return "ellos"
    if lemma.endswith(("as", "es")):
        return "tú"
    return "ella"


def load_lexicon(path: Path) -> List[LexiconRow]:
    rows: List[LexiconRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lemma = normalize_token(row.get("lemma") or row.get("word") or "")
            if not lemma:
                continue
            try:
                rank = int(float(row.get("rank") or 999999))
            except ValueError:
                rank = 999999
            rows.append(
                LexiconRow(
                    lemma=lemma,
                    rank=rank,
                    pos=normalize_token(row.get("pos") or ""),
                    translation=(row.get("translation") or "").strip(),
                )
            )
    rows.sort(key=lambda r: r.rank)
    return rows


def rank_map_from_lexicon(rows: Iterable[LexiconRow]) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for row in rows:
        ranks.setdefault(row.lemma, row.rank)
    return ranks


def load_bigram_model(models_dir: Path) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    path = models_dir / "bigrams.pkl"
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["next"], data["totals"]


def build_predecessor_index(
    bi_next: Dict[str, Dict[str, int]],
    target_words: Iterable[str],
) -> Dict[str, Dict[str, int]]:
    target_set = set(target_words)
    predecessors: Dict[str, Dict[str, int]] = {target: {} for target in target_set}
    for prev, next_counts in bi_next.items():
        if prev == "<END>":
            continue
        for nxt, count in next_counts.items():
            if nxt in target_set and prev not in BAD_INTERNAL_TOKENS:
                predecessors[nxt][prev] = count
    return predecessors


class MarkovOnlyGenerator:
    def __init__(
        self,
        lexicon_rows: List[LexiconRow],
        bi_next: Dict[str, Dict[str, int]],
        bi_totals: Dict[str, int],
        seed: int = 42,
        temperature: float = 0.9,
        top_k: int = 24,
    ):
        self.lexicon_rows = lexicon_rows
        self.rank_by_word = rank_map_from_lexicon(lexicon_rows)
        self.pos_by_word = {row.lemma: row.pos for row in lexicon_rows}
        self.bi_next = bi_next
        self.bi_totals = bi_totals
        self.random = random.Random(seed)
        self.temperature = max(0.05, temperature)
        self.top_k = max(1, top_k)
        self.predecessors: Dict[str, Dict[str, int]] = {}
        self._top_cache: Dict[int, List[Tuple[str, int]]] = {}

    def prepare_targets(self, targets: List[LexiconRow]) -> None:
        self.predecessors = build_predecessor_index(self.bi_next, [target.lemma for target in targets])

    def top_items(self, counts: Dict[str, int]) -> List[Tuple[str, int]]:
        key = id(counts)
        cached = self._top_cache.get(key)
        if cached is None:
            keep = max(self.top_k * 6, 80)
            cached = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:keep]
            self._top_cache[key] = cached
        return cached

    def weighted_choice(self, counts: Dict[str, int], banned: Optional[set] = None) -> Optional[str]:
        banned = banned or set()
        items = [
            (word, count)
            for word, count in self.top_items(counts)
            if word not in banned and count > 0
        ][: self.top_k]
        if not items:
            return None
        weights = [count ** (1.0 / self.temperature) for _, count in items]
        return self.random.choices([word for word, _ in items], weights=weights, k=1)[0]

    def sample_left(self, target: LexiconRow, max_left: int) -> List[str]:
        tokens_reversed: List[str] = []
        current = target.lemma
        seen = {current}
        for _ in range(max_left):
            counts = self.predecessors.get(current, {})
            prev = self.weighted_choice(counts, banned=seen | {"<END>"})
            if not prev or prev == "<START>":
                break
            tokens_reversed.append(prev)
            seen.add(prev)
            current = prev
        return list(reversed(tokens_reversed))

    def sample_right(self, tokens: List[str], max_right: int, rank_ceiling: int) -> List[str]:
        out = list(tokens)
        seen_tail = set(out[-2:])
        for _ in range(max_right):
            current = out[-1] if out else "<START>"
            counts = self.bi_next.get(current, {})
            if not counts:
                break
            banned = {
                word
                for word, _ in self.top_items(counts)
                if word == "<START>"
                or word in seen_tail
                or (word != "<END>" and self.rank_by_word.get(word, 999999) > rank_ceiling)
            }
            nxt = self.weighted_choice(counts, banned=banned)
            if not nxt or nxt == "<END>":
                break
            out.append(nxt)
            seen_tail = set(out[-2:])
        return out

    def article_for_noun(self, lemma: str, definite: bool = True) -> str:
        feminine = (
            (lemma.endswith("a") and not lemma.endswith("ma"))
            or lemma.endswith(("dad", "tad", "tud", "ción", "sión", "umbre"))
        )
        if definite:
            return "la" if feminine else "el"
        return "una" if feminine else "un"

    def grammar_seeds_for_target(self, target: LexiconRow) -> List[List[str]]:
        lemma = target.lemma
        if target.pos == "n":
            if lemma == "gracias":
                return [["muchas", "gracias"]]
            if lemma == "tiempo":
                return [["tengo", "tiempo"], ["hay", "tiempo"]]
            definite = self.article_for_noun(lemma, definite=True)
            indefinite = self.article_for_noun(lemma, definite=False)
            return [
                [definite, lemma, "está"],
                [definite, lemma, "es"],
                ["tengo", indefinite, lemma],
            ]
        if lemma in ARTICLES:
            if lemma in {"la", "una"}:
                return [[lemma, "casa", "está"], [lemma, "persona", "tiene"]]
            if lemma in {"las", "unas"}:
                return [[lemma, "personas", "están"]]
            if lemma in {"los", "unos"}:
                return [[lemma, "hombres", "están"]]
            return [[lemma, "hombre", "tiene"], [lemma, "mundo", "es"]]
        if lemma == "no":
            return [["no", "hay"], ["ella", "no", "tiene"]]
        if lemma == "y":
            return [["tú", "y", "yo", "estamos"], ["ella", "y", "yo", "vamos"]]
        if lemma == "que":
            return [["creo", "que"], ["ella", "dice", "que"]]
        if lemma in PREPOSITIONS:
            if lemma == "a":
                return [["voy", "a", "casa"], ["vengo", "a", "casa"]]
            if lemma == "de":
                return [["soy", "de", "aquí"], ["vengo", "de", "casa"]]
            if lemma == "en":
                return [["estoy", "en"], ["vivo", "en"]]
            if lemma == "con":
                return [["estoy", "con", "ella"], ["vivo", "con", "ella"]]
            if lemma == "por":
                return [["voy", "por", "agua"], ["trabajo", "por", "la", "mañana"]]
            if lemma == "para":
                return [["trabajo", "para", "ella"], ["es", "para", "ti"]]
            return [["hablo", lemma], ["vengo", lemma]]
        if lemma == "me":
            return [["me", "gusta"], ["me", "tiene"]]
        if lemma == "te":
            return [["te", "gusta"], ["te", "tiene"]]
        if lemma == "lo":
            return [["lo", "tengo", "claro"]]
        if lemma == "se":
            return [["se", "puede", "hacer"]]
        if lemma in {"qué", "que"}:
            return [[lemma, "es"], [lemma, "tiene"]]
        if lemma == "del":
            return [["es", "del", "mundo"], ["vengo", "del", "trabajo"]]
        if lemma == "pero":
            return [["quiero", "ir", "pero", "no", "puedo"]]
        if lemma == "si":
            return [["dime", "si", "puedes", "ir"]]
        if target.pos == "v":
            if lemma.endswith(("ar", "er", "ir")):
                return [["quiero", lemma], ["puedo", lemma], ["voy", "a", lemma]]
            return [[subject_for_verb_form(lemma), lemma]]
        if target.pos == "adj":
            return [["ella", "es", lemma], ["esto", "es", lemma]]
        if lemma in {"muy", "tan"}:
            return [["ella", "es", lemma, "buena"], ["esto", "es", lemma, "bueno"]]
        if lemma in {"bien", "aquí", "también", "nunca", "ya"}:
            return [["ella", lemma, "está"], ["ella", "está", lemma]]
        if lemma in {"esta", "esa"}:
            return [[lemma, "casa", "está"]]
        if lemma in {"este", "ese"}:
            return [[lemma, "hombre", "está"]]
        if target.pos == "prep":
            return [["vengo", lemma], ["hablo", lemma]]
        if target.pos == "conj":
            return [["ella", "va", lemma]]
        if target.pos in {"adv", "none"}:
            return [["ella", lemma], [lemma, "ella"]]
        return [[lemma]]

    def score_tokens(self, tokens: List[str], target: LexiconRow) -> float:
        score = 0.0
        for prev, nxt in zip(["<START>"] + tokens, tokens + ["<END>"]):
            counts = self.bi_next.get(prev, {})
            total = self.bi_totals.get(prev, 0)
            count = counts.get(nxt, 0)
            if total > 0 and count > 0:
                score += math.log(count / total)
            else:
                score -= 12.0
        _, min_len, max_len, _, _ = profile_for_rank(target.rank)
        ideal = (min_len + max_len) / 2.0
        score -= 0.6 * abs(len(tokens) - ideal)
        support_ranks = [self.rank_by_word.get(tok, 999999) for tok in tokens if tok != target.lemma]
        if support_ranks:
            score -= max(0, max(support_ranks) - target.rank * 2) / 2000.0
        return score

    def rejection_reasons(self, tokens: List[str], target: LexiconRow) -> List[str]:
        band, min_len, max_len, rank_ceiling, avg_ceiling = profile_for_rank(target.rank)
        reasons: List[str] = []
        if target.lemma not in tokens:
            reasons.append("missing_target")
        if not (min_len <= len(tokens) <= max_len):
            reasons.append("bad_length")
        if tokens and tokens[0] in CONTEXT_OPENERS:
            reasons.append("context_dependent_opener")
        if any(a == b for a, b in zip(tokens, tokens[1:])):
            reasons.append("repeated_token")
        if target.pos in {"prep", "conj"}:
            idx = tokens.index(target.lemma) if target.lemma in tokens else -1
            if idx <= 0 or idx >= len(tokens) - 1:
                reasons.append("dangling_function_word")
        support_ranks = [self.rank_by_word.get(tok, 999999) for tok in tokens if tok != target.lemma]
        known_support = [rank for rank in support_ranks if rank < 999999]
        if known_support:
            if max(known_support) > rank_ceiling:
                reasons.append("support_vocab_too_advanced")
            if sum(known_support) / len(known_support) > avg_ceiling:
                reasons.append("avg_support_vocab_too_advanced")
        unknown_count = sum(1 for tok in tokens if tok != target.lemma and self.rank_by_word.get(tok, 999999) >= 999999)
        if unknown_count > 1:
            reasons.append("too_many_unknown_words")
        has_verb = any(tok in COMMON_FINITE_VERBS for tok in tokens)
        if not has_verb:
            reasons.append("no_verb")
        allowed_final_pronoun = (
            len(tokens) >= 2
            and tokens[-1] in SUBJECT_PRONOUNS
            and tokens[-2] in {"con", "para"}
        )
        if tokens and tokens[-1] in BAD_FINAL_TOKENS and not allowed_final_pronoun:
            reasons.append("bad_final_token")
        if len(tokens) > 2 and tokens[-1] == "no":
            reasons.append("bad_final_no")
        allowed_final_finite = (
            len(tokens) >= 2
            and tokens[-2] == "no"
            and tokens[-1] in {"puedo", "quiero", "sé"}
        )
        if tokens and tokens[-1] in COMMON_FINITE_VERBS and not allowed_final_finite:
            reasons.append("missing_predicate_after_final_verb")
        for i, tok in enumerate(tokens[:-1]):
            if tok == "a" and tokens[i + 1] in SUBJECT_PRONOUNS | {"mí", "ti", "sí"}:
                reasons.append("bad_preposition_object")
                break
        if any(tokens[i:i + 3] == ["con", "ella", "misma"] for i in range(max(0, len(tokens) - 2))):
            reasons.append("bad_con_ella_misma")
        for i, tok in enumerate(tokens[:-2]):
            if tok == "con" and tokens[i + 1] in SUBJECT_PRONOUNS:
                reasons.append("extra_words_after_con_pronoun")
                break
        for i, tok in enumerate(tokens[:-1]):
            if tok == "a" and self.pos_by_word.get(tokens[i + 1]) == "v":
                previous = tokens[i - 1] if i > 0 else ""
                if previous not in {"voy", "vas", "va", "vamos", "van", "vengo", "vienes", "viene", "venimos", "vienen"}:
                    reasons.append("unsupported_a_infinitive")
                    break
        if len(tokens) >= 2 and tokens[-2] == "y" and tokens[-1] in SUBJECT_PRONOUNS:
            reasons.append("dangling_conjoined_subject")
        if len(tokens) >= 4 and tokens[:3] in (["tú", "y", "yo"], ["ella", "y", "yo"]):
            if tokens[3] not in {"estamos", "somos", "vamos", "tenemos", "podemos", "queremos"}:
                reasons.append("bad_conjoined_subject_agreement")
        if "nada" in tokens and "no" not in tokens and "nunca" not in tokens:
            reasons.append("negative_word_without_negation")
        if "hace" in tokens and "años" in tokens and any(tok in tokens for tok in {"lo", "se"}):
            reasons.append("bad_hace_anos_pronoun_frame")
        if target.lemma == "que" and "que" in tokens:
            idx = tokens.index("que")
            tail = tokens[idx + 1:]
            if tail != ["no"] and not any(tok in COMMON_FINITE_VERBS for tok in tail):
                reasons.append("que_without_following_clause")
        if band in {"A1", "A2"} and len([tok for tok in tokens if self.pos_by_word.get(tok) == "v"]) > 1:
            if target.lemma not in {"que", "si", "pero", "se"}:
                reasons.append("too_many_verbs_for_beginner")
        if band in {"A1", "A2"} and target.lemma != "que" and "que" in tokens:
            reasons.append("subordinate_que_for_beginner")
        if band in {"A1", "A2"} and any(tok in BEGINNER_COMPLEX_VERB_FORMS for tok in tokens):
            reasons.append("complex_verb_form_for_beginner")
        return reasons

    def generate_for_target(self, target: LexiconRow, attempts: int, seed_mode: str) -> MarkovCandidate:
        _, min_len, max_len, rank_ceiling, _ = profile_for_rank(target.rank)
        candidates: List[MarkovCandidate] = []
        near_misses: List[MarkovCandidate] = []
        grammar_seeds = self.grammar_seeds_for_target(target)

        for _ in range(attempts):
            use_grammar_seed = seed_mode == "grammar" or (
                seed_mode == "mixed" and self.random.random() < 0.65
            )
            if use_grammar_seed and grammar_seeds:
                start = list(self.random.choice(grammar_seeds))
            else:
                left_budget = self.random.randint(0, max(0, max_len - 1))
                left = self.sample_left(target, max_left=left_budget)
                start = left + [target.lemma]
            right_budget = max_len - len(start)
            tokens = self.sample_right(start, max_right=max(0, right_budget), rank_ceiling=rank_ceiling)
            tokens = [tok for tok in tokens if tok and tok not in {"<START>", "<END>"}]
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            if len(tokens) < min_len:
                continue
            reasons = self.rejection_reasons(tokens, target)
            sentence = detokenize(tokens)
            score = self.score_tokens(tokens, target)
            candidate = MarkovCandidate(
                lemma=target.lemma,
                rank=target.rank,
                pos=target.pos,
                translation=target.translation,
                sentence=sentence,
                tokens=tokens,
                score=score,
                rejection_reasons=reasons,
            )
            if reasons:
                near_misses.append(candidate)
            else:
                candidates.append(candidate)

        if candidates:
            return max(candidates, key=lambda c: c.score)
        if near_misses:
            best = max(near_misses, key=lambda c: (len(c.rejection_reasons) * -1, c.score))
            return MarkovCandidate(
                lemma=target.lemma,
                rank=target.rank,
                pos=target.pos,
                translation=target.translation,
                sentence="",
                tokens=best.tokens,
                score=best.score,
                rejection_reasons=best.rejection_reasons,
            )
        return MarkovCandidate(
            lemma=target.lemma,
            rank=target.rank,
            pos=target.pos,
            translation=target.translation,
            sentence="",
            tokens=[],
            score=0.0,
            rejection_reasons=["no_candidate"],
        )


def select_targets(
    rows: List[LexiconRow],
    min_rank: int,
    max_rank: int,
    limit: int,
    pos: Optional[str],
    lemmas: Optional[List[str]],
) -> List[LexiconRow]:
    wanted = {normalize_token(lemma) for lemma in lemmas or [] if normalize_token(lemma)}
    selected = [
        row for row in rows
        if min_rank <= row.rank <= max_rank
        and (not pos or row.pos == pos)
        and (not wanted or row.lemma in wanted)
    ]
    return selected[:limit] if limit > 0 else selected


def write_outputs(candidates: List[MarkovCandidate], out: Path, review_out: Optional[Path]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lemma",
                "rank",
                "pos",
                "translation",
                "sentence",
                "source_method",
                "score",
                "rejection_reasons",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "lemma": candidate.lemma,
                    "rank": candidate.rank,
                    "pos": candidate.pos,
                    "translation": candidate.translation,
                    "sentence": candidate.sentence,
                    "source_method": "markov_chain",
                    "score": f"{candidate.score:.6f}",
                    "rejection_reasons": "; ".join(candidate.rejection_reasons),
                }
            )

    if review_out:
        review_out.parent.mkdir(parents=True, exist_ok=True)
        with review_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["lemma", "rank", "pos", "sentence", "tokens", "score", "rejection_reasons"],
            )
            writer.writeheader()
            for candidate in candidates:
                writer.writerow(
                    {
                        "lemma": candidate.lemma,
                        "rank": candidate.rank,
                        "pos": candidate.pos,
                        "sentence": candidate.sentence,
                        "tokens": " ".join(candidate.tokens),
                        "score": f"{candidate.score:.6f}",
                        "rejection_reasons": "; ".join(candidate.rejection_reasons),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Markov-chain-only sentence generator; no retrievals.")
    parser.add_argument("--lexicon", default="stg_words_spa.csv")
    parser.add_argument("--models-dir", default="models_rebuild2")
    parser.add_argument("--out", default="outputs/markov_only_sentences.csv")
    parser.add_argument("--review-out", default="outputs/markov_only_review.csv")
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--pos", default=None)
    parser.add_argument("--lemma", action="append", default=[])
    parser.add_argument("--attempts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument(
        "--seed-mode",
        choices=["grammar", "target", "mixed"],
        default="grammar",
        help="grammar uses target-aware Markov seeds; target starts from sampled target context only.",
    )
    args = parser.parse_args()

    rows = load_lexicon(Path(args.lexicon))
    targets = select_targets(
        rows,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        limit=args.limit,
        pos=normalize_token(args.pos) if args.pos else None,
        lemmas=args.lemma,
    )
    if not targets:
        raise SystemExit("No targets selected.")

    print("Loading bigram Markov model...", flush=True)
    bi_next, bi_totals = load_bigram_model(Path(args.models_dir))
    generator = MarkovOnlyGenerator(
        rows,
        bi_next,
        bi_totals,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"Building predecessor index for {len(targets)} targets...", flush=True)
    generator.prepare_targets(targets)

    candidates: List[MarkovCandidate] = []
    for idx, target in enumerate(targets, start=1):
        candidate = generator.generate_for_target(target, attempts=args.attempts, seed_mode=args.seed_mode)
        candidates.append(candidate)
        if idx % 10 == 0 or idx == len(targets):
            print(
                f"[progress] {idx}/{len(targets)} lemma={target.lemma} "
                f"ok={bool(candidate.sentence)}",
                flush=True,
            )

    write_outputs(candidates, Path(args.out), Path(args.review_out) if args.review_out else None)

    nonempty = sum(1 for candidate in candidates if candidate.sentence)
    print(f"Generated: {len(candidates)}", flush=True)
    print(f"  markov_chain_nonempty: {nonempty}", flush=True)
    print(f"  blank_needs_review:    {len(candidates) - nonempty}", flush=True)
    print(f"Saved: {args.out}", flush=True)
    if args.review_out:
        print(f"Review: {args.review_out}", flush=True)
    for candidate in [c for c in candidates if c.sentence][:5]:
        print(f"  {candidate.lemma:<12} {candidate.sentence}", flush=True)


if __name__ == "__main__":
    main()
