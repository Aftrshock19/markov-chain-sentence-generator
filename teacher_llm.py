#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct:featherless-ai"
_SENTENCE_RE = re.compile(r"[^\n]+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_META_PREFIXES = (
    "explicación",
    "explanation",
    "aquí",
    "here are",
    "sentences:",
    "oraciones:",
    "nota:",
    "note:",
)


@dataclass
class TeacherConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 45.0
    temperature: float = 0.35
    max_tokens: int = 240

    @classmethod
    def from_env(cls) -> "TeacherConfig":
        api_key = os.environ.get("HF_TOKEN", "").strip()
        return cls(
            api_key=api_key,
            base_url=os.environ.get("HF_ROUTER_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
            model=os.environ.get("HF_ROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            timeout=float(os.environ.get("HF_ROUTER_TIMEOUT", "45") or 45),
            temperature=float(os.environ.get("HF_ROUTER_TEMPERATURE", "0.35") or 0.35),
            max_tokens=int(float(os.environ.get("HF_ROUTER_MAX_TOKENS", "240") or 240)),
        )


class TeacherLLM:
    def __init__(self, config: Optional[TeacherConfig] = None):
        self.config = config or TeacherConfig.from_env()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if OpenAI is None:
                raise RuntimeError("openai package is not installed. Install it before using the teacher backend.")
            if not self.config.api_key:
                raise RuntimeError(
                    "HF_TOKEN is missing. Set HF_TOKEN to use the offline teacher collection pipeline."
                )
            self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key, timeout=self.config.timeout)
        return self._client

    def provider_summary(self) -> str:
        return (
            f"base_url={self.config.base_url} model={self.config.model} "
            f"timeout={self.config.timeout} temperature={self.config.temperature} max_tokens={self.config.max_tokens}"
        )

    def generate_teacher_candidates(
        self,
        lemma: str,
        rank: int,
        pos: str,
        translation: Optional[str] = None,
        n: int = 20,
        band: Optional[str] = None,
    ) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=self._messages_for_request(lemma=lemma, rank=rank, pos=pos, translation=translation, n=n, band=band),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            print(f"[teacher_llm] request failed for {lemma}: {exc}", file=sys.stderr)
            return []

        text = ""
        try:
            text = response.choices[0].message.content or ""
        except Exception:
            text = ""
        return self._parse_candidates(text, lemma=lemma, n=n)

    def _messages_for_request(
        self,
        lemma: str,
        rank: int,
        pos: str,
        translation: Optional[str],
        n: int,
        band: Optional[str],
    ):
        details = [
            f"target word: {lemma}",
            f"rank: {rank}",
            f"part of speech: {pos or 'unknown'}",
            f"target CEFR-style band hint: {band or 'unknown'}",
        ]
        if translation:
            details.append(f"English gloss: {translation}")

        return [
            {
                "role": "system",
                "content": (
                    "You generate very short natural Spanish example sentences for language learners. "
                    "Return only plain sentences and nothing else."
                ),
            },
            {
                "role": "user",
                "content": self._build_user_prompt(lemma=lemma, pos=pos, n=n, details=details),
            },
        ]

    def _build_user_prompt(self, lemma: str, pos: str, n: int, details: Iterable[str]) -> str:
        pos_hint = self._pos_hint(pos)
        joined = "\n".join(f"- {item}" for item in details)
        return (
            "Generate learner-friendly Spain Spanish example sentences.\n"
            "Requirements:\n"
            f"1. Output exactly {n} sentences.\n"
            "2. Output exactly one sentence per line.\n"
            f"3. Every sentence must contain the exact word: {lemma}\n"
            "4. Use Spain Spanish.\n"
            "5. Keep each sentence short and simple.\n"
            "6. Prefer common surrounding vocabulary.\n"
            "7. Avoid subordinate clauses, idioms, literary language, rare words, and names unless necessary.\n"
            "8. Do not explain anything.\n"
            "9. Do not number the sentences.\n"
            f"10. {pos_hint}\n\n"
            "Target details:\n"
            f"{joined}\n"
        )

    def _pos_hint(self, pos: str) -> str:
        pos = (pos or "").strip().lower()
        if pos == "v":
            return "For verbs, prefer common finite or infinitive-friendly frames with simple objects or complements."
        if pos == "n":
            return "For nouns, prefer article plus noun patterns and simple predicates."
        if pos == "adj":
            return "For adjectives, prefer ser or estar patterns when natural."
        return "For function words and other items, prefer ultra simple high-frequency sentence patterns."

    def _parse_candidates(self, text: str, lemma: str, n: int) -> List[str]:
        lemma_norm = _norm(lemma)
        seen = set()
        out: List[str] = []
        for raw in _SENTENCE_RE.findall(text or ""):
            sentence = self._clean_line(raw)
            if not sentence:
                continue
            if any(sentence.lower().startswith(prefix) for prefix in _META_PREFIXES):
                continue
            if lemma_norm not in {_norm(tok) for tok in _tokenish(sentence)}:
                continue
            key = _norm(sentence)
            if key in seen:
                continue
            seen.add(key)
            out.append(sentence)
            if len(out) >= n:
                break
        return out

    def _clean_line(self, raw: str) -> str:
        line = (raw or "").strip()
        line = _BULLET_RE.sub("", line)
        line = line.strip().strip('"“”\'')
        line = re.sub(r"\s+", " ", line)
        if not line:
            return ""
        if line.count(".") == 0 and line[-1] not in "!?":
            line += "."
        return line



def _tokenish(sentence: str) -> List[str]:
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", sentence or "")



def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())



def generate_teacher_candidates(
    lemma: str,
    rank: int,
    pos: str,
    translation: Optional[str] = None,
    n: int = 20,
    band: Optional[str] = None,
    config: Optional[TeacherConfig] = None,
) -> List[str]:
    return TeacherLLM(config=config).generate_teacher_candidates(
        lemma=lemma,
        rank=rank,
        pos=pos,
        translation=translation,
        n=n,
        band=band,
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the Hugging Face router teacher backend.")
    parser.add_argument("--lemma", default="casa")
    parser.add_argument("--rank", type=int, default=500)
    parser.add_argument("--pos", default="n")
    parser.add_argument("--translation", default="house")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    teacher = TeacherLLM()
    print(teacher.provider_summary())
    rows = teacher.generate_teacher_candidates(
        lemma=args.lemma,
        rank=args.rank,
        pos=args.pos,
        translation=args.translation,
        n=args.n,
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
