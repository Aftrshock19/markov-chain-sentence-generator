#!/usr/bin/env python3
"""Load and query the corpus-derived collocation table (build_collocations.py).

Provides a count-based, no-LLM coherence signal:

  * sentence_coherence(tokens) -> features dict (mean/min PMI, violation counts)
  * incoherent_reason(tokens)  -> "incoherent_collocation" for a CONFIDENT
                                  selectional-preference violation, else None

PMI(a, b) = log( c(a,b) * N / (c(a) * c(b)) ). Positive => the two content words
attract (beber + agua); strongly negative or never-co-occurring => they repel
(hablar + agua), which is the fingerprint of fluent nonsense.

The gate is deliberately HIGH-PRECISION: it only fires when both words are
frequent enough that a near-zero co-occurrence is meaningful (not data
sparsity). That keeps it from shrinking yield on rarer target vocabulary.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Lazily import the same content-word definition the builder used.
from build_collocations import COPULAS, is_content  # noqa: E402

# A pair never seen together gets this floor instead of -inf.
_PMI_FLOOR = -6.0


class CollocationModel:
    def __init__(self, data: dict):
        self.n_sent: int = data["n_sent"]
        self.window: int = data.get("window", 4)
        self.unigram: Dict[str, int] = data["unigram"]
        self.window_pairs: Dict[Tuple[str, str], int] = data["window_pairs"]
        self.copula_pairs: Dict[Tuple[str, str], int] = data["copula_pairs"]
        # PMI normalizer: total content-token mass (falls back to n_sent).
        self.total = data.get("total_content_tokens") or self.n_sent

    @classmethod
    def load(cls, path: Path) -> "CollocationModel":
        with Path(path).open("rb") as f:
            return cls(pickle.load(f))

    # --- raw lookups ---------------------------------------------------------
    def _key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def known(self, w: str) -> bool:
        return w in self.unigram

    def _pmi(self, a: str, b: str, table: Dict[Tuple[str, str], int]) -> Optional[float]:
        """PMI for a content pair, or None when either word is unknown/rare
        (no signal). Returns the floor when both are known but never co-occur."""
        ca = self.unigram.get(a)
        cb = self.unigram.get(b)
        if not ca or not cb:
            return None
        c = table.get(self._key(a, b), 0)
        if c == 0:
            return _PMI_FLOOR
        return math.log(c * self.total / (ca * cb))

    def window_pmi(self, a: str, b: str) -> Optional[float]:
        return self._pmi(a, b, self.window_pairs)

    def copula_pmi(self, a: str, b: str) -> Optional[float]:
        return self._pmi(a, b, self.copula_pairs)

    # --- sentence-level relations -------------------------------------------
    def _bound_window_pairs(self, toks: Sequence[str]) -> List[Tuple[str, str]]:
        """All content-word pairs within `window` tokens — mirrors the builder, so
        a verb and its object still pair up across an intervening quantifier
        ("hablar un poco de agua" -> hablar+agua)."""
        content = [(i, w) for i, w in enumerate(toks) if is_content(w)]
        out = []
        for a in range(len(content)):
            ia, wa = content[a]
            for b in range(a + 1, len(content)):
                ib, wb = content[b]
                if ib - ia > self.window:
                    break
                if wa != wb:
                    out.append((wa, wb))
        return out

    def _copula_relations(self, toks: Sequence[str]) -> List[Tuple[str, str]]:
        out = []
        for i, w in enumerate(toks):
            if w in COPULAS:
                left = next((toks[j] for j in range(i - 1, -1, -1) if is_content(toks[j])), None)
                right = next((toks[j] for j in range(i + 1, len(toks)) if is_content(toks[j])), None)
                if left and right and left != right:
                    out.append((left, right))
        return out

    def sentence_coherence(self, tokens: Sequence[str]) -> Dict[str, float]:
        toks = [t.lower() for t in tokens if t]
        win = [self.window_pmi(a, b) for a, b in self._bound_window_pairs(toks)]
        win = [p for p in win if p is not None]
        cop = [self.copula_pmi(a, b) for a, b in self._copula_relations(toks)]
        cop = [p for p in cop if p is not None]
        allp = win + cop
        n_known = len(allp)
        mean_pmi = sum(allp) / n_known if n_known else 0.0
        min_pmi = min(allp) if allp else 0.0
        # A "violation" is a confidently-repelling bound pair.
        win_viol = sum(1 for p in win if p <= -1.5)
        cop_viol = sum(1 for p in cop if p <= -1.5)
        return {
            "colloc_mean_pmi": mean_pmi,
            "colloc_min_pmi": min_pmi,
            "colloc_n_known": float(n_known),
            "colloc_window_violations": float(win_viol),
            "colloc_copula_violations": float(cop_viol),
        }

    # Words rarer than this contribute no scoring signal: a zero co-occurrence
    # on a rare word is sparsity, not incoherence, so penalizing it would tank
    # perfectly good sentences ("las casas son bonitas").
    FREQ_CONF = 150

    def coherence_score(self, tokens: Sequence[str]) -> float:
        """A robust re-ranking term in [-2.5, 3]. Averages PMI over only those
        bound pairs where BOTH words are frequent enough that the count is
        trustworthy; a frequent-pair zero co-occurrence is capped at -2.5 (a
        confident "these repel"), not the -6 sparsity floor. 0.0 = no signal."""
        toks = [t.lower() for t in tokens if t]
        scores: List[float] = []
        for pairs, table in ((self._bound_window_pairs(toks), self.window_pairs),
                             (self._copula_relations(toks), self.copula_pairs)):
            for a, b in pairs:
                ca = self.unigram.get(a)
                cb = self.unigram.get(b)
                if not ca or not cb or ca < self.FREQ_CONF or cb < self.FREQ_CONF:
                    continue
                c = table.get(self._key(a, b), 0)
                pmi = -2.5 if c == 0 else max(-2.5, math.log(c * self.total / (ca * cb)))
                scores.append(min(3.0, pmi))
        return sum(scores) / len(scores) if scores else 0.0

    def incoherent_reason(
        self,
        tokens: Sequence[str],
        min_freq: int = 200,
        zero_only: bool = False,
    ) -> Optional[str]:
        """High-precision selectional-violation gate. Fires only when two FREQUENT
        content words are bound together yet (almost) never co-occur in the
        corpus — the verb->object / subject->predicate nonsense fingerprint.

        zero_only=True restricts to pairs with literally zero co-occurrence
        (maximum precision). Otherwise also fires on PMI <= -2.5.
        """
        # Only the WINDOW relation is used for the hard gate. The copula relation
        # is too sparse for noun=adjective predications ("amigos son buenos") and,
        # because pairs are stored unordered, we can't filter predicate adjectives
        # out — so it produced false positives. It still contributes to the soft
        # coherence_score, where a single capped pair can't sink a good sentence.
        toks = [t.lower() for t in tokens if t]
        for a, b in self._bound_window_pairs(toks):
            ca = self.unigram.get(a)
            cb = self.unigram.get(b)
            if not ca or not cb or ca < min_freq or cb < min_freq:
                continue  # not frequent enough for a zero to be meaningful
            c = self.window_pairs.get(self._key(a, b), 0)
            if c == 0:
                return "incoherent_collocation"
            if not zero_only:
                pmi = math.log(c * self.total / (ca * cb))
                if pmi <= -2.5:
                    return "incoherent_collocation"
        return None


_CACHE: Dict[str, CollocationModel] = {}


def load_collocations(path: Path) -> Optional[CollocationModel]:
    """Cached loader. Returns None (signal disabled) if the table is absent."""
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    p = Path(path)
    if not p.exists():
        return None
    model = CollocationModel.load(p)
    _CACHE[key] = model
    return model
