#!/usr/bin/env python3
"""Query-side of the modified Kneser-Ney trigram LM."""
import math
import pickle
from pathlib import Path
from typing import Tuple


class KNLanguageModel:
    def __init__(self, model_path: Path):
        with Path(model_path).open("rb") as f:
            m = pickle.load(f)
        self.uni = m["uni"]
        self.bi = m["bi"]
        self.tri = m["tri"]
        self.cc_uni = m["cc_uni"]
        self.cc_bi = m["cc_bi"]
        self.total_cc_uni = max(1, m["total_cc_uni"])
        self.per_bi_strata = m["per_bi_strata"]
        self.bi_followers_total = m["bi_followers_total"]
        self.per_uni_strata = m["per_uni_strata"]
        self.uni_followers_total = m["uni_followers_total"]
        self.d1_tri, self.d2_tri, self.d3_tri = m["d_tri"]
        self.d1_bi, self.d2_bi, self.d3_bi = m["d_bi"]
        self.BOS = m["BOS"]
        self.EOS = m["EOS"]
        self.V = len(self.uni)
        # Build a followers index for fast top-k retrieval during beam search.
        self._bi_followers = None
        self._tri_followers = None

    def _disc_tri(self, c: int) -> float:
        if c >= 3:
            return self.d3_tri
        if c == 2:
            return self.d2_tri
        return self.d1_tri

    def _disc_bi(self, c: int) -> float:
        if c >= 3:
            return self.d3_bi
        if c == 2:
            return self.d2_bi
        return self.d1_bi

    def p_unigram_cont(self, w: str) -> float:
        return (self.cc_uni.get(w, 0) + 1e-9) / (self.total_cc_uni + 1e-6 * self.V)

    def p_bi_cont(self, u: str, w: str) -> float:
        """Lower-order continuation probability for trigram smoothing."""
        cc = self.cc_bi.get((u, w), 0)
        denom_counts = self.per_uni_strata.get(u)
        if cc > 0 and denom_counts is not None:
            n1, n2, n3plus = denom_counts
            denom = n1 + n2 + n3plus  # distinct w following u
            if denom > 0:
                D = self._disc_bi(cc)
                gamma = (self.d1_bi * n1 + self.d2_bi * n2 + self.d3_bi * n3plus) / max(1, self.uni_followers_total.get(u, 1))
                p_lower = self.p_unigram_cont(w)
                return max(cc - D, 0) / max(1, self.uni_followers_total.get(u, 1)) + gamma * p_lower
        # backoff completely
        return self.p_unigram_cont(w)

    def p_trigram(self, u: str, v: str, w: str) -> float:
        """p_KN(w | u, v) using modified Kneser-Ney with trigram -> bigram -> unigram backoff."""
        ctx_count = self.bi.get((u, v), 0)
        strata = self.per_bi_strata.get((u, v))
        tri_count = self.tri.get((u, v, w), 0)
        if ctx_count > 0 and strata is not None:
            n1, n2, n3plus = strata
            total_followers = self.bi_followers_total.get((u, v), ctx_count)
            D = self._disc_tri(tri_count) if tri_count > 0 else 0.0
            numer = max(tri_count - D, 0) / max(1, total_followers)
            gamma = (self.d1_tri * n1 + self.d2_tri * n2 + self.d3_tri * n3plus) / max(1, total_followers)
            return numer + gamma * self.p_bi_cont(v, w)
        return self.p_bi_cont(v, w)

    def logp_trigram(self, u: str, v: str, w: str) -> float:
        p = self.p_trigram(u, v, w)
        return math.log(max(p, 1e-30))

    def sentence_logprob(self, tokens) -> float:
        ctx = [self.BOS, self.BOS]
        total = 0.0
        for w in list(tokens) + [self.EOS]:
            total += self.logp_trigram(ctx[0], ctx[1], w)
            ctx = [ctx[1], w]
        return total

    def bi_followers(self, v: str):
        """Return dict w -> c(v, w) for building candidates."""
        if self._bi_followers is None:
            from collections import defaultdict
            idx = defaultdict(dict)
            for (a, b), c in self.bi.items():
                idx[a][b] = c
            self._bi_followers = {k: dict(v) for k, v in idx.items()}
        return self._bi_followers.get(v, {})

    def tri_followers(self, u: str, v: str):
        """Return dict w -> c(u, v, w)."""
        if self._tri_followers is None:
            from collections import defaultdict
            idx = defaultdict(dict)
            for (a, b, c), n in self.tri.items():
                idx[(a, b)][c] = n
            self._tri_followers = {k: dict(val) for k, val in idx.items()}
        return self._tri_followers.get((u, v), {})


def main():
    import sys
    lm = KNLanguageModel(Path(sys.argv[1] if len(sys.argv) > 1 else "data_clean/kn_lm.pkl"))
    test_sents = [
        "yo estoy aquí",
        "tú es posible",
        "ella está bien",
        "ella hay algo",
        "más ella está bien",
        "nos está bien",
        "usted está bien",
        "qué es eso",
        "tengo una casa",
        "tengo un vez",
    ]
    for s in test_sents:
        toks = s.split()
        lp = lm.sentence_logprob(toks)
        print(f"{lp:+9.2f}  {s}")


if __name__ == "__main__":
    main()
