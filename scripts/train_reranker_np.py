#!/usr/bin/env python3
"""Drop-in replacement for train_reranker.py that uses NO sklearn/scipy.

The repo's conda env has numpy 2.x but scipy/scikit-learn compiled against
numpy 1.x, so `import sklearn` crashes. This script reproduces train_reranker.py
exactly — same data (outputs/sentances_review.csv), same label rule, same
featurize() call (no colloc, matching the deployed reranker), same deterministic
90/10 split (np.random.default_rng(42)), same StandardScaler, same output pickle
schema {feature_names, scaler_mean, scaler_scale, coef, intercept} — but fits the
balanced, L2-regularized logistic regression with a numpy-only IRLS (Newton)
solver instead of sklearn's lbfgs. C=1.0 => objective = sum_i s_i·logloss_i +
0.5·||w||^2 (bias unregularized), which is sklearn's parameterization.
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import featurize, tokenize  # noqa: E402
from kn_lm import KNLanguageModel  # noqa: E402


def load_rank_by_word(lexicon_path: Path):
    out = {}
    with lexicon_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lemma = (row.get("lemma") or row.get("word") or "").strip().lower()
            if not lemma:
                continue
            try:
                rank = int(float(row.get("rank") or 999999))
            except ValueError:
                rank = 999999
            out.setdefault(lemma, rank)
    return out


def iter_review(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sent = row.get("sentence", "")
            if not sent:
                continue
            try:
                rank = int(float(row.get("rank") or 999999))
            except ValueError:
                rank = 999999
            lemma = (row.get("lemma") or "").strip().lower()
            g = row.get("grammatical_ok") or "0"
            n = row.get("natural_ok") or "0"
            l_ok = row.get("learner_clear_ok") or "0"
            label = 1 if (g == "1" and n == "1" and l_ok == "1") else 0
            yield lemma, rank, sent, label


def auc_score(y, p):
    """Rank-based ROC AUC (Mann-Whitney U), numpy-only."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_irls(X, y, sample_weight, lam=1.0, n_iter=100, tol=1e-8):
    """Balanced, L2-regularized logistic regression via IRLS / Newton.

    Minimizes sum_i s_i·logloss_i + 0.5·lam·||w||^2 (bias column unregularized).
    X already has a leading bias column of ones.
    """
    n, d = X.shape
    beta = np.zeros(d)
    reg = lam * np.ones(d)
    reg[0] = 0.0  # don't regularize intercept
    prev = None
    for it in range(n_iter):
        z = X @ beta
        z = np.clip(z, -35, 35)
        p = 1.0 / (1.0 + np.exp(-z))
        W = sample_weight * p * (1.0 - p)
        W = np.maximum(W, 1e-9)
        grad = X.T @ (sample_weight * (p - y)) + reg * beta
        # Hessian = X^T diag(W) X + diag(reg)
        H = (X * W[:, None]).T @ X
        H[np.diag_indices_from(H)] += reg
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta -= delta
        loss = float((sample_weight * np.logaddexp(0, -(2 * y - 1) * z)).sum()
                     + 0.5 * (reg * beta * beta).sum())
        if prev is not None and abs(prev - loss) < tol * max(1.0, abs(prev)):
            print(f"  IRLS converged at iter {it} (loss={loss:.4f})", file=sys.stderr)
            break
        prev = loss
    return beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="outputs/sentances_review.csv")
    ap.add_argument("--lexicon", default="stg_words_spa.csv")
    ap.add_argument("--lm", default="data_clean/kn_lm.pkl")
    ap.add_argument("--out", default="data_clean/reranker.pkl")
    args = ap.parse_args()

    print("[reranker] loading KN-LM", file=sys.stderr)
    lm = KNLanguageModel(Path(args.lm))
    rank_by_word = load_rank_by_word(Path(args.lexicon))

    print("[reranker] extracting features from review data", file=sys.stderr)
    rows = list(iter_review(Path(args.review)))
    feats_dicts = []
    labels = []
    for i, (lemma, rank, sent, label) in enumerate(rows):
        toks = tokenize(sent)
        if not toks:
            continue
        lm_logp = lm.sentence_logprob(toks)
        f = featurize(toks, lemma, rank, lm_logp, rank_by_word)
        feats_dicts.append(f)
        labels.append(label)
        if i and i % 5000 == 0:
            print(f"  processed {i}/{len(rows)}", file=sys.stderr)

    feature_names = sorted(feats_dicts[0].keys())
    X = np.asarray([[d.get(k, 0.0) for k in feature_names] for d in feats_dicts], dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    print(f"[reranker] {len(y):,} rows  {int(y.sum()):,} good / {int((1-y).sum()):,} bad  "
          f"{len(feature_names)} features", file=sys.stderr)

    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr, te = idx[:split], idx[split:]

    mean = X[tr].mean(axis=0)
    scale = X[tr].std(axis=0)
    scale[scale == 0] = 1.0
    Xs = (X - mean) / scale

    # class_weight="balanced": w_c = n / (2 * n_c)
    n = len(y)
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    sw = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))

    Xb = np.column_stack([np.ones(n), Xs])  # bias column first
    beta = fit_irls(Xb[tr], y[tr], sw[tr], lam=1.0)

    intercept = float(beta[0])
    coef = beta[1:]

    # Evaluate on held-out split
    z_te = np.clip(Xb[te] @ beta, -35, 35)
    p_te = 1.0 / (1.0 + np.exp(-z_te))
    auc = auc_score(y[te], p_te)
    pred = (p_te >= 0.5).astype(int)
    tp = int(((pred == 1) & (y[te] == 1)).sum())
    fp = int(((pred == 1) & (y[te] == 0)).sum())
    fn = int(((pred == 0) & (y[te] == 1)).sum())
    tn = int(((pred == 0) & (y[te] == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(te)
    print(f"[reranker] test AUC={auc:.4f}  acc={acc:.3f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
          f"(tp={tp} fp={fp} fn={fn} tn={tn})", file=sys.stderr)

    nc = sorted(zip(feature_names, coef), key=lambda kv: kv[1])
    print("\nTop-8 negative (bad signal):", file=sys.stderr)
    for name, c in nc[:8]:
        print(f"  {name:<24} {c:+.3f}", file=sys.stderr)
    print("Top-8 positive (good signal):", file=sys.stderr)
    for name, c in nc[-8:]:
        print(f"  {name:<24} {c:+.3f}", file=sys.stderr)

    model = dict(
        feature_names=feature_names,
        scaler_mean=mean.tolist(),
        scaler_scale=scale.tolist(),
        coef=coef.tolist(),
        intercept=intercept,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(model, f, protocol=4)
    print(f"[reranker] saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
