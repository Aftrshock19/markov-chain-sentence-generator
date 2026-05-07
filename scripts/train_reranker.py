#!/usr/bin/env python3
"""Train a logistic-regression reranker to score Spanish sentence quality.

Uses the 39,761 human-reviewed sentences in outputs/sentances_review.csv as
training data. The target is all-three-flags-positive (grammatical + natural +
learner_clear). Features come from scripts/features.py and include the KN-LM
trigram log-probability.
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler


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
    X = np.asarray([[d[k] for k in feature_names] for d in feats_dicts], dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    # Train/test split (stratified by class, deterministic)
    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr, te = idx[:split], idx[split:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs")
    clf.fit(X_tr_s, y_tr)
    probs_te = clf.predict_proba(X_te_s)[:, 1]
    preds_te = (probs_te >= 0.5).astype(int)

    print("[reranker] test-set AUC:", roc_auc_score(y_te, probs_te), file=sys.stderr)
    print(classification_report(y_te, preds_te, digits=3), file=sys.stderr)

    # Show top features
    coefs = clf.coef_[0]
    names_and_coef = sorted(zip(feature_names, coefs), key=lambda kv: kv[1])
    print("\nTop-10 negative (bad-sentence signal):", file=sys.stderr)
    for name, c in names_and_coef[:10]:
        print(f"  {name:<22} {c:+.3f}", file=sys.stderr)
    print("\nTop-10 positive (good-sentence signal):", file=sys.stderr)
    for name, c in names_and_coef[-10:]:
        print(f"  {name:<22} {c:+.3f}", file=sys.stderr)

    model = dict(
        feature_names=feature_names,
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        coef=coefs.tolist(),
        intercept=float(clf.intercept_[0]),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(model, f, protocol=4)
    print(f"[reranker] saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
