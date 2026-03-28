#!/usr/bin/env python3
import argparse
import csv
import os
import pickle
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODEL_PATH = os.path.join("models", "reranker.pkl")
DEFAULT_VENDOR_DIR = os.path.join(os.path.dirname(__file__), ".vendor")
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
STRIP_RE = re.compile(r'^[¿¡"“”\'\(\[\{]+|[.,;:!?"”\'\)\]\}]+$')
PRONOUN_STARTERS = {
    "yo",
    "tú",
    "él",
    "ella",
    "nosotros",
    "nosotras",
    "ellos",
    "ellas",
    "usted",
    "ustedes",
}
COPULA_FORMS = {
    "ser",
    "soy",
    "eres",
    "es",
    "somos",
    "sois",
    "son",
    "fui",
    "fuiste",
    "fue",
    "fuimos",
    "fueron",
    "era",
    "eras",
    "éramos",
    "eran",
    "sea",
    "seas",
    "seamos",
    "sean",
    "será",
    "serían",
    "estar",
    "estoy",
    "estás",
    "está",
    "estamos",
    "están",
    "estaba",
    "estaban",
    "estuve",
    "estuviste",
    "estuvo",
    "estuvimos",
    "estuvieron",
    "esté",
    "estés",
    "estemos",
    "estén",
    "estaría",
    "estarán",
}


def normalize_token(token: str) -> str:
    token = (token or "").strip().lower()
    token = STRIP_RE.sub("", token)
    return token


def word_tokens(sentence: str) -> List[str]:
    out: List[str] = []
    for raw in (sentence or "").split():
        if not WORD_RE.search(raw):
            continue
        token = normalize_token(raw)
        if token:
            out.append(token)
    return out


def _value(item: Any, key: str, default: Any = "") -> Any:
    if hasattr(item, key):
        value = getattr(item, key)
    elif isinstance(item, dict):
        value = item.get(key, default)
    else:
        value = default
    return default if value is None else value


def _float_value(item: Any, key: str, default: float = 0.0) -> float:
    value = _value(item, key, default)
    if value in {"", None}:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_value(item: Any, key: str, default: int = 0) -> int:
    value = _value(item, key, default)
    if value in {"", None}:
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def candidate_feature_dict(item: Any) -> Dict[str, Any]:
    sentence = str(_value(item, "sentence", ""))
    tokens = word_tokens(sentence)
    return {
        "sentence_word_count": float(len(tokens)),
        "avg_support_rank": _float_value(item, "avg_support_rank"),
        "max_support_rank": _float_value(item, "max_support_rank"),
        "original_score": _float_value(item, "score"),
        "source_method": str(_value(item, "source_method", "")) or "__missing__",
        "template_id": str(_value(item, "template_id", "")) or "__missing__",
        "contains_aqui": int("aquí" in tokens or "aqui" in tokens),
        "contains_copula": int(any(token in COPULA_FORMS for token in tokens)),
        "starts_with_pronoun": int(bool(tokens and tokens[0] in PRONOUN_STARTERS)),
    }


def candidate_label(item: Any) -> int:
    return (
        _int_value(item, "grammatical_ok")
        + _int_value(item, "natural_ok")
        + _int_value(item, "learner_clear_ok")
    )


def load_candidate_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_training_examples(rows: Iterable[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[float]]:
    feature_rows: List[Dict[str, Any]] = []
    labels: List[float] = []
    for row in rows:
        if not str(row.get("sentence", "")).strip():
            continue
        feature_rows.append(candidate_feature_dict(row))
        labels.append(float(candidate_label(row)))
    return feature_rows, labels


def _import_sklearn():
    if os.path.isdir(DEFAULT_VENDOR_DIR) and DEFAULT_VENDOR_DIR not in sys.path:
        sys.path.insert(0, DEFAULT_VENDOR_DIR)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    return RandomForestRegressor, DictVectorizer, mean_absolute_error, r2_score, train_test_split


def train_reranker(
    candidates_csv: str,
    model_out: str = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    RandomForestRegressor, DictVectorizer, mean_absolute_error, r2_score, train_test_split = _import_sklearn()
    rows = load_candidate_rows(candidates_csv)
    feature_rows, labels = build_training_examples(rows)
    if len(feature_rows) < 5:
        raise ValueError(f"Need at least 5 usable rows to train, found {len(feature_rows)}")

    train_features, test_features, y_train, y_test = train_test_split(
        feature_rows,
        labels,
        test_size=test_size,
        random_state=random_state,
    )
    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(train_features)
    x_test = vectorizer.transform(test_features)

    model = RandomForestRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "total_rows": len(feature_rows),
    }
    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "metrics": metrics,
        "feature_names": list(vectorizer.get_feature_names_out()),
    }
    model_dir = os.path.dirname(model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump(artifact, f)
    return artifact


def load_reranker_model(model_path: str = DEFAULT_MODEL_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(model_path):
        return None
    try:
        _import_sklearn()
    except Exception:
        return None
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    if not isinstance(loaded, dict):
        return None
    if "model" not in loaded or "vectorizer" not in loaded:
        return None
    return loaded


def predict_candidate_scores(model_artifact: Dict[str, Any], candidates: Sequence[Any]) -> List[float]:
    if not candidates:
        return []
    vectorizer = model_artifact["vectorizer"]
    model = model_artifact["model"]
    feature_rows = [candidate_feature_dict(candidate) for candidate in candidates]
    matrix = vectorizer.transform(feature_rows)
    return [float(value) for value in model.predict(matrix)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a candidate sentence reranker from a candidate export CSV.")
    parser.add_argument("--candidates-csv", required=True, help="Path to the candidate export CSV.")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Pickle path for the trained reranker.")
    args = parser.parse_args()

    artifact = train_reranker(args.candidates_csv, model_out=args.model_out)
    metrics = artifact["metrics"]
    print(f"Rows: {metrics['total_rows']}")
    print(f"Train: {metrics['train_rows']}")
    print(f"Test: {metrics['test_rows']}")
    print(f"R^2: {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Saved: {args.model_out}")


if __name__ == "__main__":
    main()
