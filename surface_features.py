from typing import Any, Dict, Optional

_PERSON_NUMBER_TO_CODE = {
    ("1", "Sing"): "1sg",
    ("2", "Sing"): "2sg",
    ("3", "Sing"): "3sg",
    ("1", "Plur"): "1pl",
    ("2", "Plur"): "2pl",
    ("3", "Plur"): "3pl",
}

_SAFE_IMPERATIVE_CANONICALS = {
    "tomar",
    "beber",
    "escuchar",
    "mirar",
    "leer",
    "venir",
    "salir",
}

_SUBJUNCTIVE_HINTS = {
    "venga",
    "esté",
    "sea",
    "vaya",
    "haga",
    "diga",
}


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def infer_surface_features(
    entry: Any,
    canonical_lemma: str = "",
    morph: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    morph = dict(morph or {})
    surface = _norm(getattr(entry, "lemma", "") if not isinstance(entry, dict) else entry.get("lemma", ""))
    canonical = _norm(canonical_lemma or (getattr(entry, "canonical_lemma", "") if not isinstance(entry, dict) else entry.get("canonical_lemma", "")))
    verb_form = morph.get("VerbForm", "")
    mood = morph.get("Mood", "")
    tense = morph.get("Tense", "")
    person = str(morph.get("Person", "") or "")
    number = morph.get("Number", "")
    person_code = _PERSON_NUMBER_TO_CODE.get((person, number))

    is_infinitive = verb_form == "Inf" or (not morph and surface.endswith(("ar", "er", "ir")))
    is_gerund = verb_form == "Ger" or (not morph and surface.endswith(("ando", "iendo", "yendo")))
    is_participle = verb_form == "Part" or (not morph and surface.endswith(("ado", "ido", "to", "so", "cho")))
    is_conditional = mood == "Cnd" or tense == "Cnd" or surface.endswith(("ría", "rías", "ríamos", "ríais", "rían"))
    is_subjunctive_finite = (verb_form == "Fin" and mood == "Sub") or surface in _SUBJUNCTIVE_HINTS

    likely_imperative_candidate = False
    if mood == "Imp":
        likely_imperative_candidate = True
    elif canonical in _SAFE_IMPERATIVE_CANONICALS and verb_form == "Fin" and mood in {"", "Ind"}:
        if canonical.endswith("ar") and surface.endswith("a"):
            likely_imperative_candidate = True
        elif canonical.endswith(("er", "ir")) and surface.endswith("e"):
            likely_imperative_candidate = True

    finite_present_candidate = False
    if verb_form == "Fin" and mood in {"", "Ind"} and tense in {"", "Pres"} and not is_subjunctive_finite and not is_conditional:
        finite_present_candidate = True
    elif not morph and not (is_infinitive or is_gerund or is_participle or is_subjunctive_finite or is_conditional):
        finite_present_candidate = True

    surface_class = "unknown"
    if is_infinitive:
        surface_class = "infinitive"
    elif is_gerund:
        surface_class = "gerund"
    elif is_participle:
        surface_class = "participle"
    elif is_conditional:
        surface_class = "conditional"
    elif is_subjunctive_finite:
        surface_class = "subjunctive_finite"
    elif likely_imperative_candidate:
        surface_class = "imperative_candidate"
    elif finite_present_candidate:
        surface_class = "finite_present"

    return {
        "surface": surface,
        "canonical_lemma": canonical,
        "verb_form": verb_form,
        "mood": mood,
        "tense": tense,
        "person": person,
        "number": number,
        "person_code": person_code,
        "is_infinitive": is_infinitive,
        "is_gerund": is_gerund,
        "is_participle": is_participle,
        "is_conditional": is_conditional,
        "is_subjunctive_finite": is_subjunctive_finite,
        "likely_imperative_candidate": likely_imperative_candidate,
        "finite_present_candidate": finite_present_candidate,
        "surface_class": surface_class,
    }
