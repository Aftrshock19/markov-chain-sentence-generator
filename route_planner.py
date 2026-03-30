from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoutePlan:
    lemma: str
    pos_family: str
    surface_class: str
    semantic_subclass: Optional[str]
    allowed_route_families: List[str]
    support_constraints: Dict[str, Any] = field(default_factory=dict)
    exact_surface_required: bool = True


_SAFE_CONDITIONAL_CANONICALS = {"gustar", "deber", "haber", "poder"}
_SAFE_SUBJUNCTIVE_CANONICALS = {"estar", "venir", "ir", "salir", "hacer", "decir", "dar"}
_SAFE_PARTICIPLE_CANONICALS = {"dar", "decir", "hacer", "leer", "comprar", "buscar", "tener"}


def build_route_plan(
    entry: Any,
    canonical_lemma: str = "",
    morph: Optional[Dict[str, str]] = None,
    surface_features: Optional[Dict[str, Any]] = None,
    pos_family: str = "v",
    exact_surface_required: bool = True,
) -> RoutePlan:
    features = dict(surface_features or {})
    canonical = (canonical_lemma or features.get("canonical_lemma") or getattr(entry, "canonical_lemma", "") or getattr(entry, "lemma", "")).strip().lower()
    lemma = (getattr(entry, "lemma", "") if not isinstance(entry, dict) else entry.get("lemma", "")).strip().lower()
    families: List[str] = []
    semantic_subclass: Optional[str] = None
    surface_class = features.get("surface_class") or "unknown"

    if pos_family != "v":
        return RoutePlan(lemma=lemma, pos_family=pos_family, surface_class=surface_class, semantic_subclass=None, allowed_route_families=[], support_constraints={}, exact_surface_required=exact_surface_required)

    if features.get("is_infinitive"):
        surface_class = "infinitive"
        semantic_subclass = "licensed_infinitive"
        families = ["infinitive"]
    elif features.get("is_gerund"):
        surface_class = "gerund"
        semantic_subclass = "continuation" if canonical == "decir" else "progressive"
        families = ["gerund"]
    elif features.get("is_participle"):
        surface_class = "participle"
        semantic_subclass = "perfect_auxiliary" if canonical in _SAFE_PARTICIPLE_CANONICALS else "conservative_participle"
        families = ["participle"] if canonical in _SAFE_PARTICIPLE_CANONICALS else []
    elif features.get("is_conditional"):
        surface_class = "conditional"
        if canonical == "gustar":
            semantic_subclass = "experiencer_desiderative"
        elif canonical == "deber":
            semantic_subclass = "modal_obligation"
        elif canonical == "haber":
            semantic_subclass = "existential_haber"
        else:
            semantic_subclass = "generic_conditional"
        families = ["conditional"] if canonical in _SAFE_CONDITIONAL_CANONICALS else []
    elif features.get("is_subjunctive_finite"):
        surface_class = "subjunctive_finite"
        if canonical == "estar":
            semantic_subclass = "licensed_locative_subjunctive"
        elif canonical in {"venir", "ir", "salir"}:
            semantic_subclass = "licensed_motion_subjunctive"
        else:
            semantic_subclass = "licensed_subjunctive"
        families = ["subjunctive"] if canonical in _SAFE_SUBJUNCTIVE_CANONICALS else ["subjunctive"]
    else:
        if features.get("likely_imperative_candidate"):
            surface_class = "imperative_candidate"
            semantic_subclass = "safe_command"
            families.append("imperative")
        if features.get("finite_present_candidate"):
            if canonical == "deber":
                semantic_subclass = "modal_obligation"
            elif canonical == "quedar":
                semantic_subclass = "existential_quedar"
            elif canonical == "hacer":
                semantic_subclass = "transitive_simple"
            else:
                semantic_subclass = semantic_subclass or "finite_present"
            families.append("finite_present")

    constraints = {
        "canonical_lemma": canonical,
        "person_code": features.get("person_code"),
        "verb_form": features.get("verb_form", ""),
        "mood": features.get("mood", ""),
        "tense": features.get("tense", ""),
    }
    return RoutePlan(
        lemma=lemma,
        pos_family=pos_family,
        surface_class=surface_class,
        semantic_subclass=semantic_subclass,
        allowed_route_families=families,
        support_constraints=constraints,
        exact_surface_required=exact_surface_required,
    )
