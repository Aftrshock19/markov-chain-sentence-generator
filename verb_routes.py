VERB_ROUTE_TEMPLATE_PREFIX = "verb_route_"

VERB_ROUTE_FAMILY_BONUS = {
    "conditional": 0.55,
    "subjunctive": 0.55,
    "imperative": 0.45,
    "finite_present": 0.40,
    "infinitive": 0.40,
    "gerund": 0.35,
    "participle": 0.30,
}


def template_id_for_route(route_family: str, variant: str) -> str:
    return f"{VERB_ROUTE_TEMPLATE_PREFIX}{route_family}_{variant}"


def route_family_from_template_id(template_id: str) -> str:
    if not template_id or not template_id.startswith(VERB_ROUTE_TEMPLATE_PREFIX):
        return ""
    rest = template_id[len(VERB_ROUTE_TEMPLATE_PREFIX):]
    return rest.split("_", 1)[0] if rest else ""
