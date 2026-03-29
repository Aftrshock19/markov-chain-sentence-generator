#!/usr/bin/env python3
import argparse
import csv
import math
import os
import pickle
import random
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable, Any

from reranker import load_reranker_model, predict_candidate_scores


WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
STRIP_RE = re.compile(r'^[¿¡"“”\'\(\[\{]+|[.,;:!?"”\'\)\]\}]+$')
TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+|[¿¡.,;:!?\"“”'()\[\]{}]")
ARTICLE_SET = {"el", "la", "los", "las", "un", "una", "unos", "unas", "este", "esta", "estos", "estas"}
DEMONSTRATIVE_SET = {"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas"}
PRONOUNS = ["yo", "tú", "él", "ella", "nosotros", "ellos"]
COMMON_ADVERBS = ["bien", "mucho", "siempre", "ahora", "aquí"]
CONTEXT_DEPENDENT_OPENERS = {"si", "que", "cuando", "aunque", "porque", "como", "pero", "y", "o"}
DEICTIC_SUBJECTS = {"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "esto", "eso"}
VERB_OBJECT_PREFS = {
    "comer": ["food"],
    "beber": ["drink", "food"],
    "leer": ["text", "object"],
    "visitar": ["place"],
    "vivir": ["place"],
    "comprar": ["object", "food", "clothing"],
    "abrir": ["object"],
    "cerrar": ["object"],
    "ver": ["person", "animal", "object"],
    "conocer": ["person", "place"],
    "buscar": ["person", "object"],
    "escribir": ["text"],
    "llevar": ["clothing", "object"],
    "conducir": ["vehicle"],
    "tener": ["object", "place", "person", "animal", "food", "clothing", "text"],
    "querer": ["object", "place", "person", "animal", "food", "clothing", "text"],
    "hacer": ["object", "food", "text"],
}
ADJ_SUBJECT_PREFS = {
    "cansado": ["person", "animal"],
    "grande": ["object", "place", "animal"],
    "pequeño": ["object", "animal"],
    "difícil": ["abstract", "activity"],
    "delicioso": ["food", "drink"],
    "rápido": ["person", "animal", "vehicle"],
    "bonito": ["object", "place", "person"],
}
COPULA_FORMS = {"soy", "eres", "es", "somos", "son", "estoy", "estás", "está", "estamos", "están"}
PERSON_NUMBER_TO_CODE = {
    ("1", "Sing"): "1sg",
    ("2", "Sing"): "2sg",
    ("3", "Sing"): "3sg",
    ("1", "Plur"): "1pl",
    ("2", "Plur"): "2pl",
    ("3", "Plur"): "3pl",
}
PERSON_CODE_TO_SUBJECT = {
    "1sg": "yo",
    "2sg": "tú",
    "3sg": "ella",
    "1pl": "nosotros",
    "2pl": "ustedes",
    "3pl": "ellos",
}
SUBJECT_FEATURES = {
    "yo": {"person_code": "1sg", "gender": None, "number": "sg"},
    "tú": {"person_code": "2sg", "gender": None, "number": "sg"},
    "él": {"person_code": "3sg", "gender": "m", "number": "sg"},
    "ella": {"person_code": "3sg", "gender": "f", "number": "sg"},
    "usted": {"person_code": "3sg", "gender": None, "number": "sg"},
    "nosotros": {"person_code": "1pl", "gender": "m", "number": "pl"},
    "nosotras": {"person_code": "1pl", "gender": "f", "number": "pl"},
    "vosotros": {"person_code": "2pl", "gender": "m", "number": "pl"},
    "vosotras": {"person_code": "2pl", "gender": "f", "number": "pl"},
    "ustedes": {"person_code": "3pl", "gender": None, "number": "pl"},
    "ellos": {"person_code": "3pl", "gender": "m", "number": "pl"},
    "ellas": {"person_code": "3pl", "gender": "f", "number": "pl"},
}
PRONOUN_PERSON_NUMBER = {
    "yo": ("1", "Sing"),
    "tú": ("2", "Sing"),
    "él": ("3", "Sing"),
    "ella": ("3", "Sing"),
    "usted": ("3", "Sing"),
    "nosotros": ("1", "Plur"),
    "nosotras": ("1", "Plur"),
    "vosotros": ("2", "Plur"),
    "vosotras": ("2", "Plur"),
    "ustedes": ("3", "Plur"),
    "ellos": ("3", "Plur"),
    "ellas": ("3", "Plur"),
}
PREPOSITIONAL_SUBJECT_PRONOUNS = {"mí", "ti", "sí"}
PAST_TIME_ADVERBS = {"ayer", "anoche"}
FUTURE_TIME_ADVERBS = {"mañana"}
QUANTIFIER_DETERMINERS = {
    "todo", "toda", "todos", "todas",
    "mucho", "mucha", "muchos", "muchas",
    "otro", "otra", "otros", "otras",
}
SHORT_SEMANTIC_NONSENSE_PATTERNS = (
    "va a ser nada",
    "voy a ser nada",
    "vamos a ser nada",
    "van a ser nada",
    "es un poco de",
)
PUNCT_ATTACH_LEFT = {".", ",", ";", ":", "!", "?", "%", "…"}
PUNCT_ATTACH_RIGHT = {"¿", "¡", "(", "[", "{"}
SPECIAL_VERB_LEMMAS = {"ser", "estar", "haber", "ir", "poder", "saber", "creer", "querer", "hacer", "tener"}
PLACE_PREP_VERBS = {"ir", "venir", "vivir", "llegar", "entrar", "salir", "quedar"}
ARTICLE_FEATURES = {
    "el": ("m", "sg"),
    "la": ("f", "sg"),
    "los": ("m", "pl"),
    "las": ("f", "pl"),
    "un": ("m", "sg"),
    "una": ("f", "sg"),
    "unos": ("m", "pl"),
    "unas": ("f", "pl"),
    "este": ("m", "sg"),
    "esta": ("f", "sg"),
    "estos": ("m", "pl"),
    "estas": ("f", "pl"),
    "ese": ("m", "sg"),
    "esa": ("f", "sg"),
    "esos": ("m", "pl"),
    "esas": ("f", "pl"),
}
GOOD_EXISTENTIAL_CLASSES = {"object", "place", "person", "animal", "food", "clothing", "vehicle", "text"}
BAD_EXISTENTIAL_CLASSES = {"time", "abstract", "activity"}
LOCATABLE_NOUN_CLASSES = {"object", "place", "person", "animal", "food", "clothing", "vehicle", "text"}
SAFE_POSSESSIBLE_NOUN_CLASSES = {"object", "place", "person", "animal", "food", "clothing", "text"}
SER_ADJ_TEMPLATE_NOUN_CLASSES = {"animal", "food", "drink", "object", "clothing", "place", "person"}
WEAK_COPULA_ADJECTIVES = {"solo", "sola", "solos", "solas"}
SAFE_SER_ADJECTIVES = {
    "bueno", "malo", "grande", "pequeño", "bonito", "rápido", "feliz", "listo",
    "cansado", "delicioso", "importante", "nuevo", "fuerte", "largo", "especial",
    "posible", "necesario", "increíble", "imposible", "diferente", "difícil",
    "fácil", "pobre", "rico", "real", "simple", "terrible", "horrible",
    "público", "personal", "normal", "natural", "extraño", "divertido",
    "interesante", "peligroso", "popular", "humano", "común", "viejo", "alto",
    "bajo", "joven", "corto", "limpio", "sucio", "tranquilo", "perfecto",
    "inteligente", "serio", "seguro",
}
SAFE_OBJECT_CLASSES_BY_VERB = {
    "tener": {"object", "place", "person", "animal", "food", "clothing", "text"},
    "querer": {"object", "place", "person", "animal", "food", "clothing", "text"},
    "hacer": {"object", "food", "text"},
}
REJECT_OBJECT_CLASSES_BY_VERB = {
    "haber": BAD_EXISTENTIAL_CLASSES,
    "tener": {"time", "abstract", "activity"},
    "querer": {"time", "abstract"},
    "hacer": {"time", "abstract", "place"},
}
SAFE_OBJECT_LEMMAS_BY_VERB = {
    "hacer": ["trabajo", "comida", "pregunta", "plan", "foto"],
    "tener": ["casa", "libro", "perro", "comida", "trabajo"],
    "querer": ["casa", "libro", "perro", "comida", "ayuda"],
}
EASY_INFINITIVE_LEMMAS = ["leer", "comer", "beber", "ver", "comprar", "visitar", "buscar", "abrir", "cerrar", "llevar", "ir"]
SUPPORTED_TEMPLATE_MOODS = {"Ind", "Sub", "Cnd"}
SUBJUNCTIVE_TRIGGER_TOKENS = {"que", "ojalá", "quizá", "quizás"}
LOW_VALUE_TEMPLATE_IDS = {"seeded_noun_here", "n_a1_2", "seeded_verb_adv", "v_a1_2"}
LOW_VALUE_NOUN_TEMPLATE_LEMMAS = {"vez", "verdad", "tiempo", "pasado", "gracias"}
SAFE_HABER_PERFECT_VERBS = {"tener", "hacer", "leer", "comprar", "buscar"}
EL_ARTICLE_FEMININE_NOUNS = {"agua", "alma", "arma", "águila", "aula", "hada", "hambre"}
HERE_TEMPLATE_SAFE_LEMMAS = {"casa", "puerta"}
SAFE_LOCATION_LEMMAS = {"casa", "escuela", "trabajo", "puerta"}
SAFE_MOVEMENT_DESTINATION_LEMMAS = {"casa", "escuela", "trabajo", "puerta", "tienda", "parque"}
SAFE_POSSESSIBLE_TEMPLATE_LEMMAS = {
    "casa", "libro", "perro", "comida", "trabajo", "teléfono", "coche", "cama",
    "foto", "papel", "hermano", "hermana", "amigo", "amiga", "hija",
    "puerta", "mesa", "llave", "gato", "caja",
}
SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB = {
    "tener": {"casa", "libro", "perro", "comida", "trabajo", "teléfono", "coche", "cama", "foto", "papel", "hermano", "hermana", "amigo", "amiga", "hija"},
    "querer": {"casa", "libro", "perro", "comida", "ayuda"},
    "hacer": {"pregunta", "plan", "comida", "trabajo"},
    "comer": {"comida", "pan", "fruta", "carne", "pescado"},
    "beber": {"agua", "café", "leche", "té"},
    "leer": {"libro", "carta", "periódico", "mensaje"},
    "comprar": {"casa", "libro", "comida", "ropa", "coche"},
    "dar": {"regalo", "libro", "comida", "ayuda"},
    "tomar": {"agua", "café", "té", "leche"},
    "cambiar": {"ropa", "coche", "plan"},
    "abrir": {"puerta", "libro", "caja", "ventana"},
    "buscar": {"casa", "trabajo", "libro", "ayuda"},
    "escribir": {"carta", "libro", "mensaje"},
    "llevar": {"ropa", "comida", "libro", "bolsa"},
    "encontrar": {"casa", "trabajo", "libro", "llave"},
    "usar": {"coche", "teléfono", "libro"},
    "mirar": {"casa", "ciudad", "libro"},
    "pagar": {"comida", "casa"},
    "cerrar": {"puerta", "libro", "caja"},
    "conocer": {"ciudad", "pueblo", "casa"},
    "sacar": {"libro", "foto"},
    "poner": {"mesa", "libro"},
}
LOW_VALUE_RETRIEVED_PATTERNS = {
    "persona es ella",
    "es gracias a",
    "están en nada",
    "por supuesto que",
}
APOCOPATED_ADJECTIVE_FEATURES = {
    "buen": ("m", "sg"),
    "mal": ("m", "sg"),
    "gran": (None, "sg"),
    "primer": ("m", "sg"),
    "tercer": ("m", "sg"),
}
ARTICLELESS_LOCATION_NOUNS = {"casa"}
STARTER_SAFE_SUPPORT_NOUNS_FOR_ADJ = {
    "libro", "coche", "casa", "puerta", "perro", "niño", "niña", "hombre",
    "mujer", "amigo", "amiga", "comida", "ciudad", "escuela", "hospital",
    "hotel", "campo", "pueblo", "calle", "mesa", "caja", "llave", "gato",
    "pan", "carne", "carta", "mensaje", "ropa", "vestido",
    "tienda", "oficina", "caballo", "doctor", "profesor", "jefe",
    "rey", "reina", "papá", "agua", "café",
}
STARTER_INFINITIVE_CARRIERS = ["querer", "poder", "necesitar"]

MASCULINE_A_ENDING_NOUNS = {
    "día", "mapa", "problema", "tema", "idioma", "sistema", "programa",
    "clima", "sofá", "planeta",
}
FEMININE_O_ENDING_NOUNS = {"mano", "foto", "moto", "radio"}

STARTER_INFINITIVE_COMPLEMENTS = {
    "beber": [("agua", None)],
    "comprar": [("casa", "una"), ("libro", "un")],
    "buscar": [("casa", "una"), ("trabajo", "un")],
    "abrir": [("puerta", "una")],
    "cerrar": [("puerta", "la")],
    "leer": [("libro", "un")],
    "ver": [("ciudad", "la")],
    "hacer": [("trabajo", "un")],
    "tomar": [("agua", None)],
    "encontrar": [("casa", "una"), ("trabajo", "un")],
    "ganar": [("dinero", None)],
    "mirar": [("casa", "la"), ("ciudad", "la")],
    "conocer": [("ciudad", "la"), ("pueblo", "el")],
    "tocar": [("puerta", "la")],
    "pagar": [("comida", "la")],
    "dejar": [("casa", "la")],
}

BARE_INFINITIVE_OK = {
    "hablar", "caminar", "correr", "cantar", "bailar", "nadar", "jugar",
    "estudiar", "cocinar",
}

STARTER_INFINITIVE_REJECT = {
    "estar", "saber", "ser", "haber", "poder", "querer", "creer", "pensar",
    "parecer", "deber", "pasar",
}

STARTER_RETRIEVED_SKEPTICAL_WORDS = {
    "igual", "suficiente", "peor", "mejor", "siguiente", "libre", "duro",
    "propio", "mismo", "misma", "tuya", "tuyo", "mía", "mío", "suya", "suyo",
    "nada", "nadie", "nunca", "ahora", "mañana", "ayer", "aquello", "casi",
    "algo", "alguien",
}
STARTER_RETRIEVED_BANNED_BIGRAMS = {
    "mi vida", "tu vida", "su vida", "para nada", "más de", "más que",
    "hora de", "que eso", "lo puedo", "lo puede", "te puede", "te quiero",
    "te puedo", "la vida", "sobre ello",
}
STARTER_RETRIEVED_CONTEXT_WORDS = {
    "ello", "eso", "esto", "ya", "todavía", "siento",
}
STARTER_RETRIEVED_POSSESSIVES = {"mi", "mis", "tu", "tus", "su", "sus"}
STARTER_RETRIEVED_PERSONAL_TOKENS = {
    "yo", "tú", "él", "ella", "ellos", "ellas",
    "me", "te", "se", "nos", "os", "le", "les", "lo", "la", "los", "las",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "nuestro", "nuestra", "nuestros", "nuestras",
}
STARTER_WEAK_COPULA_PREDICATES_BY_CLASS = {
    "text": {"importante"},
}
STARTER_UNNATURAL_COPULA_PREDICATES_BY_CLASS = {
    "place": {"natural"},
}

POS_FAMILY_MAP = {
    "n": "n",
    "prop": "prop",
    "v": "v",
    "aux": "v",
    "adj": "adj",
    "adv": "adv",
    "determiner": "determiner",
    "det": "determiner",
    "art": "art",
    "pron": "pron",
    "pronoun": "pron",
    "prep": "prep",
    "adp": "prep",
    "conj": "conj",
    "cconj": "conj",
    "sconj": "conj",
    "interj": "interj",
    "num": "num",
    "contraction": "contraction",
    "letter": "letter",
    "prefix": "prefix",
    "phrase": "phrase",
    "particle": "particle",
    "none": "residual",
    "": "residual",
}

FUNCTION_WORD_FAMILIES = {
    "adv",
    "determiner",
    "art",
    "pron",
    "prep",
    "conj",
    "interj",
    "num",
    "contraction",
    "letter",
    "prefix",
    "phrase",
    "particle",
    "residual",
}

BEGINNER_SUBORDINATORS = {"y", "o", "pero"}
ADV_TIME_LEMMAS = {"ahora", "ya", "siempre", "hoy", "mañana", "ayer"}
ADV_PLACE_LEMMAS = {"aquí", "allí", "acá", "afuera", "dentro"}
DETERMINER_POSSESSIVES = {"mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras"}
DETERMINER_DEMONSTRATIVES = {"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas"}
SIMPLE_PREPOSITIONS = {"de", "en", "con", "para", "sin", "sobre", "hasta", "por"}
COORDINATING_CONJUNCTIONS = {"y", "o", "ni", "pero"}
SUBORDINATING_CONJUNCTIONS = {"porque", "cuando", "como", "si", "aunque"}
INTERJECTION_PUNCT = {"hola": "", "oh": "!", "eh": "!", "ah": "!", "wow": "!"}
NUMERAL_NOUN_CLASSES = {"object", "person", "animal", "food", "text", "place"}
RESIDUAL_QUOTE_CARRIERS = ["palabra", "letra", "forma", "expresión"]
SAFE_TEMPLATE_SUPPORT_LIMITS = {
    "num_indefinite_object": (600, 325),
    "prep_on_table": (800, 325),
    "pron_clitic_call": (500, 250),
}
SAFE_TEMPLATE_REVIEW_OVERRIDES = {
    "conj_or_time",
    "noun_exact_say_thanks",
    "noun_exact_turn",
    "num_indefinite_object",
    "prep_on_table",
    "pron_clitic_call",
    "verb_exact_ha_do_that",
}


@dataclass
class Lexeme:
    lemma: str
    rank: int
    pos: str
    translation: str = ""
    gender: Optional[str] = None
    semantic_class: Optional[str] = None
    verb_type: Optional[str] = None
    required_prep: Optional[str] = None
    is_reflexive: bool = False
    canonical_lemma: Optional[str] = None


@dataclass
class DifficultyProfile:
    band: str
    min_len: int
    max_len: int
    filler_ceil: int
    avg_ceil: int
    slot_range: Tuple[int, int]
    complexity: int

    @property
    def ideal_length(self) -> int:
        return (self.min_len + self.max_len) // 2


@dataclass
class Candidate:
    lemma: str
    rank: int
    pos: str
    band: str
    translation: str
    sentence: str
    target_form: str
    target_index: int
    support_ranks: List[int]
    avg_support_rank: float
    max_support_rank: int
    template_id: str
    source_method: str
    canonical_lemma: str = ""
    target_morph: str = ""
    score: float = 0.0
    english_sentence: str = ""
    english_source_method: str = ""
    parallel_pair_id: Optional[int] = None
    english_score: float = 0.0


BANDS = [
    (1, 800, DifficultyProfile("A1", 3, 6, 300, 180, (0, 1), 1)),
    (801, 1500, DifficultyProfile("A2", 3, 7, 600, 350, (0, 2), 2)),
    (1501, 2500, DifficultyProfile("B1", 4, 8, 1400, 800, (1, 3), 3)),
    (2501, 4000, DifficultyProfile("B2", 5, 8, 2500, 1400, (1, 4), 4)),
    (4001, 6000, DifficultyProfile("C1", 5, 9, 5000, 2500, (2, 5), 5)),
    (6001, 10**9, DifficultyProfile("C2", 6, 10, 8000, 4000, (2, 6), 6)),
]


STARTER_ELIGIBLE_POS = {"n", "v", "adj"}

STARTER_MAX_BAND = {"A1", "A2"}
STARTER_BANNED_LEMMAS = {"mierda", "puta", "muerte", "guerra"}
STARTER_BANNED_TEMPLATE_IDS = {"n_b1_1", "n_b1_2"}
STARTER_EXCLUDED_TARGET_LEMMAS = {"acuerdo", "forma", "razón", "manera", "vuelta"}
STARTER_LOW_VALUE_TARGET_LEMMAS = {"pregunta", "amor", "corazón", "miedo", "seguro", "adiós"}
STARTER_BAD_TRANSLATION_FRAGMENTS = {
    "title of respect",
    "civility",
    "public order",
    "doggy",
    "doggish",
    "solid excretory",
}
STARTER_BANNED_PATTERNS = {
    "creo que",
    "pienso que",
    "parece que",
    "es que",
    "como si",
    "de acuerdo",
    "tener razón",
    "hace muchos años que",
    "yo misma",
    "contigo",
}
STARTER_BANNED_WORD_PATTERNS = {
    "usted",
    "a ti",
    "con nosotras",
    "con vosotros",
    "con usted",
    "de quién",
    "lo sabe tu",
}
STARTER_SAFE_NOUN_CLASSES = {"object", "place", "person", "animal", "food", "clothing", "text"}
STARTER_SAFE_ADJECTIVES_BY_CLASS = {
    "object": {
        "grande", "pequeño", "bonito", "nuevo", "viejo", "limpio",
        "perfecto", "importante", "simple", "diferente", "largo", "corto",
        "increíble", "necesario", "terrible", "horrible", "interesante",
        "extraño", "real", "especial", "común", "seguro",
        "negro", "oficial", "local", "central",
    },
    "place": {
        "grande", "pequeño", "bonito", "importante", "nuevo", "viejo",
        "tranquilo", "peligroso", "público", "popular", "limpio",
        "diferente", "increíble", "terrible", "horrible", "interesante",
        "especial", "real", "común", "seguro", "nacional",
        "local", "central", "internacional", "oficial", "político",
        "profesional", "militar",
    },
    "person": {
        "bueno", "feliz", "cansado", "rápido", "fuerte", "joven", "alto",
        "bajo", "importante", "inteligente", "serio", "rico", "pobre",
        "divertido", "tranquilo", "peligroso", "extraño", "increíble",
        "especial", "perfecto", "diferente", "popular", "real", "humano",
        "común", "seguro", "terrible", "horrible",
        "profesional", "local", "oficial", "idiota",
    },
    "animal": {
        "grande", "pequeño", "rápido", "bonito", "fuerte", "joven",
        "peligroso", "tranquilo", "increíble", "extraño", "inteligente",
        "negro",
    },
    "text": {
        "grande", "pequeño", "nuevo", "largo", "corto",
        "simple", "diferente", "interesante", "perfecto", "terrible",
        "increíble", "claro", "personal", "especial", "real",
        "oficial", "profesional", "público", "local", "nacional",
        "internacional",
    },
    "food": {
        "bueno", "malo", "delicioso", "rico", "simple", "perfecto",
        "diferente", "especial", "increíble", "terrible", "horrible",
        "natural",
    },
    "drink": {
        "bueno", "malo", "delicioso", "rico", "perfecto",
        "diferente", "especial", "natural",
    },
    "clothing": {
        "grande", "pequeño", "bonito", "nuevo", "viejo", "limpio",
        "sucio", "perfecto", "diferente", "especial",
        "negro",
    },
}

STARTER_ADJ_ALLOWED_NOUN_CLASSES: Dict[str, set] = {}
for _cls, _adjs in STARTER_SAFE_ADJECTIVES_BY_CLASS.items():
    for _adj in _adjs:
        STARTER_ADJ_ALLOWED_NOUN_CLASSES.setdefault(_adj, set()).add(_cls)


IRREGULAR_PRESENT = {
    "ser": {"1sg": "soy", "2sg": "eres", "3sg": "es", "1pl": "somos", "3pl": "son"},
    "estar": {"1sg": "estoy", "2sg": "estás", "3sg": "está", "1pl": "estamos", "3pl": "están"},
    "ir": {"1sg": "voy", "2sg": "vas", "3sg": "va", "1pl": "vamos", "3pl": "van"},
    "tener": {"1sg": "tengo", "2sg": "tienes", "3sg": "tiene", "1pl": "tenemos", "3pl": "tienen"},
    "haber": {"1sg": "he", "2sg": "has", "3sg": "ha", "1pl": "hemos", "3pl": "han"},
    "poder": {"1sg": "puedo", "2sg": "puedes", "3sg": "puede", "1pl": "podemos", "3pl": "pueden"},
    "querer": {"1sg": "quiero", "2sg": "quieres", "3sg": "quiere", "1pl": "queremos", "3pl": "quieren"},
    "decir": {"1sg": "digo", "2sg": "dices", "3sg": "dice", "1pl": "decimos", "3pl": "dicen"},
    "hacer": {"1sg": "hago", "2sg": "haces", "3sg": "hace", "1pl": "hacemos", "3pl": "hacen"},
    "saber": {"1sg": "sé", "2sg": "sabes", "3sg": "sabe", "1pl": "sabemos", "3pl": "saben"},
    "poner": {"1sg": "pongo", "2sg": "pones", "3sg": "pone", "1pl": "ponemos", "3pl": "ponen"},
    "venir": {"1sg": "vengo", "2sg": "vienes", "3sg": "viene", "1pl": "venimos", "3pl": "vienen"},
    "salir": {"1sg": "salgo", "2sg": "sales", "3sg": "sale", "1pl": "salimos", "3pl": "salen"},
    "dar": {"1sg": "doy", "2sg": "das", "3sg": "da", "1pl": "damos", "3pl": "dan"},
    "ver": {"1sg": "veo", "2sg": "ves", "3sg": "ve", "1pl": "vemos", "3pl": "ven"},
    "conocer": {"1sg": "conozco", "2sg": "conoces", "3sg": "conoce", "1pl": "conocemos", "3pl": "conocen"},
    "dormir": {"1sg": "duermo", "2sg": "duermes", "3sg": "duerme", "1pl": "dormimos", "3pl": "duermen"},
    "pedir": {"1sg": "pido", "2sg": "pides", "3sg": "pide", "1pl": "pedimos", "3pl": "piden"},
    "pensar": {"1sg": "pienso", "2sg": "piensas", "3sg": "piensa", "1pl": "pensamos", "3pl": "piensan"},
    "encontrar": {"1sg": "encuentro", "2sg": "encuentras", "3sg": "encuentra", "1pl": "encontramos", "3pl": "encuentran"},
    "volver": {"1sg": "vuelvo", "2sg": "vuelves", "3sg": "vuelve", "1pl": "volvemos", "3pl": "vuelven"},
}


def get_profile(rank: int) -> DifficultyProfile:
    for lo, hi, profile in BANDS:
        if lo <= rank <= hi:
            return profile
    return BANDS[-1][2]


def allowed_support_rank(target_rank: int, profile: DifficultyProfile) -> int:
    return min(profile.filler_ceil, max(300, int(target_rank * 0.8)))


def is_word_token(token: str) -> bool:
    return bool(WORD_RE.search(token))


def normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = token.strip("¿¡\"“”'()[]{}.,;:!?")
    return token


def strip_accents(token: str) -> str:
    if not token:
        return ""
    normalized = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


class SentenceGenerator:
    def __init__(self, lexicon_path: str, models_dir: str, seed: int = 42):
        self.random = random.Random(seed)
        self.lexicon: Dict[str, Lexeme] = self._load_lexicon(lexicon_path, models_dir)
        self.lemma_forms = self._load_pickle(models_dir, "lemma_forms.pkl")
        self.lemma_contexts = self._load_pickle(models_dir, "lemma_contexts.pkl")
        self.bigrams = self._load_pickle(models_dir, "bigrams.pkl")
        self.trigrams = self._load_pickle(models_dir, "trigrams.pkl")
        self.pos_transitions = self._load_pickle(models_dir, "pos_transitions.pkl")
        self.enriched = self._load_pickle(models_dir, "enriched_lexicon.pkl")
        self.form_to_lemmas = self._build_form_index()
        self.generation_lexicon = self._build_generation_lexicon()
        self.pos_buckets = self._build_pos_buckets()
        self.pos_strategies = self._build_pos_strategies()
        self.candidate_pool_cache: Dict[str, List[Candidate]] = {}
        self.starter_candidate_pool_cache: Dict[Tuple[str, int], List[Candidate]] = {}
        self.last_candidate_export_stats: Optional[Dict[str, float]] = None
        self.last_starter_stats: Optional[Dict[str, Any]] = None
        self.last_coverage_report: Optional[Dict[str, Any]] = None
        self.reranker = load_reranker_model(os.path.join(models_dir, "reranker.pkl"))
        self.overrides: Dict[str, Dict[str, str]] = {}

    def _active_lemma(self, lemma: Optional[str]) -> bool:
        if not lemma:
            return False
        return lemma in self.lexicon or lemma in self.generation_lexicon

    def _load_pickle(self, models_dir: str, name: str):
        path = os.path.join(models_dir, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_lexicon(self, lexicon_path: str, models_dir: str) -> Dict[str, Lexeme]:
        enriched_path = os.path.join(models_dir, "enriched_lexicon.pkl")
        with open(enriched_path, "rb") as f:
            enriched = pickle.load(f)
        lexicon: Dict[str, Lexeme] = {}
        with open(lexicon_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lemma = row.get("lemma", "").strip().lower()
                if not lemma:
                    continue
                canonical = (row.get("original_lemma") or lemma).strip().lower()
                rank = int(row.get("rank", 99999) or 99999)
                base = enriched.get(canonical) or enriched.get(lemma, {})
                pos = (row.get("pos") or base.get("pos") or "").strip().lower()
                translation = (row.get("translation") or base.get("translation") or "").strip()
                lexicon[lemma] = Lexeme(
                    lemma=lemma,
                    rank=rank,
                    pos=pos,
                    translation=translation,
                    gender=base.get("gender"),
                    semantic_class=base.get("semantic_class"),
                    verb_type=base.get("verb_type"),
                    required_prep=base.get("required_prep"),
                    is_reflexive=bool(base.get("is_reflexive", False)),
                    canonical_lemma=canonical,
                )
        return lexicon

    def load_and_apply_overrides(self, path: str) -> None:
        raw_overrides: Dict[str, Dict[str, str]] = {}
        if path.endswith(".json"):
            import json as _json
            with open(path, encoding="utf-8") as f:
                raw = _json.load(f)
            for entry in raw:
                lemma = normalize_token(entry.get("lemma", ""))
                if lemma:
                    raw_overrides[lemma] = entry
        else:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lemma = normalize_token(row.get("lemma") or "")
                    if lemma:
                        raw_overrides[lemma] = {
                            k: v.strip() for k, v in row.items() if v and str(v).strip()
                        }

        expanded_overrides: Dict[str, Dict[str, str]] = {}
        for lemma, ov in raw_overrides.items():
            for key in self._override_lookup_keys(lemma, ov):
                merged = dict(ov)
                merged["lemma"] = lemma
                expanded_overrides[key] = merged

        self.overrides.update(expanded_overrides)

        applied_seed_contexts: set = set()
        for lemma, ov in raw_overrides.items():
            for key in self._override_lookup_keys(lemma, ov):
                if _truthy(ov.get("exclude_from_generation")):
                    self.lexicon.pop(key, None)
                    continue
                lex = self.lexicon.get(key)
                if not lex:
                    continue
                if ov.get("force_translation"):
                    lex.translation = ov["force_translation"]
                if ov.get("force_pos"):
                    lex.pos = ov["force_pos"].strip().lower()
                if ov.get("force_canonical_lemma"):
                    lex.canonical_lemma = normalize_token(ov["force_canonical_lemma"])
                if _truthy(ov.get("force_template_friendly")) and lex.semantic_class in ("time", "abstract", "activity", "body"):
                    lex.semantic_class = "object"
                seed_sentence = ov.get("seed_sentence")
                if seed_sentence and (key, seed_sentence) not in applied_seed_contexts:
                    self._inject_seed_context(key, seed_sentence)
                    applied_seed_contexts.add((key, seed_sentence))

        self.generation_lexicon = self._build_generation_lexicon()
        self.pos_buckets = self._build_pos_buckets()
        self.pos_strategies = self._build_pos_strategies()
        self.candidate_pool_cache.clear()
        self.starter_candidate_pool_cache.clear()
        self.last_candidate_export_stats = None
        self.last_starter_stats = None
        self.last_coverage_report = None

    def is_starter_target_eligible(self, lex: Lexeme) -> bool:
        if lex.pos not in STARTER_ELIGIBLE_POS:
            return False
        if get_profile(lex.rank).band not in STARTER_MAX_BAND:
            return False
        ov = self.overrides.get(lex.lemma, {})
        if _truthy(ov.get("exclude_from_starter_dataset")):
            return False
        if not lex.translation:
            return False
        return True

    def starter_target_translation_ok(self, lex: Lexeme) -> bool:
        t = (lex.translation or "").strip().lower()
        if not t:
            return False
        if ";" in t:
            return False
        if len(t) > 40:
            return False
        return not any(frag in t for frag in STARTER_BAD_TRANSLATION_FRAGMENTS)

    def exact_surface_present(self, candidate: "Candidate", target: Lexeme) -> bool:
        return normalize_token(candidate.target_form) == normalize_token(target.lemma)

    def _build_generation_lexicon(self) -> Dict[str, Lexeme]:
        generation: Dict[str, Lexeme] = {}
        for lex in self.lexicon.values():
            canonical = lex.canonical_lemma or lex.lemma
            candidate = Lexeme(
                lemma=canonical,
                rank=lex.rank,
                pos=lex.pos,
                translation=lex.translation,
                gender=lex.gender,
                semantic_class=lex.semantic_class,
                verb_type=lex.verb_type,
                required_prep=lex.required_prep,
                is_reflexive=lex.is_reflexive,
                canonical_lemma=canonical,
            )
            existing = generation.get(canonical)
            if existing is None or candidate.rank < existing.rank:
                generation[canonical] = candidate
        return generation

    def _build_form_index(self) -> Dict[str, List[str]]:
        form_to_lemmas: Dict[str, List[str]] = {}
        for lemma, forms in self.lemma_forms.items():
            seen = set()
            for entry in forms:
                form = normalize_token(entry.get("form", ""))
                if not form or form in seen:
                    continue
                seen.add(form)
                form_to_lemmas.setdefault(form, []).append(lemma)
        return form_to_lemmas

    def _build_pos_buckets(self) -> Dict[str, List[Lexeme]]:
        buckets: Dict[str, List[Lexeme]] = {}
        for lex in self.generation_lexicon.values():
            buckets.setdefault(lex.pos, []).append(lex)
        for values in buckets.values():
            values.sort(key=lambda x: x.rank)
        return buckets

    def _build_pos_strategies(self) -> Dict[str, Dict[str, Any]]:
        return {
            "n": {"seeded": self.seeded_template_candidate, "pure": self.pure_template_candidate},
            "v": {"seeded": self.seeded_template_candidate, "pure": self.pure_template_candidate},
            "adj": {"seeded": self.seeded_template_candidate, "pure": self.pure_template_candidate},
            "adv": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "determiner": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "art": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "pron": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "prep": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "conj": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "interj": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "num": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "prop": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "contraction": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "letter": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "prefix": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "phrase": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "particle": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
            "residual": {"seeded": self.seeded_pos_template_candidate, "pure": self.pure_pos_template_candidate},
        }

    def normalized_pos_family(self, lex: Optional[Lexeme]) -> str:
        if not lex:
            return "residual"
        raw = (lex.pos or "").strip().lower()
        lemma = normalize_token(lex.lemma)
        if raw in {"", "none"}:
            if lemma in SIMPLE_PREPOSITIONS or lemma in {"a"}:
                return "prep"
            if lemma == "no":
                return "adv"
        return POS_FAMILY_MAP.get(raw, raw or "residual")

    def strategy_for_target(self, target: Lexeme) -> Dict[str, Any]:
        return self.pos_strategies.get(self.normalized_pos_family(target), self.pos_strategies["residual"])

    def band_template_attempts(self, profile: DifficultyProfile) -> Tuple[int, int]:
        if profile.band == "A1":
            return 8, 16
        if profile.band == "A2":
            return 10, 18
        if profile.band == "B1":
            return 12, 22
        if profile.band == "B2":
            return 12, 24
        if profile.band == "C1":
            return 10, 20
        return 8, 16

    def lookup_lemma(self, surface: str) -> Optional[str]:
        s = normalize_token(surface)
        if not s:
            return None
        candidates: List[str] = []
        if s in self.lexicon:
            candidates.append(s)
            direct = self.lexicon[s]
            canonical = direct.canonical_lemma or s
            if canonical != s:
                candidates.append(canonical)
        candidates.extend(self.form_to_lemmas.get(s, []))
        if not candidates:
            return None
        unique = []
        seen = set()
        for lemma in candidates:
            if lemma in seen:
                continue
            seen.add(lemma)
            if self._active_lemma(lemma):
                unique.append(lemma)
        if not unique:
            return None
        return min(unique, key=lambda l: self.get_known_lexeme(l).rank if self.get_known_lexeme(l) else 99999)

    def lookup_rank(self, surface: str) -> int:
        lemma = self.lookup_lemma(surface)
        if not lemma:
            return 99999
        lex = self.get_known_lexeme(lemma)
        if not lex:
            return 99999
        canonical = self.canonical_lemma_for(lex)
        canonical_lex = self.generation_lexicon.get(canonical)
        if canonical_lex:
            return min(lex.rank, canonical_lex.rank)
        return lex.rank

    def lookup_pos(self, surface: str) -> str:
        lemma = self.lookup_lemma(surface)
        if not lemma:
            return ""
        lex = self.get_known_lexeme(lemma)
        return lex.pos if lex else ""

    def choose_article(self, gender: Optional[str], definite: bool = True, demonstrative: bool = False) -> str:
        gender = gender or "m"
        if demonstrative:
            return "esta" if gender == "f" else "este"
        if definite:
            return "la" if gender == "f" else "el"
        return "una" if gender == "f" else "un"

    def infer_noun_gender(self, lemma: str) -> Optional[str]:
        normalized = normalize_token(lemma)
        if normalized in MASCULINE_A_ENDING_NOUNS:
            return "m"
        if normalized in EL_ARTICLE_FEMININE_NOUNS:
            return "f"
        if normalized in FEMININE_O_ENDING_NOUNS:
            return "f"
        if normalized.endswith("a"):
            return "f"
        if normalized.endswith("o"):
            return "m"
        for entry in self.lemma_forms.get(lemma, []):
            morph = entry.get("morph") or {}
            if morph.get("Gender") == "Fem":
                return "f"
            if morph.get("Gender") == "Masc":
                return "m"
        lex = self.get_known_lexeme(lemma)
        if lex and lex.gender:
            return lex.gender
        if normalized.endswith("ión") or normalized.endswith("dad") or normalized.endswith("tud"):
            return "f"
        if normalized.endswith("or"):
            return "m"
        return None

    def safe_noun_gender(self, noun_lemma: str, fallback: Optional[str] = None) -> str:
        return self.infer_noun_gender(noun_lemma) or fallback or "m"

    def inflect_adj(self, lemma: str, gender: Optional[str], number: str = "sg") -> str:
        gender = gender or "m"
        if lemma.endswith("o"):
            stem = lemma[:-1]
            mapping = {
                ("m", "sg"): "o",
                ("f", "sg"): "a",
                ("m", "pl"): "os",
                ("f", "pl"): "as",
            }
            return stem + mapping[(gender, number)]
        if number == "pl":
            if lemma.endswith(("e", "a", "o")):
                return lemma + "s"
            return lemma + "es"
        return lemma

    def pluralize_noun(self, lemma: str) -> str:
        if lemma.endswith(("a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú")):
            return lemma + "s"
        if lemma.endswith("z"):
            return lemma[:-1] + "ces"
        return lemma + "es"

    def conjugate_present(self, lemma: str, person: str = "3sg") -> str:
        if lemma in IRREGULAR_PRESENT and person in IRREGULAR_PRESENT[lemma]:
            return IRREGULAR_PRESENT[lemma][person]
        if lemma.endswith("ar"):
            endings = {"1sg": "o", "2sg": "as", "3sg": "a", "1pl": "amos", "3pl": "an"}
        elif lemma.endswith("er"):
            endings = {"1sg": "o", "2sg": "es", "3sg": "e", "1pl": "emos", "3pl": "en"}
        elif lemma.endswith("ir"):
            endings = {"1sg": "o", "2sg": "es", "3sg": "e", "1pl": "imos", "3pl": "en"}
        else:
            return lemma
        return lemma[:-2] + endings.get(person, endings["3sg"])

    def apply_contractions(self, tokens: List[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens):
                pair = (tokens[i].lower(), tokens[i + 1].lower())
                if pair == ("a", "el"):
                    out.append("al")
                    i += 2
                    continue
                if pair == ("de", "el"):
                    out.append("del")
                    i += 2
                    continue
            out.append(tokens[i])
            i += 1
        return out

    def canonical_lemma_for(self, lex: Lexeme) -> str:
        canonical = lex.canonical_lemma or lex.lemma
        if normalize_token(canonical) == normalize_token(lex.lemma):
            return canonical
        surface = normalize_token(lex.lemma)
        if any(normalize_token(entry.get("form", "")) == surface for entry in self.lemma_forms.get(canonical, [])):
            return canonical
        return lex.lemma

    def target_form_metadata(self, lex: Lexeme) -> Dict[str, str]:
        canonical = self.canonical_lemma_for(lex)
        surface = normalize_token(lex.lemma)
        for entry in self.lemma_forms.get(canonical, []):
            if normalize_token(entry.get("form", "")) == surface:
                return entry.get("morph") or {}
        return {}

    def surface_morph(self, surface: str) -> Dict[str, str]:
        lemma = self.lookup_lemma(surface)
        lex = self.get_known_lexeme(lemma)
        if not lex:
            return {}
        canonical = self.canonical_lemma_for(lex)
        for entry in self.lemma_forms.get(canonical, []):
            if normalize_token(entry.get("form", "")) == normalize_token(surface):
                return entry.get("morph") or {}
        return {}

    def get_known_lexeme(self, lemma: Optional[str]) -> Optional[Lexeme]:
        if not lemma:
            return None
        return self.lexicon.get(lemma) or self.generation_lexicon.get(lemma)

    def word_tokens(self, sentence: str) -> List[str]:
        return [normalize_token(t) for t in sentence.split() if is_word_token(t) and normalize_token(t)]

    def sentence_tokens(self, sentence: str) -> List[str]:
        return TOKEN_RE.findall(sentence or "")

    def surface_analysis(self, surface: str) -> Dict[str, Optional[str]]:
        lemma = self.lookup_lemma(surface)
        lex = self.get_known_lexeme(lemma)
        canonical = self.canonical_lemma_for(lex) if lex else lemma
        morph = self.surface_morph(surface)
        gender = lex.gender if lex else None
        if morph.get("Gender") == "Fem":
            gender = "f"
        elif morph.get("Gender") == "Masc":
            gender = "m"
        number = "pl" if morph.get("Number") == "Plur" else "sg"
        if morph.get("Number") == "Sing":
            number = "sg"
        return {
            "lemma": canonical,
            "lex": lex,
            "pos": lex.pos if lex else "",
            "gender": gender,
            "number": number,
            "semantic_class": lex.semantic_class if lex else None,
            "morph": morph,
        }

    def surface_candidate_lemmas(self, surface: str) -> List[str]:
        normalized = normalize_token(surface)
        if not normalized:
            return []
        candidates: List[str] = []
        if normalized in self.lexicon:
            candidates.append(normalized)
            direct = self.lexicon[normalized]
            canonical = direct.canonical_lemma or normalized
            if canonical != normalized:
                candidates.append(canonical)
        candidates.extend(self.form_to_lemmas.get(normalized, []))
        unique: List[str] = []
        seen = set()
        for lemma in candidates:
            key = normalize_token(lemma)
            if not key or key in seen:
                continue
            seen.add(key)
            if self._active_lemma(lemma):
                unique.append(lemma)
        return unique

    def surface_has_multiple_candidate_lemmas(self, surface: str) -> bool:
        return len(self.surface_candidate_lemmas(surface)) > 1

    def form_entries_for_surface(self, lemma: str, surface: str) -> List[Dict[str, str]]:
        normalized_surface = normalize_token(surface)
        if not lemma or not normalized_surface:
            return []
        out: List[Dict[str, str]] = []
        seen_keys: set = set()
        canonical = None
        if lemma in self.lexicon:
            canonical = self.canonical_lemma_for(self.lexicon[lemma])
        keys = []
        if canonical and canonical != lemma:
            keys.append(canonical)
        keys.append(lemma)
        for key in keys:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            for entry in self.lemma_forms.get(key, []):
                if normalize_token(entry.get("form", "")) == normalized_surface:
                    out.append(entry.get("morph") or {})
            if out and key == canonical:
                return out
        return out

    def morph_matches_pos(self, morph: Dict[str, str], pos: str) -> bool:
        verbish = bool(morph.get("VerbForm") or morph.get("Mood") or morph.get("Tense") or morph.get("Person"))
        pronish = bool(morph.get("PronType"))
        if pos == "v":
            return verbish
        if pos in {"n", "adj"}:
            return not verbish and not pronish
        if pos == "determiner":
            return pronish or not morph
        if pos in {"pron", "pronoun"}:
            return pronish or not morph
        if pos == "adv":
            return not morph
        return True

    def requested_lemma_allows_inflected_target(self, target: Lexeme) -> bool:
        return (
            normalize_token(target.lemma) == normalize_token(self.canonical_lemma_for(target))
            and self.normalized_pos_family(target) in {"v", "n", "adj"}
            and not self.exact_surface_target_only(target)
        )

    def target_pos_hint_matches_request(self, requested_pos: str, target_pos_hint: str) -> bool:
        if not target_pos_hint:
            return True
        mapped = {
            "VERB": "v",
            "AUX": "v",
            "NOUN": "n",
            "PROPN": "n",
            "PROPER_NOUN": "prop",
            "ADJ": "adj",
            "ADV": "adv",
            "DET": "determiner",
            "PRON": "pron",
            "ADP": "prep",
            "CCONJ": "conj",
            "SCONJ": "conj",
            "INTJ": "interj",
            "NUM": "num",
        }.get(target_pos_hint.upper(), target_pos_hint.lower())
        requested_family = POS_FAMILY_MAP.get(requested_pos, requested_pos)
        mapped_family = POS_FAMILY_MAP.get(mapped, mapped)
        if requested_pos in {"pron", "pronoun"}:
            return mapped_family == "pron"
        if requested_family == "prop":
            return mapped_family in {"prop", "n"}
        return mapped_family == requested_family

    def verb_morph_is_complete(self, morph: Dict[str, str]) -> bool:
        if morph.get("VerbForm") == "Inf":
            return True
        if morph.get("VerbForm") == "Ger":
            return True
        if morph.get("VerbForm") == "Part":
            return bool(morph.get("Gender") or morph.get("Number"))
        return bool(morph.get("Person") and morph.get("Number"))

    def _subject_pronoun_before_verb(self, tokens: List[str], verb_idx: int) -> Optional[Tuple[str, str]]:
        for i in range(verb_idx - 1, -1, -1):
            if not is_word_token(tokens[i]):
                continue
            pn = PRONOUN_PERSON_NUMBER.get(normalize_token(tokens[i]))
            if pn:
                return pn
            return None
        return None

    def resolve_verb_morph_ambiguity(
        self, morphs: List[Dict[str, str]], tokens: List[str], verb_idx: int
    ) -> Optional[Dict[str, str]]:
        if not morphs or verb_idx < 0:
            return None
        unique: List[Dict[str, str]] = []
        seen_sigs: set = set()
        for m in morphs:
            sig = self.morph_signature(m)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                unique.append(m)
        if len(unique) == 1:
            return unique[0]

        moods = set(m.get("Mood", "") for m in unique)
        subject_pn = self._subject_pronoun_before_verb(tokens, verb_idx)

        has_excl = any(t in ("¡", "!") for t in tokens)
        first_word_idx = next((i for i, t in enumerate(tokens) if is_word_token(t)), -1)
        verb_is_first_word = first_word_idx == verb_idx

        preceding_words = set()
        for i in range(max(0, verb_idx - 4), verb_idx):
            if is_word_token(tokens[i]):
                preceding_words.add(normalize_token(tokens[i]))
        has_subj_trigger = bool(preceding_words & SUBJUNCTIVE_TRIGGER_TOKENS)

        remaining = list(unique)

        if len(moods) > 1:
            if "Imp" in moods and "Ind" in moods:
                if subject_pn:
                    remaining = [m for m in remaining if m.get("Mood") != "Imp"]
                elif has_excl and verb_is_first_word:
                    remaining = [m for m in remaining if m.get("Mood") == "Imp"]
                else:
                    return None
            if "Sub" in moods:
                if has_subj_trigger:
                    sub_only = [m for m in remaining if m.get("Mood") == "Sub"]
                    if sub_only:
                        remaining = sub_only
                elif subject_pn:
                    non_sub = [m for m in remaining if m.get("Mood") != "Sub"]
                    if non_sub:
                        remaining = non_sub
                    else:
                        return None
                else:
                    return None

        unique_sigs = set(self.morph_signature(m) for m in remaining)
        if len(unique_sigs) == 1:
            return remaining[0]

        verb_forms = set(m.get("VerbForm", "") for m in remaining)
        if "Inf" in verb_forms and "Fin" in verb_forms and verb_idx > 0:
            prev_token = None
            for i in range(verb_idx - 1, -1, -1):
                if is_word_token(tokens[i]):
                    prev_token = tokens[i]
                    break
            if prev_token:
                prev_morph = self.surface_morph(prev_token)
                if prev_morph.get("VerbForm") == "Fin":
                    inf_only = [m for m in remaining if m.get("VerbForm") == "Inf"]
                    if inf_only:
                        return inf_only[0]

        person_number_pairs = set((m.get("Person", ""), m.get("Number", "")) for m in remaining)
        if len(person_number_pairs) > 1:
            if subject_pn:
                resolved = [m for m in remaining if (m.get("Person"), m.get("Number")) == subject_pn]
                if len(set(self.morph_signature(m) for m in resolved)) == 1:
                    return resolved[0]
            return None

        unique_sigs = set(self.morph_signature(m) for m in remaining)
        if len(unique_sigs) == 1:
            return remaining[0]
        return None

    SAFE_MORPH_KEYS: Dict[str, set] = {
        "n": {"Gender", "Number"},
        "adj": {"Gender", "Number"},
        "determiner": {"Gender", "Number"},
        "pron": {"Gender", "Number", "Person", "Case"},
        "pronoun": {"Gender", "Number", "Person", "Case"},
        "adv": set(),
    }

    def safe_nonverb_morph(self, morph: Dict[str, str], pos: str) -> Dict[str, str]:
        allowed = self.SAFE_MORPH_KEYS.get(pos)
        if allowed is None:
            return morph
        return {k: v for k, v in morph.items() if k in allowed}

    def _parse_morph_str(self, morph_str: str) -> Dict[str, str]:
        if not morph_str:
            return {}
        out: Dict[str, str] = {}
        for pair in morph_str.split("|"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                out[k] = v
        return out

    def target_morph_for_request(
        self,
        target: Lexeme,
        surface: str,
        sentence_tokens: Optional[List[str]] = None,
        target_idx: int = -1,
    ) -> str:
        normalized_surface = normalize_token(surface)
        if not normalized_surface:
            return ""
        matching: List[Dict[str, str]] = []
        for morph in self.form_entries_for_surface(target.lemma, surface):
            if not morph:
                continue
            if self.morph_matches_pos(morph, target.pos):
                if target.pos == "v" and not self.verb_morph_is_complete(morph):
                    continue
                matching.append(morph)
        if not matching and normalized_surface == normalize_token(target.lemma):
            morph = self.target_form_metadata(target)
            if morph and self.morph_matches_pos(morph, target.pos):
                if target.pos == "v" and not self.verb_morph_is_complete(morph):
                    return ""
                matching.append(morph)
        if not matching:
            return ""
        if target.pos == "v":
            unique_sigs = set(self.morph_signature(m) for m in matching)
            if len(unique_sigs) == 1:
                return self.morph_signature(matching[0])
            if sentence_tokens is not None and target_idx >= 0:
                resolved = self.resolve_verb_morph_ambiguity(matching, sentence_tokens, target_idx)
                if resolved:
                    return self.morph_signature(resolved)
            return ""
        safe = [self.safe_nonverb_morph(m, target.pos) for m in matching]
        safe = [m for m in safe if m]
        if not safe:
            return ""
        safe_sigs = set(self.morph_signature(m) for m in safe)
        if len(safe_sigs) == 1:
            return self.morph_signature(safe[0])
        return ""

    def surface_matches_requested_target(self, target: Lexeme, surface: str) -> bool:
        normalized_surface = normalize_token(surface)
        requested = normalize_token(target.lemma)
        if not normalized_surface:
            return False
        if not self.requested_lemma_allows_inflected_target(target):
            if normalized_surface != requested:
                return False
            direct = self.lexicon.get(normalized_surface)
            if direct and direct.pos == target.pos:
                return True
            morphs = self.form_entries_for_surface(target.lemma, surface)
            return any(self.morph_matches_pos(morph, target.pos) for morph in morphs) or target.pos == "adv"
        morphs = self.form_entries_for_surface(target.lemma, surface)
        return any(self.morph_matches_pos(morph, target.pos) for morph in morphs)

    def candidate_target_matches_request(self, candidate: Candidate, tokens: Optional[List[str]] = None) -> bool:
        target = self.lexicon.get(candidate.lemma)
        if not target:
            return False
        token_sequence = list(tokens) if tokens is not None else self.sentence_tokens(candidate.sentence)
        if candidate.target_index < 0 or candidate.target_index >= len(token_sequence):
            return False
        anchored_token = token_sequence[candidate.target_index]
        if not is_word_token(anchored_token):
            return False
        if normalize_token(anchored_token) != normalize_token(candidate.target_form):
            return False
        target_form_hint = normalize_token(getattr(candidate, "_target_form_hint", ""))
        if target_form_hint and normalize_token(anchored_token) != target_form_hint:
            return False
        target_pos_hint = getattr(candidate, "_target_pos_hint", "")
        if candidate.source_method == "retrieved_corpus" and not self.target_pos_hint_matches_request(target.pos, target_pos_hint):
            return False
        if not self.surface_matches_requested_target(target, anchored_token):
            return False
        if self.surface_has_multiple_candidate_lemmas(anchored_token):
            hinted_ok = bool(target_pos_hint and self.target_pos_hint_matches_request(target.pos, target_pos_hint))
            if candidate.source_method == "retrieved_corpus":
                if not hinted_ok:
                    return False
            elif hinted_ok:
                return True
            elif not self.requested_lemma_allows_inflected_target(target):
                direct = self.lexicon.get(normalize_token(anchored_token))
                if not (direct and direct.lemma == target.lemma and direct.pos == target.pos):
                    return False
        return True

    def morph_signature(self, morph: Dict[str, str]) -> str:
        if not morph:
            return ""
        return "|".join(f"{key}={morph[key]}" for key in sorted(morph))

    def infer_adjective_features(self, adj_surface: str) -> Tuple[Optional[str], Optional[str]]:
        info = self.surface_analysis(adj_surface)
        surface = normalize_token(adj_surface)
        gender = info["gender"]
        number = info["number"]
        if surface in APOCOPATED_ADJECTIVE_FEATURES:
            inferred_gender, inferred_number = APOCOPATED_ADJECTIVE_FEATURES[surface]
            gender = gender or inferred_gender
            number = inferred_number
        if surface.endswith("os"):
            gender = gender or "m"
            number = "pl"
        elif surface.endswith("as"):
            gender = gender or "f"
            number = "pl"
        elif surface.endswith("o"):
            gender = gender or "m"
            number = number or "sg"
        elif surface.endswith("a"):
            gender = gender or "f"
            number = number or "sg"
        elif surface.endswith("es") or (surface.endswith("s") and len(surface) > 2):
            number = "pl"
        return gender, number

    def noun_is_locatable(self, noun: Optional[Lexeme]) -> bool:
        return bool(noun and noun.pos == "n" and noun.semantic_class in LOCATABLE_NOUN_CLASSES)

    def noun_is_template_friendly(self, noun: Optional[Lexeme]) -> bool:
        if not noun or noun.pos != "n":
            return False
        if normalize_token(noun.lemma) in LOW_VALUE_NOUN_TEMPLATE_LEMMAS:
            return False
        if noun.semantic_class in {"time", "abstract", "activity"}:
            return False
        return True

    def noun_supports_here_template(self, noun: Optional[Lexeme]) -> bool:
        return bool(noun and self.noun_is_template_friendly(noun) and normalize_token(noun.lemma) in HERE_TEMPLATE_SAFE_LEMMAS)

    def noun_supports_ser_adjective_template(self, noun: Optional[Lexeme]) -> bool:
        return bool(
            noun
            and self.noun_is_template_friendly(noun)
            and noun.semantic_class in SER_ADJ_TEMPLATE_NOUN_CLASSES
        )

    def noun_supports_possession_template(self, noun: Optional[Lexeme]) -> bool:
        return bool(
            noun
            and self.noun_is_template_friendly(noun)
            and noun.semantic_class in SAFE_POSSESSIBLE_NOUN_CLASSES
            and normalize_token(noun.lemma) in SAFE_POSSESSIBLE_TEMPLATE_LEMMAS
        )

    def pick_template_friendly_noun(
        self,
        rank_ceiling: int,
        semantic_classes: Optional[List[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[Lexeme]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        nouns = [
            x
            for x in self.pos_buckets.get("n", [])
            if x.rank <= rank_ceiling
            and self.noun_is_template_friendly(x)
            and normalize_token(x.lemma) not in exclude_set
            and normalize_token(self.canonical_lemma_for(x)) not in exclude_set
        ]
        if semantic_classes:
            preferred = [x for x in nouns if x.semantic_class in semantic_classes]
            picked = self.pick_from_candidates(preferred)
            if picked:
                return picked
        return self.pick_from_candidates(nouns)

    def pick_preferred_lemmas(
        self,
        lemmas: Iterable[str],
        rank_ceiling: int,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[Lexeme]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        candidates = [
            self.generation_lexicon[lemma]
            for lemma in lemmas
            if lemma in self.generation_lexicon
            and self.generation_lexicon[lemma].rank <= rank_ceiling
            and normalize_token(lemma) not in exclude_set
        ]
        return self.pick_from_candidates(candidates)

    def pick_safe_location_noun(self, rank_ceiling: int, exclude: Optional[Iterable[str]] = None) -> Optional[Lexeme]:
        return self.pick_preferred_lemmas(SAFE_LOCATION_LEMMAS, rank_ceiling, exclude=exclude)

    def pick_safe_destination_noun(self, rank_ceiling: int, exclude: Optional[Iterable[str]] = None) -> Optional[Lexeme]:
        return self.pick_preferred_lemmas(SAFE_MOVEMENT_DESTINATION_LEMMAS, rank_ceiling, exclude=exclude)

    def pick_safe_object_noun_for_verb(
        self,
        canonical: str,
        rank_ceiling: int,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[Lexeme]:
        safe_lemmas = SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB.get(canonical)
        if not safe_lemmas:
            return None
        return self.pick_preferred_lemmas(safe_lemmas, rank_ceiling, exclude=exclude)

    def adjective_target_is_template_friendly(self, target: Lexeme) -> bool:
        if target.pos != "adj":
            return True
        surface = normalize_token(target.lemma)
        canonical = normalize_token(target.canonical_lemma or target.lemma)
        if surface in APOCOPATED_ADJECTIVE_FEATURES:
            return False
        if canonical != surface:
            return False
        morph = self.target_form_metadata(target)
        if morph.get("Number") == "Plur":
            return False
        return True

    def subject_features(self, words: List[str], verb_index: int) -> Optional[Dict[str, Optional[str]]]:
        if verb_index <= 0 or not words:
            return None
        first = words[0]
        if first in SUBJECT_FEATURES:
            features = dict(SUBJECT_FEATURES[first])
            features["source"] = "pronoun"
            return features
        if first in ARTICLE_FEATURES or first in DEMONSTRATIVE_SET:
            search_space = words[1:verb_index]
        else:
            search_space = words[:verb_index]
        for token in search_space:
            info = self.surface_analysis(token)
            if info["pos"] != "n":
                continue
            return {
                "person_code": "3pl" if info["number"] == "pl" else "3sg",
                "gender": info["gender"],
                "number": info["number"],
                "source": "noun",
                "surface": token,
            }
        return None

    def first_finite_verb_index(
        self, words: List[str], target_word_idx: int = -1, target_morph_override: Optional[Dict[str, str]] = None
    ) -> int:
        for idx, token in enumerate(words):
            if self.lookup_pos(token) != "v":
                continue
            if idx == target_word_idx and target_morph_override is not None:
                morph = target_morph_override
            else:
                morph = self.surface_morph(token)
            if morph.get("VerbForm") == "Fin":
                return idx
        return -1

    def subject_verb_agreement_ok(
        self, words: List[str], verb_index: int, target_word_idx: int = -1, target_morph_override: Optional[Dict[str, str]] = None
    ) -> bool:
        if verb_index < 0:
            return False
        subject = self.subject_features(words, verb_index)
        if not subject:
            return True
        if verb_index == target_word_idx and target_morph_override is not None:
            verb_morph = target_morph_override
        else:
            verb_morph = self.surface_morph(words[verb_index])
        if verb_morph.get("VerbForm") != "Fin":
            return False
        expected_person = subject.get("person_code")
        actual_person = self.person_code_from_morph(verb_morph)
        if expected_person and actual_person and expected_person != actual_person:
            return False
        expected_number = subject.get("number")
        actual_number = "pl" if verb_morph.get("Number") == "Plur" else "sg"
        if expected_number and actual_number != expected_number:
            return False
        return True

    def copula_predicate_agreement_ok(self, words: List[str], verb_index: int) -> bool:
        if verb_index < 0:
            return True
        if self.surface_analysis(words[verb_index])["lemma"] not in {"ser", "estar"}:
            return True
        subject = self.subject_features(words, verb_index)
        if not subject:
            return True
        predicate_adj = next((token for token in words[verb_index + 1 :] if self.lookup_pos(token) == "adj"), None)
        if not predicate_adj:
            return True
        if normalize_token(predicate_adj) in APOCOPATED_ADJECTIVE_FEATURES:
            return False
        adj_gender, adj_number = self.infer_adjective_features(predicate_adj)
        if subject.get("number") and adj_number and adj_number != subject["number"]:
            return False
        if subject.get("gender") and adj_gender and adj_gender != subject["gender"]:
            return False
        return True

    def template_support_reason(self, target: Lexeme) -> str:
        if target.pos != "v":
            return ""
        canonical = self.canonical_lemma_for(target)
        morph = self.target_form_metadata(target)
        if normalize_token(target.lemma) == normalize_token(canonical):
            if morph.get("VerbForm") and morph.get("VerbForm") != "Inf":
                return "canonical target has unsupported surface verb form"
            if canonical in IRREGULAR_PRESENT or canonical.endswith(("ar", "er", "ir")):
                return ""
            return "unsupported canonical verb target"
        if not morph:
            return "missing morphology for surface verb target"
        verb_form = morph.get("VerbForm", "")
        mood = morph.get("Mood", "")
        # Allow gerunds and participles (handled by embedding templates)
        if verb_form in ("Ger", "Part"):
            return ""
        # Allow subjunctive and conditional (handled by clause-embedding)
        if verb_form == "Fin" and mood in SUPPORTED_TEMPLATE_MOODS:
            return ""
        if verb_form != "Fin":
            return f"unsupported target verb form: {verb_form or 'unknown'}"
        return f"unsupported target verb mood: {mood or 'unknown'}"

    def standalone_subjunctive_without_trigger(
        self, words: List[str], verb_index: int, target_word_idx: int = -1, target_morph_override: Optional[Dict[str, str]] = None
    ) -> bool:
        if verb_index < 0:
            return False
        if verb_index == target_word_idx and target_morph_override is not None:
            morph = target_morph_override
        else:
            morph = self.surface_morph(words[verb_index])
        if morph.get("Mood") != "Sub":
            return False
        prefix = set(words[:verb_index])
        if prefix & SUBJUNCTIVE_TRIGGER_TOKENS:
            return False
        if "tal" in prefix and "vez" in prefix:
            return False
        return True

    def low_value_here_fallback(self, words: List[str], verb_index: int) -> bool:
        if verb_index < 0 or "aquí" not in words:
            return False
        if self.surface_analysis(words[verb_index])["lemma"] != "estar":
            return False
        subject = self.subject_features(words, verb_index)
        if not subject or subject.get("source") != "noun":
            return False
        noun_surface = subject.get("surface")
        noun = self.get_known_lexeme(self.lookup_lemma(noun_surface or ""))
        return not self.noun_supports_here_template(noun)

    def contradictory_identity_statement(self, words: List[str], verb_index: int) -> bool:
        if verb_index < 0 or self.surface_analysis(words[verb_index])["lemma"] != "ser":
            return False
        subject = self.subject_features(words, verb_index)
        if not subject or subject.get("source") != "noun":
            return False
        subject_info = self.surface_analysis(subject.get("surface", ""))
        if subject_info["semantic_class"] != "person":
            return False
        for idx in range(verb_index + 1, len(words) - 1):
            if words[idx] not in ARTICLE_FEATURES:
                continue
            pred_info = self.surface_analysis(words[idx + 1])
            if pred_info["pos"] != "n" or pred_info["semantic_class"] != "person":
                continue
            if (
                subject_info["lemma"]
                and pred_info["lemma"]
                and subject_info["lemma"] != pred_info["lemma"]
                and subject_info["gender"]
                and pred_info["gender"]
                and subject_info["gender"] != pred_info["gender"]
            ):
                return True
        return False

    def retrieved_quality(self, candidate: Candidate, words: List[str], verb_index: int) -> Tuple[bool, List[float], List[str]]:
        if candidate.source_method != "retrieved_corpus":
            return False, [], []
        reasons: List[str] = []
        penalties: List[float] = []
        sentence = candidate.sentence
        normalized_sentence = " ".join(words)
        if any(ch in sentence for ch in {'"', "“", "”"}):
            reasons.append("quoted_fragment")
        if words and words[0] in CONTEXT_DEPENDENT_OPENERS:
            reasons.append("context_dependent_opener")
        if words and words[0] in DEICTIC_SUBJECTS and self.subject_features(words, verb_index) is None:
            reasons.append("deictic_subject_without_anchor")
        if any(pattern in normalized_sentence for pattern in LOW_VALUE_RETRIEVED_PATTERNS):
            reasons.append("low_value_retrieved_pattern")
        if self.standalone_subjunctive_without_trigger(words, verb_index):
            reasons.append("standalone_subjunctive")
        if self.contradictory_identity_statement(words, verb_index):
            reasons.append("contradictory_identity")
        target = self.lexicon.get(candidate.lemma)
        canonical = self.canonical_lemma_for(target) if target else normalize_token(candidate.canonical_lemma or candidate.lemma)
        noun_surface = self.first_following_noun(words, candidate.target_index if candidate.target_index >= 0 else 0)
        if canonical == "tener" and not noun_surface:
            reasons.append("weak_retrieved_verb_object")
        if canonical in SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB and noun_surface and not self.verb_object_is_semantically_safe(canonical, noun_surface):
            reasons.append("weak_retrieved_verb_object")
        if canonical in {"ir", "poder"} and noun_surface and not self.destination_is_safe(noun_surface):
            reasons.append("weak_retrieved_destination")
        if canonical == "estar" and noun_surface and not self.location_is_safe(noun_surface):
            reasons.append("weak_retrieved_location")
        if "," in sentence or ";" in sentence or ":" in sentence:
            penalties.append(0.8)
            reasons.append("multi_clause_context")
        return bool(
            {
                "quoted_fragment",
                "context_dependent_opener",
                "deictic_subject_without_anchor",
                "standalone_subjunctive",
                "contradictory_identity",
                "low_value_retrieved_pattern",
                "weak_retrieved_verb_object",
                "weak_retrieved_destination",
                "weak_retrieved_location",
            }
            & set(reasons)
        ), penalties, reasons

    def retrieved_is_level_appropriate(
        self, candidate: Candidate, words: List[str], verb_index: int
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        profile = get_profile(candidate.rank)
        band = profile.band

        if band in ("A1", "A2"):
            for idx, token in enumerate(words):
                if self.lookup_pos(token) != "v":
                    continue
                morph = self.surface_morph(token)
                mood = morph.get("Mood", "")
                if mood == "Sub":
                    reasons.append("subjunctive_in_beginner_band")
                    break
                tense = morph.get("Tense", "")
                if mood == "Cnd" or tense == "Imp":
                    reasons.append("advanced_tense_in_beginner_band")
                    break

        if band == "A1" and len(words) > 6:
            reasons.append("too_long_for_A1")
        if band == "A2" and len(words) > 8:
            reasons.append("too_long_for_A2")

        if band in ("A1", "A2"):
            finite_count = 0
            for token in words:
                if self.lookup_pos(token) != "v":
                    continue
                morph = self.surface_morph(token)
                if morph.get("VerbForm") == "Fin":
                    finite_count += 1
            if finite_count > 1:
                reasons.append("multi_clause_for_beginner")

        if band in ("A1", "A2") and ("," in candidate.sentence or ";" in candidate.sentence):
            reasons.append("punctuation_complexity_for_beginner")

        if band == "B1" and len(words) > 9:
            reasons.append("too_long_for_B1")
        if band == "B2" and len(words) > 10:
            reasons.append("too_long_for_B2")

        finite_count = 0
        for token in words:
            if self.lookup_pos(token) != "v":
                continue
            morph = self.surface_morph(token)
            if morph.get("VerbForm") == "Fin":
                finite_count += 1

        if band in ("B1", "B2") and finite_count > 2:
            reasons.append("too_many_finite_verbs")
        if band in ("C1", "C2") and finite_count > 3:
            reasons.append("too_many_finite_verbs")

        if candidate.support_ranks:
            max_support = max(candidate.support_ranks)
            support_ceiling = int(profile.filler_ceil * (2.0 if band in {"A1", "A2"} else 2.5 if band in {"B1", "B2"} else 3.0))
            if max_support > support_ceiling:
                reasons.append("support_vocab_too_advanced")

        return len(reasons) == 0, reasons

    def retrieved_is_preferred(self, candidate: Candidate) -> bool:
        if candidate.source_method != "retrieved_corpus" or not candidate.sentence:
            return False
        words = self.word_tokens(candidate.sentence)
        profile = get_profile(candidate.rank)
        verb_index = self.first_finite_verb_index(words)
        rejected, penalties, _ = self.retrieved_quality(candidate, words, verb_index)
        if rejected:
            return False
        if penalties:
            return False
        if len(words) > profile.ideal_length + 2:
            return False
        grammatical_ok, natural_ok, learner_clear_ok, _ = self.review_flags(candidate)
        return grammatical_ok == "1" and natural_ok == "1" and learner_clear_ok == "1" and candidate.score >= 7.5

    def exact_surface_required(self, candidate: Candidate) -> bool:
        target = self.lexicon.get(candidate.lemma)
        return bool(target and self.exact_surface_target_only(target))

    def strategy_validation_reasons(self, candidate: Candidate, words: List[str], finite_verb_index: int) -> List[str]:
        reasons: List[str] = []
        family = self.normalized_pos_family(self.lexicon.get(candidate.lemma))
        lowered = [normalize_token(w) for w in words]
        target_form = normalize_token(candidate.target_form)

        if self.exact_surface_required(candidate) and target_form != normalize_token(candidate.lemma):
            reasons.append("exact_surface_required")

        if family == "prep":
            if candidate.target_index <= 0 or candidate.target_index >= len(words) - 1:
                reasons.append("dangling_preposition")
        elif family == "contraction":
            if candidate.target_index > 0 and candidate.target_index >= len(words) - 1:
                reasons.append("dangling_preposition")
        elif family == "conj":
            if candidate.target_index <= 0 or candidate.target_index >= len(words) - 1:
                reasons.append("isolated_conjunction")
            if target_form in SUBORDINATING_CONJUNCTIONS and finite_verb_index < 0:
                reasons.append("subordinator_without_clause")
        elif family == "interj":
            if candidate.target_index != 0:
                reasons.append("interjection_misplaced")
        elif family == "pron":
            if target_form in {"me", "te", "se", "nos", "lo", "la", "los", "las", "le", "les"} and candidate.target_index >= len(words) - 1:
                reasons.append("dangling_clitic")
        elif family == "num":
            next_pos = self.lookup_pos(words[candidate.target_index + 1]) if candidate.target_index < len(words) - 1 else ""
            if "número" in lowered:
                pass
            elif next_pos == "v":
                pass
            elif candidate.target_index >= len(words) - 1 or next_pos not in {"n", "prop"}:
                reasons.append("numeral_without_noun")
        elif family in {"letter", "prefix", "phrase", "particle"}:
            if not any(token in {"palabra", "letra", "prefijo", "expresión", "partícula", "forma"} for token in lowered):
                reasons.append("missing_metalinguistic_frame")

        return reasons

    def strategy_score_adjustment(self, candidate: Candidate) -> float:
        family = self.normalized_pos_family(self.lexicon.get(candidate.lemma))
        adjustment = 0.0
        if self.exact_surface_required(candidate):
            adjustment += 0.4
        if candidate.source_method == "emergency_template":
            adjustment -= 0.7
        if family in {"letter", "prefix", "phrase", "particle", "residual"}:
            adjustment -= 0.3
        if family in {"adv", "prep", "conj", "interj", "num", "pron", "determiner", "art", "prop", "contraction"} and candidate.source_method != "manual_review_needed":
            adjustment += 1.4
        if family in {"interj", "num"}:
            adjustment += 0.4
        if family == "conj":
            adjustment += 1.6
        if family == "num":
            adjustment += 1.4
        if family in {"pron", "determiner", "art", "contraction"}:
            adjustment += 1.6
        if family == "pron":
            adjustment += 1.2
        if family == "art":
            adjustment += 0.4
        if family == "contraction":
            adjustment += 0.8
        return adjustment

    def pick_from_candidates(self, candidates: List[Lexeme]) -> Optional[Lexeme]:
        if not candidates:
            return None
        trimmed = candidates[:500]
        weights = [1.0 / max(1, c.rank) for c in trimmed]
        return self.random.choices(trimmed, weights=weights, k=1)[0]

    def pick_semantic_noun(
        self,
        rank_ceiling: int,
        preferred_classes: Optional[Iterable[str]] = None,
        banned_classes: Optional[Iterable[str]] = None,
        preferred_lemmas: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[Lexeme]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        nouns = [
            x
            for x in self.pos_buckets.get("n", [])
            if x.rank <= rank_ceiling
            and normalize_token(x.lemma) not in exclude_set
            and normalize_token(self.canonical_lemma_for(x)) not in exclude_set
        ]
        banned = set(banned_classes or [])
        if banned:
            nouns = [x for x in nouns if x.semantic_class not in banned]
        preferred_lemma_list = [x for x in (preferred_lemmas or []) if x in self.generation_lexicon]
        if preferred_lemma_list:
            preferred = [self.generation_lexicon[x] for x in preferred_lemma_list if self.generation_lexicon[x].rank <= rank_ceiling and x not in exclude_set]
            picked = self.pick_from_candidates(preferred)
            if picked:
                return picked
        preferred_classes = list(preferred_classes or [])
        if preferred_classes:
            preferred = [x for x in nouns if x.semantic_class in preferred_classes]
            picked = self.pick_from_candidates(preferred)
            if picked:
                return picked
        return self.pick_from_candidates(nouns)

    def pick_compatible_adjective(self, rank_ceiling: int, subject_class: Optional[str], exclude: Optional[Iterable[str]] = None) -> Optional[Lexeme]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        adjs = [
            x
            for x in self.pos_buckets.get("adj", [])
            if x.rank <= rank_ceiling
            and normalize_token(x.lemma) not in exclude_set
            and normalize_token(x.lemma) not in WEAK_COPULA_ADJECTIVES
        ]
        preferred = []
        for adj in adjs:
            allowed_classes = ADJ_SUBJECT_PREFS.get(adj.lemma)
            if allowed_classes is None:
                if adj.lemma in SAFE_SER_ADJECTIVES:
                    preferred.append(adj)
                continue
            if subject_class and subject_class in allowed_classes:
                preferred.append(adj)
        return self.pick_from_candidates(preferred or adjs)

    def pick_starter_compatible_adjective(self, rank_ceiling: int, subject_class: Optional[str], exclude: Optional[Iterable[str]] = None) -> Optional[Lexeme]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        allowed_lemmas = STARTER_SAFE_ADJECTIVES_BY_CLASS.get(subject_class or "", set())
        if not allowed_lemmas:
            return None
        adjs = [
            x
            for x in self.pos_buckets.get("adj", [])
            if x.rank <= rank_ceiling
            and normalize_token(x.lemma) in allowed_lemmas
            and normalize_token(x.lemma) not in exclude_set
        ]
        return self.pick_from_candidates(adjs)

    def pick_starter_safe_adj_support_noun(self, rank_ceiling: int, exclude: Optional[Iterable[str]] = None) -> Optional[Lexeme]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        nouns = [
            x
            for x in self.pos_buckets.get("n", [])
            if x.rank <= rank_ceiling
            and self.noun_is_template_friendly(x)
            and normalize_token(x.lemma) in STARTER_SAFE_SUPPORT_NOUNS_FOR_ADJ
            and normalize_token(x.lemma) not in exclude_set
        ]
        return self.pick_from_candidates(nouns)

    def pick_easy_infinitive(self, rank_ceiling: int, exclude: Optional[Iterable[str]] = None) -> Optional[Lexeme]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        preferred = [
            self.generation_lexicon[lemma]
            for lemma in EASY_INFINITIVE_LEMMAS
            if lemma in self.generation_lexicon
            and self.generation_lexicon[lemma].rank <= rank_ceiling
            and normalize_token(lemma) not in exclude_set
        ]
        picked = self.pick_from_candidates(preferred)
        if picked:
            return picked
        verbs = [
            x
            for x in self.pos_buckets.get("v", [])
            if x.rank <= rank_ceiling
            and x.lemma not in {"ser", "estar", "haber"}
            and normalize_token(x.lemma) not in exclude_set
            and (x.lemma in VERB_OBJECT_PREFS or x.lemma in PLACE_PREP_VERBS)
        ]
        return self.pick_from_candidates(verbs)

    def pick_safe_subject_noun(
        self,
        rank_ceiling: int,
        semantic_classes: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[Lexeme]:
        preferred_classes = list(semantic_classes or ["person", "animal", "object", "place"])
        return self.pick_template_friendly_noun(rank_ceiling, semantic_classes=preferred_classes, exclude=exclude)

    def pick_support_verb(
        self,
        rank_ceiling: int,
        exclude: Optional[Iterable[str]] = None,
        allow_special: bool = False,
    ) -> Optional[Lexeme]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        verbs = [
            x
            for x in self.pos_buckets.get("v", [])
            if x.rank <= rank_ceiling
            and normalize_token(x.lemma) not in exclude_set
            and (allow_special or self.canonical_lemma_for(x) not in {"ser", "estar", "haber"})
        ]
        return self.pick_from_candidates(verbs)

    def preferred_subject_pronouns(self, profile: DifficultyProfile) -> List[str]:
        if profile.band in {"A1", "A2"}:
            return ["ella", "él"]
        if profile.band in {"B1", "B2"}:
            return ["ella", "él", "ellos", "nosotros"]
        return ["ella", "él", "ellos", "nosotros", "yo", "tú"]

    def exact_surface_target_only(self, target: Lexeme) -> bool:
        return self.normalized_pos_family(target) in FUNCTION_WORD_FAMILIES or target.pos in {"prop", "art", "contraction", "letter", "prefix", "phrase", "particle", "", "none"}

    def person_code_from_morph(self, morph: Dict[str, str]) -> Optional[str]:
        person = str(morph.get("Person", ""))
        number = morph.get("Number")
        return PERSON_NUMBER_TO_CODE.get((person, number))

    def subject_for_target(self, target: Lexeme, fallback: str = "ella") -> Tuple[str, str]:
        morph = self.target_form_metadata(target)
        person_code = self.person_code_from_morph(morph)
        if person_code:
            return PERSON_CODE_TO_SUBJECT.get(person_code, fallback), person_code
        fallback_features = SUBJECT_FEATURES.get(fallback, SUBJECT_FEATURES["ella"])
        return fallback, fallback_features["person_code"]

    def inflect_adj_for_subject(self, lemma: str, subject: str) -> str:
        features = SUBJECT_FEATURES.get(subject, SUBJECT_FEATURES["ella"])
        return self.inflect_adj(lemma, features["gender"], features["number"])

    def target_verb_form(self, target: Lexeme, person_code: str) -> str:
        canonical = self.canonical_lemma_for(target)
        if normalize_token(target.lemma) != normalize_token(canonical):
            return target.lemma
        morph = self.target_form_metadata(target)
        if morph.get("VerbForm") and morph.get("VerbForm") != "Inf":
            return target.lemma
        return self.conjugate_present(canonical, person_code)

    def can_template_target(self, target: Lexeme) -> bool:
        if not self.adjective_target_is_template_friendly(target):
            return False
        return not self.template_support_reason(target)

    def detokenize(self, tokens: List[str]) -> str:
        parts: List[str] = []
        for token in tokens:
            if not token:
                continue
            if not parts:
                parts.append(token)
                continue
            if token in PUNCT_ATTACH_LEFT:
                parts[-1] += token
                continue
            if parts[-1] in PUNCT_ATTACH_RIGHT:
                parts[-1] += token
                continue
            parts.append(token)
        return " ".join(parts)

    def _override_lookup_keys(self, lemma: str, ov: Optional[Dict[str, str]] = None) -> List[str]:
        candidates = [
            normalize_token(lemma or ""),
            strip_accents(normalize_token(lemma or "")),
        ]
        if ov:
            canonical = normalize_token(ov.get("force_canonical_lemma", ""))
            if canonical:
                candidates.extend([canonical, strip_accents(canonical)])
        out: List[str] = []
        seen = set()
        for key in candidates:
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def determiner_surface_features(self, det_surface: str) -> Tuple[Optional[str], Optional[str]]:
        det = normalize_token(det_surface)
        explicit = {
            "mi": (None, "sg"), "tu": (None, "sg"), "su": (None, "sg"),
            "mis": (None, "pl"), "tus": (None, "pl"), "sus": (None, "pl"),
            "nuestro": ("m", "sg"), "nuestra": ("f", "sg"),
            "nuestros": ("m", "pl"), "nuestras": ("f", "pl"),
            "otro": ("m", "sg"), "otra": ("f", "sg"),
            "otros": ("m", "pl"), "otras": ("f", "pl"),
            "todo": ("m", "sg"), "toda": ("f", "sg"),
            "todos": ("m", "pl"), "todas": ("f", "pl"),
            "mucho": ("m", "sg"), "mucha": ("f", "sg"),
            "muchos": ("m", "pl"), "muchas": ("f", "pl"),
        }
        if det in ARTICLE_FEATURES:
            article_gender, article_number = ARTICLE_FEATURES[det]
            return article_gender, article_number
        if det in explicit:
            return explicit[det]
        morph = self.surface_morph(det_surface)
        gender = None
        number = None
        if morph.get("Gender") == "Fem":
            gender = "f"
        elif morph.get("Gender") == "Masc":
            gender = "m"
        if morph.get("Number") == "Plur":
            number = "pl"
        elif morph.get("Number") == "Sing":
            number = "sg"
        if number is None:
            number = "pl" if det.endswith("s") and det not in {"más", "menos"} else "sg"
        return gender, number

    def determiner_matches_noun(self, det_surface: str, noun_surface: str) -> bool:
        det_gender, det_number = self.determiner_surface_features(det_surface)
        noun = self.surface_analysis(noun_surface)
        if noun["pos"] != "n":
            return True
        noun_gender = noun["gender"] or self.infer_noun_gender(noun["lemma"] or noun_surface)
        noun_number = noun["number"]
        if det_number and noun_number and det_number != noun_number:
            return False
        if det_gender and noun_gender and det_gender != noun_gender:
            if normalize_token(noun["lemma"] or noun_surface) in EL_ARTICLE_FEMININE_NOUNS and det_gender == "m" and noun_number == "sg":
                return normalize_token(det_surface) in {"el", "un", "este", "ese"}
            return False
        return True

    def determiner_phrase_ok(self, words: List[str]) -> bool:
        det_like = set(ARTICLE_FEATURES) | DETERMINER_POSSESSIVES | DETERMINER_DEMONSTRATIVES | QUANTIFIER_DETERMINERS
        for idx, token in enumerate(words[:-1]):
            det = normalize_token(token)
            if det not in det_like:
                continue
            noun_idx = -1
            if idx + 2 < len(words) and self.lookup_pos(words[idx + 1]) == "adj" and self.lookup_pos(words[idx + 2]) == "n":
                noun_idx = idx + 2
                if not self.adjective_matches_noun(words[idx + 1], words[noun_idx]):
                    return False
            elif self.lookup_pos(words[idx + 1]) == "n":
                noun_idx = idx + 1
            elif det in {"todo", "todos", "toda", "todas"} and self.lookup_pos(words[idx + 1]) == "v":
                continue
            if noun_idx >= 0 and not self.determiner_matches_noun(token, words[noun_idx]):
                return False
        return True

    def prepositional_pronoun_used_as_subject(self, words: List[str]) -> bool:
        return bool(words and normalize_token(words[0]) in PREPOSITIONAL_SUBJECT_PRONOUNS)

    def strict_subject_pronoun_agreement_ok(
        self,
        words: List[str],
        verb_index: int,
        target_word_idx: int = -1,
        target_morph_override: Optional[Dict[str, str]] = None,
    ) -> bool:
        if verb_index < 0 or not words:
            return False
        subject_surface = normalize_token(words[0])
        if subject_surface not in PRONOUN_PERSON_NUMBER:
            return True
        if subject_surface in PREPOSITIONAL_SUBJECT_PRONOUNS:
            return False
        if verb_index == target_word_idx and target_morph_override is not None:
            verb_morph = target_morph_override
        else:
            verb_morph = self.surface_morph(words[verb_index])
        expected = PERSON_NUMBER_TO_CODE.get(PRONOUN_PERSON_NUMBER[subject_surface])
        actual = self.person_code_from_morph(verb_morph)
        if not expected or not actual:
            return False
        return expected == actual

    def temporal_verb_mismatch(
        self,
        words: List[str],
        verb_index: int,
        target_word_idx: int = -1,
        target_morph_override: Optional[Dict[str, str]] = None,
    ) -> bool:
        if verb_index < 0:
            return False
        lowered = {normalize_token(word) for word in words}
        if verb_index == target_word_idx and target_morph_override is not None:
            morph = target_morph_override
        else:
            morph = self.surface_morph(words[verb_index])
        tense = morph.get("Tense", "")
        mood = morph.get("Mood", "")
        if lowered & PAST_TIME_ADVERBS and tense == "Pres" and mood in {"", "Ind"}:
            return True
        if lowered & FUTURE_TIME_ADVERBS and tense in {"Past", "Imp"}:
            return True
        return False

    def semantic_nonsense_reasons(self, candidate: Candidate, words: List[str]) -> List[str]:
        lowered = [normalize_token(word) for word in words]
        joined = " ".join(lowered)
        reasons: List[str] = []
        if any(pattern in joined for pattern in SHORT_SEMANTIC_NONSENSE_PATTERNS):
            reasons.append("short_semantic_nonsense")
        if candidate.source_method == "stochastic_decoder" and len(words) <= 5:
            if any(phrase in joined for phrase in ("va a ser nada", "voy a ser nada", "vamos a ser nada", "van a ser nada")):
                reasons.append("stochastic_semantic_nonsense")
        return list(dict.fromkeys(reasons))

    def article_matches_noun(self, article: str, noun_surface: str) -> bool:
        features = ARTICLE_FEATURES.get(normalize_token(article))
        noun = self.surface_analysis(noun_surface)
        if not features or noun["pos"] != "n":
            return True
        article_gender, article_number = features
        noun_lemma = noun["lemma"] or normalize_token(noun_surface)
        inferred_gender = noun["gender"] or self.infer_noun_gender(noun_lemma)
        noun_gender = inferred_gender or article_gender
        noun_number = noun["number"] or article_number
        if noun_lemma in EL_ARTICLE_FEMININE_NOUNS and noun_number == "sg":
            return normalize_token(article) in {"el", "un", "este", "ese"}
        return article_gender == noun_gender and article_number == noun_number

    def adjective_matches_noun(self, adj_surface: str, noun_surface: str) -> bool:
        adj = self.surface_analysis(adj_surface)
        noun = self.surface_analysis(noun_surface)
        if adj["pos"] != "adj" or noun["pos"] != "n":
            return True
        adj_lemma = adj["lemma"] or normalize_token(adj_surface)
        noun_gender = noun["gender"]
        noun_number = noun["number"]
        adj_gender, adj_number = self.infer_adjective_features(adj_surface)
        if noun_gender and adj_gender and adj_gender != noun_gender:
            return False
        if noun_number and adj_number and adj_number != noun_number:
            return False
        allowed_classes = ADJ_SUBJECT_PREFS.get(adj_lemma)
        if allowed_classes and noun["semantic_class"] and noun["semantic_class"] not in allowed_classes:
            return False
        return True

    def subject_semantic_class(self, words: List[str], verb_index: int) -> Optional[str]:
        if verb_index <= 0:
            return None
        if words[0] in SUBJECT_FEATURES:
            return "person"
        if words[0] in ARTICLE_FEATURES:
            for idx in range(1, verb_index):
                info = self.surface_analysis(words[idx])
                if info["pos"] == "n":
                    return info["semantic_class"]
        info = self.surface_analysis(words[0])
        if info["pos"] == "n":
            return info["semantic_class"]
        return None

    def first_following_noun(self, words: List[str], start_idx: int) -> Optional[str]:
        for token in words[start_idx + 1 :]:
            info = self.surface_analysis(token)
            if info["pos"] == "n":
                return token
        return None

    def finite_verb_indices(self, words: List[str]) -> List[int]:
        indices: List[int] = []
        for idx, token in enumerate(words):
            if self.lookup_pos(token) != "v":
                continue
            morph = self.surface_morph(token)
            if morph.get("VerbForm") == "Fin" or (morph.get("Mood") and morph.get("Tense")):
                indices.append(idx)
        return indices

    def finite_verb_count(self, words: List[str]) -> int:
        return len(self.finite_verb_indices(words))

    def verb_token_count(self, words: List[str]) -> int:
        return sum(1 for token in words if self.lookup_pos(token) == "v")

    def has_unlinked_finite_verbs(self, words: List[str]) -> bool:
        finite_indices = self.finite_verb_indices(words)
        if len(finite_indices) < 2:
            return False
        lowered = [normalize_token(word) for word in words]
        bridges = COORDINATING_CONJUNCTIONS | SUBORDINATING_CONJUNCTIONS | {"que"}
        soft_tokens = {
            "no", "ya", "también", "tambien",
            "me", "te", "se", "nos", "os", "lo", "la", "los", "las", "le", "les",
            "un", "una", "unos", "unas",
            "el", "la", "los", "las",
            "mi", "mis", "tu", "tus", "su", "sus",
            "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
            "a", "de", "en", "con", "para", "por", "sin", "sobre", "hasta", "al", "del",
        }
        for left_idx, right_idx in zip(finite_indices, finite_indices[1:]):
            bridge_tokens = lowered[left_idx + 1 : right_idx]
            if not bridge_tokens:
                return True
            if any(token in bridges for token in bridge_tokens):
                continue
            hard_gap = [
                token for token in bridge_tokens
                if token not in soft_tokens
            ]
            if len(hard_gap) <= 1:
                return True
        return False

    def universal_coherence_reasons(self, words: List[str]) -> List[str]:
        reasons: List[str] = []
        if not words:
            return reasons

        lowered = [normalize_token(word) for word in words]
        det_like = set(ARTICLE_FEATURES) | DETERMINER_POSSESSIVES | DETERMINER_DEMONSTRATIVES
        trailing_bad = SIMPLE_PREPOSITIONS | {"a", "al", "del"} | COORDINATING_CONJUNCTIONS | SUBORDINATING_CONJUNCTIONS
        clitic_like = {"me", "te", "se", "nos", "os", "lo", "la", "los", "las", "le", "les"}
        leading_subordinators = {"que", "porque", "si", "cuando", "como"}

        if lowered[-1] in det_like | trailing_bad | clitic_like:
            reasons.append("dangling_function_word")
        if lowered[-1] == "no" and len(words) > 1:
            reasons.append("terminal_negation")

        for idx, token in enumerate(lowered[:-1]):
            if token in det_like and self.lookup_pos(words[idx + 1]) == "adj":
                if idx + 2 >= len(words) or self.lookup_pos(words[idx + 2]) != "n":
                    reasons.append("truncated_noun_phrase")
                    break

        if lowered[0] in leading_subordinators and self.finite_verb_count(words) < 2:
            reasons.append("orphaned_subordinate_clause")

        if self.has_unlinked_finite_verbs(words):
            reasons.append("unlinked_finite_verbs")

        for idx in range(1, len(lowered) - 2):
            if lowered[idx] != "de" or lowered[idx + 1] != "lo":
                continue
            if lowered[idx - 1] not in {"más", "menos", "mucho", "poco", "tan"}:
                continue
            if self.lookup_pos(words[idx + 2]) in {"adj", "adv"}:
                reasons.append("bad_degree_phrase")
                break

        return list(dict.fromkeys(reasons))

    def bad_copula_output(self, words: List[str]) -> bool:
        copula_idx = next((i for i, w in enumerate(words) if w in COPULA_FORMS), -1)
        if copula_idx < 0 or copula_idx + 1 >= len(words):
            return False
        subject_class = self.subject_semantic_class(words, copula_idx)
        for token in words[copula_idx + 1 :]:
            info = self.surface_analysis(token)
            if info["pos"] != "adj":
                continue
            adj_lemma = info["lemma"] or token
            if adj_lemma in WEAK_COPULA_ADJECTIVES:
                return True
            allowed_classes = ADJ_SUBJECT_PREFS.get(adj_lemma)
            if allowed_classes and subject_class and subject_class not in allowed_classes:
                return True
            if words[copula_idx] in {"es", "eres", "soy", "somos", "son", "fue", "era"} and adj_lemma not in SAFE_SER_ADJECTIVES and allowed_classes is None:
                return True
            break
        return False

    def verb_object_is_semantically_safe(self, canonical: str, noun_surface: Optional[str]) -> bool:
        if not noun_surface:
            return True
        noun = self.surface_analysis(noun_surface)
        noun_class = noun["semantic_class"]
        noun_lemma = normalize_token(noun["lemma"] or noun_surface)
        if noun_class in REJECT_OBJECT_CLASSES_BY_VERB.get(canonical, set()):
            return False
        explicit_safe_lemmas = SAFE_TEMPLATE_OBJECT_LEMMAS_BY_VERB.get(canonical)
        if explicit_safe_lemmas is not None and noun_lemma not in explicit_safe_lemmas:
            return False
        allowed_classes = SAFE_OBJECT_CLASSES_BY_VERB.get(canonical)
        if allowed_classes and noun_class and noun_class not in allowed_classes:
            return False
        return True

    def noun_surface_in_lemma_set(self, noun_surface: Optional[str], safe_lemmas: Iterable[str]) -> bool:
        if not noun_surface:
            return False
        noun = self.surface_analysis(noun_surface)
        noun_lemma = normalize_token(noun["lemma"] or noun_surface)
        return noun_lemma in {normalize_token(x) for x in safe_lemmas}

    def destination_is_safe(self, noun_surface: Optional[str]) -> bool:
        return self.noun_surface_in_lemma_set(noun_surface, SAFE_MOVEMENT_DESTINATION_LEMMAS)

    def location_is_safe(self, noun_surface: Optional[str]) -> bool:
        return self.noun_surface_in_lemma_set(noun_surface, SAFE_LOCATION_LEMMAS)

    def review_flags(self, candidate: Candidate) -> Tuple[str, str, str, str]:
        if not candidate.sentence:
            return "0", "0", "0", "manual review needed"
        words = self.word_tokens(candidate.sentence)
        verb_index = self.first_finite_verb_index(words)
        valid, _ = self.validate(candidate)
        if not valid:
            notes = self.universal_coherence_reasons(words)
            if not notes:
                notes = [candidate.template_id or "failed_validation"]
            return "0", "0", "0", "; ".join(dict.fromkeys(x for x in notes if x))
        grammatical = "1"
        natural = "1"
        learner_clear = "1"
        notes: List[str] = []
        rejected_retrieval, retrieval_penalties, retrieval_reasons = self.retrieved_quality(candidate, words, verb_index)
        if candidate.source_method != "retrieved_corpus" and candidate.template_id:
            notes.append(candidate.template_id)
        if retrieval_reasons:
            notes.extend(retrieval_reasons)
        if rejected_retrieval or retrieval_penalties:
            natural = "0"
            learner_clear = "0"
        if self.low_value_here_fallback(words, verb_index):
            natural = "0"
            learner_clear = "0"
            notes.append("low_value_here_fallback")
        if candidate.template_id in LOW_VALUE_TEMPLATE_IDS:
            natural = "0"
            if candidate.template_id in {"seeded_noun_here", "n_a1_2", "seeded_verb_adv"}:
                learner_clear = "0"
        if candidate.template_id.endswith("_haber_perfect"):
            natural = "0"
            learner_clear = "0"
        if self.review_override_template(candidate):
            return grammatical, natural, learner_clear, "; ".join(dict.fromkeys(x for x in notes if x))
        if candidate.source_method != "retrieved_corpus" and candidate.score < 8.0:
            natural = "0"
        if candidate.score < 7.5:
            learner_clear = "0"
        if candidate.source_method == "retrieved_corpus" and candidate.score < 8.5:
            natural = "0"
        if candidate.source_method == "retrieved_corpus" and candidate.score < 7.5:
            learner_clear = "0"
        return grammatical, natural, learner_clear, "; ".join(dict.fromkeys(x for x in notes if x))

    def _inject_seed_context(self, lemma: str, sentence: str) -> None:
        """Inject a manually authored seed sentence into lemma_contexts."""
        tokens = self.sentence_tokens(sentence)
        if not tokens:
            return
        target_surface = lemma.strip().lower()
        target_idx = -1
        for i, tok in enumerate(tokens):
            if normalize_token(tok) == target_surface:
                target_idx = i
                break
        if target_idx < 0:
            for i, tok in enumerate(tokens):
                if not is_word_token(tok):
                    continue
                candidate_lemma = self.lookup_lemma(tok)
                if candidate_lemma and normalize_token(candidate_lemma) == target_surface:
                    target_idx = i
                    break
        if target_idx < 0:
            return
        ctx = {
            "tokens": tokens,
            "index": target_idx,
            "target_form": tokens[target_idx],
            "target_pos": "",
            "left": normalize_token(tokens[target_idx - 1]) if target_idx > 0 else "",
            "right": normalize_token(tokens[target_idx + 1]) if target_idx + 1 < len(tokens) else "",
            "source": "seed_override",
        }
        self.lemma_contexts.setdefault(lemma, []).insert(0, ctx)

    def contexts_for_target(self, target: Lexeme) -> List[Dict]:
        seen = set()
        out: List[Dict] = []
        keys = [target.lemma]
        canonical = self.canonical_lemma_for(target)
        if canonical not in keys:
            keys.append(canonical)
        for key in keys:
            for ctx in self.lemma_contexts.get(key, []):
                marker = (tuple(ctx.get("tokens") or []), ctx.get("index", -1))
                if marker in seen:
                    continue
                seen.add(marker)
                out.append(ctx)
        return out

    def participle_form(self, lemma: str) -> Optional[str]:
        for entry in self.lemma_forms.get(lemma, []):
            morph = entry.get("morph") or {}
            if morph.get("VerbForm") == "Part" and morph.get("Number", "Sing") == "Sing":
                gender = morph.get("Gender")
                if gender in {None, "Masc"}:
                    return entry.get("form")
        if lemma.endswith("ar"):
            return lemma[:-2] + "ado"
        if lemma.endswith(("er", "ir")):
            return lemma[:-2] + "ido"
        return None

    def location_phrase(self, noun: Lexeme) -> List[str]:
        if noun.lemma in ARTICLELESS_LOCATION_NOUNS:
            return ["en", noun.lemma]
        article = self.choose_article(self.safe_noun_gender(noun.lemma, noun.gender), definite=True)
        return ["en", article, noun.lemma]

    def destination_phrase(self, noun: Lexeme) -> List[str]:
        if noun.lemma in ARTICLELESS_LOCATION_NOUNS:
            return ["a", noun.lemma]
        article = self.choose_article(self.safe_noun_gender(noun.lemma, noun.gender), definite=True)
        return ["a", article, noun.lemma]

    def special_verb_candidate(self, target: Lexeme, allowed: int, template_id: str, source_method: str) -> Optional[Candidate]:
        canonical = self.canonical_lemma_for(target)
        subject, person_code = self.subject_for_target(target)
        verb = self.target_verb_form(target, person_code)
        morph = self.target_form_metadata(target)
        verb_form = morph.get("VerbForm", "")
        mood = morph.get("Mood", "")
        tense = morph.get("Tense", "")

        if canonical == "haber":
            if normalize_token(verb) == "hay":
                noun = self.pick_semantic_noun(
                    allowed,
                    preferred_classes=GOOD_EXISTENTIAL_CLASSES,
                    banned_classes=BAD_EXISTENTIAL_CLASSES,
                    exclude={target.lemma, canonical},
                )
                if not noun:
                    return None
                article = self.choose_article(self.safe_noun_gender(noun.lemma, noun.gender), definite=False)
                return self.build_candidate(target, [verb, article, noun.lemma], f"{template_id}_haber_existential", source_method, 0)
            if tense == "Imp":
                noun = self.pick_semantic_noun(
                    allowed,
                    preferred_classes=GOOD_EXISTENTIAL_CLASSES,
                    banned_classes=BAD_EXISTENTIAL_CLASSES,
                    exclude={target.lemma, canonical},
                )
                if not noun:
                    return None
                article = self.choose_article(self.safe_noun_gender(noun.lemma, noun.gender), definite=False)
                return self.build_candidate(target, [verb, article, noun.lemma], f"{template_id}_haber_existential", source_method, 0)
            return self.build_candidate(target, [subject, verb, "hecho", "eso"], f"{template_id}_haber_perfect", source_method, 1)

        if canonical == "ser":
            if verb_form == "Part":
                return self.build_candidate(target, ["ha", verb, "difícil"], f"{template_id}_ser_part", source_method, 1)
            if mood == "Sub":
                return self.build_candidate(target, ["quiero", "que", verb, "claro"], f"{template_id}_ser_subj", source_method, 2)
            adj = self.pick_compatible_adjective(allowed, "person", exclude={target.lemma, canonical})
            if adj:
                return self.build_candidate(
                    target,
                    [subject, verb, self.inflect_adj_for_subject(adj.lemma, subject)],
                    f"{template_id}_ser_adj",
                    source_method,
                    1,
                )
            return None

        if canonical == "estar":
            if verb_form == "Ger":
                return self.build_candidate(target, ["ella", "está", verb], f"{template_id}_estar_gerund", source_method, 2)
            place = self.pick_safe_location_noun(allowed, exclude={target.lemma, canonical})
            if place:
                loc = self.location_phrase(place)
                return self.build_candidate(target, [subject, verb] + loc, f"{template_id}_estar_place", source_method, 1)
            return None

        if canonical == "ir":
            place = self.pick_safe_destination_noun(allowed, exclude={target.lemma, canonical})
            if not place:
                return None
            dest = self.destination_phrase(place)
            return self.build_candidate(target, [subject, verb] + dest, f"{template_id}_ir_place", source_method, 1)

        if canonical == "poder":
            if mood == "Cnd":
                return self.build_candidate(target, [subject, verb, "ir"], f"{template_id}_poder_conditional", source_method, 1)
            place = self.pick_safe_destination_noun(allowed, exclude={target.lemma, canonical, "ir"})
            if not place:
                return None
            dest = self.destination_phrase(place)
            return self.build_candidate(target, [subject, verb, "ir"] + dest, f"{template_id}_poder_ir_place", source_method, 1)

        if canonical == "querer":
            noun = self.pick_safe_object_noun_for_verb(canonical, allowed, exclude={target.lemma, canonical})
            if noun:
                article = self.choose_article(self.safe_noun_gender(noun.lemma, noun.gender), definite=False)
                return self.build_candidate(target, [subject, verb, article, noun.lemma], f"{template_id}_querer_obj", source_method, 1)
            return None

        if canonical == "hacer":
            if verb_form == "Ger":
                return self.build_candidate(target, ["ella", "está", verb, "eso"], f"{template_id}_hacer_gerund", source_method, 2)
            noun = self.pick_safe_object_noun_for_verb(canonical, allowed, exclude={target.lemma, canonical})
            if noun:
                article = self.choose_article(self.safe_noun_gender(noun.lemma, noun.gender), definite=False)
                return self.build_candidate(target, [subject, verb, article, noun.lemma], f"{template_id}_hacer_obj", source_method, 1)
            return None

        if canonical == "tener":
            noun = self.pick_safe_object_noun_for_verb(canonical, allowed, exclude={target.lemma, canonical})
            if noun:
                article = self.choose_article(self.safe_noun_gender(noun.lemma, noun.gender), definite=False)
                return self.build_candidate(target, [subject, verb, article, noun.lemma], f"{template_id}_tener_obj", source_method, 1)
            return None

        if canonical == "saber":
            return self.build_candidate(target, [subject, verb, "la", "verdad"], f"{template_id}_saber_truth", source_method, 1)

        if canonical == "creer":
            return self.build_candidate(target, [subject, verb, "eso"], f"{template_id}_creer_object", source_method, 1)

        return None

    def pick_candidate(
        self,
        pos: str,
        rank_ceiling: int,
        semantic_classes: Optional[List[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[Lexeme]:
        exclude = {normalize_token(x) for x in (exclude or []) if x}
        bucket = self.pos_buckets.get(pos, [])
        candidates = [
            x
            for x in bucket
            if x.rank <= rank_ceiling
            and normalize_token(x.lemma) not in exclude
            and normalize_token(self.canonical_lemma_for(x)) not in exclude
        ]
        if semantic_classes:
            preferred = [x for x in candidates if x.semantic_class in semantic_classes]
            if preferred:
                candidates = preferred
        if not candidates:
            return None
        weights = [1.0 / max(1, c.rank) for c in candidates[:500]]
        trimmed = candidates[:500]
        return self.random.choices(trimmed, weights=weights, k=1)[0]

    def score_sequence(self, tokens: List[str]) -> float:
        words = ["<START>"] + [normalize_token(t) for t in tokens if is_word_token(t)] + ["<END>"]
        if len(words) <= 2:
            return -10.0
        tri_counts = self.trigrams["counts"]
        bi_counts = self.bigrams["counts"]
        total = 0.0
        for i in range(1, len(words)):
            if i >= 2 and (words[i - 2], words[i - 1], words[i]) in tri_counts:
                total += math.log(tri_counts[(words[i - 2], words[i - 1], words[i])] + 1)
            elif (words[i - 1], words[i]) in bi_counts:
                total += math.log(bi_counts[(words[i - 1], words[i])] + 1)
            else:
                total -= 10.0
        return total / max(1, len(words) - 1)

    def _adjust_index_for_contractions(self, original_tokens: List[str], contracted: List[str], old_index: int) -> int:
        if old_index < 0 or old_index >= len(original_tokens):
            return old_index
        target_surface = normalize_token(original_tokens[old_index])
        if not target_surface:
            return -1
        i = 0
        j = 0
        while i < len(original_tokens) and j < len(contracted):
            if i == old_index:
                if normalize_token(contracted[j]) == target_surface:
                    return j
                return -1
            cont_norm = normalize_token(contracted[j])
            orig_norm = normalize_token(original_tokens[i])
            if orig_norm == cont_norm:
                i += 1
                j += 1
            elif cont_norm in ("al", "del") and i + 1 < len(original_tokens):
                i += 2
                j += 1
            else:
                i += 1
                j += 1
        return -1

    def build_candidate(
        self,
        target: Lexeme,
        tokens: List[str],
        template_id: str,
        source_method: str,
        target_index: Optional[int] = None,
        target_pos_hint: str = "",
        target_form_hint: str = "",
    ) -> Optional[Candidate]:
        nonempty_tokens = [t for t in tokens if t]
        if target_index is not None:
            filtered_index = 0
            orig_idx = 0
            remap = -1
            for i, t in enumerate(tokens):
                if not t:
                    continue
                if i == target_index:
                    remap = filtered_index
                    break
                filtered_index += 1
            if remap >= 0:
                target_index = remap
            else:
                target_index = -1
        clean_tokens = self.apply_contractions(nonempty_tokens)
        if target_index is not None and target_index >= 0:
            target_index = self._adjust_index_for_contractions(nonempty_tokens, clean_tokens, target_index)
        word_tokens = [t for t in clean_tokens if is_word_token(t)]
        if not word_tokens:
            return None
        if target_index is None:
            target_index = next((i for i, t in enumerate(clean_tokens) if normalize_token(t) == normalize_token(target.lemma)), -1)
        target_form = clean_tokens[target_index] if 0 <= target_index < len(clean_tokens) else target.lemma
        support_ranks: List[int] = []
        for i, tok in enumerate(clean_tokens):
            if not is_word_token(tok):
                continue
            if i == target_index:
                continue
            support_ranks.append(self.lookup_rank(tok))
        avg_support = statistics.mean(support_ranks) if support_ranks else 0.0
        max_support = max(support_ranks) if support_ranks else 0
        sentence = self.detokenize(clean_tokens)
        sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        canonical_target = self.canonical_lemma_for(target)
        candidate = Candidate(
            lemma=target.lemma,
            rank=target.rank,
            pos=target.pos,
            band=get_profile(target.rank).band,
            translation=target.translation,
            sentence=sentence,
            target_form=target_form,
            target_index=target_index,
            support_ranks=support_ranks,
            avg_support_rank=avg_support,
            max_support_rank=max_support,
            template_id=template_id,
            source_method=source_method,
            canonical_lemma=canonical_target,
            target_morph=self.target_morph_for_request(target, target_form, sentence_tokens=clean_tokens, target_idx=target_index),
        )
        setattr(candidate, "_target_pos_hint", target_pos_hint or target.pos or "")
        setattr(candidate, "_target_form_hint", target_form_hint or target_form)
        if target.pos in {"v", "n", "adj"} and not candidate.target_morph:
            return None
        if not self.candidate_target_matches_request(candidate, tokens=clean_tokens):
            return None
        candidate.target_form = clean_tokens[candidate.target_index]
        return candidate

    def manual_review_candidate(self, target: Lexeme) -> Candidate:
        return Candidate(
            lemma=target.lemma,
            rank=target.rank,
            pos=target.pos,
            band=get_profile(target.rank).band,
            translation=target.translation,
            sentence="",
            target_form=target.lemma,
            target_index=-1,
            support_ranks=[],
            avg_support_rank=0.0,
            max_support_rank=0,
            template_id="",
            source_method="manual_review_needed",
        )

    def candidate_sentence_key(self, candidate: Candidate) -> str:
        return " ".join(self.word_tokens(candidate.sentence))

    def dedupe_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        best_by_sentence: Dict[str, Candidate] = {}
        for candidate in candidates:
            key = self.candidate_sentence_key(candidate)
            if not key:
                continue
            existing = best_by_sentence.get(key)
            if existing is None or candidate.score > existing.score:
                best_by_sentence[key] = candidate
        return sorted(best_by_sentence.values(), key=lambda c: c.score, reverse=True)

    def support_rank_caps(
        self,
        candidate: Candidate,
        profile: DifficultyProfile,
        max_allowed: int,
        avg_allowed: int,
    ) -> Tuple[int, int]:
        caps = SAFE_TEMPLATE_SUPPORT_LIMITS.get(candidate.template_id)
        if not caps:
            return max_allowed, avg_allowed
        cap_max, cap_avg = caps
        return max(max_allowed, cap_max), max(avg_allowed, cap_avg)

    def review_override_template(self, candidate: Candidate) -> bool:
        return candidate.template_id in SAFE_TEMPLATE_REVIEW_OVERRIDES

    def haber_perfect_surface_ok(self, candidate: Candidate, words: List[str]) -> bool:
        target = self.lexicon.get(candidate.lemma)
        if not target or self.canonical_lemma_for(target) != "haber":
            return False
        if normalize_token(candidate.target_form or candidate.lemma) == "hay":
            return False
        target_word_idx = next(
            (i for i, word in enumerate(words) if normalize_token(word) == normalize_token(candidate.target_form or candidate.lemma)),
            -1,
        )
        if target_word_idx < 0 or target_word_idx >= len(words) - 1:
            return False
        participle = words[target_word_idx + 1]
        part_morph = self.surface_morph(participle)
        if part_morph.get("VerbForm") != "Part":
            return False
        part_lemma = normalize_token(self.lookup_lemma(participle) or participle)
        if part_lemma not in SAFE_HABER_PERFECT_VERBS:
            return False
        noun_surface = self.first_following_noun(words, target_word_idx + 1)
        if noun_surface:
            return self.verb_object_is_semantically_safe(part_lemma, noun_surface)
        trailing = words[target_word_idx + 2 :]
        if any(self.lookup_pos(token) == "pron" for token in trailing):
            return True
        return part_lemma == "hacer"

    def exact_surface_template_candidate(
        self,
        target: Lexeme,
        source_method: str,
    ) -> Optional[Candidate]:
        surface = normalize_token(target.lemma)
        canonical = normalize_token(self.canonical_lemma_for(target))
        specs: List[Tuple[str, List[str], int]] = []

        if target.pos == "v":
            if surface == "ha" and canonical == "haber":
                specs = [("verb_exact_ha_do_that", ["ha", "hecho", "eso"], 0)]
            elif surface == "han" and canonical == "haber":
                specs = [("verb_exact_han_done_that", ["han", "hecho", "eso"], 0)]
            elif surface == "había" and canonical == "haber":
                specs = [("verb_exact_habia_problem", ["había", "un", "problema"], 0)]
            elif surface == "sé" and canonical == "saber":
                specs = [("verb_exact_se_know_that", ["yo", "sé", "eso"], 1)]
            elif surface == "soy" and canonical == "ser":
                specs = [("verb_exact_soy_feliz", ["yo", "soy", "feliz"], 1)]
            elif surface == "eres" and canonical == "ser":
                specs = [("verb_exact_eres_bueno", ["tú", "eres", "bueno"], 1)]
            elif surface == "son" and canonical == "ser":
                specs = [("verb_exact_son_buenos", ["ellos", "son", "buenos"], 1)]
            elif surface == "hace" and canonical == "hacer":
                specs = [("verb_exact_hace_plan", ["ella", "hace", "un", "plan"], 1)]
            elif surface == "decir" and canonical == "decir":
                specs = [("verb_exact_decir_that", ["quiero", "decir", "eso"], 1)]
            elif surface == "siento" and canonical == "sentir":
                specs = [("verb_exact_lo_siento", ["lo", "siento"], 1)]
            elif surface == "sea" and canonical == "ser":
                specs = [("verb_exact_sea_claro", ["quiero", "que", "sea", "claro"], 2)]
            elif surface == "sido" and canonical == "ser":
                specs = [("verb_exact_sido_dificil", ["ha", "sido", "difícil"], 1)]
            elif surface == "podría" and canonical == "poder":
                specs = [("verb_exact_podria_ir", ["ella", "podría", "ir"], 1)]
            elif surface == "haciendo" and canonical == "hacer":
                specs = [("verb_exact_haciendo_eso", ["ella", "está", "haciendo", "eso"], 2)]
            elif surface == "haces" and canonical == "hacer":
                specs = [("verb_exact_haces_eso", ["tú", "haces", "eso"], 1)]
            elif surface == "hemos" and canonical == "haber":
                specs = [("verb_exact_hemos_hecho", ["hemos", "hecho", "eso"], 0)]
            elif surface == "debo" and canonical == "deber":
                specs = [("verb_exact_debo_ir", ["yo", "debo", "ir"], 1)]
            elif surface == "ven" and canonical == "venir":
                specs = [("verb_exact_ven_aqui", ["ven", "aquí"], 0)]
            elif surface == "ve" and canonical == "ver":
                specs = [("verb_exact_ve_eso", ["ve", "eso"], 0)]
            elif surface == "da" and canonical == "dar":
                specs = [("verb_exact_da_eso", ["ella", "da", "eso"], 1)]
            elif surface == "digo" and canonical == "decir":
                specs = [("verb_exact_digo_eso", ["yo", "digo", "eso"], 1)]
            elif surface == "pensé" and canonical == "pensar":
                specs = [("verb_exact_pense_eso", ["yo", "pensé", "eso"], 1)]
            elif surface == "dije" and canonical == "decir":
                specs = [("verb_exact_dije_eso", ["yo", "dije", "eso"], 1)]
        elif target.pos == "n":
            if surface == "vez":
                specs = [("noun_exact_turn", ["es", "mi", "vez"], 2)]
            elif surface == "gracias":
                specs = [("noun_exact_say_thanks", ["ella", "dice", "gracias"], 2)]
            elif surface == "vida":
                specs = [("noun_exact_life", ["la", "vida", "es", "buena"], 1)]
            elif surface == "día":
                specs = [("noun_exact_day", ["el", "día", "es", "largo"], 1)]
            elif surface == "favor":
                specs = [("noun_exact_favor", ["te", "hago", "un", "favor"], 3)]
            elif surface == "lugar":
                specs = [("noun_exact_place", ["es", "un", "lugar", "tranquilo"], 2)]
            elif surface == "nombre":
                specs = [("noun_exact_name", ["ese", "nombre", "es", "común"], 1)]
            elif surface == "mañana":
                specs = [("noun_exact_morning", ["la", "mañana", "es", "tranquila"], 1)]
            elif surface == "mamá":
                specs = [("noun_exact_mama", ["mi", "mamá", "está", "aquí"], 1)]
            elif surface == "parte":
                specs = [("noun_exact_part", ["es", "parte", "del", "plan"], 1)]
            elif surface == "nombre":
                specs = [("noun_exact_name", ["mi", "nombre", "es", "claro"], 1)]
            elif surface == "mamá":
                specs = [("noun_exact_mama", ["mi", "mamá", "está", "aquí"], 1)]
            elif surface == "lugar":
                specs = [("noun_exact_place", ["es", "un", "lugar", "tranquilo"], 2)]
            elif surface == "mañana":
                specs = [("noun_exact_morning", ["la", "mañana", "es", "tranquila"], 1)]
            elif surface == "noche":
                specs = [("noun_exact_night", ["la", "noche", "es", "larga"], 1)]
            elif surface == "hora":
                specs = [("noun_exact_hour", ["es", "la", "hora"], 2)]
            elif surface == "fin":
                specs = [("noun_exact_end", ["es", "el", "fin"], 2)]
            elif surface == "agua":
                specs = [("noun_exact_water", ["el", "agua", "está", "aquí"], 1)]
            elif surface == "mano":
                specs = [("noun_exact_hand", ["la", "mano", "está", "aquí"], 1)]
            elif surface == "noche":
                specs = [("noun_exact_night", ["la", "noche", "es", "larga"], 1)]
            elif surface == "supuesto":
                specs = [("noun_exact_of_course", ["por", "supuesto"], 1)]
            elif surface == "general":
                specs = [("noun_exact_general", ["en", "general", "es", "claro"], 1)]
            elif surface == "minutos":
                specs = [("noun_exact_minutes", ["faltan", "minutos"], 1)]
            elif surface == "días":
                specs = [("noun_exact_days", ["pasan", "días"], 1)]
            elif surface == "horas":
                specs = [("noun_exact_hours", ["pasan", "horas"], 1)]
            elif surface == "veces":
                specs = [("noun_exact_times", ["muchas", "veces"], 1)]
            elif surface == "personas":
                specs = [("noun_exact_people", ["muchas", "personas", "llegan"], 1)]
            elif surface == "hombres":
                specs = [("noun_exact_men", ["muchos", "hombres", "llegan"], 1)]
            elif surface == "chicos":
                specs = [("noun_exact_boys", ["muchos", "chicos", "llegan"], 1)]
            elif surface == "ojos":
                specs = [("noun_exact_eyes", ["sus", "ojos", "están", "aquí"], 1)]
            elif surface == "buenas":
                specs = [("noun_exact_good_evening", ["buenas", "noches"], 0)]
            elif surface == "señora":
                specs = [("noun_exact_lady", ["la", "señora", "está", "aquí"], 1)]
            elif surface == "primera":
                specs = [("noun_exact_first", ["la", "primera", "parte"], 1)]
            elif surface == "pasado":
                specs = [("noun_exact_past", ["el", "pasado", "importa"], 1)]
            elif surface == "cierto":
                specs = [("noun_exact_true", ["es", "cierto"], 1)]
            elif surface == "claro":
                specs = [("noun_exact_clear", ["está", "claro"], 1)]
            elif surface == "pronto":
                specs = [("noun_exact_soon", ["hasta", "pronto"], 1)]
            elif surface == "ah":
                specs = [("noun_exact_ah", ["ah"], 0)]
            elif surface == "eh":
                specs = [("noun_exact_eh", ["eh"], 0)]

        for template_id, tokens, target_index in specs:
            cand = self.build_candidate(target, tokens, template_id, source_method, target_index)
            if cand:
                return cand
        return None

    def validate(self, candidate: Candidate) -> Tuple[bool, List[float]]:
        profile = get_profile(candidate.rank)
        allowed = allowed_support_rank(candidate.rank, profile)
        allowed, avg_allowed = self.support_rank_caps(candidate, profile, allowed, profile.avg_ceil)
        penalties: List[float] = []
        if not self.candidate_target_matches_request(candidate):
            return False, penalties
        words = self.word_tokens(candidate.sentence)
        if not (profile.min_len <= len(words) <= profile.max_len):
            return False, penalties
        target_forms = {normalize_token(candidate.lemma)}
        target = self.lexicon.get(candidate.lemma)
        canonical_target = self.canonical_lemma_for(target) if target else candidate.lemma
        if normalize_token(candidate.lemma) == normalize_token(canonical_target):
            target_forms.add(normalize_token(candidate.target_form))
            target_forms.update({normalize_token(f.get("form", "")) for f in self.lemma_forms.get(candidate.lemma, [])})
        if not any(w in target_forms for w in words):
            return False, penalties
        if candidate.max_support_rank > allowed:
            return False, penalties
        if candidate.avg_support_rank > avg_allowed:
            return False, penalties
        for a, b in zip(words, words[1:]):
            if a == b:
                return False, penalties

        target_morph_override = self._parse_morph_str(candidate.target_morph) if candidate.target_morph else None
        target_word_idx = -1
        if candidate.pos == "v" and target_morph_override and candidate.target_index >= 0:
            sentence_tokens = self.sentence_tokens(candidate.sentence)
            wi = 0
            for si, st in enumerate(sentence_tokens):
                if not is_word_token(st):
                    continue
                if si == candidate.target_index:
                    target_word_idx = wi
                    break
                wi += 1

        finite_verb_index = self.first_finite_verb_index(words, target_word_idx, target_morph_override)
        if finite_verb_index < 0:
            return False, penalties
        if self.prepositional_pronoun_used_as_subject(words):
            return False, penalties
        if not self.strict_subject_pronoun_agreement_ok(words, finite_verb_index, target_word_idx, target_morph_override):
            return False, penalties
        if not self.subject_verb_agreement_ok(words, finite_verb_index, target_word_idx, target_morph_override):
            return False, penalties
        if not self.copula_predicate_agreement_ok(words, finite_verb_index):
            return False, penalties
        if not self.determiner_phrase_ok(words):
            return False, penalties
        if self.temporal_verb_mismatch(words, finite_verb_index, target_word_idx, target_morph_override):
            return False, penalties
        if self.semantic_nonsense_reasons(candidate, words):
            return False, penalties

        for i, word in enumerate(words[:-1]):
            if word in ARTICLE_FEATURES:
                if i + 2 < len(words) and self.lookup_pos(words[i + 1]) == "adj" and self.lookup_pos(words[i + 2]) == "n":
                    if not self.article_matches_noun(word, words[i + 2]):
                        return False, penalties
                    if not self.adjective_matches_noun(words[i + 1], words[i + 2]):
                        return False, penalties
                elif self.lookup_pos(words[i + 1]) == "n" and not self.article_matches_noun(word, words[i + 1]):
                    return False, penalties
        for a, b in zip(words, words[1:]):
            if self.lookup_pos(a) == "adj" and self.lookup_pos(b) == "n" and not self.adjective_matches_noun(a, b):
                return False, penalties
            if self.lookup_pos(a) == "n" and self.lookup_pos(b) == "adj" and not self.adjective_matches_noun(b, a):
                return False, penalties
        if candidate.pos == "v" and candidate.source_method != "retrieved_corpus":
            if target:
                if self.template_support_reason(target):
                    return False, penalties
                person_code = self.person_code_from_morph(self.target_form_metadata(target))
                if person_code and words:
                    subject = normalize_token(words[0])
                    expected = SUBJECT_FEATURES.get(subject, {}).get("person_code")
                    if expected and expected != person_code:
                        return False, penalties
                canonical = self.canonical_lemma_for(target)
                if canonical == "haber" and words and words[0] == "hay":
                    noun_surface = self.first_following_noun(words, 0)
                    noun = self.surface_analysis(noun_surface or "")
                    if noun["semantic_class"] in BAD_EXISTENTIAL_CLASSES:
                        return False, penalties
                if canonical == "haber" and words and words[0] != "hay" and not self.haber_perfect_surface_ok(candidate, words):
                    if not (target_morph_override and target_morph_override.get("Tense") == "Imp" and words and normalize_token(words[0]) == normalize_token(candidate.target_form)):
                        return False, penalties
                noun_surface = self.first_following_noun(words, candidate.target_index if candidate.target_index >= 0 else 0)
                if canonical == "haber" and self.haber_perfect_surface_ok(candidate, words):
                    noun_surface = self.first_following_noun(words, 1)
                    participle = words[1] if len(words) > 1 else ""
                    part_lemma = normalize_token(self.lookup_lemma(participle) or participle)
                    if noun_surface and not self.verb_object_is_semantically_safe(part_lemma, noun_surface):
                        return False, penalties
                elif not self.verb_object_is_semantically_safe(canonical, noun_surface):
                    return False, penalties
                if canonical in {"ir", "poder"} and noun_surface and not self.destination_is_safe(noun_surface):
                    return False, penalties
                if canonical == "estar" and noun_surface and not self.location_is_safe(noun_surface):
                    return False, penalties
        if candidate.pos == "n" and candidate.source_method != "retrieved_corpus":
            if target and " aquí" in candidate.sentence.lower() and not self.noun_supports_here_template(target):
                return False, penalties
            if target and self.lookup_lemma(words[1] if len(words) > 1 else "") == target.lemma and " es " in candidate.sentence.lower() and not self.noun_is_template_friendly(target):
                return False, penalties
        if candidate.pos == "adj" and candidate.source_method != "retrieved_corpus" and normalize_token(candidate.lemma) in APOCOPATED_ADJECTIVE_FEATURES:
            return False, penalties
        if candidate.pos == "adj" and candidate.source_method != "retrieved_corpus" and not self.adjective_target_is_template_friendly(target or self.lexicon.get(candidate.lemma)):
            return False, penalties
        if self.standalone_subjunctive_without_trigger(words, finite_verb_index, target_word_idx, target_morph_override):
            return False, penalties
        if self.low_value_here_fallback(words, finite_verb_index):
            return False, penalties
        rejected_retrieval, retrieval_penalties, _ = self.retrieved_quality(candidate, words, finite_verb_index)
        if rejected_retrieval:
            return False, penalties
        penalties.extend(retrieval_penalties)
        strategy_reasons = self.strategy_validation_reasons(candidate, words, finite_verb_index)
        if strategy_reasons:
            return False, penalties
        if self.bad_copula_output(words):
            return False, penalties
        coherence_reasons = self.universal_coherence_reasons(words)
        if coherence_reasons:
            return False, penalties
        if len(words) == profile.min_len:
            penalties.append(0.2)
        if len(words) == profile.max_len:
            penalties.append(0.2)
        return True, penalties

    def score(self, candidate: Candidate, penalties: List[float]) -> float:
        profile = get_profile(candidate.rank)
        value = 3.0
        if candidate.source_method == "retrieved_corpus":
            value += 5.0
        elif candidate.source_method == "seeded_template":
            value += 1.0
        if candidate.avg_support_rank < candidate.rank:
            value += 1.5
        if candidate.max_support_rank < candidate.rank:
            value += 1.0
        value += self.score_sequence(candidate.sentence.split())
        value -= sum(penalties)
        value -= 0.3 * abs(len([t for t in candidate.sentence.split() if is_word_token(t)]) - profile.ideal_length)
        value += self.strategy_score_adjustment(candidate)
        candidate.score = value
        return value

    def try_candidate(self, cand: Optional[Candidate]) -> Optional[Candidate]:
        if not cand:
            return None
        ok, penalties = self.validate(cand)
        if not ok:
            return None
        self.score(cand, penalties)
        return cand

    def starter_support_limits(self, candidate: Candidate) -> Tuple[int, float]:
        band = get_profile(candidate.rank).band
        if band == "A1":
            return 350, 180.0
        if band == "A2":
            return 600, 300.0
        profile = get_profile(candidate.rank)
        return profile.filler_ceil, float(profile.avg_ceil)

    def starter_validate(self, candidate: Candidate) -> Tuple[bool, List[float]]:
        profile = get_profile(candidate.rank)
        allowed, avg_ceiling = self.starter_support_limits(candidate)
        penalties: List[float] = []
        if not self.candidate_target_matches_request(candidate):
            return False, penalties
        words = self.word_tokens(candidate.sentence)
        if not (profile.min_len <= len(words) <= profile.max_len):
            return False, penalties
        target_forms = {normalize_token(candidate.lemma)}
        target = self.lexicon.get(candidate.lemma)
        canonical_target = self.canonical_lemma_for(target) if target else candidate.lemma
        if normalize_token(candidate.lemma) == normalize_token(canonical_target):
            target_forms.add(normalize_token(candidate.target_form))
            target_forms.update({normalize_token(f.get("form", "")) for f in self.lemma_forms.get(candidate.lemma, [])})
        if not any(word in target_forms for word in words):
            return False, penalties
        if candidate.max_support_rank > allowed:
            return False, penalties
        if candidate.avg_support_rank > avg_ceiling:
            return False, penalties
        for a, b in zip(words, words[1:]):
            if a == b:
                return False, penalties
        target_morph_override = self._parse_morph_str(candidate.target_morph) if candidate.target_morph else None
        target_word_idx = -1
        if candidate.pos == "v" and target_morph_override and candidate.target_index >= 0:
            sentence_tokens = self.sentence_tokens(candidate.sentence)
            word_idx = 0
            for sent_idx, token in enumerate(sentence_tokens):
                if not is_word_token(token):
                    continue
                if sent_idx == candidate.target_index:
                    target_word_idx = word_idx
                    break
                word_idx += 1
        finite_verb_index = self.first_finite_verb_index(words, target_word_idx, target_morph_override)
        if finite_verb_index < 0:
            return False, penalties
        if not self.subject_verb_agreement_ok(words, finite_verb_index, target_word_idx, target_morph_override):
            return False, penalties
        if not self.copula_predicate_agreement_ok(words, finite_verb_index):
            return False, penalties
        for i, word in enumerate(words[:-1]):
            if word in ARTICLE_FEATURES:
                if i + 2 < len(words) and self.lookup_pos(words[i + 1]) == "adj" and self.lookup_pos(words[i + 2]) == "n":
                    if not self.article_matches_noun(word, words[i + 2]):
                        return False, penalties
                    if not self.adjective_matches_noun(words[i + 1], words[i + 2]):
                        return False, penalties
                elif self.lookup_pos(words[i + 1]) == "n" and not self.article_matches_noun(word, words[i + 1]):
                    return False, penalties
        for a, b in zip(words, words[1:]):
            if self.lookup_pos(a) == "adj" and self.lookup_pos(b) == "n" and not self.adjective_matches_noun(a, b):
                return False, penalties
            if self.lookup_pos(a) == "n" and self.lookup_pos(b) == "adj" and not self.adjective_matches_noun(b, a):
                return False, penalties
        if candidate.pos == "v" and candidate.source_method != "retrieved_corpus":
            if target:
                if self.template_support_reason(target):
                    return False, penalties
                person_code = self.person_code_from_morph(self.target_form_metadata(target))
                if person_code and words:
                    subject = words[0]
                    expected = SUBJECT_FEATURES.get(subject, {}).get("person_code")
                    if expected and expected != person_code:
                        return False, penalties
                canonical = self.canonical_lemma_for(target)
                if canonical == "haber" and words and words[0] == "hay":
                    noun_surface = self.first_following_noun(words, 0)
                    noun = self.surface_analysis(noun_surface or "")
                    if noun["semantic_class"] in BAD_EXISTENTIAL_CLASSES:
                        return False, penalties
                if canonical == "haber" and words and words[0] != "hay":
                    return False, penalties
                noun_surface = self.first_following_noun(words, candidate.target_index if candidate.target_index >= 0 else 0)
                if not self.verb_object_is_semantically_safe(canonical, noun_surface):
                    return False, penalties
                if canonical in {"ir", "poder"} and noun_surface and not self.destination_is_safe(noun_surface):
                    return False, penalties
                if canonical == "estar" and noun_surface and not self.location_is_safe(noun_surface):
                    return False, penalties
        if candidate.pos == "n" and candidate.source_method != "retrieved_corpus":
            if target and " aquí" in candidate.sentence.lower() and not self.noun_supports_here_template(target):
                return False, penalties
            if target and self.lookup_lemma(words[1] if len(words) > 1 else "") == target.lemma and " es " in candidate.sentence.lower() and not self.noun_is_template_friendly(target):
                return False, penalties
        if candidate.pos == "adj" and candidate.source_method != "retrieved_corpus" and normalize_token(candidate.lemma) in APOCOPATED_ADJECTIVE_FEATURES:
            return False, penalties
        if candidate.pos == "adj" and candidate.source_method != "retrieved_corpus" and not self.adjective_target_is_template_friendly(target or self.lexicon.get(candidate.lemma)):
            return False, penalties
        if self.standalone_subjunctive_without_trigger(words, finite_verb_index, target_word_idx, target_morph_override):
            return False, penalties
        if self.low_value_here_fallback(words, finite_verb_index):
            return False, penalties
        rejected_retrieval, retrieval_penalties, _ = self.retrieved_quality(candidate, words, finite_verb_index)
        if rejected_retrieval:
            return False, penalties
        penalties.extend(retrieval_penalties)
        if self.bad_copula_output(words):
            return False, penalties
        coherence_reasons = self.universal_coherence_reasons(words)
        if coherence_reasons:
            return False, penalties
        if len(words) == profile.min_len:
            penalties.append(0.2)
        if len(words) == profile.max_len:
            penalties.append(0.2)
        return True, penalties

    def retrieve_candidates(self, target: Lexeme) -> List[Candidate]:
        contexts = self.contexts_for_target(target)
        out: List[Candidate] = []
        for ctx in contexts:
            tokens = ctx.get("tokens") or []
            idx = ctx.get("index", -1)
            if not tokens or idx < 0 or idx >= len(tokens):
                continue
            cand = self.build_candidate(
                target,
                tokens,
                template_id="retrieved",
                source_method="retrieved_corpus",
                target_index=idx,
                target_pos_hint=ctx.get("target_pos", ""),
                target_form_hint=ctx.get("target_form", ""),
            )
            if not cand:
                continue
            if not self.candidate_target_matches_request(cand, tokens=tokens):
                continue
            ok, penalties = self.validate(cand)
            if ok:
                self.score(cand, penalties)
                out.append(cand)
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:10]

    def strong_retrieved_candidate(self, candidate: Candidate) -> bool:
        return self.retrieved_is_preferred(candidate)

    def collect_candidates_for_lemma(self, lemma: str, max_candidates_per_lemma: Optional[int] = None) -> List[Candidate]:
        lemma = lemma.strip().lower()
        if lemma not in self.lexicon:
            raise KeyError(f"Lemma not in lexicon: {lemma}")
        if lemma not in self.candidate_pool_cache:
            target = self.lexicon[lemma]
            self.candidate_pool_cache[lemma] = self.generate_strategy_candidates(target)

        pool = list(self.candidate_pool_cache[lemma])
        if max_candidates_per_lemma is not None:
            return pool[:max(0, max_candidates_per_lemma)]
        return pool

    def seeded_template_candidate(self, target: Lexeme) -> Optional[Candidate]:
        exact = self.exact_surface_template_candidate(target, "seeded_template")
        if exact:
            return exact
        contexts = self.contexts_for_target(target)
        if not contexts:
            return None
        ctx = self.random.choice(contexts)
        left = normalize_token(ctx.get("left", ""))
        right = normalize_token(ctx.get("right", ""))
        allowed = allowed_support_rank(target.rank, get_profile(target.rank))
        canonical = self.canonical_lemma_for(target)

        if target.pos == "n":
            if not self.noun_is_template_friendly(target):
                return None
            tgt_gender = self.safe_noun_gender(target.lemma, target.gender)
            article = left if left in ARTICLE_SET else self.choose_article(tgt_gender, definite=True)
            right_lemma = self.lookup_lemma(right)
            right_lex = self.get_known_lexeme(right_lemma)
            if self.noun_supports_ser_adjective_template(target) and right_lex and right_lex.pos == "adj" and right_lex.rank <= allowed:
                adj = self.inflect_adj(right_lex.lemma, tgt_gender)
                tokens = [article, target.lemma, "es", adj]
                return self.build_candidate(target, tokens, "seeded_noun_adj", "seeded_template", 1)
            if not self.noun_supports_here_template(target):
                return None
            tokens = [article, target.lemma, "está", "aquí"]
            return self.build_candidate(target, tokens, "seeded_noun_here", "seeded_template", 1)

        if target.pos == "v":
            if canonical in SPECIAL_VERB_LEMMAS:
                special = self.special_verb_candidate(target, allowed, "seeded_special", "seeded_template")
                if special:
                    return special
            subject, person = self.subject_for_target(target, fallback=left if left in SUBJECT_FEATURES else "ella")
            verb = self.target_verb_form(target, person)
            obj = self.pick_safe_object_noun_for_verb(canonical, allowed, exclude={target.lemma, canonical})
            if obj:
                article = self.choose_article(self.safe_noun_gender(obj.lemma, obj.gender), definite=True)
                tokens = [subject, verb, article, obj.lemma]
                return self.build_candidate(target, tokens, "seeded_verb_obj", "seeded_template", 1)
            return None

        if target.pos == "adj":
            allowed_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(normalize_token(target.lemma))
            if not allowed_classes:
                return None
            left_lemma = self.lookup_lemma(left)
            left_lex = self.get_known_lexeme(left_lemma)
            if (left_lex and left_lex.pos == "n" and left_lex.rank <= allowed
                    and self.noun_is_template_friendly(left_lex)
                    and normalize_token(left_lex.lemma) in STARTER_SAFE_SUPPORT_NOUNS_FOR_ADJ
                    and left_lex.semantic_class in allowed_classes
                    and self.inflect_adj(target.lemma, self.safe_noun_gender(left_lex.lemma, left_lex.gender)) == target.lemma):
                noun = left_lex
                article = self.choose_article(self.safe_noun_gender(noun.lemma, noun.gender), definite=True)
                tokens = [article, noun.lemma, "es", target.lemma]
                return self.build_candidate(target, tokens, "seeded_adj_noun", "seeded_template", 3)
            noun = self.pick_starter_safe_adj_support_noun(allowed, exclude={target.lemma})
            if noun and noun.semantic_class in allowed_classes:
                noun_gender = self.safe_noun_gender(noun.lemma, noun.gender)
                if self.inflect_adj(target.lemma, noun_gender) == target.lemma:
                    article = self.choose_article(noun_gender, definite=True)
                    tokens = [article, noun.lemma, "es", target.lemma]
                    return self.build_candidate(target, tokens, "seeded_adj_fallback", "seeded_template", 3)
        return None

    def pure_template_candidate(self, target: Lexeme, starter_mode: bool = False) -> Optional[Candidate]:
        profile = get_profile(target.rank)
        allowed = allowed_support_rank(target.rank, profile)
        canonical = self.canonical_lemma_for(target)

        exact = self.exact_surface_template_candidate(target, "template_generated")
        if exact:
            return exact

        if target.pos == "n":
            if not self.noun_is_template_friendly(target):
                return None
            if starter_mode:
                choices = ["n_a1_1", "n_a1_2", "n_a1_3"]
            else:
                choices = ["n_a1_1", "n_a1_2", "n_a1_3"] if profile.band in {"A1", "A2"} else ["n_b1_1", "n_b1_2"]
            template = self.random.choice(choices)
            if template == "n_a1_1":
                if not self.noun_is_template_friendly(target):
                    return None
                tgt_gender = self.safe_noun_gender(target.lemma, target.gender)
                if starter_mode:
                    adj = self.pick_starter_compatible_adjective(allowed, target.semantic_class, exclude={target.lemma})
                else:
                    adj = self.pick_compatible_adjective(allowed, target.semantic_class, exclude={target.lemma})
                if not adj:
                    return None
                article = self.choose_article(tgt_gender, definite=True)
                tokens = [article, target.lemma, "es", self.inflect_adj(adj.lemma, tgt_gender)]
                return self.build_candidate(target, tokens, template, "template_generated", 1)
            if template == "n_a1_2":
                if not self.noun_supports_here_template(target):
                    return None
                article = self.choose_article(self.safe_noun_gender(target.lemma, target.gender), definite=True)
                tokens = [article, target.lemma, "está", "aquí"]
                return self.build_candidate(target, tokens, template, "template_generated", 1)
            if template == "n_a1_3":
                if not self.noun_supports_possession_template(target):
                    return None
                subject = self.random.choice(["yo", "él", "ella"])
                verb = self.conjugate_present("tener", "1sg" if subject == "yo" else "3sg")
                article = self.choose_article(self.safe_noun_gender(target.lemma, target.gender), definite=False)
                tokens = [subject, verb, article, target.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 3)
            if template == "n_b1_1":
                verb = self.pick_candidate("v", allowed, exclude={target.lemma, "ser", "estar", "tener"})
                place = self.pick_safe_location_noun(allowed, exclude={target.lemma})
                if not verb or not place:
                    return None
                article = self.choose_article(self.safe_noun_gender(target.lemma, target.gender), definite=True)
                place_article = self.choose_article(self.safe_noun_gender(place.lemma, place.gender), definite=True)
                tokens = ["ella", self.conjugate_present(verb.lemma, "3sg"), article, target.lemma, "en", place_article, place.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 3)
            if template == "n_b1_2":
                if not self.noun_supports_ser_adjective_template(target):
                    return None
                tgt_gender = self.safe_noun_gender(target.lemma, target.gender)
                adj = self.pick_compatible_adjective(allowed, target.semantic_class, exclude={target.lemma})
                verb = self.pick_candidate("v", allowed, exclude={target.lemma, "ser", "estar", "tener"})
                if not adj or not verb:
                    return None
                article = self.choose_article(tgt_gender, definite=True)
                tokens = [article, self.inflect_adj(adj.lemma, tgt_gender), target.lemma, self.conjugate_present(verb.lemma, "3sg")]
                return self.build_candidate(target, tokens, template, "template_generated", 2)

        if target.pos == "v":
            is_infinitive_target = (normalize_token(target.lemma) == normalize_token(canonical)
                                    and not (self.target_form_metadata(target).get("VerbForm") and self.target_form_metadata(target).get("VerbForm") != "Inf"))
            if not (starter_mode and is_infinitive_target) and canonical in SPECIAL_VERB_LEMMAS:
                special = self.special_verb_candidate(target, allowed, "template_special", "template_generated")
                if special:
                    return special
            if starter_mode and is_infinitive_target:
                choices = ["v_a1_1", "v_a1_inf", "v_a1_4"]
            elif profile.band in {"A1", "A2"}:
                choices = ["v_a1_1", "v_a1_2", "v_a1_4"]
            else:
                choices = ["v_b1_1"]
            template = self.random.choice(choices)
            subject, person = self.subject_for_target(target)
            verb = self.target_verb_form(target, person)
            if template == "v_a1_inf":
                if canonical in STARTER_INFINITIVE_REJECT:
                    return None
                carrier = self.random.choice(STARTER_INFINITIVE_CARRIERS)
                carrier_form = self.conjugate_present(carrier, "3sg")
                carrier_subject = self.random.choice(["ella", "él"])
                complements = STARTER_INFINITIVE_COMPLEMENTS.get(canonical)
                if complements:
                    noun_lemma, art = self.random.choice(complements)
                    tokens = [carrier_subject, carrier_form, target.lemma]
                    if art:
                        tokens.append(art)
                    tokens.append(noun_lemma)
                    return self.build_candidate(target, tokens, "v_a1_inf", "template_generated", 2)
                if canonical in BARE_INFINITIVE_OK:
                    tokens = [carrier_subject, carrier_form, target.lemma]
                    return self.build_candidate(target, tokens, "v_a1_inf", "template_generated", 2)
                return None
            if template == "v_a1_1":
                obj = self.pick_safe_object_noun_for_verb(canonical, allowed, exclude={target.lemma, canonical})
                if not obj:
                    return None
                article = self.choose_article(self.safe_noun_gender(obj.lemma, obj.gender), definite=False)
                tokens = [subject, verb, article, obj.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 1)
            if template == "v_a1_2":
                return None
            if template == "v_a1_4":
                if canonical not in PLACE_PREP_VERBS:
                    return None
                prep = target.required_prep
                if not prep:
                    return None
                place = self.pick_safe_destination_noun(allowed, exclude={target.lemma, canonical})
                if not place:
                    return None
                dest = self.destination_phrase(place)
                tokens = [subject, verb] + dest
                return self.build_candidate(target, tokens, template, "template_generated", 1)
            if template == "v_b1_1":
                obj = self.pick_safe_object_noun_for_verb(canonical, allowed, exclude={target.lemma, canonical})
                extra = self.pick_safe_location_noun(allowed, exclude={target.lemma, obj.lemma if obj else ""})
                if not obj or not extra:
                    return None
                art1 = self.choose_article(self.safe_noun_gender(obj.lemma, obj.gender), definite=True)
                art2 = self.choose_article(self.safe_noun_gender(extra.lemma, extra.gender), definite=True)
                prep = target.required_prep or self.random.choice(["para", "en", "con"])
                tokens = [subject, verb, art1, obj.lemma, prep, art2, extra.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 1)

        if target.pos == "adj":
            choices = ["adj_a1_1", "adj_a1_3"] if profile.band in {"A1", "A2"} else ["adj_b1_1"]
            template = self.random.choice(choices)
            if template == "adj_a1_1":
                pref_classes = ADJ_SUBJECT_PREFS.get(target.lemma)
                noun = None
                if starter_mode:
                    allowed_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(normalize_token(target.lemma))
                    if not allowed_classes:
                        return None
                    candidates_for_adj = [
                        x for x in self.pos_buckets.get("n", [])
                        if x.rank <= allowed
                        and self.noun_is_template_friendly(x)
                        and normalize_token(x.lemma) in STARTER_SAFE_SUPPORT_NOUNS_FOR_ADJ
                        and x.semantic_class in allowed_classes
                        and normalize_token(x.lemma) != normalize_token(target.lemma)
                    ]
                    noun = self.pick_from_candidates(candidates_for_adj)
                    if not noun:
                        return None
                else:
                    noun = self.pick_template_friendly_noun(allowed, semantic_classes=pref_classes, exclude={target.lemma})
                    if not noun:
                        noun = self.pick_template_friendly_noun(allowed, exclude={target.lemma})
                if not noun:
                    return None
                noun_gender = self.safe_noun_gender(noun.lemma, noun.gender)
                if self.inflect_adj(target.lemma, noun_gender) != target.lemma:
                    return None
                article = self.choose_article(noun_gender, definite=True)
                tokens = [article, noun.lemma, "es", target.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 3)
            if template == "adj_a1_3":
                if starter_mode:
                    allowed_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(normalize_token(target.lemma))
                    if not allowed_classes or "person" not in allowed_classes:
                        return None
                subject = self.random.choice(["él"])
                adj_form = self.inflect_adj_for_subject(target.lemma, subject)
                if adj_form != target.lemma:
                    return None
                tokens = [subject, "es", target.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 2)
            if template == "adj_b1_1":
                noun = self.pick_template_friendly_noun(allowed, exclude={target.lemma})
                extra = self.pick_template_friendly_noun(allowed, semantic_classes=["place"], exclude={target.lemma, noun.lemma if noun else ""})
                if not noun or not extra:
                    return None
                if self.inflect_adj(target.lemma, noun.gender) != target.lemma:
                    return None
                article1 = self.choose_article(noun.gender, definite=True)
                article2 = self.choose_article(extra.gender, definite=True)
                tokens = [article1, noun.lemma, "es", target.lemma, "en", article2, extra.lemma]
                return self.build_candidate(target, tokens, template, "template_generated", 3)
        return None

    def _build_from_specs(
        self,
        target: Lexeme,
        specs: List[Tuple[str, List[str], int]],
        seeded: bool = False,
        emergency: bool = False,
    ) -> Optional[Candidate]:
        ordered = list(specs)
        if emergency:
            ordered = ordered[:2]
        elif not seeded and len(ordered) > 1:
            ordered = self.random.sample(ordered, len(ordered))
        for template_id, tokens, target_index in ordered:
            cand = self.build_candidate(target, tokens, template_id, "template_generated", target_index)
            if cand:
                return cand
        return None

    def _safe_nominal_support(
        self,
        surface: str,
        allowed: int,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[Tuple[str, str]]:
        exclude_set = {normalize_token(x) for x in (exclude or []) if x}
        det_gender, det_number = self.determiner_surface_features(surface)
        preferred_by_gender = {
            "f": ["casa", "idea", "mesa", "puerta"],
            "m": ["libro", "trabajo", "amigo", "hotel"],
        }
        preferred = preferred_by_gender.get(det_gender or "m", preferred_by_gender["m"])
        noun = self.pick_preferred_lemmas(preferred, allowed, exclude=exclude_set)
        if not noun:
            candidates = [
                x for x in self.pos_buckets.get("n", [])
                if x.rank <= allowed
                and self.noun_is_template_friendly(x)
                and normalize_token(x.lemma) not in exclude_set
                and normalize_token(self.canonical_lemma_for(x)) not in exclude_set
                and (det_gender is None or self.safe_noun_gender(x.lemma, x.gender) == det_gender)
            ]
            noun = self.pick_from_candidates(candidates)
        if not noun:
            noun = self.pick_template_friendly_noun(allowed, exclude=exclude_set)
        if not noun:
            return None
        noun_surface = self.pluralize_noun(noun.lemma) if det_number == "pl" else noun.lemma
        verb = "están" if det_number == "pl" else "está"
        return noun_surface, verb

    def seeded_pos_template_candidate(self, target: Lexeme) -> Optional[Candidate]:
        return self.pos_specific_template_candidate(target, seeded=True, emergency=False)

    def pure_pos_template_candidate(self, target: Lexeme, starter_mode: bool = False) -> Optional[Candidate]:
        return self.pos_specific_template_candidate(target, seeded=False, emergency=False)

    def emergency_pos_template_candidate(self, target: Lexeme) -> Optional[Candidate]:
        return self.pos_specific_template_candidate(target, seeded=False, emergency=True)

    def pos_specific_template_candidate(self, target: Lexeme, seeded: bool = False, emergency: bool = False) -> Optional[Candidate]:
        profile = get_profile(target.rank)
        allowed = allowed_support_rank(target.rank, profile)
        family = self.normalized_pos_family(target)
        surface = target.lemma
        exclude = {target.lemma, self.canonical_lemma_for(target)}

        if family == "adv":
            if surface == "cómo":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_how_be", ["cómo", "está", "él"], 0),
                        ("adv_how_do", ["cómo", "lo", "haces"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "sólo":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_only_go_home", ["sólo", "voy", "a", "casa"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "tan":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_so_good", ["es", "tan", "bueno"], 1),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "entonces":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_then_go_home", ["entonces", "voy", "a", "casa"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "no":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_negation_know", ["no", "lo", "sé"], 0),
                        ("adv_negation_go", ["no", "voy", "hoy"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "muy":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_very_good", ["está", "muy", "bien"], 1),
                        ("adv_very_clear", ["es", "muy", "claro"], 1),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "menos":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_less_work", ["ahora", "trabajo", "menos"], 2),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "casi":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_almost_ready", ["ya", "casi", "está"], 1),
                        ("adv_almost_home", ["ya", "casi", "llego"], 1),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "además":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_besides_clear", ["además", "es", "claro"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "siempre":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_always_here", ["siempre", "está", "aquí"], 0),
                        ("adv_always_know", ["siempre", "lo", "sé"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "ayer":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_yesterday_left", ["ayer", "me", "fui"], 0),
                        ("adv_yesterday_arrived", ["ayer", "llegó"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "hoy":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_today_go", ["voy", "hoy"], 1),
                        ("adv_today_arrive", ["llega", "hoy"], 1),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "ahora":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_now_home", ["ahora", "me", "voy", "a", "casa"], 0),
                        ("adv_now_talk", ["ahora", "hablo", "con", "ella"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface in ADV_PLACE_LEMMAS:
                return self._build_from_specs(
                    target,
                    [
                        ("adv_place_be", ["ella", "está", surface], 2),
                        ("adv_place_live", ["yo", "vivo", surface], 2),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "así":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_manner_speak", ["ella", "habla", "así"], 2),
                        ("adv_manner_be", ["todo", "es", "así"], 2),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "bien":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_well_be", ["ella", "está", "bien"], 2),
                        ("adv_well_speak", ["ella", "habla", "bien"], 2),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface == "ya":
                return self._build_from_specs(
                    target,
                    [
                        ("adv_already_know", ["ya", "lo", "sé"], 0),
                        ("adv_already_here", ["ya", "está", "aquí"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface in ADV_TIME_LEMMAS:
                specs = {
                    "ayer": [
                        ("adv_time_yesterday_go", ["ayer", "fui", "a", "casa"], 0),
                    ],
                    "siempre": [
                        ("adv_time_always_home", ["siempre", "voy", "a", "casa"], 0),
                    ],
                    "mañana": [
                        ("adv_time_tomorrow_home", ["mañana", "voy", "a", "casa"], 0),
                    ],
                    "hoy": [
                        ("adv_time_today_home", ["hoy", "voy", "a", "casa"], 0),
                    ],
                    "ahora": [
                        ("adv_now_home", ["ahora", "me", "voy", "a", "casa"], 0),
                    ],
                }.get(surface, [
                    ("adv_time_arrive", ["ella", "llega", surface], 2),
                ])
                return self._build_from_specs(
                    target,
                    specs,
                    seeded=seeded,
                    emergency=emergency,
                )
            # Degree adverbs: "Es [muy/bastante/demasiado] bueno."
            degree_adverbs = {
                "muy", "bastante", "demasiado", "casi",
                "realmente", "totalmente", "completamente",
                "simplemente", "probablemente", "exactamente",
            }
            if surface in degree_adverbs:
                return self._build_from_specs(
                    target,
                    [
                        ("adv_degree_good", ["es", surface, "bueno"], 1),
                        ("adv_degree_important", ["es", surface, "importante"], 1),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )

            # Accompaniment adverbs
            accompaniment_adverbs = {"conmigo", "contigo", "además", "encima", "incluso"}
            if surface in accompaniment_adverbs:
                return self._build_from_specs(
                    target,
                    [
                        ("adv_accompaniment_come", ["ella", "viene", surface], 2),
                        ("adv_accompaniment_be", ["ella", "está", surface], 2),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )

            # Quantity adverbs
            quantity_adverbs = {"poco", "mucho", "menos", "más", "tanto", "cuanto"}
            if surface in quantity_adverbs:
                return self._build_from_specs(
                    target,
                    [
                        ("adv_quantity_work", ["ella", "trabaja", surface], 2),
                        ("adv_quantity_eat", ["ella", "come", surface], 2),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )

            return self._build_from_specs(
                target,
                [
                    ("adv_general_speak", ["ella", "habla", surface], 2),
                    ("adv_general_live", ["yo", "vivo", surface], 2),
                ],
                seeded=seeded,
                emergency=emergency,
            )

        if family in {"determiner", "art"}:
            if surface == "todo":
                return self._build_from_specs(
                    target,
                    [
                        ("det_total_pronoun", ["todo", "está", "bien"], 0),
                        ("det_total_work", ["todo", "el", "trabajo", "está", "bien"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface in {"todos", "todas"}:
                noun_surface = "amigos" if surface == "todos" else "casas"
                return self._build_from_specs(
                    target,
                    [
                        ("det_total_plural_pron", [surface, "están", "aquí"], 0),
                        ("det_total_plural_nominal", [surface, noun_surface, "están", "aquí"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface in {"mucho", "mucha"}:
                noun_surface = "trabajo" if surface == "mucho" else "comida"
                return self._build_from_specs(
                    target,
                    [
                        ("det_quantity_mass", ["hay", surface, noun_surface], 1),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if surface in {"muchos", "muchas"}:
                noun_surface = "amigos" if surface == "muchos" else "casas"
                return self._build_from_specs(
                    target,
                    [
                        ("det_quantity_plural", [surface, noun_surface, "están", "aquí"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            support = self._safe_nominal_support(surface, allowed, exclude=exclude)
            if not support:
                return None
            noun_surface, verb = support
            return self._build_from_specs(
                target,
                [
                    ("det_anchor_here", [surface, noun_surface, verb, "aquí"], 0),
                    ("det_anchor_clear", [surface, noun_surface, verb, "bien"], 0),
                ],
                seeded=seeded,
                emergency=emergency,
            )

        if family == "pron":
            pron = surface
            if pron in SUBJECT_FEATURES:
                return self._build_from_specs(
                    target,
                    [
                        ("pron_subject_here", [pron, self.conjugate_present("estar", SUBJECT_FEATURES[pron]["person_code"]), "aquí"], 0),
                        ("pron_subject_home", [pron, self.conjugate_present("ir", SUBJECT_FEATURES[pron]["person_code"]), "a", "casa"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if pron in {"lo", "la", "los", "las"}:
                subject = "yo" if pron in {"lo", "la"} else "nosotros"
                person = SUBJECT_FEATURES[subject]["person_code"]
                return self._build_from_specs(
                    target,
                    [
                        ("pron_object_see", [subject, pron, self.conjugate_present("ver", person)], 1),
                        ("pron_object_know", [subject, pron, self.conjugate_present("conocer", person)], 1),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if pron in {"le", "les"}:
                return self._build_from_specs(
                    target,
                    [
                        ("pron_io_tell", [pron, "digo", "la", "verdad"], 0),
                        ("pron_io_write", [pron, "escribo", "ahora"], 0),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if pron in {"me", "te", "se", "nos", "os"}:
                subject = "ella" if pron not in {"nos", "os"} else "ellos"
                person = SUBJECT_FEATURES[subject]["person_code"]
                specs = [
                    ("pron_clitic_call", [subject, pron, self.conjugate_present("llamar", person)], 1),
                ]
                if pron == "se":
                    specs.insert(0, ("pron_clitic_leave", ["ella", "se", "va"], 1))
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            if pron in {"esto", "eso", "aquello"}:
                return self._build_from_specs(
                    target,
                    [
                        ("pron_neuter_for_you", [pron, "es", "para", "ti"], 0),
                        ("pron_neuter_like", ["me", "gusta", pron], 2),
                    ],
                    seeded=seeded,
                    emergency=emergency,
                )
            if pron in {"algo", "nada", "nadie", "alguien"}:
                specs = [("pron_existential", ["hay", pron], 1)]
                if pron in {"nada", "nadie"}:
                    specs = [("pron_negated_existential", ["no", "hay", pron], 2)]
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            if pron in {"qué", "quién"}:
                verb = "es" if pron != "quién" else "viene"
                specs = [("pron_question", [pron, verb], 0)]
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            if pron == "que":
                specs = [("pron_relative", ["la", "casa", "que", "veo"], 2)]
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            if pron == "cuál":
                specs = [("pron_which_name", ["cuál", "es", "tu", "nombre"], 0)]
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            if pron == "ello":
                specs = [("pron_ello_object", ["no", "hablo", "de", "ello"], 3)]
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            if pron in {"mí", "ti"}:
                specs = [("pron_prepositional_for", ["es", "para", pron], 2)]
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            if pron == "dónde":
                specs = [("pron_where_be", ["dónde", "está"], 0)]
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            if pron == "sí":
                specs = [("pron_reflexive_for_self", ["lo", "hace", "por", "sí", "mismo"], 3)]
                return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)
            return None


        if family in {"prep", "contraction"}:
            if surface == "con":
                specs = [
                    ("prep_with_her", ["hablo", "con", "ella"], 1),
                    ("prep_with_him", ["voy", "con", "él"], 1),
                ]
            elif surface == "para":
                specs = [
                    ("prep_for_you", ["es", "para", "ti"], 1),
                    ("prep_work_for_him", ["trabajo", "para", "él"], 1),
                ]
            elif surface == "del":
                specs = [
                    ("prep_del_book", ["hablo", "del", "libro"], 1),
                    ("prep_del_work", ["salgo", "del", "trabajo"], 1),
                ]
            elif surface == "al":
                specs = [
                    ("prep_al_work", ["voy", "al", "trabajo"], 1),
                    ("prep_al_hotel", ["llego", "al", "hotel"], 1),
                ]
            elif surface == "a":
                specs = [
                    ("prep_to_home", ["voy", "a", "casa"], 1),
                    ("prep_call_friend", ["llamo", "a", "mi", "amigo"], 1),
                ]
            elif surface == "de":
                specs = [
                    ("prep_of_work", ["hablo", "de", "mi", "trabajo"], 1),
                    ("prep_of_her", ["el", "libro", "es", "de", "ella"], 3),
                    ("prep_part_of_plan", ["es", "parte", "de", "mi", "plan"], 2),
                ]
            elif surface == "en":
                specs = [
                    ("prep_in_home", ["ella", "está", "en", "casa"], 2),
                    ("prep_in_house", ["vivo", "en", "esta", "casa"], 1),
                ]
            elif surface == "por":
                specs = [
                    ("prep_because_of_that", ["es", "por", "eso"], 1),
                    ("prep_for_you_reason", ["lo", "hago", "por", "ti"], 2),
                ]
            elif surface == "sin":
                specs = [
                    ("prep_without_money", ["estoy", "sin", "dinero"], 1),
                    ("prep_without_him", ["voy", "sin", "él"], 1),
                ]
            elif surface == "sobre":
                specs = [
                    ("prep_on_table", ["el", "libro", "está", "sobre", "la", "mesa"], 3),
                ]
            elif surface == "hasta":
                specs = [
                    ("prep_until_today", ["trabajo", "hasta", "hoy"], 1),
                    ("prep_until_tomorrow", ["espero", "hasta", "mañana"], 1),
                ]
            elif surface == "entre":
                specs = [
                    ("prep_between", ["está", "entre", "la", "casa", "y", "la", "escuela"], 1),
                ]
            elif surface == "durante":
                specs = [
                    ("prep_during", ["trabajo", "durante", "el", "día"], 1),
                ]
            elif surface == "contra":
                specs = [
                    ("prep_against", ["no", "tengo", "nada", "contra", "él"], 3),
                ]
            elif surface == "bajo":
                specs = [
                    ("prep_under", ["el", "libro", "está", "bajo", "la", "mesa"], 3),
                ]
            elif surface == "según":
                specs = [
                    ("prep_according", ["según", "él", "es", "verdad"], 0),
                ]
            elif surface == "ante":
                specs = [
                    ("prep_before", ["está", "ante", "la", "puerta"], 1),
                ]
            elif surface == "tras":
                specs = [
                    ("prep_after", ["viene", "tras", "la", "comida"], 1),
                ]
            elif surface == "hacia":
                specs = [
                    ("prep_toward", ["voy", "hacia", "la", "casa"], 1),
                ]
            else:
                specs = [
                    ("prep_generic_with_target", ["hablo", surface, "ella"], 1),
                ]
            return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)

        if family == "conj":
            if surface == "y":
                specs = [
                    ("conj_and_clauses", ["ella", "va", "y", "él", "va"], 2),
                    ("conj_and_return", ["voy", "y", "vuelvo"], 1),
                ]
            elif surface == "o":
                specs = [
                    ("conj_or_time", ["voy", "hoy", "o", "mañana"], 2),
                ]
            elif surface == "pero":
                specs = [
                    ("conj_but_cant", ["quiero", "ir", "pero", "no", "puedo"], 2),
                    ("conj_but_stay", ["ella", "viene", "pero", "yo", "me", "quedo"], 2),
                ]
            elif surface == "porque":
                specs = [
                    ("conj_because_tired", ["no", "voy", "porque", "estoy", "cansado"], 2),
                    ("conj_because_home", ["me", "voy", "porque", "es", "tarde"], 2),
                ]
            elif surface == "si":
                specs = [
                    ("conj_if_know", ["no", "sé", "si", "viene"], 2),
                ]
            elif surface == "cuando":
                specs = [
                    ("conj_when_call", ["te", "llamo", "cuando", "llego"], 2),
                    ("conj_when_arrive", ["cuando", "llega", "ella"], 0),
                ]
            elif surface == "como":
                specs = [
                    ("conj_as_you", ["lo", "hago", "como", "tú"], 2),
                    ("conj_as_before", ["todo", "sale", "como", "antes"], 2),
                ]
            elif surface == "sino":
                specs = [
                    ("conj_sino_but", ["no", "es", "malo", "sino", "bueno"], 3),
                ]
            elif surface == "mientras":
                specs = [
                    ("conj_mientras_while", ["ella", "lee", "mientras", "yo", "cocino"], 2),
                ]
            elif surface == "pues":
                specs = [
                    ("conj_pues_then", ["no", "quiero", "ir", "pues", "me", "quedo"], 3),
                ]
            elif surface == "ni":
                specs = [
                    ("conj_ni_neither", ["no", "tengo", "ni", "idea"], 2),
                ]
            elif surface in {"aunque", "como"}:
                specs = [
                    ("conj_subord_clause", ["voy", surface, "no", "quiero"], 1),
                ]
            else:
                specs = [
                    ("conj_generic_link", ["ella", "viene", surface, "yo", "salgo"], 2),
                ]
            return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)

        if family == "interj":
            punct = INTERJECTION_PUNCT.get(surface, "!")
            candidate = self._build_from_specs(
                target,
                [
                    ("interj_here", [surface, ",", "ella", "está", "aquí"], 0),
                    ("interj_arrive", [surface, ",", "ella", "llega"], 0),
                ],
                seeded=seeded,
                emergency=emergency,
            )
            if candidate and punct and candidate.sentence and candidate.sentence[-1] == ".":
                candidate.sentence = candidate.sentence[:-1] + punct
            return candidate

        if family == "num":
            if surface in {"un", "una"}:
                noun = "casa" if surface == "una" else "libro"
                specs = [("num_indefinite_object", ["tengo", surface, noun], 1)]
            elif surface == "uno":
                specs = [("num_one_subject", ["uno", "viene", "hoy"], 0)]
            else:
                noun = "libros"
                specs = [
                    ("num_count_object", ["tengo", surface, noun], 1),
                    ("num_count_subject", [surface, noun, "llegan"], 0),
                ]
            return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)

        if family == "prop":
            if profile.band in {"A1", "A2"}:
                specs = [("prop_here", [surface, "está", "aquí"], 0)]
            elif profile.band in {"B1", "B2"}:
                specs = [("prop_visit", [surface, "visita", "la", "ciudad"], 0)]
            else:
                specs = [("prop_say", [surface, "dice", "que", "todo", "cambia"], 0)]
            return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)

        if family == "letter":
            return self._build_from_specs(target, [("letter_metalinguistic_clause", ["la", "letra", surface, "es", "común"], 2)], seeded=seeded, emergency=emergency)

        if family == "prefix":
            return self._build_from_specs(target, [("prefix_metalinguistic_clause", ["el", "prefijo", surface, "es", "útil"], 2)], seeded=seeded, emergency=emergency)

        if family == "phrase":
            return self._build_from_specs(target, [("phrase_metalinguistic_clause", ["la", "expresión", surface, "es", "clara"], 2)], seeded=seeded, emergency=emergency)

        if family == "particle":
            return self._build_from_specs(target, [("particle_metalinguistic_clause", ["la", "partícula", surface, "aparece", "aquí"], 2)], seeded=seeded, emergency=emergency)

        if family == "residual":
            if surface == "no":
                specs = [
                    ("residual_negation_know", ["no", "lo", "sé"], 0),
                    ("residual_negation_go", ["no", "voy", "hoy"], 0),
                ]
            elif surface == "sí":
                specs = [("residual_yes_want", ["sí", "lo", "quiero"], 0)]
            else:
                specs = [("residual_metalinguistic_clause", ["la", "forma", surface, "es", "clara"], 2)]
            return self._build_from_specs(target, specs, seeded=seeded, emergency=emergency)

        return None

    def generate_strategy_candidates(self, target: Lexeme) -> List[Candidate]:
        profile = get_profile(target.rank)
        strategy = self.strategy_for_target(target)
        candidates: List[Candidate] = []
        candidates.extend(self.retrieve_candidates(target))

        seeded_attempts, pure_attempts = self.band_template_attempts(profile)
        seeded_builder = strategy.get("seeded")
        pure_builder = strategy.get("pure")

        if self.normalized_pos_family(target) in {"n", "v", "adj"}:
            if not self.can_template_target(target):
                seeded_attempts = 0
                pure_attempts = 0

        if seeded_builder:
            for _ in range(seeded_attempts):
                cand = seeded_builder(target)
                cand = self.try_candidate(cand)
                if cand:
                    candidates.append(cand)

        if pure_builder:
            for _ in range(pure_attempts):
                cand = pure_builder(target)
                cand = self.try_candidate(cand)
                if cand:
                    candidates.append(cand)

        emergency = self.try_candidate(self.emergency_pos_template_candidate(target))
        if emergency:
            emergency.source_method = "emergency_template"
            candidates.append(emergency)
        return self.dedupe_candidates(candidates)

    def generate_for_lemma(self, lemma: str) -> Candidate:
        lemma = lemma.strip().lower()
        if lemma not in self.lexicon:
            raise KeyError(f"Lemma not in lexicon: {lemma}")
        target = self.lexicon[lemma]
        candidates = self.collect_candidates_for_lemma(lemma)
        if candidates:
            chosen = self.select_best_candidate(candidates)
            if self.candidate_is_general_publishable(chosen):
                return chosen
        return self.manual_review_candidate(target)

    def generate_sentence_for_target(self, target_lemma: str, target_rank: int) -> Dict[str, Any]:
        """Generate one structured sentence result for a target lemma and rank.

        The returned band is always derived from the existing BANDS thresholds via
        get_profile(target_rank), so callers can treat this as the canonical
        sentence-generation contract for app or CSV use.
        """
        lemma = (target_lemma or "").strip().lower()
        band = get_profile(target_rank).band
        if not lemma:
            return {
                "lemma": lemma,
                "rank": target_rank,
                "band": band,
                "pos": "",
                "sentence": "",
                "target_form": "",
                "canonical_lemma": "",
                "source_method": "invalid_request",
                "template_id": "",
                "score": 0.0,
                "publishable": False,
                "failure_reason": "missing_target_lemma",
            }
        if lemma not in self.lexicon:
            return {
                "lemma": lemma,
                "rank": target_rank,
                "band": band,
                "pos": "",
                "sentence": "",
                "target_form": "",
                "canonical_lemma": "",
                "source_method": "missing_lemma",
                "template_id": "",
                "score": 0.0,
                "publishable": False,
                "failure_reason": "lemma_not_in_lexicon",
            }

        candidate = self.generate_for_lemma(lemma)
        target = self.lexicon[lemma]
        publishable = self.candidate_is_publishable(candidate)
        failure_reason = ""
        if candidate.source_method == "manual_review_needed":
            failure_reason = "manual_review_needed"
        elif not publishable:
            grammatical_ok, natural_ok, learner_clear_ok, notes = self.review_flags(candidate)
            reasons = []
            if grammatical_ok != "1":
                reasons.append("not_grammatical")
            if natural_ok != "1":
                reasons.append("not_natural")
            if learner_clear_ok != "1":
                reasons.append("not_learner_clear")
            if notes:
                reasons.append(notes)
            failure_reason = "; ".join(reasons) if reasons else "not_publishable"

        return {
            "lemma": lemma,
            "rank": target_rank,
            "band": band,
            "pos": target.pos,
            "sentence": candidate.sentence,
            "target_form": candidate.target_form,
            "canonical_lemma": candidate.canonical_lemma or self.canonical_lemma_for(target),
            "source_method": candidate.source_method,
            "template_id": candidate.template_id,
            "score": candidate.score,
            "publishable": publishable,
            "failure_reason": failure_reason,
        }

    def search_starter_candidates_for_lemma(
        self,
        lemma: str,
        max_attempts: int = 300,
        max_candidates_per_lemma: Optional[int] = None,
        stop_on_first_publishable: bool = False,
    ) -> List[Candidate]:
        lemma = lemma.strip().lower()
        if lemma not in self.lexicon:
            raise KeyError(f"Lemma not in lexicon: {lemma}")
        max_attempts = max(0, max_attempts)
        cache_key = (lemma, max_attempts)
        if not stop_on_first_publishable and cache_key in self.starter_candidate_pool_cache:
            pool = list(self.starter_candidate_pool_cache[cache_key])
            if max_candidates_per_lemma is not None:
                return pool[:max(0, max_candidates_per_lemma)]
            return pool

        target = self.lexicon[lemma]
        candidates: List[Candidate] = []
        candidates.extend(self.retrieve_candidates(target))

        if stop_on_first_publishable:
            publishable_retrieved = [cand for cand in candidates if self.candidate_is_starter_publishable(cand)]
            if publishable_retrieved:
                return [max(publishable_retrieved, key=lambda cand: self.starter_total_score(cand))]

        if target.pos in {"n", "v", "adj"} and self.can_template_target(target):
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                if self.random.random() < 0.4:
                    cand = self.seeded_template_candidate(target)
                else:
                    cand = self.pure_template_candidate(target, starter_mode=True)
                if cand:
                    ok, penalties = self.starter_validate(cand)
                    if ok:
                        self.score(cand, penalties)
                        candidates.append(cand)
                        if stop_on_first_publishable and self.candidate_is_starter_publishable(cand):
                            return [cand]
                if attempts % 25 == 0:
                    candidates = self.dedupe_candidates(candidates)

        pool = self.dedupe_candidates(candidates)
        if not stop_on_first_publishable:
            self.starter_candidate_pool_cache[cache_key] = list(pool)
        if max_candidates_per_lemma is not None:
            return pool[:max(0, max_candidates_per_lemma)]
        return pool

    def collect_starter_candidates_for_lemma(self, lemma: str, max_candidates_per_lemma: Optional[int] = None) -> List[Candidate]:
        return self.search_starter_candidates_for_lemma(
            lemma,
            max_attempts=120,
            max_candidates_per_lemma=max_candidates_per_lemma,
            stop_on_first_publishable=False,
        )

    def starter_total_score(self, candidate: Candidate) -> float:
        return candidate.score + self.starter_bonus(candidate)

    def score_candidate_for_output(self, candidate: Candidate) -> Candidate:
        ok, penalties = self.starter_validate(candidate)
        if ok:
            self.score(candidate, penalties)
            return candidate
        ok, penalties = self.validate(candidate)
        if ok:
            self.score(candidate, penalties)
        return candidate

    def emergency_starter_candidate(self, target: Lexeme, max_attempts: int = 40) -> Optional[Candidate]:
        fallback: Optional[Candidate] = None
        if target.pos in {"n", "v", "adj"} and self.can_template_target(target):
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                if self.random.random() < 0.4:
                    cand = self.seeded_template_candidate(target)
                else:
                    cand = self.pure_template_candidate(target, starter_mode=True)
                if not cand:
                    continue
                cand.source_method = "emergency_template"
                cand.template_id = cand.template_id or "starter_emergency"
                setattr(cand, "_starter_emergency", True)
                self.score_candidate_for_output(cand)
                if self.validate(cand)[0]:
                    return cand
                if fallback is None:
                    fallback = cand

        if target.pos == "n":
            article = self.choose_article(self.safe_noun_gender(target.lemma, target.gender), definite=True)
            cand = self.build_candidate(target, [article, target.lemma, "está", "aquí"], "starter_emergency_noun", "emergency_template", 1)
            if cand:
                setattr(cand, "_starter_emergency", True)
                return self.score_candidate_for_output(cand)

        if target.pos == "adj":
            cand = self.build_candidate(target, ["él", "es", target.lemma], "starter_emergency_adj", "emergency_template", 2)
            if cand:
                setattr(cand, "_starter_emergency", True)
                return self.score_candidate_for_output(cand)
            cand = self.build_candidate(target, ["ellos", "son", target.lemma], "starter_emergency_adj_plural", "emergency_template", 2)
            if cand:
                setattr(cand, "_starter_emergency", True)
                return self.score_candidate_for_output(cand)

        if target.pos == "v":
            canonical = self.canonical_lemma_for(target)
            morph = self.target_form_metadata(target)
            if morph.get("VerbForm") == "Inf" or normalize_token(target.lemma) == normalize_token(canonical):
                carrier = self.random.choice(STARTER_INFINITIVE_CARRIERS)
                carrier_form = self.conjugate_present(carrier, "3sg")
                cand = self.build_candidate(target, ["ella", carrier_form, target.lemma], "starter_emergency_inf", "emergency_template", 2)
                if cand:
                    setattr(cand, "_starter_emergency", True)
                    return self.score_candidate_for_output(cand)
            subject, person = self.subject_for_target(target)
            verb = self.target_verb_form(target, person)
            cand = self.build_candidate(target, [subject, verb], "starter_emergency_verb", "emergency_template", 1)
            if cand:
                setattr(cand, "_starter_emergency", True)
                return self.score_candidate_for_output(cand)

        return fallback

    def starter_output_metadata(self, candidate: Candidate) -> Dict[str, Any]:
        strict_starter_ok = self.candidate_is_starter_publishable(candidate)
        publishable = strict_starter_ok
        if candidate.source_method == "manual_review_needed":
            return {
                "publishable": False,
                "strict_starter_ok": False,
                "quality_tier": "manual_review_needed",
                "failure_reason": "manual_review_needed",
            }
        if strict_starter_ok:
            return {
                "publishable": True,
                "strict_starter_ok": True,
                "quality_tier": "strict_publishable",
                "failure_reason": "",
            }
        reasons = self.starter_rejection_reasons(candidate)
        failure_reason = "; ".join(reasons) if reasons else "not_strict_publishable"
        quality_tier = "emergency_template" if getattr(candidate, "_starter_emergency", False) or candidate.source_method == "emergency_template" else "fallback_candidate"
        return {
            "publishable": publishable,
            "strict_starter_ok": strict_starter_ok,
            "quality_tier": quality_tier,
            "failure_reason": failure_reason,
        }

    def generate_starter_for_lemma(self, lemma: str, starter_max_attempts: int = 300) -> Candidate:
        lemma = lemma.strip().lower()
        if lemma not in self.lexicon:
            raise KeyError(f"Lemma not in lexicon: {lemma}")
        target = self.lexicon[lemma]
        pool = self.search_starter_candidates_for_lemma(
            lemma,
            max_attempts=starter_max_attempts,
            stop_on_first_publishable=False,
        )
        publishable = [cand for cand in pool if self.candidate_is_starter_publishable(cand)]
        if publishable:
            return max(publishable, key=lambda cand: self.starter_total_score(cand))
        if pool:
            return max(pool, key=lambda cand: self.starter_total_score(cand))
        emergency = self.emergency_starter_candidate(target)
        if emergency:
            return emergency
        return self.manual_review_candidate(target)

    def select_best_candidate(self, candidates: List[Candidate]) -> Candidate:
        if self.reranker:
            try:
                predictions = predict_candidate_scores(self.reranker, candidates)
            except Exception:
                predictions = []
            if len(predictions) == len(candidates):
                ranked = zip(candidates, predictions)
                return max(ranked, key=lambda item: (item[1], item[0].score))[0]
        preferred_retrieved = [c for c in candidates if self.retrieved_is_preferred(c)]
        if preferred_retrieved:
            return max(preferred_retrieved, key=lambda c: c.score)
        return max(candidates, key=lambda c: c.score)

    def candidate_is_general_publishable(self, candidate: Candidate) -> bool:
        if not candidate.sentence:
            return False
        grammatical_ok, natural_ok, learner_clear_ok, _ = self.review_flags(candidate)
        return grammatical_ok == "1" and natural_ok == "1" and learner_clear_ok == "1"

    def candidate_is_publishable(self, candidate: Candidate) -> bool:
        return self.candidate_is_general_publishable(candidate)

    def is_clean_starter_target(self, lex: Lexeme) -> bool:
        if not self.is_starter_target_eligible(lex):
            return False
        canonical = normalize_token(self.canonical_lemma_for(lex))
        surface = normalize_token(lex.lemma)
        if surface in STARTER_EXCLUDED_TARGET_LEMMAS:
            return False
        if surface in STARTER_LOW_VALUE_TARGET_LEMMAS:
            return False
        morph = self.target_form_metadata(lex)

        if lex.pos == "n":
            if surface != canonical:
                return False
            if morph.get("Number") == "Plur":
                return False
            if lex.semantic_class not in STARTER_SAFE_NOUN_CLASSES:
                return False
            return True

        if lex.pos == "adj":
            if surface != canonical:
                return False
            if morph.get("Number") == "Plur":
                return False
            if surface in APOCOPATED_ADJECTIVE_FEATURES:
                return False
            return True

        if lex.pos == "v":
            return surface == canonical

        return False

    def starter_morph_ok(self, candidate: Candidate) -> bool:
        morph = self._parse_morph_str(candidate.target_morph)
        if candidate.pos != "v":
            return True
        if not morph:
            return False

        verb_form = morph.get("VerbForm")
        mood = morph.get("Mood")
        tense = morph.get("Tense")

        if verb_form in {"Part", "Ger"}:
            return False
        if mood == "Sub":
            return False
        if tense in {"Past", "Imp", "Fut"}:
            return False
        if verb_form == "Inf":
            return True
        return mood == "Ind" and tense == "Pres"

    def starter_target_surface_ok(self, candidate: Candidate) -> bool:
        target = self.lexicon.get(candidate.lemma)
        if not target:
            return False
        target_form = normalize_token(candidate.target_form)
        lemma = normalize_token(candidate.lemma)
        morph = self._parse_morph_str(candidate.target_morph)

        if candidate.pos == "n":
            if target_form != lemma:
                return False
            return morph.get("Number") != "Plur"

        if candidate.pos == "adj":
            if target_form != lemma:
                return False
            if target_form in APOCOPATED_ADJECTIVE_FEATURES:
                return False
            if morph.get("Number") == "Plur":
                return False
            if morph.get("Gender") == "Fem":
                return False
            return True

        if candidate.pos == "v":
            return True

        return False

    def starter_structure_reasons(self, candidate: Candidate) -> List[str]:
        reasons: List[str] = []
        sentence = candidate.sentence
        words = self.word_tokens(sentence)
        lowered = " ".join(words)

        if "?" in sentence or "¿" in sentence:
            reasons.append("question_form")
        if "!" in sentence or "¡" in sentence:
            reasons.append("exclamation_form")
        if any(pattern in lowered for pattern in STARTER_BANNED_PATTERNS):
            reasons.append("banned_discourse_pattern")
        padded = f" {lowered} "
        if any(f" {pattern} " in padded for pattern in STARTER_BANNED_WORD_PATTERNS):
            reasons.append("banned_discourse_pattern")
        if "," in sentence or ";" in sentence or ":" in sentence:
            reasons.append("punctuation_complexity")
        if words and words[0] in CONTEXT_DEPENDENT_OPENERS:
            reasons.append("discourse_opener")

        finite_verbs = 0
        for token in words:
            pos = self.lookup_pos(token)
            if pos != "v":
                continue
            morph = self.surface_morph(token)
            if not morph:
                continue
            verb_form = morph.get("VerbForm")
            mood = morph.get("Mood")
            tense = morph.get("Tense")
            if not verb_form and not mood and not tense:
                continue
            if verb_form in {"Part", "Ger"}:
                reasons.append("participle_or_gerund")
            if mood == "Sub":
                reasons.append("subjunctive")
            if mood == "Imp":
                reasons.append("imperative_form")
            if tense in {"Past", "Imp", "Fut"}:
                reasons.append("non_present_tense")
            if mood == "Cnd":
                reasons.append("non_present_tense")
            if verb_form == "Fin":
                finite_verbs += 1
                if not (mood == "Ind" and tense == "Pres"):
                    if "non_present_tense" not in reasons and "subjunctive" not in reasons and "imperative_form" not in reasons:
                        reasons.append("non_present_tense")
        if finite_verbs > 1:
            reasons.append("multiple_finite_verbs")

        return list(dict.fromkeys(reasons))

    def starter_structure_ok(self, candidate: Candidate) -> bool:
        return not self.starter_structure_reasons(candidate)

    def starter_translation_ok(self, candidate: Candidate) -> bool:
        t = (candidate.translation or "").strip().lower()
        if not t:
            return False
        if ";" in t:
            return False
        if len(t) > 40:
            return False
        return not any(fragment in t for fragment in STARTER_BAD_TRANSLATION_FRAGMENTS)

    def starter_retrieved_ok(self, candidate: Candidate) -> bool:
        if candidate.source_method != "retrieved_corpus":
            return True
        sentence = candidate.sentence
        words = self.word_tokens(sentence)
        lowered = " ".join(words)

        padded = f" {lowered} "
        if any(f" {p} " in padded for p in STARTER_BANNED_WORD_PATTERNS):
            return False
        if any(ch in sentence for ch in "?¿!¡;:,"):
            return False
        if any(ch in sentence for ch in {'"', '\u201c', '\u201d', "'"}):
            return False
        if words and words[0] in CONTEXT_DEPENDENT_OPENERS:
            return False
        if len(words) > 6:
            return False
        if "no" in words:
            return False
        if "bien" in words:
            return False
        if any(word in STARTER_RETRIEVED_CONTEXT_WORDS for word in words):
            return False
        if any(word in STARTER_RETRIEVED_POSSESSIVES for word in words):
            return False
        if any(word in STARTER_RETRIEVED_PERSONAL_TOKENS for word in words):
            return False
        if candidate.pos == "n" and words and words[0] in COPULA_FORMS:
            return False
        if candidate.pos == "adj" and words and words[0] in COPULA_FORMS:
            return False
        if self.starter_structure_reasons(candidate):
            return False
        verb_index = self.first_finite_verb_index(words)
        if verb_index >= 0:
            verb_morph = self.surface_morph(words[verb_index])
            if verb_morph.get("Person") == "1":
                return False
        rejected, penalties, _ = self.retrieved_quality(candidate, words, verb_index)
        if rejected or penalties:
            return False
        level_ok, _ = self.retrieved_is_level_appropriate(candidate, words, verb_index)
        if not level_ok:
            return False
        skeptical_count = sum(1 for w in words if w in STARTER_RETRIEVED_SKEPTICAL_WORDS)
        if skeptical_count >= 2:
            return False
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        if any(bg in STARTER_RETRIEVED_BANNED_BIGRAMS for bg in bigrams):
            return False
        non_target_words = [w for w in words if normalize_token(w) != normalize_token(candidate.lemma)]
        if non_target_words:
            unknown_count = sum(1 for w in non_target_words if self.lookup_rank(w) > 500)
            if unknown_count > 0:
                return False
        for w in words:
            info = self.surface_analysis(w)
            if info["pos"] == "adj" and info["lemma"]:
                adj_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(normalize_token(info["lemma"]))
                if adj_classes is not None:
                    subj_class = self.subject_semantic_class(words, verb_index) if verb_index >= 0 else None
                    if subj_class and subj_class not in adj_classes:
                        return False
        if words and words[-1] in STARTER_RETRIEVED_SKEPTICAL_WORDS:
            return False
        return True

    def starter_semantic_quality_reasons(self, candidate: Candidate) -> List[str]:
        reasons: List[str] = []
        words = self.word_tokens(candidate.sentence)
        target = self.lexicon.get(candidate.lemma)
        canonical = self.canonical_lemma_for(target) if target else normalize_token(candidate.lemma)
        verb_index = self.first_finite_verb_index(words)
        lowered = " ".join(words)

        if candidate.pos == "v" and candidate.source_method != "retrieved_corpus":
            morph = self._parse_morph_str(candidate.target_morph)
            if morph.get("VerbForm") == "Inf":
                has_complement = False
                ti = candidate.target_index
                stokens = self.sentence_tokens(candidate.sentence)
                if ti >= 0 and ti < len(stokens):
                    after_target = [t for t in stokens[ti + 1:] if is_word_token(t)]
                    has_complement = len(after_target) > 0
                if not has_complement and canonical not in BARE_INFINITIVE_OK:
                    reasons.append("weak_bare_infinitive")
                if canonical in STARTER_INFINITIVE_REJECT:
                    reasons.append("weak_bare_infinitive")
                if canonical == "tener":
                    reasons.append("weak_template_filler")

        if candidate.pos == "adj" or (candidate.source_method != "retrieved_corpus" and any(
            normalize_token(w) in COPULA_FORMS for w in words
        )):
            copula_idx = next((i for i, w in enumerate(words) if w in COPULA_FORMS), -1)
            if copula_idx >= 0:
                subj_class = self.subject_semantic_class(words, copula_idx)
                for token in words[copula_idx + 1:]:
                    info = self.surface_analysis(token)
                    if info["pos"] != "adj":
                        continue
                    adj_lemma = normalize_token(info["lemma"] or token)
                    allowed_classes = STARTER_ADJ_ALLOWED_NOUN_CLASSES.get(adj_lemma)
                    if allowed_classes is not None:
                        if subj_class and subj_class not in allowed_classes:
                            reasons.append("bad_adj_subject_semantics")
                    else:
                        if subj_class:
                            reasons.append("bad_adj_subject_semantics")
                    if subj_class and adj_lemma in STARTER_WEAK_COPULA_PREDICATES_BY_CLASS.get(subj_class, set()):
                        reasons.append("weak_copula_predicate")
                    if subj_class and adj_lemma in STARTER_UNNATURAL_COPULA_PREDICATES_BY_CLASS.get(subj_class, set()):
                        reasons.append("unnatural_starter_claim")
                    break

        if candidate.source_method != "retrieved_corpus":
            for i, word in enumerate(words[:-1]):
                if word in ARTICLE_FEATURES and self.lookup_pos(words[i + 1]) == "n":
                    if not self.article_matches_noun(word, words[i + 1]):
                        reasons.append("article_noun_gender_mismatch")
                        break

        if candidate.source_method == "retrieved_corpus":
            skeptical_count = sum(1 for w in words if w in STARTER_RETRIEVED_SKEPTICAL_WORDS)
            if skeptical_count >= 2:
                reasons.append("context_shaped_retrieval")
            if words and words[-1] in STARTER_RETRIEVED_SKEPTICAL_WORDS:
                reasons.append("context_shaped_retrieval")
            if "no" in words or "bien" in words or any(word in STARTER_RETRIEVED_CONTEXT_WORDS for word in words):
                reasons.append("context_shaped_retrieval")
            if any(word in STARTER_RETRIEVED_POSSESSIVES for word in words):
                reasons.append("context_shaped_retrieval")
            if any(word in STARTER_RETRIEVED_PERSONAL_TOKENS for word in words):
                reasons.append("context_shaped_retrieval")
            if verb_index >= 0:
                verb_morph = self.surface_morph(words[verb_index])
                if verb_morph.get("Person") == "1":
                    reasons.append("context_shaped_retrieval")

        if lowered in {"él puede cambiar", "ella puede cambiar", "él puede leer", "ella puede leer"}:
            reasons.append("weak_template_filler")

        return list(dict.fromkeys(reasons))

    def starter_bonus(self, candidate: Candidate) -> float:
        bonus = 0.0
        words = self.word_tokens(candidate.sentence)

        if len(words) <= 5:
            bonus += 1.0

        if candidate.source_method == "seeded_template":
            bonus += 0.4

        target = self.lexicon.get(candidate.lemma)
        if target and target.pos == "n" and target.semantic_class in {"object", "place", "person", "animal"}:
            bonus += 0.8

        lowered = " ".join(words)
        padded = f" {lowered} "
        if any(pattern in lowered for pattern in STARTER_BANNED_PATTERNS):
            bonus -= 3.0
        if any(f" {p} " in padded for p in STARTER_BANNED_WORD_PATTERNS):
            bonus -= 3.0

        return bonus

    def starter_rejection_reasons(self, candidate: Candidate) -> List[str]:
        reasons: List[str] = []
        target = self.lexicon.get(candidate.lemma)
        valid, _ = self.starter_validate(candidate)

        if not valid:
            reasons.append("not_general_publishable")
        if not target or not self.is_clean_starter_target(target):
            reasons.append("unclean_target")
        if target and not self.exact_surface_present(candidate, target):
            reasons.append("exact_surface_not_matched")
        if get_profile(candidate.rank).band not in STARTER_MAX_BAND:
            reasons.append("band_too_high")
        if target and target.pos == "n" and target.semantic_class not in STARTER_SAFE_NOUN_CLASSES:
            reasons.append("abstract_target")
        if not self.starter_target_surface_ok(candidate):
            reasons.append("target_surface_not_base")
        if not self.starter_morph_ok(candidate):
            reasons.append("advanced_morphology")
        structure_reasons = self.starter_structure_reasons(candidate)
        if structure_reasons:
            reasons.extend(structure_reasons)
        if not self.starter_translation_ok(candidate):
            reasons.append("bad_translation")
        if not self.starter_retrieved_ok(candidate):
            reasons.append("retrieved_sentence_too_marked")
        if candidate.template_id in STARTER_BANNED_TEMPLATE_IDS:
            reasons.append("banned_template")
        words = self.word_tokens(candidate.sentence)
        if len(words) > 6:
            reasons.append("sentence_too_long")
        band = get_profile(candidate.rank).band
        if band == "A1" and candidate.max_support_rank > 350:
            reasons.append("support_vocab_too_hard")
        if band == "A2" and candidate.max_support_rank > 600:
            reasons.append("support_vocab_too_hard")
        sentence_lemmas = {
            normalize_token(self.lookup_lemma(word) or word)
            for word in words
        }
        if sentence_lemmas & STARTER_BANNED_LEMMAS or normalize_token(candidate.lemma) in STARTER_BANNED_LEMMAS:
            reasons.append("banned_content")
        semantic_reasons = self.starter_semantic_quality_reasons(candidate)
        reasons.extend(semantic_reasons)
        return reasons

    def candidate_is_starter_publishable(self, candidate: Candidate) -> bool:
        if not candidate.sentence:
            return False
        if candidate.source_method == "manual_review_needed":
            return False
        valid, _ = self.starter_validate(candidate)
        if not valid:
            return False
        target = self.lexicon.get(candidate.lemma)
        if not target or not self.is_clean_starter_target(target):
            return False
        if not self.exact_surface_present(candidate, target):
            return False
        if get_profile(candidate.rank).band not in STARTER_MAX_BAND:
            return False
        if not self.starter_target_surface_ok(candidate):
            return False
        if not self.starter_morph_ok(candidate):
            return False
        if not self.starter_structure_ok(candidate):
            return False
        if not self.starter_translation_ok(candidate):
            return False
        if not self.starter_retrieved_ok(candidate):
            return False
        if candidate.template_id in STARTER_BANNED_TEMPLATE_IDS:
            return False
        if candidate.score < 6.5:
            return False
        words = self.word_tokens(candidate.sentence)
        if len(words) > 6:
            return False
        band = get_profile(candidate.rank).band
        if band == "A1" and candidate.max_support_rank > 350:
            return False
        if band == "A2" and candidate.max_support_rank > 600:
            return False
        sentence_lemmas = {
            normalize_token(self.lookup_lemma(word) or word)
            for word in words
        }
        if sentence_lemmas & STARTER_BANNED_LEMMAS:
            return False
        if normalize_token(candidate.lemma) in STARTER_BANNED_LEMMAS:
            return False
        if self.starter_semantic_quality_reasons(candidate):
            return False
        return not self.starter_rejection_reasons(candidate)

    def starter_review_notes(self, candidate: Candidate) -> str:
        if not candidate.sentence:
            return "no_sentence"
        notes: List[str] = []
        _, _, _, base_notes = self.review_flags(candidate)
        if base_notes:
            notes.append(base_notes)
        words = self.word_tokens(candidate.sentence)
        verb_index = self.first_finite_verb_index(words)
        _, level_reasons = self.retrieved_is_level_appropriate(candidate, words, verb_index)
        notes.extend(level_reasons)
        ov = self.overrides.get(candidate.lemma, {})
        if _truthy(ov.get("exclude_from_starter_dataset")):
            notes.append("excluded_by_override")
        notes.extend(self.starter_rejection_reasons(candidate))
        return "; ".join(notes) if notes else ""

    def generate_batch(
        self,
        limit: int,
        out_csv: str,
        min_rank: int = 1,
        max_rank: int = 10**9,
        pos_filter: Optional[str] = None,
        lemma_filter: Optional[List[str]] = None,
        mvp_only: bool = False,
        candidates_out: Optional[str] = None,
        max_candidates_per_lemma: int = 10,
    ) -> List[Candidate]:
        self.last_candidate_export_stats = None
        if lemma_filter:
            rows = []
            seen = set()
            for lemma in lemma_filter:
                key = lemma.strip().lower()
                if not key or key in seen or key not in self.lexicon:
                    continue
                seen.add(key)
                rows.append(self.lexicon[key])
        else:
            rows = list(self.lexicon.values())
            rows.sort(key=lambda x: x.rank)
        rows = [x for x in rows if min_rank <= x.rank <= max_rank]
        if mvp_only:
            rows = [x for x in rows if x.pos in {"n", "v", "adj"} and get_profile(x.rank).band in {"A1", "A2", "B1"}]
        if pos_filter:
            rows = [x for x in rows if x.pos == pos_filter]
        rows = rows[:limit]
        generated: List[Candidate] = []
        candidate_rows: List[Candidate] = []
        lemmas_with_candidates = 0
        for lex in rows:
            try:
                generated.append(self.generate_for_lemma(lex.lemma))
                if candidates_out:
                    pool = self.collect_candidates_for_lemma(lex.lemma, max_candidates_per_lemma=max_candidates_per_lemma)
                    if pool:
                        lemmas_with_candidates += 1
                        candidate_rows.extend(pool)
            except Exception as exc:
                print(f"[warn] failed to generate for {lex.lemma}: {exc}", file=sys.stderr)
                generated.append(self.manual_review_candidate(lex))
        self.write_csv(generated, out_csv)
        if candidates_out:
            self.write_candidates_csv(candidate_rows, candidates_out)
            avg_candidates = float(len(candidate_rows)) / lemmas_with_candidates if lemmas_with_candidates else 0.0
            self.last_candidate_export_stats = {
                "lemmas_processed": float(len(rows)),
                "candidate_rows_written": float(len(candidate_rows)),
                "lemmas_with_candidates": float(lemmas_with_candidates),
                "avg_candidates_per_lemma_with_candidates": avg_candidates,
            }
        return generated

    def _rows_for_coverage(
        self,
        limit: int,
        min_rank: int = 1,
        max_rank: int = 10**9,
        pos_filter: Optional[str] = None,
        lemma_filter: Optional[List[str]] = None,
    ) -> List[Lexeme]:
        if lemma_filter:
            rows = [self.lexicon[l.strip().lower()] for l in lemma_filter if l.strip().lower() in self.lexicon]
        else:
            rows = sorted(self.lexicon.values(), key=lambda x: x.rank)
        rows = [x for x in rows if min_rank <= x.rank <= max_rank]
        if pos_filter:
            rows = [x for x in rows if x.pos == pos_filter]
        if limit > 0:
            rows = rows[:limit]
        return rows

    def build_coverage_report(
        self,
        limit: int = 200,
        min_rank: int = 1,
        max_rank: int = 10**9,
        pos_filter: Optional[str] = None,
        lemma_filter: Optional[List[str]] = None,
        samples_per_bucket: int = 2,
    ) -> Dict[str, Any]:
        rows = self._rows_for_coverage(limit, min_rank=min_rank, max_rank=max_rank, pos_filter=pos_filter, lemma_filter=lemma_filter)
        raw_pos_counts: Dict[str, int] = {}
        family_counts: Dict[str, int] = {}
        for lex in self.lexicon.values():
            pos = lex.pos or "<empty>"
            raw_pos_counts[pos] = raw_pos_counts.get(pos, 0) + 1
            family = self.normalized_pos_family(lex)
            family_counts[family] = family_counts.get(family, 0) + 1
        by_pos: Dict[str, Dict[str, int]] = {}
        by_band: Dict[str, Dict[str, int]] = {}
        source_mix: Dict[str, Dict[str, int]] = {}
        samples: Dict[str, List[str]] = {}

        for lex in rows:
            pos = lex.pos or "<empty>"
            band = get_profile(lex.rank).band
            by_pos.setdefault(pos, {"total": 0, "publishable": 0, "manual_review": 0})
            by_band.setdefault(band, {"total": 0, "publishable": 0, "manual_review": 0})
            source_mix.setdefault(pos, {})
            by_pos[pos]["total"] += 1
            by_band[band]["total"] += 1
            try:
                candidate = self.generate_for_lemma(lex.lemma)
            except Exception:
                candidate = self.manual_review_candidate(lex)
            publishable = self.candidate_is_general_publishable(candidate)
            if publishable:
                by_pos[pos]["publishable"] += 1
                by_band[band]["publishable"] += 1
            if candidate.source_method == "manual_review_needed":
                by_pos[pos]["manual_review"] += 1
                by_band[band]["manual_review"] += 1
            source_mix[pos][candidate.source_method] = source_mix[pos].get(candidate.source_method, 0) + 1
            sample_key = f"{pos}|{band}"
            samples.setdefault(sample_key, [])
            if candidate.sentence and len(samples[sample_key]) < samples_per_bucket:
                samples[sample_key].append(f"{lex.lemma}: {candidate.sentence}")

        report = {
            "rows_considered": len(rows),
            "raw_pos_counts": dict(sorted(raw_pos_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "by_pos": by_pos,
            "by_band": by_band,
            "source_mix": source_mix,
            "samples": samples,
        }
        self.last_coverage_report = report
        return report

    def smoke_coverage(
        self,
        limit: int = 60,
        min_rank: int = 1,
        max_rank: int = 10**9,
        pos_filter: Optional[str] = None,
        lemma_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._rows_for_coverage(limit=0, min_rank=min_rank, max_rank=max_rank, pos_filter=pos_filter, lemma_filter=lemma_filter)
        selected: List[Lexeme] = []
        seen_buckets: set = set()
        for lex in rows:
            bucket = (lex.pos or "<empty>", get_profile(lex.rank).band)
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            selected.append(lex)
            if len(selected) >= limit:
                break
        results: List[Dict[str, Any]] = []
        for lex in selected:
            result = self.generate_sentence_for_target(lex.lemma, lex.rank)
            band_ok = result["band"] == get_profile(lex.rank).band
            sentence_words = self.word_tokens(result.get("sentence", ""))
            sentence_has_target = bool(sentence_words) and (
                normalize_token(result.get("target_form", "")) in sentence_words
                or normalize_token(lex.lemma) in sentence_words
            )
            results.append(
                {
                    "lemma": lex.lemma,
                    "pos": lex.pos,
                    "band": get_profile(lex.rank).band,
                    "source_method": result.get("source_method", ""),
                    "publishable": bool(result.get("publishable")),
                    "band_ok": band_ok,
                    "sentence_has_target": sentence_has_target,
                    "sentence": result.get("sentence", ""),
                }
            )
        return results

    def load_gold_set(self, path: str) -> List[str]:
        lemmas: List[str] = []
        seen = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                lemma = line.strip().lower()
                if not lemma or lemma.startswith("#") or lemma in seen:
                    continue
                seen.add(lemma)
                lemmas.append(lemma)
        return lemmas

    def generate_starter_dataset(
        self,
        limit: int,
        out_csv: str,
        review_out: Optional[str] = None,
        quarantine_out: Optional[str] = None,
        pos_filter: Optional[str] = None,
        lemma_filter: Optional[List[str]] = None,
        mvp_only: bool = False,
        candidates_out: Optional[str] = None,
        max_candidates_per_lemma: int = 10,
        starter_max_attempts: int = 300,
    ) -> List[Candidate]:
        stats: Dict[str, Any] = {
            "targets_in_scope": 0,
            "main_rows_written": 0,
            "strict_publishable_rows": 0,
            "retrieved_strict_rows": 0,
            "template_strict_rows": 0,
            "fallback_rows": 0,
            "emergency_rows": 0,
            "manual_review_only_failures": 0,
            "diagnostic_not_starter_eligible": 0,
            "diagnostic_outside_starter_band": 0,
            "diagnostic_excluded_by_override": 0,
            "diagnostic_no_translation": 0,
            "diagnostic_bad_translation": 0,
            "diagnostic_unclean_target_surface": 0,
        }
        starter_reason_counts: Dict[str, int] = {}

        if lemma_filter:
            all_rows: List[Lexeme] = []
            seen: set = set()
            for lemma in lemma_filter:
                key = lemma.strip().lower()
                if not key or key in seen or key not in self.lexicon:
                    continue
                seen.add(key)
                all_rows.append(self.lexicon[key])
        else:
            all_rows = list(self.lexicon.values())
            all_rows.sort(key=lambda x: x.rank)
        if pos_filter:
            all_rows = [x for x in all_rows if x.pos == pos_filter]
        if mvp_only:
            all_rows = [x for x in all_rows if x.pos in {"n", "v", "adj"} and get_profile(x.rank).band in {"A1", "A2", "B1"}]
        scoped_rows = all_rows[:limit]
        stats["targets_in_scope"] = len(scoped_rows)

        for lex in scoped_rows:
            if not self.is_starter_target_eligible(lex):
                stats["diagnostic_not_starter_eligible"] += 1
            if get_profile(lex.rank).band not in STARTER_MAX_BAND:
                stats["diagnostic_outside_starter_band"] += 1
            ov = self.overrides.get(lex.lemma, {})
            if _truthy(ov.get("exclude_from_starter_dataset")):
                stats["diagnostic_excluded_by_override"] += 1
            if not lex.translation:
                stats["diagnostic_no_translation"] += 1
            if not self.starter_target_translation_ok(lex):
                stats["diagnostic_bad_translation"] += 1
            if not self.is_clean_starter_target(lex):
                stats["diagnostic_unclean_target_surface"] += 1

        main_rows: List[Candidate] = []
        review_rows: List[Candidate] = []
        quarantine_rows: List[Candidate] = []
        candidate_rows: List[Candidate] = []

        for lex in scoped_rows:
            try:
                if candidates_out:
                    pool = self.search_starter_candidates_for_lemma(
                        lex.lemma,
                        max_attempts=starter_max_attempts,
                        max_candidates_per_lemma=max_candidates_per_lemma,
                        stop_on_first_publishable=False,
                    )
                    if pool:
                        candidate_rows.extend(pool)
                best = self.generate_starter_for_lemma(lex.lemma, starter_max_attempts=starter_max_attempts)
            except Exception as exc:
                print(f"[warn] starter: failed for {lex.lemma}: {exc}", file=sys.stderr)
                best = self.manual_review_candidate(lex)

            main_rows.append(best)
            meta = self.starter_output_metadata(best)
            stats["main_rows_written"] += 1

            if meta["quality_tier"] == "strict_publishable":
                stats["strict_publishable_rows"] += 1
                if best.source_method == "retrieved_corpus":
                    stats["retrieved_strict_rows"] += 1
                else:
                    stats["template_strict_rows"] += 1
            elif meta["quality_tier"] == "fallback_candidate":
                stats["fallback_rows"] += 1
            elif meta["quality_tier"] == "emergency_template":
                stats["emergency_rows"] += 1
            else:
                stats["manual_review_only_failures"] += 1

            if not meta["strict_starter_ok"]:
                reasons = self.starter_rejection_reasons(best)
                for reason in reasons:
                    starter_reason_counts[reason] = starter_reason_counts.get(reason, 0) + 1
                if best.source_method == "manual_review_needed":
                    review_rows.append(best)
                else:
                    valid, _ = self.validate(best)
                    target_lex = self.lexicon.get(best.lemma)
                    is_near_miss = (
                        valid
                        and best.sentence
                        and target_lex
                        and self.exact_surface_present(best, target_lex)
                        and self.starter_structure_ok(best)
                    )
                    if is_near_miss:
                        quarantine_rows.append(best)
                    else:
                        review_rows.append(best)

        self.write_starter_main_csv(main_rows, out_csv)
        if review_out:
            self.write_starter_review_csv(review_rows, review_out)
        if quarantine_out and quarantine_rows:
            self.write_starter_quarantine_csv(quarantine_rows, quarantine_out)
        if candidates_out:
            self.write_candidates_csv(candidate_rows, candidates_out)

        stats["reason_counts"] = starter_reason_counts
        stats["quarantine"] = len(quarantine_rows)
        self.last_starter_stats = stats
        return main_rows

    def write_csv(self, rows: List[Candidate], out_csv: str) -> None:
        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "band",
            "translation",
            "sentence",
            "english_sentence",
            "target_form",
            "canonical_lemma",
            "target_morph",
            "target_index",
            "support_ranks",
            "avg_support_rank",
            "max_support_rank",
            "template_id",
            "source_method",
            "english_source_method",
            "parallel_pair_id",
            "score",
            "english_score",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                d = asdict(row)
                d["support_ranks"] = " ".join(str(x) for x in row.support_ranks)
                writer.writerow(d)

    def write_starter_main_csv(self, rows: List[Candidate], out_csv: str) -> None:
        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "band",
            "translation",
            "sentence",
            "english_sentence",
            "target_form",
            "canonical_lemma",
            "target_morph",
            "target_index",
            "support_ranks",
            "avg_support_rank",
            "max_support_rank",
            "template_id",
            "source_method",
            "english_source_method",
            "parallel_pair_id",
            "score",
            "english_score",
            "publishable",
            "quality_tier",
            "failure_reason",
            "strict_starter_ok",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                d = asdict(row)
                d["support_ranks"] = " ".join(str(x) for x in row.support_ranks)
                d.update(self.starter_output_metadata(row))
                writer.writerow(d)

    def write_review_csv(self, rows: List[Candidate], out_csv: str) -> None:
        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "sentence",
            "source_method",
            "template_id",
            "score",
            "canonical_lemma",
            "target_morph",
            "grammatical_ok",
            "natural_ok",
            "learner_clear_ok",
            "notes",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                grammatical_ok, natural_ok, learner_clear_ok, notes = self.review_flags(row)
                writer.writerow(
                    {
                        "lemma": row.lemma,
                        "rank": row.rank,
                        "pos": row.pos,
                        "sentence": row.sentence,
                        "source_method": row.source_method,
                        "template_id": row.template_id,
                        "score": row.score,
                        "canonical_lemma": row.canonical_lemma,
                        "target_morph": row.target_morph,
                        "grammatical_ok": grammatical_ok,
                        "natural_ok": natural_ok,
                        "learner_clear_ok": learner_clear_ok,
                        "notes": notes,
                    }
                )

    def write_starter_review_csv(self, rows: List[Candidate], out_csv: str) -> None:
        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "translation",
            "sentence",
            "source_method",
            "template_id",
            "score",
            "canonical_lemma",
            "target_morph",
            "grammatical_ok",
            "natural_ok",
            "learner_clear_ok",
            "starter_notes",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                grammatical_ok, natural_ok, learner_clear_ok, _ = self.review_flags(row)
                starter_notes = self.starter_review_notes(row)
                writer.writerow(
                    {
                        "lemma": row.lemma,
                        "rank": row.rank,
                        "pos": row.pos,
                        "translation": row.translation,
                        "sentence": row.sentence,
                        "source_method": row.source_method,
                        "template_id": row.template_id,
                        "score": row.score,
                        "canonical_lemma": row.canonical_lemma,
                        "target_morph": row.target_morph,
                        "grammatical_ok": grammatical_ok,
                        "natural_ok": natural_ok,
                        "learner_clear_ok": learner_clear_ok,
                        "starter_notes": starter_notes,
                    }
                )

    def write_starter_quarantine_csv(self, rows: List[Candidate], out_csv: str) -> None:
        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "translation",
            "sentence",
            "source_method",
            "template_id",
            "score",
            "rejection_reasons",
            "semantic_reasons",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                rej = self.starter_rejection_reasons(row)
                sem = self.starter_semantic_quality_reasons(row)
                writer.writerow(
                    {
                        "lemma": row.lemma,
                        "rank": row.rank,
                        "pos": row.pos,
                        "translation": row.translation,
                        "sentence": row.sentence,
                        "source_method": row.source_method,
                        "template_id": row.template_id,
                        "score": row.score,
                        "rejection_reasons": "; ".join(rej),
                        "semantic_reasons": "; ".join(sem),
                    }
                )

    def write_candidates_csv(self, rows: List[Candidate], out_csv: str) -> None:
        fieldnames = [
            "lemma",
            "rank",
            "pos",
            "band",
            "translation",
            "sentence",
            "target_form",
            "canonical_lemma",
            "target_morph",
            "target_index",
            "support_ranks",
            "avg_support_rank",
            "max_support_rank",
            "template_id",
            "source_method",
            "score",
            "grammatical_ok",
            "natural_ok",
            "learner_clear_ok",
            "notes",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                grammatical_ok, natural_ok, learner_clear_ok, notes = self.review_flags(row)
                writer.writerow(
                    {
                        "lemma": row.lemma,
                        "rank": row.rank,
                        "pos": row.pos,
                        "band": row.band,
                        "translation": row.translation,
                        "sentence": row.sentence,
                        "target_form": row.target_form,
                        "canonical_lemma": row.canonical_lemma,
                        "target_morph": row.target_morph,
                        "target_index": row.target_index,
                        "support_ranks": " ".join(str(x) for x in row.support_ranks),
                        "avg_support_rank": row.avg_support_rank,
                        "max_support_rank": row.max_support_rank,
                        "template_id": row.template_id,
                        "source_method": row.source_method,
                        "score": row.score,
                        "grammatical_ok": grammatical_ok,
                        "natural_ok": natural_ok,
                        "learner_clear_ok": learner_clear_ok,
                        "notes": notes,
                    }
                )


def generate_sentence_for_target(
    generator: SentenceGenerator, target_lemma: str, target_rank: int
) -> Dict[str, Any]:
    """Canonical public sentence-generation API.

    Example CLI usage:
    python generate.py --lexicon stg_words_spa.csv --models-dir models --target-lemma casa --target-rank 500
    """
    return generator.generate_sentence_for_target(target_lemma, target_rank)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spanish sentence generator using monolingual corpus artifacts.",
        epilog="Example: python generate.py --lexicon stg_words_spa.csv --models-dir models --target-lemma casa --target-rank 500",
    )
    parser.add_argument("--lexicon", required=True, help="Path to stg_words_spa.csv")
    parser.add_argument("--models-dir", required=True, help="Directory containing .pkl artifacts")
    parser.add_argument("--out", default="generated_sentences.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=100, help="Number of lexicon rows to generate")
    parser.add_argument("--min-rank", type=int, default=1, help="Minimum rank to include")
    parser.add_argument("--max-rank", type=int, default=10**9, help="Maximum rank to include")
    parser.add_argument("--pos", default=None, help="Optional POS filter: n, v, adj")
    parser.add_argument("--lemma", action="append", help="Generate only these lemma(s). Can be repeated.")
    parser.add_argument("--gold-set", default=None, help="Path to a newline-delimited lemma list for repeatable evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mvp-only", action="store_true", help="Limit generation to A1-B1 nouns, verbs, and adjectives.")
    parser.add_argument("--review-out", default=None, help="Optional review CSV export path.")
    parser.add_argument("--starter-dataset", action="store_true", help="Generate a starter app-seeding dataset and matching review CSV.")
    parser.add_argument("--candidates-out", default=None, help="Optional multi-candidate CSV export path.")
    parser.add_argument("--max-candidates-per-lemma", type=int, default=10, help="Maximum candidate rows to keep per lemma in the candidate export.")
    parser.add_argument("--starter-max-attempts", type=int, default=300, help="Maximum generation attempts per lemma for starter dataset search.")
    parser.add_argument("--lexicon-overrides", action="append", default=[], help="Path to a lexicon overrides CSV or JSON file. Can be repeated.")
    parser.add_argument("--quarantine-out", default=None, help="Optional quarantine CSV for near-miss starter rows.")
    parser.add_argument("--target-lemma", default=None, help="Generate one structured sentence result for this lemma.")
    parser.add_argument("--target-rank", type=int, default=None, help="Rank to use when deriving the output band for --target-lemma.")
    parser.add_argument("--coverage-report", action="store_true", help="Print a POS/band coverage report over a bounded batch.")
    parser.add_argument("--coverage-limit", type=int, default=200, help="Number of lemmas to sample for coverage reporting.")
    parser.add_argument("--coverage-samples-per-bucket", type=int, default=2, help="Maximum sample sentences to print per POS/band bucket.")
    parser.add_argument("--smoke-coverage", action="store_true", help="Run a bounded smoke check across POS/band buckets.")
    args = parser.parse_args()

    gen = SentenceGenerator(args.lexicon, args.models_dir, seed=args.seed)
    for override_path in args.lexicon_overrides:
        gen.load_and_apply_overrides(override_path)
        print(f"Loaded {len(gen.overrides)} total lexicon overrides after {override_path}")
    if args.target_lemma is not None:
        import json

        target_rank = args.target_rank if args.target_rank is not None else gen.lexicon.get(args.target_lemma.strip().lower(), Lexeme("", 99999, "")).rank
        result = generate_sentence_for_target(gen, args.target_lemma, target_rank)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    lemma_filter = list(args.lemma or [])
    if args.gold_set:
        lemma_filter.extend(gen.load_gold_set(args.gold_set))
    if not lemma_filter:
        lemma_filter = None
    review_out = args.review_out
    if args.starter_dataset and not review_out:
        stem, ext = os.path.splitext(args.out)
        review_out = f"{stem}_review{ext or '.csv'}"
    quarantine_out = getattr(args, "quarantine_out", None)
    if args.starter_dataset and not quarantine_out:
        stem, ext = os.path.splitext(args.out)
        quarantine_out = f"{stem}_quarantine{ext or '.csv'}"
    if args.coverage_report:
        report = gen.build_coverage_report(
            limit=args.coverage_limit,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
            pos_filter=args.pos,
            lemma_filter=lemma_filter,
            samples_per_bucket=args.coverage_samples_per_bucket,
        )
        print("=== Coverage Report ===")
        print(f"Rows considered: {report['rows_considered']:,}")
        print("\nRaw POS inventory:")
        for pos, count in sorted(report["raw_pos_counts"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {pos:<12} {count:,}")
        print("\nNormalized strategy families:")
        for family, count in sorted(report["family_counts"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {family:<12} {count:,}")
        print("\nBy POS:")
        for pos, stats in sorted(report["by_pos"].items()):
            total = stats["total"] or 1
            print(
                f"  {pos:<12} total={stats['total']:,} publishable={stats['publishable']:,} "
                f"publishable_rate={stats['publishable'] / total * 100:.1f}% manual_review={stats['manual_review']:,}"
            )
        print("\nBy band:")
        for band, stats in sorted(report["by_band"].items()):
            total = stats["total"] or 1
            print(
                f"  {band:<3} total={stats['total']:,} publishable={stats['publishable']:,} "
                f"publishable_rate={stats['publishable'] / total * 100:.1f}% manual_review={stats['manual_review']:,}"
            )
        print("\nSource mix by POS:")
        for pos, mix in sorted(report["source_mix"].items()):
            parts = ", ".join(f"{k}={v}" for k, v in sorted(mix.items()))
            print(f"  {pos:<12} {parts}")
        print("\nSamples:")
        for bucket, items in sorted(report["samples"].items()):
            print(f"  {bucket}")
            for item in items:
                print(f"    {item}")
        return
    if args.smoke_coverage:
        results = gen.smoke_coverage(
            limit=args.coverage_limit,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
            pos_filter=args.pos,
            lemma_filter=lemma_filter,
        )
        print("=== Smoke Coverage ===")
        failures = 0
        for row in results:
            ok = row["band_ok"] and row["sentence_has_target"]
            if not ok:
                failures += 1
            print(
                f"  {'PASS' if ok else 'FAIL'} {row['lemma']:<15} pos={row['pos']:<12} "
                f"band={row['band']:<3} source={row['source_method']:<18} sentence={row['sentence']}"
            )
        print(f"\nBuckets checked: {len(results):,}")
        print(f"Failures: {failures:,}")
        return
    if args.starter_dataset:
        rows = gen.generate_starter_dataset(
            limit=args.limit,
            out_csv=args.out,
            review_out=review_out,
            quarantine_out=quarantine_out,
            pos_filter=args.pos,
            lemma_filter=lemma_filter,
            mvp_only=args.mvp_only,
            candidates_out=args.candidates_out,
            max_candidates_per_lemma=args.max_candidates_per_lemma,
            starter_max_attempts=args.starter_max_attempts,
        )
    else:
        rows = gen.generate_batch(
            limit=args.limit,
            out_csv=args.out,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
            pos_filter=args.pos,
            lemma_filter=lemma_filter,
            mvp_only=args.mvp_only,
            candidates_out=args.candidates_out,
            max_candidates_per_lemma=args.max_candidates_per_lemma,
        )
        if review_out:
            gen.write_review_csv(rows, review_out)

    if args.starter_dataset and hasattr(gen, "last_starter_stats") and gen.last_starter_stats:
        s = gen.last_starter_stats
        print("\n=== Starter Dataset Report ===")
        print(f"  Targets in scope:              {s['targets_in_scope']:,}")
        print(f"  Main rows written:             {s['main_rows_written']:,}")
        print(f"  ---")
        print(f"  Strict publishable rows:       {s['strict_publishable_rows']:,}")
        print(f"    from templates:              {s.get('template_strict_rows', 0):,}")
        print(f"    from retrieved:              {s.get('retrieved_strict_rows', 0):,}")
        print(f"  Fallback rows:                 {s['fallback_rows']:,}")
        print(f"  Emergency rows:                {s['emergency_rows']:,}")
        print(f"  Manual-review-only failures:   {s['manual_review_only_failures']:,}")
        pub_rate = s["strict_publishable_rows"] / s["targets_in_scope"] * 100 if s["targets_in_scope"] else 0
        print(f"  Strict publish rate:           {pub_rate:.1f}%")
        print(f"  Saved main output: {args.out}")
        if review_out:
            print(f"  Saved review:      {review_out}")
        if quarantine_out:
            print(f"  Saved quarantine:  {quarantine_out}")
        if args.candidates_out:
            print(f"  Saved candidates:  {args.candidates_out}")
        print("\n  Diagnostic flags across scoped targets:")
        print(f"    not starter-eligible:        {s.get('diagnostic_not_starter_eligible', 0):,}")
        print(f"    outside starter band:        {s.get('diagnostic_outside_starter_band', 0):,}")
        print(f"    excluded by override:        {s.get('diagnostic_excluded_by_override', 0):,}")
        print(f"    no translation:              {s.get('diagnostic_no_translation', 0):,}")
        print(f"    bad translation:             {s.get('diagnostic_bad_translation', 0):,}")
        print(f"    unclean target surface:      {s.get('diagnostic_unclean_target_surface', 0):,}")
        reason_counts = s.get("reason_counts", {})
        if reason_counts:
            print("\n  Top starter rejection reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {reason:<35} {count:,}")
        publishable_rows = [r for r in rows if gen.candidate_is_starter_publishable(r)]
        preview = publishable_rows[:5]
        if preview:
            print("\n  Starter preview:")
            for row in preview:
                print(f"    {row.lemma:<15} [{row.source_method:<18}] {row.sentence}")
    else:
        reviewed = sum(1 for r in rows if r.source_method == "manual_review_needed")
        retrieved = sum(1 for r in rows if r.source_method == "retrieved_corpus")
        seeded = sum(1 for r in rows if r.source_method == "seeded_template")
        templated = sum(1 for r in rows if r.source_method == "template_generated")
        print(f"Generated: {len(rows):,}")
        print(f"  retrieved_corpus:     {retrieved:,}")
        print(f"  seeded_template:      {seeded:,}")
        print(f"  template_generated:   {templated:,}")
        print(f"  manual_review_needed: {reviewed:,}")
        print(f"Saved: {args.out}")
        if review_out:
            print(f"Review: {review_out}")
        if args.candidates_out and gen.last_candidate_export_stats:
            stats = gen.last_candidate_export_stats
            print(f"Candidates: {args.candidates_out}")
            print(f"  lemmas_processed: {int(stats['lemmas_processed']):,}")
            print(f"  candidate_rows_written: {int(stats['candidate_rows_written']):,}")
            print(f"  avg_candidates_per_lemma_with_candidates: {stats['avg_candidates_per_lemma_with_candidates']:.2f}")
        preview = [r for r in rows if r.sentence][:5]
        if preview:
            print("\nPreview:")
            for row in preview:
                print(f"  {row.lemma:<15} [{row.source_method:<18}] {row.sentence}")


if __name__ == "__main__":
    main()
