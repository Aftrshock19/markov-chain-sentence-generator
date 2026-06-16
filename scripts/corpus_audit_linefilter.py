#!/usr/bin/env python3
"""Read-only SBWCE sample audit with CORRECTED stage ordering.

Read-only: input never modified, no cleaned corpus written. Only output is the
report at OUTPUT. Login-node safe (1000-line sample, senter-only). No pip, no sbatch.

  STAGE A (line-level, BEFORE segmentation): drop any input LINE firing
           has_space_token OR legal_marker.
  STAGE B: senter (es_core_news_sm, senter-only) on Stage-A survivors only.
  STAGE C: remaining per-sentence predicates on Stage-B sentences, incl. the
           new missing_space_weld.

missing_space_weld: per user decision, CLAUSE 1 ONLY (token >20 chars with no
internal hyphen/digit). The spec's clause 2 (spaCy-vocab miss rate) is INERT on
es_core_news_sm: the model ships no vectors and no lexeme_prob table, so prob is
the default -20 and is_oov is True for every token, while `tok in nlp.vocab` is
True for every token (vocab auto-vivifies). Neither distinguishes a real word
from a weld, so clause 2 is excluded and reported, not silently broken.
"""
import re, unicodedata, random
import spacy

INPUT  = "/projects/b35cg/corpora/sbwce_sample1000.txt"
OUTPUT = "/lfs1i3/home/b35cg/samtoughan.b35cg/markov-chain-sentence-generator/outputs/audit_sbwce_linefilter.md"
SEED, K = 1234, 15

ZW = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)  # soft hyphen + zero-width

def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = s.translate(ZW)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^-\s?", "", s, count=1)
    return s

# ---------- Stage A predicates (line-level) ----------
RE_SPACE = re.compile(r"\bSPACE\b")
LEGAL = ["Nº","N°","ARTÍCULO","ARTICULO","VISTO","CONSIDERANDO","Expediente","CCT",
         "Decreto","Resolución","RESUELVE","período de sesiones","Señor Presidente"]
def has_space_token(s): return bool(RE_SPACE.search(s))
def legal_marker(s):    return any(m in s for m in LEGAL)

# ---------- Stage C predicates (per sentence) ----------
RE_BRACKET = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")
RE_OCR_IFORL = re.compile(r"[a-záéíóúñ]I[a-záéíóúñ]")
RE_DIGIT_IN_ALPHA = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]\d|\d[A-Za-zÁÉÍÓÚÑáéíóúñ]")
EN = ["the","and","with","whole","miss","Hail","you","for","this","that","of","your"]
RE_EN = re.compile(r"\b(" + "|".join(EN) + r")\b")
RE_TOKEN = re.compile(r"\S+")
RE_ALPHA = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]")
RE_DIGIT = re.compile(r"\d")

def tokens(s): return RE_TOKEN.findall(s)

def is_allcaps(s):
    toks = tokens(s)
    if len(toks) <= 3: return False
    if not any(RE_ALPHA.search(t) for t in toks): return False
    return s == s.upper()

def missing_space_weld(s):
    # clause 1 only: a token >20 chars with no internal hyphen and no digit
    for t in tokens(s):
        if len(t) > 20 and "-" not in t and not RE_DIGIT.search(t):
            return True
    return False

STAGE_C = ["subtitle_furniture","ocr_iforl","digit_in_alpha","high_digit_ratio",
           "length_out_of_window","english_content_token","missing_space_weld"]

def eval_c(s):
    n = len(s); digits = sum(c.isdigit() for c in s); tc = len(tokens(s))
    p = {}
    p["subtitle_furniture"]   = ("♪" in s or "*" in s or bool(RE_BRACKET.search(s))
                                 or s.endswith(":") or is_allcaps(s))
    p["ocr_iforl"]            = bool(RE_OCR_IFORL.search(s))
    p["digit_in_alpha"]       = bool(RE_DIGIT_IN_ALPHA.search(s))
    p["high_digit_ratio"]     = (digits / n) > 0.10 if n else False
    p["length_out_of_window"] = tc < 4 or tc > 30
    en = RE_EN.findall(s)
    p["english_content_token"] = bool(en)
    p["missing_space_weld"]   = missing_space_weld(s)
    return p, tc, en

TBUCKETS = [("1",1,1),("2",2,2),("3",3,3),("4-7",4,7),("8-15",8,15),("16-30",16,30),("31+",31,10**9)]
def bucket(tc):
    for nm,lo,hi in TBUCKETS:
        if lo <= tc <= hi: return nm
    return "31+"
def pct(n,d): return f"{(100.0*n/d):.2f}%" if d else "0%"

def main():
    nlp = spacy.load("es_core_news_sm",
                     disable=["tagger","parser","ner","lemmatizer","attribute_ruler","morphologizer"])
    nlp.enable_pipe("senter")
    nlp.max_length = 2_000_000
    method = (f"spaCy es_core_news_sm v{spacy.__version__}, senter only "
              f"(pipes={nlp.pipe_names}), nlp.pipe(batch_size=200)")

    with open(INPUT, encoding="utf-8", errors="replace") as f:
        raw = [l.rstrip("\n") for l in f]
    norm_lines = [norm(l) for l in raw]
    n_in = len(norm_lines)

    # ----- Stage A -----
    fire_space = fire_legal = 0
    survivors = []
    for s in norm_lines:
        a = has_space_token(s); b = legal_marker(s)
        if a: fire_space += 1
        if b: fire_legal += 1
        if not (a or b):
            survivors.append(s)
    n_surv = len(survivors)
    n_dropped = n_in - n_surv

    # ----- Stage B -----
    sents = []
    for doc in nlp.pipe(survivors, batch_size=200):
        for sp in doc.sents:
            t = re.sub(r"^-\s?", "", sp.text.strip(), count=1)
            if t:
                sents.append(t)
    n_sents = len(sents)

    # ----- Stage C -----
    pcount = {p: 0 for p in STAGE_C}
    pfire  = {p: [] for p in STAGE_C}
    none_list = []
    atleast1 = atleast2 = 0
    tbuck = {nm: 0 for nm, _, _ in TBUCKETS}
    en_counts = {}
    for sent in sents:
        p, tc, en = eval_c(sent)
        tbuck[bucket(tc)] += 1
        for tk in en: en_counts[tk] = en_counts.get(tk, 0) + 1
        nf = 0
        for pred in STAGE_C:
            if p[pred]:
                pcount[pred] += 1; nf += 1; pfire[pred].append(sent)
        if nf >= 1: atleast1 += 1
        if nf >= 2: atleast2 += 1
        if nf == 0: none_list.append(sent)

    rnd = random.Random(SEED)
    def sample(lst): return rnd.sample(lst, min(K, len(lst))) if lst else []

    # ----- report -----
    L = []
    L.append("# Corpus audit — SBWCE (line-filter → segment → per-sentence)")
    L.append("")
    L.append(f"Source: `{INPUT}`  —  read-only; input unmodified, no cleaned corpus written.")
    L.append("Normalisation (eval only): NFC + soft-hyphen/zero-width strip + whitespace collapse + single leading dash strip. Case preserved.")
    L.append("")
    L.append("## Stage A — line-level filter (BEFORE segmentation)")
    L.append("")
    L.append("Drop any input LINE firing `has_space_token` OR `legal_marker`.")
    L.append("")
    L.append("| metric | lines | % of input |")
    L.append("|---|---|---|")
    L.append(f"| input lines | {n_in:,} | 100.00% |")
    L.append(f"| firing has_space_token | {fire_space:,} | {pct(fire_space,n_in)} |")
    L.append(f"| firing legal_marker | {fire_legal:,} | {pct(fire_legal,n_in)} |")
    L.append(f"| dropped (either) | {n_dropped:,} | {pct(n_dropped,n_in)} |")
    L.append(f"| **surviving** | **{n_surv:,}** | **{pct(n_surv,n_in)}** |")
    L.append("")
    L.append("_has_space_token and legal_marker overlap heavily (legal blocks carry both), so 'dropped' < their sum._")
    L.append("")
    L.append("## Stage B — segmentation of survivors")
    L.append("")
    L.append(f"- **Method:** {method}")
    L.append(f"- **Surviving lines in:** {n_surv:,}")
    L.append(f"- **Sentences out:** {n_sents:,}")
    L.append(f"- **Sentences per line (weld factor on non-legal lines):** {n_sents/n_surv:.2f}" if n_surv else "- n/a")
    L.append("")
    L.append("## Stage C — per-sentence predicates (over Stage-B sentences)")
    L.append("")
    L.append("| predicate | sentences firing | % |")
    L.append("|---|---|---|")
    for p in STAGE_C:
        L.append(f"| {p} | {pcount[p]:,} | {pct(pcount[p],n_sents)} |")
    L.append("")
    L.append("_`has_space_token` and `legal_marker` are not in Stage C — they are the Stage-A line filter._")
    L.append("_`missing_space_weld` = clause 1 only (token >20 chars, no internal hyphen/digit). "
             "Clause 2 (spaCy-vocab miss rate) is INERT on es_core_news_sm — no vectors / no lexeme_prob table, "
             "so is_oov is True and `tok in nlp.vocab` is True for every token; it is excluded, not computed._")
    L.append("")
    L.append("## Co-occurrence (Stage-C predicates)")
    L.append("")
    L.append(f"- firing ≥1 predicate (would-drop): {atleast1:,} ({pct(atleast1,n_sents)})")
    L.append(f"- firing ≥2 predicates: {atleast2:,} ({pct(atleast2,n_sents)})")
    L.append(f"- firing 0 predicates (estimated survival): {n_sents-atleast1:,} ({pct(n_sents-atleast1,n_sents)})")
    L.append("")
    L.append("## Token-count histogram (over Stage-B sentences)")
    L.append("")
    L.append("| tokens | sentences | % |")
    L.append("|---|---|---|")
    for nm,_,_ in TBUCKETS:
        L.append(f"| {nm} | {tbuck[nm]:,} | {pct(tbuck[nm],n_sents)} |")
    L.append("")
    L.append("## english_content_token — which tokens fired")
    L.append("")
    if en_counts:
        L.append("| token | sentence hits |"); L.append("|---|---|")
        for tk,c in sorted(en_counts.items(), key=lambda x:-x[1]):
            L.append(f"| {tk} | {c:,} |")
    else:
        L.append("_none fired_")
    L.append("")
    L.append(f"## missing_space_weld — {pcount['missing_space_weld']:,} firing (up to {K} samples)")
    L.append("")
    s = sample(pfire["missing_space_weld"])
    if not s: L.append("_none_")
    for sent in s: L.append(f"- `{sent}`")
    L.append("")
    L.append(f"## Zero-predicate would-keep set (up to {K} samples)")
    L.append("")
    s = sample(none_list)
    if not s: L.append("_none_")
    for sent in s: L.append(f"- `{sent}`")
    L.append("")
    L.append("## Combined note")
    L.append("")
    L.append(f"Stage A dropped {pct(n_dropped,n_in)} of input lines. Of the {n_surv:,} survivors, "
             f"senter produced {n_sents:,} sentences (weld {n_sents/n_surv:.2f}). "
             f"Estimated final survival = Stage-B sentences firing zero Stage-C predicates: "
             f"**{pct(n_sents-atleast1,n_sents)}** ({n_sents-atleast1:,} of {n_sents:,}).")
    L.append("")

    with open(OUTPUT, "w", encoding="utf-8") as fo:
        fo.write("\n".join(L))
    print(f"A: in={n_in} dropped={n_dropped} ({pct(n_dropped,n_in)}) surv={n_surv}")
    print(f"   space_fire={fire_space} legal_fire={fire_legal}")
    print(f"B: sents={n_sents} weld={n_sents/n_surv:.2f}")
    print(f"C: >=1={pct(atleast1,n_sents)} survival0={pct(n_sents-atleast1,n_sents)} weld_fires={pcount['missing_space_weld']}")
    print("over30:", tbuck["31+"], pct(tbuck["31+"],n_sents))
    print("wrote", OUTPUT)

if __name__ == "__main__":
    main()
