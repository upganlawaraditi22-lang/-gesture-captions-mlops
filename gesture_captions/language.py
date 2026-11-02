"""Sign gloss -> natural sentence in English / Hindi / Marathi.

Sign languages have their own grammar: a signer produces a sequence of
content words ("gloss"), e.g.  I  WANT  WATER  PLEASE.
This module is a small rule-based translator:

  1. normalise   : map signs to concepts (WAVE -> HELLO, NO_NO -> NO, ...)
  2. parse       : split the gloss into clauses and fill a semantic frame
                   {subject, verb, object, polite, question}
  3. realise     : generate the sentence from per-language templates,
                   handling word order (English SVO vs Hindi/Marathi SOV),
                   question forms and gender agreement (Marathi हवा/हवी/हवे).

Unknown signs fall back to their own word, so new gestures never break it.
"""

LANGS = ["en", "hi", "mr"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

# ---------------------------------------------------------------- lexicon
NORMALISE = {"WAVE": "HELLO", "NO_NO": "NO", "BYE": "GOODBYE"}

STANDALONE = {   # signs that form a sentence on their own
    "HELLO":        {"en": "Hello!",        "hi": "नमस्ते!",      "mr": "नमस्कार!"},
    "GOOD_MORNING": {"en": "Good morning!", "hi": "सुप्रभात!",     "mr": "सुप्रभात!"},
    "GOODBYE":      {"en": "Goodbye!",      "hi": "अलविदा!",      "mr": "निरोप घेतो!"},
    "THANK_YOU":    {"en": "Thank you!",    "hi": "धन्यवाद!",     "mr": "धन्यवाद!"},
    "YES":          {"en": "Yes.",          "hi": "हाँ।",          "mr": "हो."},
    "NO":           {"en": "No.",           "hi": "नहीं।",         "mr": "नाही."},
    "GOOD_MARKS":   {"en": "Sir, give me good marks.", "hi": "सर, मुझे अच्छे अंक दीजिए।", "mr": "सर, मला चांगले गुण द्या."},
}

SUBJECTS = {"I", "YOU"}
VERBS = {"WANT", "NEED", "GO", "COME", "DRINK", "HELP"}

# nouns: English with article, Hindi, Marathi + Marathi gender (m/f/n)
NOUNS = {
    "WATER": {"en": "some water", "hi": "पानी", "mr": "पाणी", "g": "n"},
    "FOOD":  {"en": "some food",  "hi": "खाना", "mr": "जेवण", "g": "n"},
    "HELP":  {"en": "help",       "hi": "मदद",  "mr": "मदत",  "g": "f"},
    "HOME":  {"en": "home",       "hi": "घर",   "mr": "घरी",  "g": "n"},
}
MR_HAVA = {"m": "हवा", "f": "हवी", "n": "हवे"}


def _word(label):
    return label.replace("_", " ").lower()


# ---------------------------------------------------------------- parsing
def parse(gloss):
    """gloss: list of sign labels -> list of clauses (dicts)."""
    clauses, cur = [], {}

    def flush():
        nonlocal cur
        if any(k in cur for k in ("subj", "verb", "obj", "extra")):
            clauses.append(cur)
        elif cur.get("polite") and clauses and clauses[-1]["type"] == "frame":
            clauses[-1]["polite"] = True          # trailing PLEASE
        elif cur.get("polite"):
            clauses.append({"type": "standalone", "key": "PLEASE"})
        cur = {}

    for raw in gloss:
        s = NORMALISE.get(raw, raw)
        if s in STANDALONE:
            flush()
            clauses.append({"type": "standalone", "key": s})
            continue
        cur.setdefault("type", "frame")
        if s == "PLEASE":
            cur["polite"] = True
        elif s in SUBJECTS:
            if "subj" in cur and ("verb" in cur or "obj" in cur):
                flush(); cur["type"] = "frame"
            cur["subj"] = s
        elif s == "HELP" and "verb" in cur:      # "I NEED HELP" -> HELP is the object
            cur["obj"] = s
        elif s in VERBS:
            if "verb" in cur and cur["verb"] not in ("WANT", "NEED"):
                flush(); cur["type"] = "frame"
            if "verb" in cur:                    # WANT GO / NEED DRINK -> modal + action
                cur["modal"], cur["verb"] = cur["verb"], s
            else:
                cur["verb"] = s
        elif s in NOUNS:
            if "obj" in cur:
                flush(); cur["type"] = "frame"
            cur["obj"] = s
        elif "verb" in cur and "obj" not in cur and "objw" not in cur:
            cur["objw"] = s                      # unknown / spelled word after a verb = its object
        else:
            cur.setdefault("extra", []).append(s)
    flush()
    return clauses


# ---------------------------------------------------------------- realisation
def _obj(c, lang):
    if c.get("obj"):
        return NOUNS[c["obj"]][lang]
    if c.get("objw"):
        return _word(c["objw"]).capitalize()
    return None


def _en(c):
    if c["type"] == "standalone":
        return "Please." if c["key"] == "PLEASE" else STANDALONE[c["key"]]["en"]
    subj, verb, obj = c.get("subj"), c.get("verb"), c.get("obj")
    modal, pol, q = c.get("modal"), c.get("polite"), c.get("subj") == "YOU"
    extra = " ".join(_word(e) for e in c.get("extra", []))
    o = _obj(c, "en") or ""

    if verb is None and obj and subj is None:            # WATER (PLEASE)
        o = o.replace("some ", "")
        o = o[0].upper() + o[1:]
        return f"{o}, please." if pol else f"{o}."
    if verb == "HELP" and not obj:
        return "Can I help you?" if q else "Please help me!"
    if verb == "COME":
        return "Please come here." if not q else "Will you come here?"
    if verb == "GO":
        dest = " home" if obj == "HOME" else (f" to {o}" if o else "")
        base = "want to go" if modal != "NEED" else "need to go"
        if q:
            return f"Do you {base}{dest}?"
        return f"I {base}{dest}" + (", please." if pol else ".")
    if verb == "DRINK":
        thing = o.replace("some ", "") if o else "water"
        return f"Do you want to drink {thing}?" if q else f"I would like to drink {thing}" + (", please." if pol else ".")
    if verb in ("WANT", "NEED") or obj or c.get("objw"):
        v = verb or "WANT"
        if q:
            return f"Do you {v.lower()} {o or 'something'}?"
        if v == "WANT":
            body = f"I would like {o}" if o else "I want something"
        else:
            body = f"I need {o}" if o else "I need something"
        return body + (", please." if pol else ".")
    words = " ".join(_word(w) for w in [subj, verb] if w) + (" " + extra if extra else "")
    words = words.strip() or "..."
    return words[0].upper() + words[1:] + "."


def _hi(c):
    if c["type"] == "standalone":
        return "कृपया।" if c["key"] == "PLEASE" else STANDALONE[c["key"]]["hi"]
    subj, verb, obj, modal = c.get("subj"), c.get("verb"), c.get("obj"), c.get("modal")
    q, pol = subj == "YOU", c.get("polite")
    who = "आपको" if q else "मुझे"
    pre = "क्या " if q else ("कृपया, " if pol else "")
    end = "?" if q else "।"
    o = _obj(c, "hi")

    if verb is None and obj and subj is None:
        return f"{o}, कृपया।" if pol else f"{o}।"
    if verb == "HELP" and not obj:
        return "क्या मैं आपकी मदद करूँ?" if q else "कृपया मेरी मदद कीजिए!"
    if verb == "COME":
        return "क्या आप यहाँ आएँगे?" if q else "कृपया यहाँ आइए।"
    if verb == "GO":
        return f"{pre}{who} {o or ''} जाना है{end}".replace("  ", " ")
    if verb == "DRINK":
        return f"{pre}{who} {o or 'पानी'} पीना है{end}"
    if verb in ("WANT", "NEED") or obj or c.get("objw"):
        return f"{pre}{who} {o or 'कुछ'} चाहिए{end}"
    return " ".join(_word(w).capitalize() for w in [subj, verb] + c.get("extra", []) if w) + "।"


def _mr(c):
    if c["type"] == "standalone":
        return "कृपया." if c["key"] == "PLEASE" else STANDALONE[c["key"]]["mr"]
    subj, verb, obj = c.get("subj"), c.get("verb"), c.get("obj")
    q, pol = subj == "YOU", c.get("polite")
    who = "तुम्हाला" if q else "मला"
    pre = "कृपया, " if pol and not q else ""
    end = " का?" if q else "."
    o = _obj(c, "mr")

    if verb is None and obj and subj is None:
        return f"{o}, कृपया." if pol else f"{o}."
    if verb == "HELP" and not obj:
        return "मी तुमची मदत करू का?" if q else "कृपया मला मदत करा!"
    if verb == "COME":
        return "तुम्ही इकडे याल का?" if q else "कृपया इकडे या."
    if verb == "GO":
        dest = "घरी " if obj == "HOME" else (f"{o}ला " if o else "")
        return f"{pre}{who} {dest}जायचे आहे{end}"
    if verb == "DRINK":
        return f"{pre}{who} {o or 'पाणी'} प्यायचे आहे{end}"
    if verb in ("WANT", "NEED") or obj or c.get("objw"):
        g = NOUNS[obj]["g"] if obj else "n"
        return f"{pre}{who} {o or 'काहीतरी'} {MR_HAVA[g]} आहे{end}"
    return " ".join(_word(w).capitalize() for w in [subj, verb] + c.get("extra", []) if w) + "."


REALISE = {"en": _en, "hi": _hi, "mr": _mr}


def merge_spelling(gloss):
    """Join finger-spelled letters into one word: [I, NEED, LETTER_R, LETTER_A, LETTER_M] -> [I, NEED, RAM].
    A spelled word that is a known sign (e.g. W-A-T-E-R) is then translated like that sign."""
    out, buf = [], ""
    for g in gloss:
        if g.startswith(("LETTER_", "DIGIT_")):
            buf += g.split("_", 1)[1]
            continue
        if buf:
            out.append(buf); buf = ""
        out.append(g)
    if buf:
        out.append(buf)
    return out


def translate(gloss, lang="en"):
    """['I','WANT','WATER','PLEASE'] -> 'I would like some water, please.'"""
    gloss = merge_spelling(gloss)
    if not gloss:
        return ""
    return " ".join(REALISE[lang](c) for c in parse(gloss))


def gloss_text(gloss):
    return " ".join(g.replace("_", " ") for g in merge_spelling(gloss))


if __name__ == "__main__":
    tests = [
        ["HELLO", "GOOD_MORNING"],
        ["I", "WANT", "WATER", "PLEASE"],
        ["I", "NEED", "HELP"],
        ["YOU", "WANT", "FOOD"],
        ["I", "WANT", "GO", "HOME"],
        ["WAVE", "I", "NEED", "FOOD", "THANK_YOU"],
        ["YOU", "NEED", "HELP"],
        ["COME"], ["DRINK", "WATER"], ["HELP"], ["WATER", "PLEASE"],
        ["YES"], ["NO_NO"], ["RONALDO_SIUUUUU"], ["I", "WANT", "FOOD", "YOU", "WANT", "WATER"],
    ]
    for t in tests:
        print(gloss_text(t))
        for lang in LANGS:
            print(f"   {lang}: {translate(t, lang)}")
