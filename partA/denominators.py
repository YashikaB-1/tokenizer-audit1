#!/usr/bin/env python3
"""
denominators.py -- A3's central question: what is the denominator supposed to
hold constant across languages, and which candidates actually do it?

Method. The corpus is parallel: all 997 rows carry the SAME 997 units of
meaning in every language. So for identical semantic content we can just look
at how much each candidate denominator varies across languages. A denominator
that varies is injecting a language-dependent factor into every ratio computed
with it -- that factor is indistinguishable, in the output number, from a
genuine tokenizer effect.

    fertility_lang / fertility_eng
        = (T_l / D_l) / (T_e / D_e)
        = (T_l / T_e)  x  (D_e / D_l)
          ^^^^^^^^^^^     ^^^^^^^^^^^
          what you want   contamination, unless D is constant across languages

Usage:  python denominators.py
"""

import os
import statistics
import unicodedata

import regex

HERE = os.path.dirname(os.path.abspath(__file__))
CORP = os.path.join(HERE, "corpus")
LANGS = ["eng", "hin", "ben", "mar", "kan", "tam", "tel"]


def read(lang):
    path = os.path.join(CORP, lang + ".txt")
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                out.append(unicodedata.normalize("NFC", line))
    return out


def main():
    corp = {l: read(l) for l in LANGS}
    n = len(corp["eng"])
    assert all(len(v) == n for v in corp.values())

    units = {
        "parallel sentence": lambda lines: len(lines),
        "whitespace word": lambda lines: sum(len(l.split()) for l in lines),
        "grapheme cluster": lambda lines: sum(len(regex.findall(r"\X", l)) for l in lines),
        "UTF-8 byte": lambda lines: sum(len(l.encode("utf-8")) for l in lines),
        "Unicode codepoint": lambda lines: sum(len(l) for l in lines),
    }

    print("=" * 92)
    print("Denominator size for IDENTICAL CONTENT (%d parallel sentences)" % n)
    print("=" * 92)
    print("If a denominator held meaning constant, every column below would be flat.\n")

    hdr = "%-20s" % "denominator" + "".join("%9s" % l for l in LANGS) + "%11s%9s" % ("max/min", "CV%")
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for name, fn in units.items():
        vals = [fn(corp[l]) / n for l in LANGS]   # per parallel sentence
        spread = max(vals) / min(vals)
        cv = statistics.pstdev(vals) / statistics.mean(vals) * 100
        summary[name] = (spread, cv)
        print("%-20s" % name + "".join("%9.2f" % v for v in vals)
              + "%10.2fx%8.1f%%" % (spread, cv))

    print("\n(units are 'per parallel sentence', so each column is the same content)")

    print("\n" + "=" * 92)
    print("Contamination factor D_eng/D_lang -- what each denominator ADDS to the ratio")
    print("=" * 92)
    print("A ratio computed with denominator D equals (true token ratio) x (D_eng/D_lang).")
    print("Only the parallel sentence gives 1.00 everywhere.\n")

    hdr2 = "%-20s" % "denominator" + "".join("%9s" % l for l in LANGS)
    print(hdr2)
    print("-" * len(hdr2))
    for name, fn in units.items():
        vals = {l: fn(corp[l]) for l in LANGS}
        print("%-20s" % name + "".join("%9.2f" % (vals["eng"] / vals[l]) for l in LANGS))

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    ranked = sorted(summary.items(), key=lambda kv: kv[1][1])
    for name, (spread, cv) in ranked:
        verdict = ("holds content constant BY CONSTRUCTION"
                   if cv < 1e-9 else
                   "varies %.2fx across languages -- contaminates every ratio" % spread)
        print("  %-20s %s" % (name, verdict))
    print("""
Why 'per whitespace word' is the worst of the four for THIS decision:
  * Hindi splits postpositions off as separate orthographic words, so it has
    MORE words per unit of meaning than English.
  * Kannada/Tamil/Telugu are agglutinative: one word packs several English
    words' worth of morphemes, so they have FEWER words per unit of meaning.
  The denominator therefore moves in OPPOSITE directions for Indo-Aryan and
  Dravidian, which is exactly the axis the report claimed to be measuring.""")


if __name__ == "__main__":
    main()
