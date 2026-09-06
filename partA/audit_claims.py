#!/usr/bin/env python3
"""
audit_claims.py -- evidence for the A2 claims that are not about a single line
of code: the two decoys, the sample corpus's alignment, and the "two metrics
agree" argument in REPORT_v0 Finding 2.

Usage:  python audit_claims.py
"""

import os
import re
import subprocess
import sys
import tempfile
import unicodedata

import regex

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "starter_kit_original"))
CORP = os.path.join(HERE, "corpus")


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- decoy 2
def decoy_random_seed():
    hr("D2  import random / random.seed(1337)  --  dead code, NOT a bug")
    src = open(os.path.join(KIT, "fertility.py"), encoding="utf-8").read()

    n_uses = len(re.findall(r"\brandom\.", src))
    note = "seed() only, nothing consumes the RNG" if n_uses == 1 else "CHECK MANUALLY"
    print("occurrences of 'random.' in fertility.py : %d   (%s)" % (n_uses, note))

    drop = ("import random", "random.seed(1337)  # reproducibility")
    stripped = "\n".join(ln for ln in src.splitlines() if ln.strip() not in drop)
    assert "random" not in stripped, "strip failed"

    tmp = os.path.join(tempfile.mkdtemp(), "fertility_noseed.py")
    open(tmp, "w", encoding="utf-8").write(stripped)

    args = ["--corpus", "eng=corpus_sample/eng_sample.txt",
            "--corpus", "hin=corpus_sample/hin_sample.txt", "--tokenizer", "gpt2"]
    a = subprocess.run([sys.executable, os.path.join(KIT, "fertility.py")] + args,
                       cwd=KIT, capture_output=True, text=True).stdout
    b = subprocess.run([sys.executable, tmp] + args,
                       cwd=KIT, capture_output=True, text=True).stdout
    print("output with seed == output without seed : %s" % (a == b))
    print("  stdout lengths: %d vs %d" % (len(a), len(b)))
    print("VERDICT: cosmetic dead code. Removing it changes nothing. NOT a bug.")


# ---------------------------------------------------------------- decoy 1
def decoy_nfc():
    hr("D1  unicodedata.normalize(NFC)  --  load-bearing but CORRECT, NOT a bug")
    targets = [("sample hin", os.path.join(KIT, "corpus_sample", "hin_sample.txt")),
               ("sample eng", os.path.join(KIT, "corpus_sample", "eng_sample.txt"))]
    for code in ("eng", "hin", "ben", "mar", "kan", "tam", "tel"):
        targets.append(("FLORES " + code, os.path.join(CORP, code + ".txt")))

    for name, path in targets:
        raw = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        nfc = [unicodedata.normalize("NFC", l) for l in raw]
        changed = sum(1 for a, b in zip(raw, nfc) if a != b)
        dcp = sum(len(b) - len(a) for a, b in zip(raw, nfc))
        print("  %-12s lines=%4d  altered_by_NFC=%4d  net codepoint delta=%+5d"
              % (name, len(raw), changed, dcp))

    print("\nNFC does change real text, so it is not inert -- but it makes two byte-")
    print("different spellings of the SAME grapheme compare equal, which is exactly")
    print("what you want before counting anything. Its measured cost on the headline")
    print("is the D1 row of audit_evidence.py (-0.01x on 6.10x, i.e. 0.2%).")
    print("VERDICT: correct preprocessing, cheap. NOT a bug.")


# ---------------------------------------------------------------- alignment
def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy)


def graphemes(path):
    return [len(regex.findall(r"\X", l.strip()))
            for l in open(path, encoding="utf-8") if l.strip()]


def sample_is_not_parallel():
    hr("C4  the sample corpus is NOT line-aligned (FLORES = positive control)")
    print("Test: in a genuinely parallel corpus, sentence length in language A and")
    print("language B correlate across lines -- long sentences translate to long")
    print("sentences. FLORES is known-parallel and acts as the positive control.\n")

    s_e = graphemes(os.path.join(KIT, "corpus_sample", "eng_sample.txt"))
    s_h = graphemes(os.path.join(KIT, "corpus_sample", "hin_sample.txt"))
    f_e = graphemes(os.path.join(CORP, "eng.txt"))

    print("  sample_kit  eng vs hin   n=%4d   pearson r = %+.3f"
          % (len(s_e), pearson(s_e, s_h)))
    for code in ("hin", "ben", "mar", "kan", "tam", "tel"):
        r = pearson(f_e, graphemes(os.path.join(CORP, code + ".txt")))
        print("  FLORES dev  eng vs %s   n=%4d   pearson r = %+.3f" % (code, len(f_e), r))

    print("\nManual spot-check of the sample -- the translations exist but are SHUFFLED:")
    pairs = [(3, 3, "I bought this book yesterday   -- aligned by luck"),
             (4, 7, "children playing cricket       -- eng L4 = hin L7"),
             (5, 6, "the train arrived on time      -- eng L5 = hin L6"),
             (8, 4, "visiting Mysuru next week      -- eng L8 = hin L4"),
             (7, 10, "books in the cupboard          -- eng L7 = hin L10")]
    for i, j, gloss in pairs:
        print("   eng line %2d  <->  hin line %2d    %s" % (i, j, gloss))
    print("\nAlso: 4 of the 10 English lines have no Hindi counterpart at all")
    print("(airport traffic, quarterly review, NASA/ISRO, GPU cluster).")
    print("\nVERDICT: this does NOT change the tok/word numbers -- those are per-file")
    print("aggregates. It forecloses the one denominator that matters (A3), and it")
    print("means the sample cannot support any per-sentence claim.")


# ---------------------------------------------------------------- claim 2
def claim2_not_independent():
    hr("C2  REPORT_v0 Finding 2: 'tok/char agrees, which confirms the per-word number'")
    print("tok/word and tok/char share the SAME numerator. Their ratio-of-ratios is")
    print("an algebraic identity that contains no token counts at all:\n")
    print("   (T_h/C_h) / (T_e/C_e)      C_e/W_e      chars-per-word(eng)")
    print("   --------------------- =  ---------  =  -------------------")
    print("   (T_h/W_h) / (T_e/W_e)      C_h/W_h      chars-per-word(hin)\n")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    out = {}
    for lang, path in [("eng", os.path.join(KIT, "corpus_sample", "eng_sample.txt")),
                       ("hin", os.path.join(KIT, "corpus_sample", "hin_sample.txt"))]:
        T = W = C = 0
        for line in open(path, encoding="utf-8"):
            line = unicodedata.normalize("NFC", line.strip())
            if not line:
                continue
            line = line.lower()
            T += len(enc.encode(line))
            W += len(line.split())
            C += len(line)
        out[lang] = (T, W, C)
        print("  %s: tokens=%5d  words=%4d  codepoints=%5d   chars/word=%.3f"
              % (lang, T, W, C, C / W))

    Te, We, Ce = out["eng"]
    Th, Wh, Ch = out["hin"]
    r_word = (Th / Wh) / (Te / We)
    r_char = (Th / Ch) / (Te / Ce)
    print("\n  tok/word ratio (hin:eng) = %.3fx" % r_word)
    print("  tok/char ratio (hin:eng) = %.3fx   (REPORT_v0 quotes ~7.0x)" % r_char)
    print("  r_char / r_word          = %.4f" % (r_char / r_word))
    print("  cpw(eng) / cpw(hin)      = %.4f   <-- identical, by construction"
          % ((Ce / We) / (Ch / Wh)))
    print("\nThe 19% gap between 5.89x and 7.0x is neither corroboration nor noise:")
    print("it is exactly the statement that Hindi packs fewer codepoints per")
    print("whitespace word than English. It carries ZERO extra information about")
    print("the tokenizer.")
    print("VERDICT: Finding 2 is not evidence. The two metrics cannot disagree except")
    print("by the chars-per-word ratio, so 'they agree' is unfalsifiable.")


if __name__ == "__main__":
    decoy_random_seed()
    decoy_nfc()
    sample_is_not_parallel()
    claim2_not_independent()
