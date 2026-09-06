# A2 — Audit of `fertility.py` and of the metric

**Reproduce everything here with three commands:**

```bash
python partA/audit_evidence.py                      # ablations on the intern's own sample corpus
python partA/audit_evidence.py --corpus-dir corpus  # same ablations on the 997-sentence FLORES set
python partA/audit_claims.py                        # decoys, corpus alignment, the "two metrics agree" identity
```

Saved output: [`results/a2_ablation_sample.txt`](results/a2_ablation_sample.txt), [`results/a2_ablation_flores.txt`](results/a2_ablation_flores.txt), [`results/a2_claims.txt`](results/a2_claims.txt).

## 0. Baseline: the report is reproducible

```
$ cd starter_kit_original && python fertility.py \
    --corpus eng=corpus_sample/eng_sample.txt \
    --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
eng    1.27   0.226
hin    7.45   1.579
hin is 5.89x the fertility of eng (worse tokenization)
```

Exactly the table in REPORT_v0 §1. Every delta below is measured against this.

## 1. Headline

**The three code bugs are real and they barely matter.** Fixing all of them moves the headline from 5.89× to 6.11× on the intern's sample, and from 6.10× to 6.35× on a real corpus — about 4%. Anyone who "audits" this script by listing code bugs has found the least important thing wrong with it.

**What actually breaks the report is the metric and the reasoning built on it.** The denominator does not hold content constant; Finding 2's "confirmation" is an algebraic identity that cannot fail; Finding 3's root-cause claim is false by a factor of 6.5×; and the recommendation covers Dravidian traffic that was never measured and is 2× worse than the number it budgeted from.

## 2. Method

Ablation. `audit_evidence.py` re-implements v0's `analyze()` with each behaviour behind a switch, flips exactly one at a time, and reports the change in the headline hin:eng ratio. Anything measuring 0.00 is reported as harmless, not as a bug.

## 3. Code bugs — claimed, with measured effect

| # | Flaw | Sample corpus | FLORES 997 | Direction |
|---|---|---|---|---|
| **F1** | `line.split(" ")` instead of `line.split()` | 5.89 → 5.92 (**+0.03**) | 6.10 → 6.10 (**+0.00**) | understates the gap |
| **F2** | `line.lower()` before tokenizing | 5.89 → 6.06 (**+0.17**) | 6.10 → 6.33 (**+0.23**) | understates the gap |
| **F3** | mean of per-line ratios, not corpus totals | 5.89 → 5.91 (**+0.02**) | 6.10 → 6.12 (**+0.02**) | understates the gap |
| | **all three together** | 5.89 → **6.11** (+3.7%) | 6.10 → **6.35** (+4.1%) | |

### F1 — `split(" ")` splits on *each* space, not on whitespace runs

`"a  b".split(" ")` is `['a', '', 'b']` — three words, one of them empty. The empty string inflates the denominator and deflates fertility.

Evidence that the data actually triggers it: exactly one line in *each* sample file contains a double space — English line 7 (`books  in`) counts 8 words instead of 7, Hindi line 10 (`किताबें  अलमारी`) counts 6 instead of 5.

Measured effect: **+0.03× on the sample, +0.00× on FLORES** (FLORES has no double spaces). This is a genuine bug with a genuinely negligible effect on the reported number. I am claiming it as a latent correctness defect, not as a distortion of the conclusion — on a corpus with messier whitespace it would bite harder, and `split()` is free.

### F2 — `.lower()` is an asymmetric transform (largest of the three)

Lowercasing changes the token count *only for languages that have case*. Devanagari, Bengali, Kannada, Tamil and Telugu have none, so the transform rescales the English baseline and leaves every Indic number untouched.

Measured on FLORES: English fertility 1.237 → 1.283 (**+3.7%**) with lowercasing on; Hindi 7.823 → 7.823 (**unchanged, exactly**). GPT-2's BPE has merges for capitalised word forms, so `NASA`→`nasa` and `Bengaluru`→`bengaluru` fragment into more tokens.

Direction: because it inflates only the denominator language, `.lower()` makes Hindi look **better** than it is. It understates the gap by 3.7% (6.33 → 6.10).

This is the one I would call a real methodological bug rather than a typo: the comment says "so casing doesn't add noise to the comparison", and the effect is precisely to add a language-dependent term to the comparison.

### F3 — macro-averaging per-line ratios

`sum(per_line_fertility)/n` weights a 4-word sentence exactly as much as a 40-word one. The quantity a cost model needs is total tokens ÷ total words. Measured effect **+0.02×** on both corpora — real, correct to fix, immaterial here. Fixed in `fertility_v1.py`; I am not claiming it changes any conclusion.

## 4. The conceptual problem — the code is right, the quantity is wrong

> **`tokens ÷ whitespace words` does not hold anything constant across languages, so a ratio of it is not a cost ratio.**

A cross-language ratio decomposes exactly:

```
fertility_lang     T_lang / D_lang       T_lang       D_eng
--------------  =  ---------------  =   -------  ×  -------
fertility_eng      T_eng  / D_eng        T_eng        D_lang
                                        ^^^^^^^      ^^^^^^^
                                        what you     contamination, unless D is
                                        want         constant across languages
```

`D = whitespace words` is not constant. Measured on 997 sentences of **identical content** (`denominators.py`):

| denominator | eng | hin | ben | mar | kan | tam | tel | max/min | CV |
|---|---|---|---|---|---|---|---|---|---|
| parallel sentence | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00×** | **0.0%** |
| whitespace word | 21.02 | 24.68 | 18.81 | 18.12 | 15.48 | 16.18 | 16.44 | 1.59× | 16.1% |
| grapheme cluster | 125.57 | 82.65 | 78.36 | 76.33 | 86.44 | 94.75 | 73.79 | 1.70× | 18.8% |
| UTF-8 byte | 125.67 | 323.61 | 334.50 | 336.69 | 358.48 | 399.99 | 339.82 | 3.18× | 25.7% |

And the contamination term `D_eng/D_lang` that the word denominator injects into every ratio:

| | eng | hin | ben | mar | kan | tam | tel |
|---|---|---|---|---|---|---|---|
| whitespace word | 1.00 | **0.85** | 1.12 | 1.16 | **1.36** | **1.30** | **1.28** |

**The contamination runs in opposite directions for the two families.** Hindi splits postpositions into separate orthographic words, so it has *more* words per unit of meaning and the metric flatters it by 15%. Kannada, Tamil and Telugu are agglutinative — one word carries several English words' worth of morphemes — so they have *fewer* words per unit of meaning and the metric penalises them by 28–36%. The report's chosen denominator moves against the exact axis it claims to measure.

Concretely, for identical content under GPT-2: Tamil is **20.0×** English per whitespace word, but **15.4×** per parallel sentence. The word denominator overstates Tamil's real cost by 30%.

## 5. Errors in the report's reasoning (not in the code)

### C2 — Finding 2 is unfalsifiable, not corroborating

> *"The tok/char column agrees: 1.579 vs 0.226 = 7.0× worse per character, which confirms the per-word number."*

`tok/word` and `tok/char` share the same numerator. Their ratio-of-ratios is an identity containing no token counts at all:

```
(T_h/C_h)/(T_e/C_e)     C_e/W_e     chars-per-word(eng)
-------------------  =  -------  =  -------------------
(T_h/W_h)/(T_e/W_e)     C_h/W_h     chars-per-word(hin)
```

Measured on the intern's own corpus (`audit_claims.py`): `r_char/r_word = 1.2081`, `cpw(eng)/cpw(hin) = 1.2081`. Identical to four decimals, because it is algebra, not measurement.

So the 19% gap between "5.89×" and "7.0×" is neither corroboration nor noise — it is exactly the statement that Hindi packs fewer codepoints per whitespace word than English. Two metrics that cannot disagree except through a fixed property of the two languages provide one piece of evidence, not two. **The report's stated reason for confidence ("the two metrics agree, so the result is robust") is the part of the report with the least support.**

### C3 — Finding 3 is false, by a factor of 6.5×

> *"Root cause: Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."*

Directly falsifiable, and false. Same 997 sentences, same content, cost multiplier vs English per parallel sentence (full table in A3):

| tokenizer | hin | kan | tam |
|---|---|---|---|
| gpt2 (what the report used) | 7.45× | 13.59× | 15.43× |
| xlm-roberta-base | 1.26× | 1.37× | 1.35× |
| bloom-560m | 1.29× | 1.33× | 1.29× |
| sarvam-1 | **1.14×** | **1.24×** | **1.15×** |

Hindi goes from 7.45× to 1.14× — a **6.5× improvement** — with the text held byte-identical and only the tokenizer changed. It is overwhelmingly a property of the tokenizer. GPT-2's byte-level BPE has essentially no Devanagari merges and degrades to near-byte-level encoding on Indic script; a tokenizer trained with Indic data in it does not.

This matters because Finding 3 is what licenses the recommendation. "Any tokenizer will struggle" implies the only lever is routing to a different *model*. The measurement says the cheapest lever is the *vocabulary*.

### C4 — the sample corpus is not parallel, and n = 10

Documented with evidence in [CORPUS.md](CORPUS.md) and `audit_claims.py` §C4. It does not change the reported tok/word numbers. It does mean the corpus could never have supported the correct denominator, and 10 sentences of hand-written smoke text cannot support a capacity decision.

### C5 — the recommendation generalises from a language that was never measured

> *"Route all Indic traffic to a separate Indic-specialized tokenizer/model and budget 6× serving cost for Hindi. No further measurement needed."*

The report measured exactly one Indic language. Measured on GPT-2 per parallel sentence: Hindi 7.45×, **Tamil 15.43×**, Kannada 13.59×, Telugu 13.04×. A capacity plan built on "6× for Indic" under-provisions Tamil by **2.6×**. "No further measurement needed" is the single most expensive sentence in the report.

## 6. Things that look wrong but are fine — **not** bugs

I checked these because they are the sort of thing an audit reflexively flags. Each is reported with its measured effect, which is why I am *not* claiming them.

### D1 — `unicodedata.normalize("NFC", line)`

**Looks suspicious:** the script mutates the text before measuring it, which could plausibly change token counts and inflate or deflate a language.

**It is not inert** — I checked rather than asserting. It alters 90/997 Hindi lines, 586/997 Bengali, 10 Kannada, 3 Telugu, 2 Tamil, 0 English.

**Measured effect on the headline: −0.01× on 6.10×, i.e. 0.2%.** And the transform is *correct*: NFC makes two byte-different spellings of the same grapheme (composed vs decomposed nukta forms) compare and tokenize identically, which is what you want before counting anything. Removing it would make the measurement depend on which normalisation the corpus happened to ship in.

**Verdict: correct preprocessing, negligible cost. Kept in `fertility_v1.py`.**

### D2 — `import random` / `random.seed(1337)  # reproducibility`

**Looks suspicious:** a seeded RNG in a benchmark suggests sampling or subsetting somewhere, and raises the question of what varies run to run.

**Evidence:** `random.` appears exactly once in the file — the `seed()` call itself. Nothing consumes the RNG. `audit_claims.py` strips both lines, runs the original and the stripped version, and diffs stdout: **byte-identical, 244 bytes each.**

**Verdict: cargo-cult dead code. Cosmetic, not a defect in the numbers. Not claimed as a bug.**

### D3 — `add_special_tokens=False` in the HF path

Correct, and load-bearing: leaving it at the default would add 2 special tokens per line, inflating fertility on short sentences and doing so unequally across tokenizers. It also never touched the reported numbers, which were produced with `gpt2` via tiktoken. Not a bug.

## 7. Summary

| | claim | evidence | material to the conclusion? |
|---|---|---|---|
| F1 | `split(" ")` counts empty strings as words | +0.03× / +0.00× | no — latent |
| F2 | `.lower()` is asymmetric across scripts | +0.17× / +0.23× | marginally |
| F3 | macro-average of per-line ratios | +0.02× / +0.02× | no |
| **C1** | **tok/word holds nothing constant; contamination 0.85×–1.36×** | denominator table | **yes — this is the report's core error** |
| **C2** | **Finding 2's "confirmation" is an identity (1.2081 = 1.2081)** | algebra + measurement | **yes — removes the stated basis for confidence** |
| **C3** | **Finding 3 false: 7.45× → 1.14× by changing tokenizer alone** | 5-tokenizer sweep | **yes — inverts the recommendation** |
| C4 | sample corpus is not parallel; n=10 | content + r=0.51 vs 0.90 | yes for method, no for the numbers |
| C5 | "6× for Indic" never measured Dravidian; Tamil is 15.4× | A3 table | **yes — under-provisions by 2.6×** |
| D1 | NFC — **not a bug** | −0.01×, and correct | — |
| D2 | `random.seed` — **not a bug** | byte-identical output | — |
| D3 | `add_special_tokens=False` — **not a bug** | correct, unused in v0 | — |
