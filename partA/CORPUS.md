# A1 — The evaluation corpus

**Build it yourself:** `python partA/build_corpus.py` (downloads ~26 MB, verifies SHA-256, writes `partA/corpus/`).

## What I used and why

**FLORES-200, `dev` split, 997 sentences × 7 languages.**

| | |
|---|---|
| Source | FLORES-200, `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` |
| SHA-256 | `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6` (asserted by the build script) |
| Split | `dev` (997 sentences). `devtest` is held back so I have an untouched second split to re-run on during the defense. |
| Languages | eng, hin, ben, mar (Indo-Aryan) + **kan, tam, tel (Dravidian)** |
| Domain | Wikinews 348 / Wikivoyage 348 / Wikibooks 301 — encyclopedic and news prose |
| Licence | CC-BY-SA 4.0 |
| Alignment | **line *i* is the same sentence in every file** — asserted, not assumed (`build_corpus.py` refuses to write unequal line counts) |

The requirement was ≥4 languages including English, Hindi and two Dravidian. I took seven, because the *routing* question in A4 and the language list in Part C both span Indo-Aryan and Dravidian, and the whole point of A3 is that those two families behave in **opposite** directions under the metric the report used. Three Dravidian languages instead of two costs nothing and makes that claim a pattern rather than an anecdote.

**Why parallel matters more than size here.** The single most important property is not sentence count, it is that the 997 rows carry *the same meaning* in every language. That is the only thing that makes a denominator available which holds content constant across languages, which is the entire argument of A3. A large non-parallel corpus could not answer the question at all.

## Preprocessing

Deliberately almost none:

- Blank lines dropped — and dropped **from every language together**, so alignment survives. (`fertility.py`'s reader drops blanks per file, which would silently shear the alignment.)
- **No lowercasing** — see A2/F2.
- **No** punctuation stripping, digit normalisation, or de-duplication.
- Unicode NFC is applied at measurement time, not in the corpus files, so the files on disk stay byte-identical to FLORES.

## Corpus size

Per language, over all 997 sentences:

| lang | family | script | sentences | words | grapheme clusters | UTF-8 bytes |
|---|---|---|---|---|---|---|
| eng | Germanic | Latin | 997 | 20,954 | 125,194 | 125,290 |
| hin | Indo-Aryan | Devanagari | 997 | 24,607 | 82,404 | 322,640 |
| ben | Indo-Aryan | Bengali | 997 | 18,756 | 78,123 | 333,496 |
| mar | Indo-Aryan | Devanagari | 997 | 18,065 | 76,101 | 335,677 |
| kan | **Dravidian** | Kannada | 997 | 15,430 | 86,177 | 357,408 |
| tam | **Dravidian** | Tamil | 997 | 16,134 | 94,467 | 398,795 |
| tel | **Dravidian** | Telugu | 997 | 16,388 | 73,568 | 338,804 |

Read the `words` column across rows: **identical content, 15,430 to 24,607 whitespace words depending on language.** Hindi has 17% *more* words than English; Kannada has 26% *fewer*. That 1.59× spread is the A3 result in embryo.

## The starter kit's sample corpus is not a corpus

`corpus_sample/` is described in the brief as "parallel line-by-line". It is not, and I checked rather than assuming (`audit_claims.py`, section C4):

- English line 4 ("children playing cricket") is Hindi line **7**; English line 5 is Hindi line **6**; English line 8 is Hindi line **4**; English line 7 is Hindi line **10**.
- 4 of the 10 English lines have no Hindi counterpart at all (airport traffic, quarterly review, NASA/ISRO, GPU cluster).
- Length correlation eng↔hin is r = +0.51 on the sample, against +0.90 to +0.92 for every FLORES pair (positive control). With n=10 that correlation alone is weak evidence; the content inspection above is the real evidence, and the correlation is corroboration only.

This does not change the tok/word figures in REPORT_v0 — those are per-file aggregates and never touch the pairing. It does mean the sample can support no per-sentence claim whatsoever.

## What this corpus cannot tell you

FLORES is professionally translated Wikimedia prose: news, travel guides and textbooks. Every number downstream is conditional on that, and four limitations follow.

**Domain.** This is edited, formal, written register. Our production traffic is conversational assistant text — short turns, code-switching, Roman-script Hindi and Kannada, product names, emoji, typos. Romanised Indic input in particular would land in a completely different part of every tokenizer's vocabulary and could invert the ranking in A3; nothing here measures it. That caveat has teeth for Part C specifically, whose entire goal is *casual* register — the corpus that validates a tokenizer choice here is the wrong corpus for validating a casualness intervention.

**Translationese.** FLORES targets are translations *from English*, so they inherit English sentence segmentation and information packing. This plausibly makes the non-English sides look more English-shaped than native text would, which if anything **understates** the true cross-language token gap. The direction of that bias is worth stating: my A3 multipliers are more likely floors than ceilings.

**Sample size.** 997 sentences is enough to make the aggregate ratios stable but not to say anything about tails — the long or unusual inputs that actually blow up a context budget. I report no per-sentence variance claims and no p-values.

**Generation is unmeasured.** Everything here tokenizes *existing text*. Serving cost is dominated by tokens the model *generates*, and generation length in a given language is a property of the model's decoding behaviour, not of this corpus. A3's multipliers are input-side and prefill-side; A4 says explicitly what would have to be measured to extend them to output cost.
