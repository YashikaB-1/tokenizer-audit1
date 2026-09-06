# NOTEBOOK

Chronological log. Entries are in the order the work actually happened, including the
things that turned out wrong and the paths that went nowhere. All work in this log was done
on **2026-09-07** across a few sittings, **AI-assisted throughout** — entries 1–19 are the
audit, 20–25 are defense hardening. The first person below is the working voice of that
session, not a claim that a human did it unaided: see [AI_USAGE.md](AI_USAGE.md) for who
wrote and caught what. Every number quoted here is reproducible with the command next to it.

---

### 1 — Environment and reproduction, before anything else

Before forming any opinion about `fertility.py`, I wanted the v0 numbers to come out of
my own machine. `tiktoken` was missing (`pip install tiktoken` → 0.14.0); `transformers`
5.15.0 and `regex` were already present.

```
$ cd starter_kit_original
$ python fertility.py --corpus eng=corpus_sample/eng_sample.txt \
                      --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
eng 1.27  0.226 / hin 7.45  1.579 / hin is 5.89x
```

Exact match with REPORT_v0 §1. Good — the report is reproducible, so every claim I make
can be a measured delta against this rather than an assertion.

### 2 — First read of the script: four suspects

Reading `fertility.py` I flagged, in order of how confident I felt:

1. `line.split(" ")` — splits on single spaces, so whitespace runs produce empty "words".
2. `line.lower()` — a transform applied to all languages but only *doing* anything to cased ones.
3. `sum(per_line)/n` — macro-averaging ratios instead of corpus totals.
4. `random.seed(1337)` — nothing in the file looked random. Suspicious.

Plus one thing that felt wrong at a level above the code: **tokens-per-whitespace-word is
not a quantity you can compare across languages**, because a "word" is not the same amount
of anything in Hindi and in Kannada. Parked that; wanted measurements first.

### 3 — Probing the corpus for whether suspects 1 and 4 even fire

```
eng: lines=10 NFC_changed=0 double_space_lines=[7]
hin: lines=10 NFC_changed=0 double_space_lines=[10]
```

Two findings:

- The double space exists — exactly one line in each file. Suspect 1 does fire on this data.
- **NFC changed nothing on either sample file.** My immediate conclusion: *"NFC is inert here — that's the planted harmless thing."*

**This conclusion was wrong, and entry 7 corrects it.** Worth recording as written, because
I believed it for about an hour and it was based on a 10-sentence sample.

### 4 — Built the ablation harness rather than arguing from the code

Decision: no claimed flaw goes in the report without an ablation. `audit_evidence.py`
re-implements v0's `analyze()` with every behaviour behind a switch, flips one at a time,
prints the delta on the headline ratio. Anything measuring 0.00 gets reported as harmless.

### 5 — The result that reorganised the whole audit

```
$ python audit_evidence.py                       # the intern's own sample corpus
V0 (reproduces REPORT_v0)   1.265  7.448  5.89x
F1 split() not split(" ")   1.283  7.598  5.92x  (+0.03)
F2 no .lower()              1.229  7.448  6.06x  (+0.17)
F3 corpus-level ratio       1.253  7.403  5.91x  (+0.02)
F1+F2+F3                    1.231  7.525  6.11x  (+0.23)
```

**Surprise, and it cost me my planned framing.** I had expected the code bugs to be the
story — that fixing them would meaningfully move 5.89×. They move it to 6.11×. **3.7%.**

Revision: the audit is not "here are three bugs". The bugs are real and nearly irrelevant.
If the report's conclusion is wrong it has to be wrong for a reason that is not in the code.
Pivoted to attacking the metric and the three Findings directly, and demoted F1–F3 to a
table with honest "does this change the conclusion? no" annotations.

Also noted: F4a/F4b (grapheme and byte denominators) changed `tok/char` but left fertility
at exactly 1.265/7.448 — a good sign the harness really is isolating one variable.

### 6 — Direction check on `.lower()`, which I had backwards in my head

I assumed lowercasing would make English tokenize *more* efficiently (fewer tokens), which
would mean v0 **overstated** the gap. Measured: English fertility 1.237 → **1.283** with
lowercasing on — GPT-2 has merges for capitalised forms, so `NASA`→`nasa` and
`Bengaluru`→`bengaluru` fragment into *more* tokens. Hindi: unchanged to the last digit,
since Devanagari has no case.

So the direction is the opposite of my guess: v0's `.lower()` inflates the English
denominator and **understates** the Hindi gap. Lesson recorded because I would have written
the wrong direction into the memo if I had reasoned instead of measured.

### 7 — Corpus build, and the NFC conclusion from entry 3 falls over

Needed a real corpus. `datasets` not installed; rather than add a heavy dependency I pulled
the canonical FLORES-200 tarball directly (25.6 MB) and pinned its SHA-256 in
`build_corpus.py` so the build is verifiable. Took `dev` (997 sentences) and held `devtest`
back so there is an untouched split to re-run on during the defense. Seven languages —
eng/hin/ben/mar + kan/tam/tel — because Part C names six Indic languages and the
Indo-Aryan/Dravidian split turned out to matter (entry 10).

Re-ran the ablations on real data:

```
$ python audit_evidence.py --corpus-dir corpus
F1  +0.00   <- the split bug does nothing here; FLORES has no double spaces
F2  +0.23   <- biggest code bug, same direction as entry 6
F3  +0.02
D1 (drop NFC) -0.01,  lines altered by NFC: {'eng': 0, 'hin': 90}
```

**Two corrections to earlier beliefs.**

First: **NFC is not inert.** It alters 90/997 Hindi lines — and when I checked the other
languages, **586/997 Bengali**. My entry-3 conclusion was an artefact of a 10-sentence
sample. Revised verdict: NFC is load-bearing *and correct* — it makes composed and
decomposed spellings of the same grapheme compare equal — and its measured cost on the
headline is −0.01× (0.2%). So it is still "the thing that looks suspicious but is fine",
but for a completely different and more interesting reason than I first wrote down. Had I
stopped at the sample I would have shipped "NFC is a no-op", which is false.

Second: F1 measures **+0.00** on FLORES. It is a real bug that this data never triggers.
Reported as a latent defect, explicitly not as a distortion of the reported number.

### 8 — Nailing down the actual harmless decoy

`random.seed(1337)` needed better evidence than "I don't see any randomness". `audit_claims.py`
strips the two lines, runs original and stripped side by side, diffs stdout: **byte-identical,
244 bytes.** `random.` appears exactly once in the file — the seed call itself. That is a
provable decoy rather than an asserted one.

### 9 — The tokenizer sweep, and two dead ends

Wanted a multilingual and an Indic-specific tokenizer. Tried six:

- `google/gemma-2b` → **gated repo, no access. Dead end.**
- `google/mt5-base` → **SentencePiece parse error on download. Dead end**, didn't chase it.
- `xlm-roberta-base`, `bigscience/bloom-560m`, `sarvamai/sarvam-1`, `Qwen/Qwen2.5-0.5B` → all loaded.

Four working tokenizers plus gpt2 is more than the required two, so I stopped there rather
than debugging mt5.

### 10 — The measurement that kills Finding 3

```
$ python fertility_v1.py --preset all
COST MULTIPLIER vs eng for IDENTICAL CONTENT
lang     gpt2   qwen2.5   xlm-r   bloom  sarvam-1
hin     7.45x    4.44x   1.26x   1.29x     1.14x
tam    15.43x    6.10x   1.35x   1.29x     1.15x
```

REPORT_v0 Finding 3: *"any tokenizer will struggle... a property of the script, not the
tokenizer."* Byte-identical text; only the vocabulary changes; Hindi goes 7.45× → 1.14×.
Falsified by a factor of 6.5.

Second thing I had not anticipated: **the report never measured a Dravidian language.**
Tamil is 15.43× under the report's own tokenizer, against the "6× for Hindi" it told
leadership to budget. That is the finding with the largest actual cost attached, and it
came out of a language the report simply never looked at.

### 11 — Making the denominator argument measurable instead of rhetorical

The conceptual claim from entry 2 was still just an argument. Turned it into a measurement:
if the corpus is parallel, the 997 rows carry the same meaning in every language, so any
denominator that *varies* across languages is injecting a factor into every ratio.

```
$ python denominators.py
denominator          max/min    CV
parallel sentence      1.00x   0.0%   <- constant by construction
whitespace word        1.59x  16.1%
grapheme cluster       1.70x  18.8%
UTF-8 byte             3.18x  25.7%

contamination D_eng/D_lang:  hin 0.85 | kan 1.36 | tam 1.30 | tel 1.28
```

The result I did not expect: the contamination runs in **opposite directions** for the two
families. Hindi splits postpositions into separate words, so the word denominator *flatters*
it by 15%; Dravidian languages are agglutinative, so it *penalises* them by 28–36%. The
report's chosen denominator moves against the exact axis it claims to measure.

### 12 — A prediction, then its confirmation

If the contamination is purely `words_eng/words_lang`, it must be identical for every
tokenizer. Checked against the CSV:

```
ratio(per word)/ratio(per sentence):  hin  ben  mar  kan  tam  tel
gpt2                                  0.85 1.12 1.16 1.36 1.30 1.28
sarvam-1                              0.85 1.12 1.16 1.36 1.30 1.28
```

Identical across tokenizers that differ by 6.5× in efficiency, because it is a property of
the languages and nothing else. Cleanest confirmation in the whole of Part A.

Bonus find while reading that table: under sarvam-1, Hindi measures **0.97 tokens/word
relative to English — apparently cheaper than English** — while per parallel sentence it is
1.14×, i.e. more expensive. v0's metric can flip the sign of the answer.

### 13 — Sample corpus alignment: a test that half worked

The brief calls the sample corpora "parallel line-by-line". Reading them, they are not —
English line 4 (cricket) is Hindi line 7, English line 8 (Mysuru) is Hindi line 4, and four
English lines have no Hindi counterpart at all.

Tried to make that quantitative with a length-correlation test, using FLORES as a positive
control:

```
sample_kit eng vs hin  n=10   r = +0.511
FLORES dev eng vs hin  n=997  r = +0.915   (and +0.90..+0.92 for all six pairs)
```

**Partial dead end.** r = 0.51 is lower than 0.92 but with n = 10 it is not a real result —
the confidence interval is enormous. I kept the test in because the FLORES control is
informative, but demoted it to corroboration and made the content inspection the actual
evidence. Recording this because the tempting move was to quote "r=0.51 vs 0.92" as if it
were a finding.

### 14 — Part B: the GB/GiB fork, resolved by the log

KV cache per token was unambiguous: `2 × 28 × 8 × 128 × 2 = 114,688 B/token`. The 8 is the
GQA KV-head count; using the 24 query heads gives 344,064 and is clearly the trap.

Capacity was not unambiguous, because "24 GB" could be 24×10⁹ or 24×2³⁰:

- 24 GiB → 119,526 tokens → **29.2** sequences
- 24 GB → 105,329 tokens → **25.7** sequences

Rather than pick one, I inverted `kv_cache_util` on every row to recover the pool size the
hardware actually had. The three unsaturated long-context rows all give **exactly 105,703
tokens**. So the decimal reading is right (+0.8%) and the binary reading is out by 14%.
Pleasing: the ambiguity was resolvable from the data instead of by assertion.

### 15 — A sloppy average, caught

First pass averaged the implied pool over all unsaturated rows and got 25.9 sequences — but
the batch-1 and batch-2 rows implied 76,800 and 153,600 tokens, which is nonsense. Cause:
`kv_cache_util` is logged to two decimals, so at util = 0.01 a single ulp is ±50%. Excluded
rows below util 0.05 and said so in the output. Final: 104,468 observed vs 105,329 predicted,
**+0.82%**.

### 16 — B3: two hypotheses, one survives

Hypothesis A: `reported_tok_s` = generated tokens / wall. Hypothesis B: (prompt+gen) / wall.
Tested both against all 13 rows:

```
mean |error| vs (prompt+gen)/wall : 0.01%
mean |error| vs (gen only)/wall   : 76.3%
```

Settled. The counter bills prompt tokens as throughput, and on the 3584/512 rows **88% of
the number is prompt**. Both of §2's conclusions fall out of that one misreading, which is
what the question said to expect.

### 17 — "Two independent ways", and a cross-check I briefly mistook for a contradiction

Route 1: `24 × 512 / 61.16 = 200.92 tok/s`. Route 2: `1607.4 × 512/4096 = 200.93`. Agree to
0.005%.

I then tried a third route from `itl_ms_p50`: `24 / 0.09607 = 249.8 tok/s`, and for a few
minutes treated the mismatch as a failed check. It is not a contradiction — ITL measures
*steady-state decode rate*, which excludes prefill, while goodput divides by the whole wall
clock. The ratio 200.9/249.8 = 0.80 says 80% of wall clock is spent decoding and 20% is
prefill (ttft 500 ms) and scheduling. Reframed it as an upper-bound cross-check, which is
what it actually is, and it became useful in B2 rather than embarrassing.

### 18 — B2: the column that rules out the alternatives

The anomaly is easy (throughput peaks at batch 24, falls at 32 and 48). The mechanism needed
a column that distinguishes "preemption thrash" from "GPU ran out of compute or bandwidth":

`itl_ms_p50` goes 96.07 → 101.79 → 100.00 across a 2× batch increase — **+4%**. Per-token
decode speed is essentially flat, so nothing is saturated on compute or bandwidth; either
would show up there first. Meanwhile `wall_clock_s` goes 61.16 → 151.41 s. Time that should
be decoding (`gen_len × itl`) is 49.2 s at b24 and 51.2 s at b48 — so the decoding fraction
of wall clock collapses from **80% to 34%**.

And the batch where it breaks is 32, which needs 125% of the KV pool computed in B1 from the
model spec alone. The B1 arithmetic and the B2 anomaly are the same fact seen twice.

### 19 — Part C: the constraint I initially mis-ranked

My first instinct was that the A100-for-2-weeks was the tight constraint. Doing the
arithmetic killed that: data generation ~1 GPU-hour, LoRA training ~1.2 GPU-hours,
**under 4 hours against 336 available.**

The actual constraint is the reviewer: 30 hours total → ~900 pairwise judgements → and the
reviewer covers **2 of the 6 languages**. Rewrote the memo around that: reserve the 400-judgement
final gate first, squeeze everything else around it, and make shipping the four unreviewed
languages conditional on a judge-vs-human κ ≥ 0.6 rather than on a proxy score. Rejected
path (b) on TTFT grounds — a serial rewriter doubles time-to-first-token, and Part B shows
p50 TTFT is already 500 ms on long-context traffic.

---

### 20 — Held-out replication: does any of this survive data I never saw?

Everything so far was developed against FLORES `dev`. `devtest` (1012 disjoint sentences)
was built on day one and deliberately never opened. Added `--out-dir` to `build_corpus.py`
so the primary corpus could not be clobbered, then ran the full Part A on it.

```
$ python replicate.py
largest cross-split disagreement on any of 30 A3 multipliers: 2.54% (bloom/kan)

                     dev      devtest
gpt2 hin           7.45x        7.42x
gpt2 tam          15.43x       15.54x
sarvam-1 hin       1.14x        1.13x
F2 (.lower())      +0.23        +0.21
F1+F2+F3            6.35         6.34
NFC hin lines         90           93
```

Everything reproduces. Recorded the limit of what this shows: dev and devtest are the same
domain, drawn the same way, translated the same way. It is a **sampling** check, not a
domain-transfer check — the Romanised/conversational caveat in CORPUS.md is completely
untouched by it.

### 21 — Corpus-size counterfactual

Anticipating "what if your corpus were smaller": rebuilt at n=100 and compared.
gpt2/hin 7.31 vs 7.45 vs 7.42 (n=100 / 997 / held-out); gpt2/tam 15.07 / 15.43 / 15.54.
A 10× size change moves multipliers a few percent and inverts nothing.

### 22 — B1 sensitivity, and a nice falsification

Wrote `sensitivity.py` to sweep the two assumptions B1 takes on faith.

Overhead 1.0 → 3.0 GB: capacity stays **23–28 sequences, and batch 32 never fits**. So B2's
mechanism does not depend on the overhead being right — it only needs the ceiling between 24
and 32, which every plausible value gives. Solving backwards from the log implies 1.70 GB vs
the spec's assumed 1.60.

Better than expected: sizing the cache by the 24 **query** heads predicts 8.6 concurrent
sequences, and the log has batch 24 running with zero preemptions. So the log doesn't just
*agree* with the GQA arithmetic, it **rules out** the alternative. That is a stronger
statement than I had in B1 and I promoted it into DEFENSE.md.

### 23 — Built `probe.py`, and found a bug that would have killed the live demo

The defense says "run your script with this input I'm about to paste", so I wrote a tool that
tokenizes arbitrary text across all five tokenizers.

**First run crashed.** `UnicodeEncodeError: 'charmap' codec` — the Windows console is cp1252
and cannot encode Devanagari or Kannada. Every earlier script had been safe only because none
of them ever *printed* corpus text. Fixed by forcing UTF-8 on stdout/stderr. This would have
been a live failure in front of the panel.

### 24 — The best single piece of evidence in the audit, found by accident

Probing one Kannada word:

```
$ python probe.py -v "ಬೆಂಗಳೂರು"          # "Bengaluru", 4 graphemes, 24 UTF-8 bytes
gpt2        24 tokens   1.000 tok/byte   <- at byte level
xlm-r        1 token
bloom        1 token
sarvam-1     4 tokens
```

**GPT-2 emits exactly one token per UTF-8 byte.** Not "approximately byte-level" — 24 tokens
for 24 bytes, 1.000. That is the mechanism behind the whole Finding-3 falsification, visible
on one line, on one word, in one command. Added an automatic `<- at byte level` flag to
`probe.py` when tok/byte ≥ 0.98.

I had been asserting "GPT-2 degrades to near-byte-level on Indic script" since entry 10 on the
strength of aggregate ratios. This is the first time I actually *showed* it.

### 25 — Wrote DEFENSE.md

Consolidated the seven derivations, ten counterfactuals with measured answers, and the live
commands. Two counterfactuals I could answer only by conceding, and both are written that way:
the fp8 extrapolation, and B4's `prompt_tokens_total` prediction if vLLM swaps rather than
recomputes.

## Open threads / what I would do with more time

- **Output-side cost is unmeasured.** Every A3 multiplier is input-side. The real serving
  cost is generated tokens, and generation length per language is a model behaviour no
  tokenization study can reach. This is A4's stated biggest caveat and it is a genuine hole,
  not a hedge.
- **Romanised Indic is completely uncovered.** A large share of real Hindi/Kannada traffic is
  typed in Latin script and would tokenize nothing like the Devanagari in FLORES. I would
  expect it to change the A3 ranking, and I have no measurement either way.
- **F1's magnitude is corpus-dependent** and I only have two corpora. A messier-whitespace
  corpus would give it a bigger number; I did not construct one, because manufacturing input
  to inflate a bug's apparent severity is the opposite of the point.
- The fp8-KV prediction in B2 extrapolates a 4-point fit past its range. Flagged as the
  weaker of the two proposals, and I would want a measured run before quoting it.
- `google/mt5-base` and `google/gemma-2b` never loaded (entry 9); a gemma comparison would
  have been a useful third data point for the Finding-3 falsification.
