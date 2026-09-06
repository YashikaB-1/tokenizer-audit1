# A4 — Recommendation memo: tokenizer cost and Indic routing

**To:** Leadership / serving · **Re:** REPORT_v0 §1 · **Status:** supersedes §1 and its recommendation

## Do not ship REPORT_v0 §1

Its headline number is roughly reproducible; the three findings built on it are not. Recommendation 1 ("budget 6× for Hindi") under-provisions Tamil by 2.6×. Recommendation 2 ("no further measurement needed") is what makes it dangerous.

## Corrected headline numbers

Cost multiplier vs English **for identical content** — 997 parallel FLORES-200 sentences, tokens per parallel sentence ([A3](A3_ANALYSIS.md)):

| tokenizer | hin | ben | mar | kan | tam | tel |
|---|---|---|---|---|---|---|
| **gpt2** (current, what §1 measured) | **7.45×** | 9.66× | 7.79× | 13.59× | **15.43×** | 13.04× |
| **sarvam-1** (Indic-aware) | **1.14×** | 1.11× | 1.07× | 1.24× | **1.15×** | 1.16× |

Three corrections:

1. **"6× for Indic" is not one number.** On our current tokenizer the range is 7.5×–15.4×. §1 measured only Hindi — the *cheapest* Indic language in the set — and generalised to all of them.
2. **§1's root cause is wrong.** It blames Devanagari ("any tokenizer will struggle"). Holding the text byte-identical and changing only the vocabulary takes Hindi from 7.45× to 1.14×. It is a vocabulary problem, not a script problem.
3. **§1's stated reason for confidence is circular.** "The two metrics agree, so the result is robust" — tok/word and tok/char share a numerator; their agreement is an algebraic identity (measured: 1.2081 = 1.2081) and cannot fail.

## Recommendation

**Do not route Indic traffic to a separate model on the strength of §1. Fix the vocabulary first, then re-derive the routing question.**

1. **Now — never quote fertility without naming the tokenizer.** The deck should read "Tamil costs 15.4× English *under our current tokenizer*". The multiplier is a property of the pair.
2. **Now — re-plan capacity against 13–15×, not 6×**, for anything Tamil-, Kannada- or Telugu-facing. This is a today problem regardless of what we decide next.
3. **Next — evaluate an Indic-aware vocabulary as the primary lever.** Measured headroom is 6.5× on Hindi and 13× on Tamil, which dwarfs any routing gain and is cheaper than standing up a second serving path. Not free: it means retraining or transplanting the vocabulary, and scoping *that* is the next piece of work.
4. **Only then — revisit routing.** With an Indic-aware tokenizer the multipliers land at 1.07×–1.24× and the case for a separate Indic path largely evaporates. Deciding routing first is deciding it on numbers about to move by 6×.

## Biggest caveat

**Everything above is input-side.** These multipliers are measured on text that already exists, so they are correct for prefill, context budget and per-request KV footprint. Serving cost is dominated by tokens the model *generates*, and generation length per language is a model behaviour no tokenization study can measure. If the model is more verbose in Kannada, the true multiplier exceeds 13.59× and nothing here would show it.

Secondary: FLORES is edited Wikimedia prose. Real traffic is conversational, code-switched and often Romanised — Hindi typed in Latin script tokenizes nothing like Devanagari, and no number here covers it. Being translated *from* English also makes these figures more likely floors than ceilings.

## The one metric to monitor in production

> **Tokens per completed request, segmented by detected language, tracked as a ratio to the English median — on the live tokenizer, weekly.**

Expect ~7.5 for Hindi and ~15 for Tamil on the current stack. It is the right alarm because it is measured on **real traffic** (so it fails loudly if the domain caveat bites — if live Hindi is half Romanised, this diverges from 7.5 immediately), it is **end-to-end** (it counts generated tokens, closing the input-side gap above), it is **denominated in requests** — the production form of "per parallel sentence", the one denominator that holds user intent constant — and it is the **direct cost driver**: tokens per request × requests is the GPU bill.

Alarm if any language moves >20% week-over-week, or if a newly launched language exceeds 1.5× the ratio predicted here. Either means this analysis has stopped describing production.
