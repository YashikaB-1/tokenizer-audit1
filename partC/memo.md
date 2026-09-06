# Part C — Decision memo: casual register in six Indic languages

**Recommendation: path (a), SFT via LoRA — gated on first beating a serious path (c) prompt baseline, measured on day 1. Reject path (b).**

## Assumptions

1. Served model is **FLM-4B-Instruct** (the Part B stack): 4.2 B params, fp16, `max_model_len` 4096.
2. "Too formal" is a **register** problem, not a correctness one — answers are right and stiff.
3. **No external API budget** ⇒ synthetic pairs must be self-generated on our own A100. No stronger teacher to distil from.
4. The reviewer covers **Hindi and Kannada only — 2 of 6 languages.** Tamil, Telugu, Bengali, Marathi get no native-speaker signal unless we build a proxy. **This, not the GPU, is the binding constraint.**
5. Casualness is judged as a **blind pairwise preference**; humans rank register far more consistently than they rate it.

## Back-of-envelope arithmetic

**GPU is not the constraint.** Data: 6 langs × 3,000 = **18,000 pairs**, ~150 output tokens each ≈ 2.7 M tokens, doubled to ~5.4 M by a casualize rewrite pass; at ~2,500 tok/s batched on an A100 (≈3–4× the L4 goodput measured in Part B) that is **~1 GPU-hour**. LoRA training: 18 k × ~400 tokens × 3 epochs ≈ 21.6 M tokens; 6·N·T ≈ 5.4×10¹⁷ FLOPs at ~40% MFU on 312 TFLOPS ≈ **1.2 GPU-hours**. **Under 4 GPU-hours against 336 available.**

**Reviewer time is the constraint.** 30 h total ÷ ~2 min per blind judgement ≈ **900 judgements for the whole project**:

| | Hindi | Kannada | purpose |
|---|---|---|---|
| Day-1 baseline + prompt variants | 100 | 100 | the number everything is measured against |
| Week-1 SFT pilot | 100 | 50 | kill/continue gate |
| Week-2 iteration | 100 | 100 | one correction round |
| **Final launch gate (reserved first)** | **200** | **200** | the only ship/no-ship judgements |

At n=200, a 50%→65% shift is detectable at ~4σ (SE ≈ 3.5 pts). That is why 400 judgements are reserved before anything else is scheduled.

**Serving cost.** (a) merges the LoRA: **zero added latency or GPU.** (b) adds a serial ≤1 B pass over the full response — ~+25% GPU and, decisively, it **doubles TTFT**, since nothing streams until the main model finishes. Part B already shows p50 TTFT at 500 ms on long-context traffic. **(b) is rejected on latency architecture, not quality.**

One cross-link to Part A: Kannada costs **13.6×** and Tamil **15.4×** as many tokens as equivalent English on our current tokenizer, so in-language few-shot exemplars are expensive on every request. Keep the system prompt in English.

## Success metric and threshold

> **Blind pairwise native-speaker preference vs current production, 200 held-out realistic prompts per language: ≥ 65% win rate in *both* Hindi and Kannada, with ≤ 5% adequacy regression.**

Adequacy is a separate binary — *does it still answer the question, nothing lost or invented?* — because the obvious failure mode is buying casualness with meaning.

For the four unreviewed languages: an LLM-as-judge **calibrated against the human labels on Hindi and Kannada**, shipped only if it reaches **Cohen's κ ≥ 0.6** there. Below that we have no evidence for those four and they stay on the current prompt. Stating this now prevents the week-3 conversation where a proxy gets promoted to evidence because the deadline arrived.

## Kill criterion

- **End of day 7:** if the LoRA checkpoint does not beat the day-1 prompt baseline by **≥ 10 points of Hindi win rate** (n=100), abandon (a) and ship (c). Hindi has our best data and best coverage — if SFT cannot win there, it will not win on the four languages we cannot see.
- **Any checkpoint:** adequacy regression **> 5%** kills the run regardless of casualness gain.
- **End of day 10:** judge-human **κ < 0.6** ⇒ stop trying to ship six languages; scope the launch to two.

## First experiment, day 1

**Measure the baseline and the prompt ceiling before training anything.** Assemble 100 held-out realistic prompts per language (short conversational turns, not FLORES prose). Generate four conditions with the *current* model: (0) production prompt, (1) English system prompt instructing casual register, (2) +3 in-language exemplars, (3) +explicit "avoid textbook forms". Reviewer spends **2 h** on blind pairs, Hindi and Kannada, condition 0 vs best of 1–3. Run the judge over the same pairs and compute κ.

Cost: 2 of 30 reviewer hours, ~15 GPU-minutes. Returns the baseline number, whether **(c) alone already clears 65%** (in which case we finish in week 1 and hand back the A100), a calibrated judge or early warning that we won't have one, and a real measurement of reviewer throughput — currently the assumption the entire plan rests on.
