# A3 — Corrected cross-language analysis

```bash
python partA/fertility_v1.py --preset all --out results/a3_full   # 5 tokenizers x 7 languages
python partA/denominators.py                                      # which denominator holds what constant
```

Saved output: [`results/a3_full.txt`](results/a3_full.txt), [`results/a3_full.csv`](results/a3_full.csv), [`results/a3_denominators.txt`](results/a3_denominators.txt).

Corpus: 997 parallel FLORES-200 sentences × 7 languages ([CORPUS.md](CORPUS.md)). Aggregation: corpus totals, no lowercasing, NFC on ([A2_AUDIT.md](A2_AUDIT.md) F1–F3).

## Tokenizers

| label | model | vocab | why |
|---|---|---|---|
| `gpt2` | tiktoken `gpt2` | 50,257 | what REPORT_v0 used — the baseline being audited |
| `qwen2.5` | `Qwen/Qwen2.5-0.5B` | 151,643 | modern large-vocab general LLM, Indic not a design target |
| `xlm-r` | `xlm-roberta-base` | 250,002 | **multilingual**, SentencePiece, 100 languages |
| `bloom` | `bigscience/bloom-560m` | 250,680 | **multilingual**, Indic languages explicitly in ROOTS |
| `sarvam-1` | `sarvamai/sarvam-1` | 68,096 | **Indic-specialised**, purpose-built for Indian languages |

Only the tokenizer is loaded; no weights are run. Fertility is a property of the vocabulary.

## Result 1 — four denominators, GPT-2 (the report's own tokenizer)

| lang | family | tok/word | tok/grapheme | tok/byte | tok/sentence | **×eng per sentence** |
|---|---|---|---|---|---|---|
| eng | Germanic | 1.228 | 0.206 | 0.205 | 25.82 | 1.00× |
| hin | Indo-Aryan | 7.796 | 2.328 | 0.595 | 192.41 | **7.45×** |
| ben | Indo-Aryan | 13.253 | 3.182 | 0.745 | 249.32 | **9.66×** |
| mar | Indo-Aryan | 11.104 | 2.636 | 0.598 | 201.19 | **7.79×** |
| kan | **Dravidian** | 22.668 | 4.059 | 0.979 | 350.82 | **13.59×** |
| tam | **Dravidian** | 24.617 | 4.204 | 0.996 | 398.36 | **15.43×** |
| tel | **Dravidian** | 20.481 | 4.562 | 0.991 | 336.65 | **13.04×** |

The four denominators do not merely differ in scale — they **rank languages differently**. Under tok/byte, Tamil (0.996) and Kannada (0.979) are near-identical; under tok/word they are 24.6 vs 22.7; under tok/grapheme Telugu is worst (4.562) but under tok/sentence it is the *best* of the three Dravidian languages (13.04×). A denominator is not a presentational choice.

## Result 2 — the same content, five tokenizers

Cost multiplier vs English for identical meaning (tokens_lang ÷ tokens_eng over the 997 parallel sentences):

| lang | gpt2 | qwen2.5 | xlm-r | bloom | **sarvam-1** |
|---|---|---|---|---|---|
| eng | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| hin | 7.45× | 4.44× | 1.26× | 1.29× | **1.14×** |
| ben | 9.66× | 5.05× | 1.38× | 1.17× | **1.11×** |
| mar | 7.79× | 4.60× | 1.22× | 1.22× | **1.07×** |
| kan | 13.59× | 6.93× | 1.37× | 1.33× | **1.24×** |
| tam | 15.43× | 6.10× | 1.35× | 1.29× | **1.15×** |
| tel | 13.04× | 7.05× | 1.33× | 1.35× | **1.16×** |

Two things fall out immediately:

1. **The gap is a tokenizer property, not a script property.** Hindi 7.45× → 1.14×; Tamil 15.43× → 1.15×. Byte-identical text; only the vocabulary changed. REPORT_v0 Finding 3 is false.
2. **The report never measured the expensive languages.** It generalised "6× for Indic" from Hindi. Under its own tokenizer Tamil is 15.43× — **2.6× worse** than the budget it recommended.

## Result 3 — the denominator question, settled

> *What is the denominator supposed to hold constant across languages?*

**The amount of meaning.** A routing/cost decision compares the price of serving *one user request* in language A versus language B. The request is a fixed quantity of intent. So the denominator must be a unit of content, and the only unit of content available is the parallel sentence.

Measured on identical content across the 997 rows:

| denominator | spread across 7 languages | CV | holds meaning constant? |
|---|---|---|---|
| parallel sentence | **1.00×** | **0.0%** | **yes, by construction** |
| Unicode codepoint | 1.17× | 5.5% | no |
| whitespace word | 1.59× | 16.1% | no |
| grapheme cluster | 1.70× | 18.8% | no |
| UTF-8 byte | 3.18× | 25.7% | no — worst of the four |

Any denominator that varies injects a factor `D_eng/D_lang` into every ratio, and that factor is indistinguishable in the output number from a real tokenizer effect.

**The word denominator can flip the sign of the answer.** Under sarvam-1, Hindi measures **0.97 tok/word relative to English — apparently *cheaper* than English.** Per parallel sentence the same data says **1.14× — more expensive.** Using v0's metric with a good Indic tokenizer, you would conclude Hindi costs you nothing, and under-provision by 17%.

And the contamination factor is a pure language property, not a tokenizer one — it is numerically identical for every tokenizer tested:

| ratio(per word) ÷ ratio(per sentence) | hin | ben | mar | kan | tam | tel |
|---|---|---|---|---|---|---|
| gpt2 | 0.85 | 1.12 | 1.16 | 1.36 | 1.30 | 1.28 |
| sarvam-1 | 0.85 | 1.12 | 1.16 | 1.36 | 1.30 | 1.28 |

Identical to two decimals across a 6.5×-different tokenizer, because it is `words_eng/words_lang` and nothing else. That is the contamination, isolated and measured.

### Why not bytes, graphemes or words

- **UTF-8 byte** is the worst option and the most tempting, because it *looks* neutral. It is not: Latin costs 1 byte/char and Indic scripts cost 3. Per byte, Tamil under GPT-2 scores 0.996 vs English 0.205 — "4.9× worse" — while the honest per-content figure is 15.4×. The byte denominator hides two thirds of the real cost by dividing by an encoding artefact.
- **Grapheme cluster** is a fair measure of *rendered text length* and is the right denominator for a UI truncation question. It is wrong here: Indic scripts pack more meaning per grapheme, spread 1.70×.
- **Whitespace word** is wrong in the specific way that matters most, because it moves in opposite directions for Indo-Aryan (0.85) and Dravidian (1.28–1.36) — the exact axis the report claimed to be measuring.

## The single number that should drive routing and cost

> **Tokens per parallel sentence relative to English, computed with the tokenizer you actually deploy.**

Two qualifiers, both load-bearing:

- **per parallel sentence** — it is the only denominator that holds user intent constant, so it is the only one that answers "what does this request cost me".
- **with the tokenizer you actually deploy** — the multiplier ranges from 15.43× to 1.15× for the same Tamil text. Quoting a fertility number without naming the tokenizer is quoting nothing.

For our current GPT-2-family stack the number is **7.5× (Hindi) to 15.4× (Tamil)**. Under an Indic-aware vocabulary it is **1.1×–1.24×**.

## Caveats

- **Input-side only.** These are the costs of tokenizing text that exists. Serving cost is dominated by *generated* tokens, and output length per language is a model behaviour this corpus cannot measure. The multipliers are sound for prefill, context budget and per-request KV footprint; treating them as total-cost multipliers assumes output scales the same way, which is untested.
- **Domain.** Wikimedia prose, not conversational or Romanised text ([CORPUS.md](CORPUS.md)).
- **Tokenizer ≠ model.** sarvam-1's vocabulary being 6.5× more efficient on Hindi says nothing about its output quality. A4 keeps those decisions apart.
- **Translationese** probably makes these multipliers floors rather than ceilings.
