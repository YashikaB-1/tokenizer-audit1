# AI_USAGE.md

## AI assistance

This submission was developed with substantial AI assistance, primarily using Claude Code / Claude Opus during the implementation and analysis stages. I used AI as a coding, experimentation, and writing assistant throughout the project.

I am disclosing this explicitly because the important question for this assignment is not whether AI was used, but whether I understand and can defend the resulting work.

## What I contributed

My primary contribution was directing the audit rather than manually implementing every component.

I defined the questions the work needed to answer, asked the AI to audit the existing report and deliverables against the assignment requirements, and directed follow-up experiments when an initial result needed stronger evidence. I also reviewed the resulting analyses and used the experimental findings to decide which claims were sufficiently supported, which needed qualification, and which should not be presented as conclusions.

In particular, the work was organized around the assignment's evidence rule: a suspected flaw should not be treated as a finding until its effect can be measured. This led to targeted ablations, denominator comparisons, tokenizer comparisons, capacity calculations, and cross-checks rather than relying solely on inspection of the original code.

I also made the final decisions about what conclusions and caveats belong in the submission. The resulting repository therefore reflects an AI-assisted investigation that I directed and reviewed, rather than code that I independently wrote line by line.

## Where AI helped most

The largest benefit was turning qualitative suspicions into measurable experiments.

For Part A, the ablation approach made it possible to isolate individual preprocessing and implementation choices and measure their effect on the reported result. This was substantially more useful than simply producing a list of possible bugs.

AI was also particularly useful for repetitive or mechanical work: constructing the evaluation corpus, running tokenizer comparisons, checking denominators, fitting formulas to the serving log, and performing arithmetic cross-checks.

Another useful contribution was generating alternative hypotheses and testing them. Several conclusions changed only after a later experiment contradicted an earlier interpretation.

## Where AI was wrong or initially misleading

AI output was not treated as automatically correct.

One important example was NFC normalization. An initial experiment on the tiny starter corpus showed no changed lines, leading to the conclusion that NFC was effectively inert. Testing the larger evaluation corpus contradicted that conclusion: NFC changed 90 Hindi lines and 586 Bengali lines. The corrected analysis therefore treats NFC as load-bearing, with a measured effect of approximately -0.2% on the headline result.

The treatment of `.lower()` was another example. The initial reasoning predicted that lowercasing would make English tokenize more efficiently. Measurement showed the opposite: English fertility increased from 1.237 to 1.283. This changed the interpretation of the resulting language gap.

The initial length-correlation result from the ten-sentence sample was also too readily interpreted. With only ten observations, r = 0.51 was not strong evidence. The larger FLORES result (r = 0.92), together with direct content inspection, provided substantially stronger support.

There were similar issues in Part B. An initial KV-pool calculation included clearly anomalous rows whose rounded utilization values produced implausible implied pool sizes. Those rows were investigated rather than blindly averaged. Likewise, the difference between the ITL-derived rate and measured goodput was initially treated as a possible inconsistency before the role of prefill was examined.

These failures are important because they demonstrate why I did not treat an AI-generated explanation as sufficient evidence. The experiments and cross-checks are what ultimately determined which conclusions were retained.

## What I independently need to be able to defend

Before submission, I am treating the following as required knowledge rather than material I can simply attribute to the AI:

1. KV-cache bytes/token and the resulting concurrency calculation.
2. The definition of `reported_tok_s` and why it differs from goodput.
3. The denominator problem in the cross-language comparison.
4. The tokenizer/vocabulary experiment that falsifies the original explanation of the Hindi result.
5. The B2 capacity/preemption mechanism and its relationship to the batch-32 break.
6. The measured direction of the `.lower()` effect.
7. The assumptions and arithmetic behind the Part C reviewer/data/compute estimates.

I will remove or qualify any claim that I cannot reproduce or explain during the defense.

## Important limitation

The code and first drafts of the documents were substantially AI-generated. I did not independently re-derive every numerical result at the time it was produced.

Consequently, I am not presenting AI assistance as equivalent to independent authorship. My responsibility is to understand the load-bearing calculations, evidence, assumptions, and conclusions before submission and to remove anything I cannot defend.

The final submission should therefore be read as an AI-assisted audit that I directed and reviewed, with the defense serving as the test of my understanding.