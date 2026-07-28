# Qwen3-4B for Grounded Sinhala QA — Improvement Roadmap & Research Framing

This document covers (1) where the project currently stands, (2) a failure analysis of the
zero-shot Qwen baseline, (3) a prioritized plan to improve it, and (4) how to present the work
as a valid research contribution for a final-year undergraduate thesis on **hallucination
mitigation for low-resource (Sinhala) question answering**.

---

## 1. Where the project stands

Two pipelines have been evaluated on the same 153-row external test split
(`new_split_v2/test.jsonl`: 135 answerable, 18 unanswerable):

| Metric | SinLlama 3B (CPT + QA-FT v6) | Qwen3-4B-Instruct (zero-shot) |
|---|---|---|
| Exact match (all) | **32.68 %** (50/153) | 13.73 % (21/153) |
| Mean token F1 (all) | **0.648** | 0.606 |
| Answerable exact | **23.70 %** (32/135) | 4.44 % (6/135) |
| Answerable mean F1 | 0.601 | 0.576 |
| Answerable rows with F1 ≥ 0.5 | 66.7 % (90/135) | 62.2 % (84/135) |
| Mean F1 on *attempted* answers | 0.649 | **0.659** |
| Unanswerable exact (refusal) | **100 %** (18/18) | 83.3 % (15/18) |
| False-answer rate (unanswerable) | **0 %** | 16.7 % (3/18) |
| Over-abstention (answerable) | **7.4 %** (10/135, all gate) | 12.6 % (17/135, 14 raw + 3 gate) |

> Numbers computed from `qwen/qwen-4B-instruct.txt` and `qwen/3B-v6-results.txt`.
> ⚠️ Data-hygiene note: the footer of `3B-v6-results.txt` references
> `/tmp/sinllama_1b_qa_v7_results.jsonl`. Verify which run produced that file (3B-v6 vs 1B-v7)
> before citing it in the thesis — a mislabeled baseline is a viva-killer.

**Head-to-head on answerable rows (F1 ≥ 0.5 = "solved"):** both solve 64, Llama-only 26,
Qwen-only 20, neither 25. The two models are *complementary* — 46 questions are solved by
exactly one of them. An oracle over the two would solve 110/135 (81.5 %), far above either
alone. This is direct evidence that Qwen is not simply "worse" — it has different knowledge
and different failure modes, and there is large headroom.

### What the zero-shot result actually means

The SinLlama pipeline required: tokenizer extension → continual pre-training (CPT) on a
Sinhala corpus → QA LoRA fine-tuning → merge. Qwen3-4B-Instruct got **none** of that and
still matches it on token F1 (0.606 vs 0.648) and *beats* it on attempted-answer F1
(0.659 vs 0.649). The gap is concentrated in exactly three places, all fixable:

1. **Style mismatch, not knowledge mismatch.** Most of the exact-match gap is formatting.
   Qwen produces the right content without the dataset's reference style (the final
   copula `ය.` and dataset phrasing conventions):
   - Ref: `කැප්පෙටිපොළ හා මඩුගල්ලේ ය.` → Pred: `කැප්පෙටිපොළ හා මඩුගල්ලේ දෙදෙනා` (F1 0.8, EM ✗)
   - Ref: `රාජාධිරාජසිංහ රජු ය.` → Pred: `රාජාධිරාජසිංහ රජු` (F1 1.0, EM ✗)

   A short fine-tune on the 1,370-row train split fixes style. EM should roughly triple from
   this alone (this is what QA-FT did for Llama).

2. **Over-refusal on answerable questions** (14/135 raw refusals vs 0 for the tuned Llama).
   The instruct model is too cautious. Fine-tuning with answerable examples calibrates this.

3. **Hallucination on unanswerable questions** (3/18 vs 0). This is the interesting one —
   see below.

### The key research insight already in your data

All 3 of Qwen's false answers on unanswerable questions had **lexical support ≥ 0.8** — they
sailed through the grounding gate:

> Q: *"විමලධර්මසූරිය රජු මිය ගියේ කුමන රෝහලේ ප්‍රතිකාර ලබමින් සිටියදීද?"* (which **hospital**?)
> Pred: *"...ක්‍රිස්තු වර්ෂ 1604 දී උණ රෝගයකින් විය."* (support 0.8 — copied from context, but
> answers *when/how he died*, not *which hospital*)

The lexical gate verifies **provenance** ("is this text from the context?") but not
**relevance** ("does this text answer the question?"). A model can hallucinate an answer by
copying grounded-but-irrelevant context. This failure class — call it *relevance
hallucination* — is invisible to lexical grounding by construction, and it motivates the two
strongest components of your contribution: refusal-aware training and question-conditioned
answer verification. Lead with this observation in the thesis; it turns your gate from "a
heuristic we used" into "a baseline whose measured blind spot motivates our method."

---

## 2. Prioritized improvement plan

Ordered by expected-impact-per-GPU-hour. Steps 1–3 are the thesis core; 4–5 are stretch.

### P0 — Refusal-aware LoRA fine-tuning of Qwen3-4B-Instruct  *(the big win)*

Supervised LoRA fine-tuning on `new_split_v2/train.jsonl` (1,370 rows: 1,213 answerable,
157 unanswerable), **through the Qwen chat template** with completion-only loss (train only on
the assistant turn). This is the same recipe class as your Llama v6 QA-FT, adapted:

- **Prompt:** the same system prompt + `Context / Question / Answer` user turn used in
  `qwen3-4b-instruct-2507-test-split-inference.ipynb`, so train and eval distributions match.
- **Targets:** canonical answers; every unanswerable target is the canonical refusal string
  `මෙම ප්‍රශ්නයට පිළිතුරු දීමට ප්‍රමාණවත් තොරතුරු නොමැත.`
- **Refusal-aware augmentation (R-Tuning adapted — this is the novelty-bearing step):**
  don't rely only on the 157 natural unanswerable rows (11.5 % of train). Synthesize hard
  negatives whose *target is the refusal*:
  - **Context-swap negatives:** pair a question with a context from a different chapter
    (topically distant) → refuse.
  - **Near-miss negatives (the hard ones):** pair a question with a context from the *same
    chapter* that does not contain the answer — this directly attacks the relevance-
    hallucination failure mode found above.
  - **Attribute-mismatch negatives:** questions asking for an attribute the context doesn't
    state about an entity it *does* mention (your "hospital" example is exactly this class).
  Keep the final answerable:unanswerable ratio around 70:30. R-Tuning showed refusal
  training generalizes as a meta-skill; GRAIT refines *which* negatives to pick — you can
  cite both and describe your selection as textbook-structure-aware (grade/chapter metadata
  gives you a principled distance measure most RAIT papers don't have).
- **LoRA config (community-standard for Qwen3-4B):** r=32–64, α=2r, dropout 0.05, targets
  `q/k/v/o/gate/up/down_proj`, bf16, lr 1e-4–2e-4 cosine, 2–3 epochs, effective batch ≈ 32.
  Fits on a single L40S/A100 (~10–20 GB with LoRA); TRL `SFTTrainer` with
  `completion_only_loss=True` works as in your v6 notebook, or Unsloth for speed.
- **Expected effect:** EM into the 30–45 % range (style alignment), false-answer rate toward
  0, over-refusal down. Every fabrication needs the model to *overcome trained refusal*, not
  just slip past a lexical filter.

### P1 — Context-aware decoding (CAD)  *(training-free, cheap, publishable ablation)*

Shi et al.'s context-aware decoding contrasts the output distribution **with** and
**without** the context and sharpens the difference:

```
logits_final = (1 + α) · logits(prompt with context) − α · logits(prompt without context)
```

α ≈ 0.5–1.0. It needs two forward passes per token, no training, and directly targets the
mechanism behind hallucination-under-context: the model trusting parametric memory over the
evidence. Reported gains of ~14 % factuality for LLaMA on summarization; applying it to
Sinhala grounded QA is unexplored territory. Implement it as a `LogitsProcessor` in your
inference notebook and evaluate zero-shot ± CAD and fine-tuned ± CAD (2×2). Even a *negative*
result ("CAD helps zero-shot but is redundant after refusal-aware FT") is a real finding.
DoLa is the related layer-contrast alternative if you want a second training-free baseline.

### P2 — Question-conditioned answer verification  *(fixes the gate's measured blind spot)*

Replace/augment the lexical provenance gate with a relevance check. Cheapest options first:

1. **Question-term coverage:** require that the *sentence in the context supporting the
   answer* also covers the question's focus terms (the interrogative's target — "රෝහල",
   "කීයක්", "කවුද" types). Pure-Python extension of your existing gate.
2. **Self-verification pass:** a second prompt to the same model — "Does this answer, with
   this context, actually answer this question? Answer only yes/no" — abstain on "no". Two
   generations per item; no extra model.
3. **Multilingual NLI / embedding cross-encoder** (e.g., a multilingual MiniLM) scoring
   (question + answer) vs context — heavier, only if 1–2 underperform.

Evaluate all abstention mechanisms (trained refusal / lexical gate / relevance gate /
combinations) as **selective QA**: plot risk–coverage curves (x = fraction of questions
answered, y = error rate among answered). This is the standard, rigorous way to present
abstention trade-offs, and it converts your "gate vs no gate" choice into an analysis section.

### P3 — Optional stretch (only if time remains)

- **Preference optimization (DPO):** build preference pairs where *chosen* = correct answer or
  refusal, *rejected* = the model's own hallucinated outputs harvested from validation runs.
  Small, targeted, and very thesis-friendly ("the model learns from its own hallucinations").
- **Ensemble/oracle analysis:** you already have 46 complementary wins; a simple
  confidence-based router between SinLlama-3B and Qwen would quantify the headroom.
- **Qwen3.5 / larger Qwen:** a one-run comparison to check whether the recipe transfers.

### What about extending the Qwen tokenizer (the SinLlama recipe)?

Short answer: technically yes, strategically no — but *measure and discuss it*, because the
question is guaranteed to come up at the viva, and the measurement strengthens your story.

**The measured facts** (Qwen3 tokenizer on `new_split_v2/test.jsonl`):

| | Qwen3-4B tokenizer |
|---|---|
| Sinhala fertility | **9.19 tokens/word** (0.69 chars/token) |
| English reference | 1.08 tokens/word |
| Single words | `ලංකාව` = 7 tokens, `ප්‍රශ්නය` = 12 tokens |
| Consequence | gold answers reach 261 tokens → `MAX_NEW_TOKENS=320`; ~8.5× more decode steps per Sinhala answer than per English answer |

Qwen has essentially **no Sinhala vocabulary** and falls back to byte-level BPE — and *still*
matches the CPT-adapted Llama 3B on token F1. Byte-level BPE is lossless (no unknown tokens),
so poor tokenization is an **efficiency tax, not an accuracy ceiling** — your zero-shot result
is the proof. That is itself a reportable finding, because the SinLlama paper
(arXiv:2508.09115) merged a tiktoken-trained Sinhala tokenizer into Llama-3-8B and ran CPT on
10.7M sentences (304M tokens), but reports **no fertility numbers and no ablation** isolating
the tokenizer's contribution from CPT's. Your experiment separates what theirs conflates.

**Why not just extend Qwen's tokenizer?** New tokens get randomly-initialized embeddings that
are useless until trained — extension *forces* continual pre-training, which is exactly the
cost your Qwen arm exists to avoid. Worse, you'd have to extend Qwen3-4B-**Base** (CPT on raw
Sinhala text erodes instruction-following via catastrophic forgetting), then redo instruction
tuning yourself — at which point you've rebuilt the SinLlama pipeline on a different base and
destroyed the two-pipeline comparison that is your thesis narrative.

**If you want a tokenizer-shaped experiment at undergrad scale** (stretch / future work, not
the critical path): lightweight vocabulary adaptation *without* full CPT — add the top-N
Sinhala tokens, initialize their embeddings as weighted combinations of the subword embeddings
they replace (FOCUS-style; WECHSEL and Zero-Shot Tokenizer Transfer are the related methods),
train **only the embedding rows** (transformer frozen) on a modest Sinhala corpus, then apply
your refusal-aware LoRA on top. Honest framing: this tests whether the fertility tax can be
removed cheaply; with only ~300M tokens of available Sinhala data it may well *underperform*
byte-level fallback on accuracy, and either outcome is a valid ablation result.

**Recommended treatment in the thesis:** a "Tokenizer analysis" subsection reporting the
fertility table above, the inference-cost implication, and the argument for why extension was
scoped out — plus RQ1's zero-shot result as evidence that multilingual instruct models
tolerate extreme fertility. That converts a potential examiner attack into a contribution.

### Metric & data fixes to do alongside (cheap, improves the thesis regardless)

- **Report a style-insensitive EM** (strip the final copula `ය./යි.` and punctuation on both
  sides — your Llama inference notebook already does this; the v6 normalizer does not) *and*
  strict EM. Right now part of the Llama-vs-Qwen EM gap is a normalizer artifact.
- **18 unanswerable test rows is too few** for a hallucination-mitigation headline claim
  (each row = 5.6 percentage points). Expand to ≥ 50 using the same negative-synthesis
  procedure as P0 (held-out chapters only, manually checked — an afternoon of work), and
  categorize them (context-swap / near-miss / attribute-mismatch) so you can report *which
  hallucination type* each method fixes. This is a benchmark contribution in itself.
- **Significance tests:** McNemar's test on EM (paired, same items), paired bootstrap for F1.
  With n=153 these are one-liners and reviewers/examiners will ask for them.
- **Fixed seeds + 2–3 runs** for the fine-tuning experiments; report mean ± sd.

---

## 3. Framing it as a research contribution

### The gap (verified against the literature, July 2026)

- Sinhala QA resources exist — SiQuAD (translated SQuAD 1.1, ~16k pairs, best F1 73 %,
  span-extraction-style evaluation — paywalled, verify the model list via the library before
  citing), SinhalaMMLU (multiple-choice evaluation only), and SinLlama (Llama-3-8B +
  tokenizer extension + CPT — evaluated on *classification*, not QA). **All of it is
  answerable-only; none studies hallucination, abstention, or unanswerable questions, and no
  found work performs generative QA fine-tuning natively in Sinhala.**
- Closest prior work — cite and differentiate: Kiridana & Dias (Univ. of Moratuwa),
  *"Developing a Question Answering System for the Sri Lankan School Education System"* —
  proposes school-curriculum Sinhala QA (your exact domain) but is a **proposal only** (no
  implementation, no dataset, no results), and its design pivots through English translation
  with an encoder model (XLM-R + LoRA + FAISS, mBART back-translation), with no unanswerable
  or hallucination handling. It independently validates the problem; everything it leaves
  open is what this project delivers —
  https://dl.lib.uom.lk/server/api/core/bitstreams/e41844f8-2566-417c-a92e-d1dded3eb8ef/content
- Multilingual hallucination work (Mu-SHROOM SemEval-2025, MultiWikiQHalluA covering 215
  languages, CausalAbstain for multilingual abstention) consistently reports that
  hallucination is *worse* in low-resource languages — but Sinhala grounded-QA abstention is
  not covered by any of them.
- R-Tuning/RAIT and CAD are established for English; neither has a published application to
  Sinhala (or, as far as searchable, any Indo-Aryan low-resource language) grounded QA.

So your honest, defensible novelty claims are:

### Contribution list (thesis wording)

1. **A Sinhala grounded-QA benchmark with unanswerable questions** — the first
   SQuAD-2.0-style (answerable + typed-unanswerable) evaluation set for Sinhala, built from
   grade 6–11 history textbooks, with a faithfulness-oriented metric suite (EM/F1,
   false-answer rate, over-abstention, no-answer P/R/F1, risk–coverage).
2. **The first systematic study of hallucination mitigation for Sinhala QA**, comparing two
   pipeline families under one benchmark:
   *(a)* Sinhala-adapted English LLM (tokenizer extension + CPT + QA-FT: your SinLlama 1B/3B), vs
   *(b)* multilingual instruction-tuned LLM (Qwen3-4B) + lightweight faithfulness adaptations.
3. **A faithfulness-adaptation recipe for (b):** refusal-aware LoRA fine-tuning with
   textbook-structure-aware negative synthesis (context-swap / near-miss / attribute-mismatch),
   optionally combined with context-aware decoding and a relevance-verification gate —
   evaluated component-by-component.
4. **An empirical finding about lexical grounding:** provenance-based gating cannot detect
   relevance hallucinations (measured: 3/3 residual hallucinations passed the gate at
   support ≥ 0.8), motivating question-conditioned verification.
5. **A compute-cost analysis:** the multilingual-instruct route reaches equal or better
   faithfulness without any continual pre-training — a practically important result for
   low-resource-language practitioners with limited GPU budgets.

### The thesis narrative (one paragraph you can adapt)

> Continual pre-training is the established route to low-resource-language LLMs, but it is
> the most expensive step in the pipeline. We show that for grounded Sinhala QA, a
> mid-size multilingual instruction-tuned model already matches a CPT-adapted model of
> comparable size at zero training cost — but hallucinates on unanswerable questions in a
> way that lexical grounding checks provably cannot catch. We close this gap with a
> lightweight, refusal-aware LoRA recipe whose negative examples are synthesized from
> textbook structure, plus training-free context-aware decoding, and evaluate on the first
> Sinhala QA benchmark with typed unanswerable questions. The result is a
> hallucination-mitigation pipeline for low-resource QA that needs hours, not weeks, of GPU
> time.

### What *not* to claim (viva safety)

- Don't claim to have invented refusal-aware tuning, CAD, or grounding gates — cite
  R-Tuning, GRAIT, and Shi et al. and claim the **adaptation, combination, and first
  application to Sinhala**, plus the benchmark and the empirical findings. Undergraduate
  (and most published) contributions are exactly this shape; examiners punish overclaiming
  far more than modest claims.
- Don't compare "Qwen 4B" vs "Llama 3B" as if size-matched — acknowledge the parameter gap
  and lean on the *compute-to-adapt* comparison (CPT+FT hours vs LoRA-only hours), which is
  the axis your evidence actually supports. Include your 1B results as the size-scaling data
  point.
- Don't rest the headline on 18 unanswerable items — expand the set (see metric fixes).
- Don't claim QA fine-tuning of Llama (1B/3B) as a novel contribution — Sinhala QA
  fine-tuning is already published (SiQuAD: mono/cross/multilingual, F1 73 %), and SinLlama
  already showed CPT-then-fine-tune beats plain Llama on Sinhala tasks. Your 1B/3B runs are
  the **reference systems** (row H) that make the real contributions measurable. The
  defensible differentiators, with every qualifier load-bearing: *generative* (not
  extractive), *unanswerable questions + refusal* (SiQuAD is answerable-only SQuAD 1.1),
  *original textbook-derived data* (not translated), *hallucination-focused evaluation*
  (first for Sinhala).

### Research questions (put these verbatim in the proposal/thesis)

- **RQ1:** Can a multilingual instruction-tuned LLM match a Sinhala-CPT-adapted LLM on
  grounded Sinhala QA without any Sinhala pre-training? *(already answered: ≈ yes on F1;
  your zero-shot run is the evidence)*
- **RQ2:** Does refusal-aware fine-tuning with structure-aware synthetic negatives reduce
  the false-answer rate on unanswerable questions without increasing over-abstention?
- **RQ3:** Does context-aware decoding further improve faithfulness, before and after
  fine-tuning?
- **RQ4:** Which abstention mechanism — trained refusal, lexical provenance gating, or
  question-conditioned relevance verification — gives the best risk–coverage trade-off?

### Experiment matrix (the ablation table your results chapter is built from)

| # | System | Trains? | Answers RQ |
|---|---|---|---|
| A | Qwen3-4B zero-shot (current) | – | RQ1 baseline |
| B | A + lexical gate (current) | – | RQ4 |
| C | A + CAD | – | RQ3 |
| D | Qwen3-4B + plain QA LoRA (no synthetic negatives) | ✓ | RQ2 ablation |
| E | Qwen3-4B + refusal-aware LoRA (full recipe) | ✓ | RQ2 |
| F | E + CAD | ✓ | RQ3 |
| G | E + relevance verification | ✓ | RQ4 |
| H | SinLlama 1B / 3B v6 (existing results) | ✓ | RQ1 reference |

Report per system: strict EM, style-insensitive EM, token F1, answerable EM/F1,
false-answer rate, over-abstention, no-answer P/R/F1, risk–coverage AUC; McNemar / bootstrap
significance vs baseline A and reference H. D-vs-E is the ablation that proves the synthetic
negatives matter — don't skip D, it's what makes E a *contribution* rather than a recipe.

---

## 4. Practical notes

- **Order of work:** expand unanswerable test set → P0 (D then E) → re-run inference
  notebook → P1 → P2 → risk–coverage analysis → (P3 if time). P0 with the existing v6-style
  TRL code is 1–2 days of work; CAD is ~50 lines.
- **Reuse the eval harness:** `qwen3-4b-instruct-2507-test-split-inference.ipynb` already
  computes every metric above and writes the v6-format report; point it at each fine-tuned
  checkpoint so all systems share one scorer. Keep `MAX_NEW_TOKENS=320` for Qwen — gold
  answers reach 261 Qwen tokens (no Sinhala vocabulary → byte-level BPE).
- **Versions:** Qwen3 needs `transformers>=4.51`; TRL ≥ 0.26 for `completion_only_loss`.
  Left-padding for batched generation. Greedy decoding for all reported numbers.
- **Honesty in reporting:** publish the exact prompts, seeds, and the metric definitions in
  an appendix; note that test contexts are gold (retrieval is out of scope) — that scoping
  sentence preempts the "but what about RAG?" question.

## 5. References

- R-Tuning: Instructing LLMs to Say "I Don't Know" (NAACL 2024 Outstanding Paper) — https://aclanthology.org/2024.naacl-long.394/ · code: https://github.com/shizhediao/R-Tuning
- GRAIT: Gradient-Driven Refusal-Aware Instruction Tuning — https://arxiv.org/abs/2502.05911
- Trusting Your Evidence: Context-Aware Decoding (Shi et al., NAACL 2024) — https://aclanthology.org/2024.naacl-short.69/
- No-Worse Context-Aware Decoding (2026) — https://arxiv.org/abs/2604.16686
- DoLa: Decoding by Contrasting Layers (ICLR 2024) — https://arxiv.org/abs/2309.03883
- SinLlama — an LLM for Sinhala (tokenizer extension + CPT) — https://arxiv.org/abs/2508.09115
- SiQuAD: QA in a Low-Resource Language: Dataset & Adaptations for Sinhala — https://link.springer.com/chapter/10.1007/978-3-032-04339-9_22
- SinhalaMMLU benchmark — https://arxiv.org/abs/2509.03162
- Mu-SHROOM: SemEval-2025 Task 3, multilingual hallucination — https://helsinki-nlp.github.io/shroom/
- MultiWikiQHalluA: multilingual hallucination benchmark, 215 languages (2026) — https://arxiv.org/abs/2605.02504
- CausalAbstain: multilingual trustworthy abstention — https://arxiv.org/abs/2506.00519
- Hallucination in conversations for low-resource languages — https://arxiv.org/abs/2507.22720
- Unsloth Qwen3 fine-tuning guide — https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
- FOCUS: embedding initialization for new vocabularies (EMNLP 2023) — https://aclanthology.org/2023.emnlp-main.829/
- WECHSEL: cross-lingual transfer of embeddings (NAACL 2022) — https://aclanthology.org/2022.naacl-main.293/
- Zero-Shot Tokenizer Transfer (NeurIPS 2024) — https://arxiv.org/abs/2405.07883
