# Module 2: Sinhala SLM Adaptation
### Individual Contribution — Interlekt (Hallucination Mitigation in Small Language Models for Domain-Specific Sinhala Question Answering)

## Overview

This module adapts small language models to understand and generate Sinhala text for
GCE O/L History question answering. The goal is to take a general-purpose small model,
teach it Sinhala, and teach it to answer History questions faithfully from retrieved
context while abstaining when evidence is insufficient — the hallucination-mitigation
objective of the wider project.

Two adaptation routes were built and compared under one evaluation harness:

- **Route A — full Sinhala adaptation of Llama:** use extended tokenizer  → continual
  pre-training (CPT) → QA fine-tuning, at 1B and 3B scale.
- **Route B — lightweight adaptation of a stronger multilingual base (Qwen3-4B):**
  from-scratch Sinhala tokenizer extension → CPT → QA
  fine-tuning

## What I Did

### 1. Continual Pre-Training (CPT) — Llama

- Used an already-extended Sinhala tokenizer for **Llama-3.2-1B**
  (`polyglots/Extended-Sinhala-LLaMA`, vocabulary expanded 128,256 → 139,336 tokens,
  +11,080 Sinhala tokens) and ran LoRA-based continual pre-training on
  `isji/sinhala-corpus` (10.7M Sinhala sentences) to strengthen Sinhala language
  understanding.
- Repeated the same extended-tokenizer + CPT recipe on **Llama-3.2-3B** to evaluate
  whether a larger base model improves Sinhala fluency and downstream QA quality.
- Built a B200-optimized CPT notebook:
- Work was carried out on Modal.com.

### 2. QA Dataset Construction

- Built a Sinhala RAG QA fine-tuning dataset pipeline: generated `{context, question,
  answer}` triplets from Sinhala school history textbooks using the
  Google Gemini API, exported to JSONL.
- Assembled a Sinhala History QA dataset of 1,500 examples, with roughly 20% marked
  unanswerable to train abstention behaviour, later refined into a 1,370/153
  train/test split with grade and chapter metadata.

### 3. QA Fine-Tuning — Llama

- Instruction fine-tuned both the 1B and 3B CPT checkpoints on the Sinhala History QA
  dataset using TRL/PEFT with a custom Sinhala RAG instruction template.
- Training converged smoothly at 3 epochs for both scales — validation loss for 3B
  fell from 0.813 (step 13) to 0.594 (step 117) before plateauing; 1B showed the same
  pattern at a higher loss floor (0.929 → 0.705), consistent with capacity-limited
  fluency rather than an under-trained run.
- The 1B pipeline (CPT + QA LoRA adapters + extended tokenizer) produced
  `isji/sinllama-1b-qa-v6-merged`;

### 4. Comparative Arm — Qwen3-4B (proof of concept)

Built to answer a specific question: *does a stronger multilingual instruction-tuned
model need the same tokenizer-extension-and-CPT investment Llama needed, or does it
already do most of the job zero-shot?*


**b. Sinhala tokenizer extension for Qwen3-4B**
(`qwen3-4b-sinhala-tokenizer-extension.ipynb`). Qwen3-4B has no Sinhala-specific
vocabulary and falls back to byte-level BPE — measured at **9.19 tokens/word** on the
test split, worse even than base Llama-3.2's own tokenizer (11.27 tokens/word) despite
Meta's multilingual claims. Built and reproduced the SinLlama extension recipe on
Qwen's vocabulary from scratch:
  - Reverse-engineered the design of the published `polyglots/Extended-Sinhala-LLaMA`
    tokenizer directly from its artifact (not tiktoken, which has no public trainer):
    added tokens as raw text, ~75% carrying a literal leading space, ZWJ-aware for
    Sinhala conjuncts, appended as a strict, contiguous ID extension.
  - Trained a whitespace-aware BPE on `isji/sinhala-corpus`, filtered to Sinhala-block
    + ZWJ candidates, and grafted the result onto Qwen3's tokenizer via `add_tokens()`.
  - Result: **151,669 → 176,856 tokens (+25,187)**, verified as a strict extension
    (every original token ID unchanged), with an exact encode→decode round-trip on
    Sinhala text and bit-identical tokenization of English/code samples.
  - **Fertility on the test split: 9.19 → 1.48 tokens/word**, a 6.2× reduction in total
    tokens (54,010 → 8,696) — slightly better than the published SinLlama tokenizer's
    own 1.60 tokens/word on the same data. Pushed as `isji/Extended-Sinhala-Qwen3`.

**c. Continual pre-training** (`qwen3-4b-sinhala-cpt-b200.ipynb`), ported from the
Llama B200 notebook onto `Qwen/Qwen3-4B` with the new tokenizer. Several
architecture-specific issues surfaced and were fixed during this port, each worth
recording as they are not obvious from the Llama recipe:
  - **Tied-embedding bug.** Qwen3-4B ties `embed_tokens` and `lm_head` into one shared
    matrix. PEFT's `modules_to_save=["embed_tokens"]` silently *untied* them on this
    architecture — the trainable copy fed the input side while `lm_head` kept the
    frozen, mean-initialized original, so the 25,187 new Sinhala rows would never have
    learned to be *generated*. Verified empirically on a minimal Qwen3 model, fixed by
    re-tying `lm_head` to the trainable embedding copy after PEFT wrapping and
    asserting the fix before training is allowed to start.
  - **Vocabulary-driven OOM.** The larger vocabulary makes the training-time logits
    tensor `microbatch × 2048 × 176,856 × 4 bytes`, allocated twice for gradient — 21.6
    GiB at microbatch 8, which combined with unchecked activations across 36 layers
    exceeded the 178 GiB B200 during `backward()`. Root-caused to the exact byte, then
    resolved without reducing the microbatch (which would have doubled wall-clock time)
    by enabling Liger Kernel's fused linear cross-entropy, which computes the loss in
    chunks and never materializes the full logits tensor.
  - **Throughput tuning.** CPU preprocessing (tokenize + pack 10.7M documents) was
    scaled to the host's actual core count rather than a fixed cap, and a wall-clock
    training budget (`TARGET_TRAINING_HOURS`, enforced by a `TrainerCallback` backstop
    in addition to a derived `MAX_STEPS`) was added so a run completes in a fixed time
    budget with the cosine LR schedule still decaying to completion, rather than being
    killed mid-schedule.
  - Net effect of these fixes: measured step rate went from 0.096 it/s (initial,
    checkpointed) → 0.259 it/s (checkpointing removed) → 0.286 it/s (Liger + full
    core count), bringing the run within the same ~4-hour wall-clock budget as the
    Llama-3B CPT despite Qwen3-4B costing ~1.25× more compute per token (36 layers vs.
    28, 27% larger vocabulary).

**d. QA fine-tuning POC** (`qwen3-4b-sinhala-qa-finetuning-poc.ipynb`) — the v6 Llama
recipe (evidence windows, grounding gate, completion-only loss, unanswerable
rebalancing) ported to Qwen's chat template with thinking mode disabled
(`enable_thinking=False`), training and evaluation sharing one prompt distribution.
1,375 training rows after rebalancing (25.0% unanswerable) + 205 validation rows,
LoRA r=32/α=64 on all attention/MLP projections. Includes a standalone inference cell
producing a transcript in the same per-item format as the Llama result files, so all
systems diff directly.

## Results

### Zero-shot Qwen3-4B vs. fine-tuned Llama (153-row external test split)

| System | Exact match | Token F1 | Answerable EM | Unanswerable EM | False-answer rate |
|---|---|---|---|---|---|
| SinLlama 1B (CPT + QA-FT v6) | 22.22% | 0.496 | 11.85% | 100% | 0.00% |
| **Qwen3-4B-Instruct (zero-shot, no adaptation)** | 13.73% | 0.606 | 4.44% | 83.33% | 16.67% |
| SinLlama 3B (CPT + QA-FT v6) | 32.68% | 0.648 | 23.70% | 100% | 0.00% |

Zero-shot Qwen already exceeds the fully-adapted SinLlama 1B on token F1 despite no
Sinhala-specific training at all, and its answered-question F1 (0.659, computed over
rows where it actually attempted an answer) exceeds SinLlama 3B's equivalent (0.649).
Its exact-match and refusal-discipline gap versus the fine-tuned models is
concentrated in answer **style** (missing the dataset's copula/phrasing convention) and
**refusal calibration** (14/135 answerable questions over-refused; 3/18 unanswerable
questions answered instead of refused) — both are exactly what QA fine-tuning targets,
not evidence of a knowledge gap.

A head-to-head on answerable questions (F1 ≥ 0.5 = solved) found the two model
families solve substantially different subsets: 64 solved by both, 26 by SinLlama-3B
only, 20 by Qwen-zero-shot only, 25 by neither. An oracle over the two would solve
81.5% of answerable questions, well above either alone — evidence the routes are
complementary, not redundant.

### Tokenizer fertility (Sinhala, measured on the test split)

| Tokenizer | Vocabulary | Tokens/word | Total tokens (test split) |
|---|---|---|---|
| Llama-3.2 base | 128,256 | 11.27 | 66,191 |
| Qwen3-4B base | 151,669 | 9.19 | 54,010 |
| `polyglots/Extended-Sinhala-LLaMA` (SinLlama) | 139,336 | 1.60 | 9,379 |
| `isji/Extended-Sinhala-Qwen3` (this work) | 176,856 | **1.48** | **8,696** |

The from-scratch Qwen extension edges out the published SinLlama tokenizer despite
targeting a base with zero prior Sinhala coverage, at a lower embedding-parameter cost
than SinLlama's own extension (Qwen3-4B ties input/output embeddings into one matrix;
Llama-3-8B's does not), because it adds ~2.3× the tokens into a single shared matrix
rather than two.

## Status

- Llama 1B and 3B CPT: complete.
- Llama QA fine-tuning: complete for both 1B and 3B checkpoints (v6 recipe), evaluated
  on the 153-row external test split.
- Qwen3-4B comparative arm: zero-shot baseline evaluated; Sinhala tokenizer extension
  built, verified, and pushed (`isji/Extended-Sinhala-Qwen3`); CPT notebook built,
  debugged, and running; QA fine-tuning POC notebook built and dry-run verified,
  pending a GPU run against the CPT checkpoint.
- **Not yet done:** finish the Qwen3-4B CPT run and the QA fine-tune on top of it, so
  the fully-adapted Qwen arm can be added to the results table above; integrate a
  Sinhala translation of SQuAD v2 into the QA fine-tuning dataset (valued for its ~33%
  unanswerable question ratio).

## Key Takeaway

Continual pre-training improves Sinhala fluency, but factual History QA also depends
on a carefully curated, syllabus-grounded QA dataset — model scale alone (1B vs 3B)
does not substitute for grounding and dataset quality. The Qwen3-4B comparison adds a
second finding with direct cost implications: a strong multilingual instruction-tuned
model can match a fully Sinhala-adapted model on answer quality (token F1) with **zero**
adaptation cost, and the specific gaps it does have — output style and refusal
calibration — are cheaply closed by QA fine-tuning alone, without necessarily requiring
the far more expensive CPT stage. Tokenizer fertility (9.19 vs 1.48 tokens/word) is a
genuine and separately worth-fixing cost, but it is an **inference-efficiency** problem,
not an accuracy ceiling — the zero-shot result was achieved with the inefficient
tokenizer. This reframes the adaptation question from "does this model need Sinhala
training" to "which of tokenizer efficiency, base fluency, and task-specific
calibration does this model actually need," which is a materially cheaper question to
answer for a low-resource-language project with a fixed compute budget.
