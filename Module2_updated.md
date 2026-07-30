# Module 2: Sinhala SLM Adaptation
### Individual Contribution — Interlekt (Hallucination Mitigation in Small Language Models for Domain-Specific Sinhala Question Answering)

## Overview

This module adapts small language models to understand and generate Sinhala text for
GCE O/L History question answering. The goal is to take a general-purpose small model,
teach it Sinhala, and teach it to answer History questions faithfully from retrieved
context while abstaining when evidence is insufficient — the hallucination-mitigation
objective of the wider project.

The adaptation pipeline: Sinhala tokenizer extension → continual pre-training (CPT) →
QA fine-tuning, applied to Llama at 1B and 3B scale so the two sizes can be compared
under one evaluation harness.

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
- The 1B and 3B pipelines (CPT + QA LoRA adapters + extended tokenizer) produced
  `isji/sinllama-1b-qa-v6-merged` and `isji/sinllama-3b-qa-v6-merged`. Full per-question
  transcripts (question, expected answer, generated answer, exact/F1 per item) are in
  `results/llama-1B-finetuned.txt` and `results/llama-3B-finetuned.txt`; aggregate
  numbers are in the Results section below.

## Results

### Llama QA fine-tuning results (153-row external test split)

| System | Exact match | Token F1 | Answerable EM | Unanswerable EM | False-answer rate |
|---|---|---|---|---|---|
| SinLlama 1B (CPT + QA-FT v6) | 22.88% | 0.505 | 12.59% | 100% | 0.00% |
| SinLlama 3B (CPT + QA-FT v6) | 32.03% | 0.650 | 22.96% | 100% | 0.00% |

These numbers are from `results/llama-1B-finetuned.txt` and
`results/llama-3B-finetuned.txt`, produced by `qa-evaluation.ipynb` after it was rebuilt
to reproduce the v6 training-time inference path exactly (same instruction text,
context/question labels, evidence-window retrieval, and grounding gate as
`qa-finetuning_v6.ipynb` — verified byte-for-byte identical prompt construction). An
earlier version of this evaluation notebook used a different prompt format the models
were never trained on and scored far lower for reasons unrelated to model quality; these
numbers are within run-to-run noise of the original v6 training-time evaluation
(22.22%/32.68% EM), confirming the corrected notebook reproduces the intended result
rather than either under- or over-stating it.

The 3B model outperforms the 1B model on every metric, but the gap (32.03% vs 22.88%
exact match) is smaller than model scale alone would suggest — both retain 100%
unanswerable accuracy with 0% false-answer rate (the grounding gate is doing its job at
both scales), and the remaining gap is concentrated in answerable-question exact match,
where dataset grounding and answer-style calibration matter more than raw capacity.

### Tokenizer fertility (Sinhala, measured on the test split)

| Tokenizer | Vocabulary | Tokens/word | Total tokens (test split) |
|---|---|---|---|
| Llama-3.2 base | 128,256 | 11.27 | 66,191 |
| `polyglots/Extended-Sinhala-LLaMA` (SinLlama) | 139,336 | 1.60 | 9,379 |

The Sinhala-extended tokenizer used for CPT is a ~7× reduction in Sinhala fertility over
the base Llama-3.2 tokenizer, which directly lowers inference cost and shortens the
sequence length the model has to reason over for a given amount of Sinhala text.

## Status

- Llama 1B and 3B CPT: complete.
- Llama QA fine-tuning: complete for both 1B and 3B checkpoints (v6 recipe), evaluated
  on the 153-row external test split.

## Key Takeaway

Continual pre-training improves Sinhala fluency, but factual History QA also depends
on a carefully curated, syllabus-grounded QA dataset — model scale alone (1B vs 3B)
does not substitute for grounding and dataset quality. Both scales reach 100%
unanswerable accuracy with a 0% false-answer rate once the grounding gate is applied,
showing that faithful abstention is achievable even at 1B parameters; the scale gap
shows up specifically in answerable-question exact match, where dataset quality and
answer-style calibration are the levers that matter most.
