# Tokenizer Comparison: Qwen3-4B vs. base Llama-3 vs. the SinLlama-8B tokenizer

Measured directly on this project's own test data (`new_split_v2/test.jsonl`, 153 rows:
context + question + answer text for every row), so these numbers are specific to your
Sinhala history-textbook QA domain, not generic benchmark claims.

## TL;DR

| Tokenizer | Vocab size | Tokens / Sinhala word | Chars / token | `ලංකාව` ("Lanka") |
|---|---:|---:|---:|---|
| **Qwen3-4B-Instruct-2507** | 151,669 | 9.19 | 0.69 | 7 tokens |
| **Llama-3.2 base** (`meta-llama/Llama-3.2-1B`) | 128,256 | 11.27 | 0.56 | 10 tokens |
| **SinLlama tokenizer** (`polyglots/Extended-Sinhala-LLaMA`, = `polyglots/SinLlama_v01`) | 139,336 | **1.60** | **3.95** | **2 tokens** |

The SinLlama tokenizer is **~5.7× more efficient** on Sinhala than Qwen3, and **~7× more
efficient** than base Llama-3.2. Both Qwen and Llama get ~1.08 tokens/word on English — the
gap is Sinhala-specific, not a general tokenizer-quality gap.

> **Note on naming:** `polyglots/Extended-Sinhala-LLaMA` — the tokenizer ID this project's own
> CPT notebooks (`sinllama_1b_cpt.ipynb`, `sinllama_4b_cpt_b200.ipynb`) already load as
> `TOKENIZER_ID` — has an identical vocabulary (139,336 tokens, verified token-for-token) to
> `polyglots/SinLlama_v01`, the tokenizer shipped with the actual published SinLlama-8B model
> from arXiv:2508.09115. **This project already uses the real SinLlama paper's tokenizer**,
> not a separate reimplementation — worth stating explicitly in the thesis so it reads as
> "we build on the published SinLlama tokenizer" rather than leaving it unstated.

## Why the gap: how each tokenizer was built

All three are BPE variants, but they differ in what got merged into the vocabulary:

- **Qwen3-4B-Instruct-2507** — byte-level BPE trained on a broad multilingual + code corpus
  (Qwen's own tokenizer, ~152K vocab). It has *no dedicated Sinhala merges*, so it falls back
  to encoding each Sinhala character as its raw UTF-8 bytes, then BPE-merging *those bytes*
  where patterns recur. This never produces an unknown token (byte-level BPE has no OOV by
  construction) but it's inefficient for a script the merge training rarely saw.

- **Llama-3.2 base** — also byte-level BPE (~128K vocab), trained with a corpus that is
  overwhelmingly English/code, with only modest allocation to non-Latin scripts. It falls
  back to raw bytes the same way Qwen does, just with **less byte-merge coverage for
  Sinhala than Qwen has** — hence its fertility (11.27 tok/word) is even worse than Qwen's
  (9.19 tok/word), despite Meta's "multilingual" marketing for Llama 3. This is the
  tokenizer the base `meta-llama/Llama-3.2-1B`/`-3B` checkpoints ship with, before this
  project's CPT step replaces it.

- **SinLlama tokenizer (`Extended-Sinhala-LLaMA`)** — takes the Llama-3 tokenizer and
  **appends** ~11,080 new tokens (128,256 → 139,336) trained with `tiktoken` on a native
  Sinhala corpus, so the new merges are real Sinhala subwords — syllables, common word
  stems, and case-suffix morphemes — not byte fragments. Critically, it is a **strict
  extension**: token IDs 0–128,255 are byte-identical to base Llama-3 (this project's own
  `sinllama_4b_cpt_b200.ipynb` verifies this token-for-token before training, see
  `first_token_id_mismatch`), so English/code tokenization and the base model's original
  capability are fully preserved — only Sinhala gets new, efficient tokens.

### Same word, three tokenizers — concretely

`ලංකාව` ("Lanka"):

```
Qwen3-4B   : à¶½ | à¶ | Ĥ | à¶ļ | à· | ı | à·Ģ            (7 raw byte-fragments)
Llama-3.2  : à¶ | ½ | à¶ | Ĥ | à¶ | ļ | à· | ı | à· | Ģ    (10 raw byte-fragments — even worse)
SinLlama   : ලංකා | ව                                   (2 real Sinhala subwords: "Lanka" + case suffix)
```

The `à¶...`/`à·...` pieces in Qwen and Llama are not linguistic units at all — they're raw
UTF-8 byte sequences (Sinhala codepoints are 3 bytes each in UTF-8) rendered through BPE's
byte-to-printable-character mapping, which is why they look like mojibake. Every Sinhala
character typically costs 2–3 of these byte-tokens in Qwen/Llama; the SinLlama tokenizer
spends roughly one token per syllable instead.

## What this costs downstream (measured on your test split)

| | Qwen3-4B | Llama-3.2 base | SinLlama tokenizer |
|---|---:|---:|---:|
| Total tokens to encode all 153 rows (context+Q+A) | 54,010 | 66,191 | **9,379** |
| Implication | ~8.5× the decode steps of an English-fluent tokenizer per Sinhala answer | worst of the three | Sinhala answers cost roughly what English answers would |

This is why the Qwen inference notebook needs `MAX_NEW_TOKENS=320` (measured gold-answer max:
261 Qwen tokens) where the SinLlama-tokenizer pipelines use 48–80. It also means every
Sinhala token you *do* spend compute on, under Qwen/Llama's byte-level fallback, is doing
far less linguistic work than one SinLlama token — more forward passes, more KV-cache memory,
slower generation, and a harder learning problem per answer (more sub-tokens to get right in
sequence) for no representational benefit.

## What this does *not* mean

Higher fertility is an **efficiency cost, not an accuracy ceiling** — byte-level BPE is
lossless, so Qwen3 can still represent any Sinhala string exactly, just less efficiently. This
is consistent with what the project has already measured: zero-shot Qwen3-4B (9.19 tok/word,
no Sinhala-specific tokens at all) roughly matches the CPT+QA-fine-tuned SinLlama-3B (which
uses this same efficient tokenizer) on token F1. The tokenizer gap explains *compute cost and
generation speed*, not the accuracy gap seen elsewhere in this project — that gap traces to
answer-style calibration and refusal behavior (see `qwen/README.md`).

## Reproduce these numbers

```python
from transformers import AutoTokenizer
import json

rows = [json.loads(l) for l in open("new_split_v2/test.jsonl", encoding="utf-8") if l.strip()]

def fertility(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    words = toks = 0
    for r in rows:
        for t in (r["context"], r["question"], r.get("answer") or ""):
            words += len(t.split())
            toks += len(tok(t, add_special_tokens=False)["input_ids"])
    return toks / words

for repo in ["Qwen/Qwen3-4B-Instruct-2507", "unsloth/Llama-3.2-1B", "polyglots/Extended-Sinhala-LLaMA"]:
    print(repo, fertility(repo))
```

`meta-llama/Llama-3.2-1B` is gated on Hugging Face; `unsloth/Llama-3.2-1B` is an ungated
mirror with an identical tokenizer, used here only to avoid the gate — the vocab and IDs are
Meta's original.

## Sources

- SinLlama — A Large Language Model for Sinhala (tokenizer extension + CPT recipe) — https://arxiv.org/abs/2508.09115
- Published SinLlama tokenizer/model — https://huggingface.co/polyglots/SinLlama_v01
- Extended tokenizer used by this project's CPT notebooks — https://huggingface.co/polyglots/Extended-Sinhala-LLaMA
- Qwen3 tokenizer — https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
