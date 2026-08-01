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

## End-to-end inference-time impact (measured on GPU, real query)

Ran one real example — the query and 5 retrieved context chunks used throughout this
project's Qwen single-inference testing — through both models on GPU
(`qwen/qwen3-4b-inference-time-comparison.ipynb`), each loading its own real tokenizer from
its own pushed Hugging Face repo. Not a synthetic benchmark, not a stub.

| | Base (`Qwen3-4B-Instruct-2507`, greedy) | Improved (`qwen3-4b-sinhala-qa-cpt-v2-merged`, greedy) | Improved (same model, beam=6 production) |
|---|---:|---:|---:|
| Tokenizer vocab | 151,669 | 176,856 | 176,856 |
| Prompt tokens | 6,008 | **1,135** | 1,135 |
| Output tokens | 255.0 | 11.0 | 6.0 |
| Mean generation time | 14.473 s | **0.747 s** | 1.944 s |
| Decode rate | 17.6 tok/s | 14.7 tok/s | 3.1 tok/s |

**Prompt-token reduction: 6,008 → 1,135, a 5.29× reduction (81.1% fewer tokens)** for the
*identical* context + question text — the cleanest number in this table, since it involves
no generation at all, just each model's own tokenizer encoding the same input. It confirms
the fertility numbers above hold at real retrieval length, not just on short isolated words.
The observed vocab sizes (151,669 vs. 176,856, a difference of exactly 25,187) also confirm
the merged, pushed model genuinely carries the extended tokenizer end to end.

**Generation time, decoding held constant (greedy vs. greedy): 14.473s → 0.747s, a 19.4×
speedup (94.8% less time).** Even switched to its production config (beam search, slower by
design), the improved model still finishes in 1.944s — **7.4× faster than the base model**.

### Two things this same example also shows — worth knowing before citing it

- **The wall-clock win is not a per-token decode-speed effect.** Decode rate is actually
  slightly *lower* for the improved model (14.7 tok/s vs. the base model's 17.6 tok/s) — a
  larger merged vocabulary (176,856 vs. 151,669) makes each decode step's LM-head projection
  marginally more expensive. The 19.4× wall-clock win instead comes entirely from needing
  far fewer tokens end to end: a 5.29× shorter prompt (less prefill compute) and far fewer
  output tokens for an equivalent answer (see next point). For a paper claim, cite the
  *prompt-token* number as the tokenizer's effect in isolation, and the *wall-clock* number
  as the combined effect of tokenizer + fine-tuning together — not the tokenizer alone.

- **The base model's 255-token output on this example is not a real answer** — it opens by
  echoing the question back rather than answering, and runs on toward its 320-token cap.
  This matches what `Qwen_model_results/Qwen3-4B-Instruct-2507.txt` already shows for
  zero-shot Qwen3-4B at full dataset scale (EM 20.26%, frequent non-extractive rambling), so
  it's not an artifact of this one run. But it means part of the 14.473s reflects that
  failure mode, not purely slower tokenization — the prompt-token number above is the more
  defensible efficiency claim of the two.

- **The beam-search (production) answer for this example — "ගම්පොළ ලෙන යි" — names a place
  that does not appear anywhere in the 5 retrieved chunks** (which name Pahiyangala/
  Bulathsinhala and Udarancha-madama, not Gampola). Greedy decoding of the *same* model on
  the *same* input instead answered "උඩරංචාමඩමේ නේවාසික ස්ථානය යි," a close, grounded match
  to the chunk-2 sentence the question is actually asking about. This timing notebook does
  not apply the grounding gate from `qwen3-4b-sinhala-qa-on-cpt.ipynb`, so nothing here
  re-scores or refuses the beam-search answer. Beam search still measures better in aggregate
  over the full 153-row test split (EM 33.99%, see
  `Qwen_model_results/qwen3-4b-sinhala-qa-cpt-v2-merged.txt`) — one example doesn't overturn
  that — but it's a concrete, worth-checking illustration of exactly the failure mode the
  grounding gate exists to catch.

### Reproduce this measurement

`qwen/qwen3-4b-inference-time-comparison.ipynb` loads both models sequentially on one GPU,
times 1 warm-up + 5 measured `model.generate()` calls per configuration with
`torch.cuda.synchronize()` bracketing each call, and reports prompt tokens, output tokens,
mean/median/stdev time, and tokens/sec for all three configurations above.

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
