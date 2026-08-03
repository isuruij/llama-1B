# CPT Notebook Comparison: 1B vs. 3B vs. Qwen3-4B

How `sinllama_1b_cpt.ipynb`, `sinllama_4b_cpt_b200.ipynb` (trains Llama-3.2-**3B**, despite
the filename), and `qwen/qwen3-4b-sinhala-cpt-b200.ipynb` differ. Verified directly against
each notebook's cells — not from memory.

## 1. Target model, tokenizer, and dataset

| | 1B | 3B (`sinllama_4b_cpt_b200.ipynb`) | Qwen3-4B |
|---|---|---|---|
| Base model | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-3B` | `Qwen/Qwen3-4B` |
| Tokenizer | `polyglots/Extended-Sinhala-LLaMA` (published SinLlama) | same | `isji/Extended-Sinhala-Qwen3` (built for this project) |
| Corpus | `isji/sinhala-corpus` | same | same |

## 2. Training mechanism — the biggest structural difference

- **1B**: clones an external repo (`Chinese-LLaMA-Alpaca`, `github.com/isuruij/llama-1B`)
  and shells out to its `run_clm_pt_with_peft.py` via `!torchrun`. All hyperparameters are
  CLI flags, not Python config. Section 3 of this notebook is literally "Clone your repo."
- **3B & Qwen**: fully self-contained — no cloned repo, everything done in-notebook via
  `AutoModelForCausalLM` + `peft.get_peft_model` + a `Trainer` object. This is a real
  architecture change between the 1B notebook and the two later ones, not just cosmetic —
  there is no "Clone your repo" step in either later notebook because there's nothing left
  to clone.

## 3. Sequence length / batch composition

All three land on the same 65,536 tokens/update, just assembled differently:

| | 1B | 3B | Qwen |
|---|---|---|---|
| Sequence length | 512 | 2,048 | 2,048 |
| Batch × grad-accum | 16 × 8 | 8 × 4 | 8 × 4 (Liger active) / 4 × 8 (fallback) |
| Precision | plain `bf16` | FP32 master + BF16 autocast | FP32 master + BF16 autocast |

## 4. LoRA config

Rank/alpha/dropout/target_modules are identical across all three: `r=8, alpha=32,
dropout=0.05`, same 7 projection layers (`q/k/v/o/gate/up/down_proj`). What differs is
`modules_to_save` handling:

- **1B**: unconditionally saves both `embed_tokens` and `lm_head` (simple, always-both,
  set via the `--modules_to_save embed_tokens,lm_head` CLI flag).
- **3B**: conditional — checks whether embeddings are actually tied first
  (`input_weight.data_ptr() == output_weight.data_ptr()`), saves just `embed_tokens` if so,
  both if not.
- **Qwen**: same conditional logic, **plus an extra manual re-tie step** the other two
  don't have — a block that detects and fixes a PEFT bug where `modules_to_save` silently
  breaks a tied `lm_head`/`embed_tokens` pair (PEFT's `ModulesToSaveWrapper` deep-copies the
  wrapped module), then re-ties `lm_head.weight` to the trainable embedding copy and
  asserts the tie holds before allowing training to start. This is Qwen-specific because
  `tie_word_embeddings=True` on Qwen3-4B is what exposed the bug.

## 5. Qwen-only additions (none present in either Llama notebook)

- **Liger Kernel** (`USE_LIGER_KERNEL`, runtime Qwen3-support detection, automatic
  microbatch 8→4 fallback) — needed because Qwen's 176,856-token extended vocabulary makes
  the training-time logits tensor ~21.6 GiB at microbatch 8, which OOMs without it.
- **Wall-clock training budget** (`TARGET_TRAINING_HOURS = 4.0` → derived `MAX_STEPS`) —
  the 3B notebook just runs a fixed epoch (`MAX_STEPS = -1`), no time cap.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**, set before CUDA initializes, to
  reduce allocator fragmentation — a direct consequence of that large logits tensor.
- **Preprocessing workers scaled up**: `min(64, cpu_count - 4)` vs. the 3B notebook's
  `min(8, cpu_count // 2)` — the 3B notebook leaves a B200 host's ~129 cores mostly idle;
  Qwen's config comment calls this out explicitly.
- **Pinned exact dependency versions** (`transformers==4.57.6`, `datasets==4.4.1`,
  `peft==0.17.1`, ... via `uv pip`) vs. the Llama notebooks' looser `pip install`.
- **`PUSH_TO_HUB = True` by default**, with a real target repo (`isji/qwen3-4b-cpt`) — the
  3B notebook defaults this off with a placeholder repo name (`isji/sinllama-4b-cpt-b200`).

## Summary

The 1B notebook is architecturally the odd one out — external script, short sequences
(512), simple always-both embedding handling. The 3B and Qwen notebooks share near-identical
structure (same section headings, same LoRA numbers, same tokens/update budget): Qwen is a
direct evolution of the 3B notebook, carrying the extra engineering its larger vocabulary
and the tied-embedding bug specifically forced.
