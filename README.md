# kiln

A from-scratch LLM inference engine in Rust. It loads Hugging Face Llama-architecture checkpoints (safetensors) and runs them on the CPU. PyTorch, llama.cpp, `tokenizers` and BLAS are not used: the tokenizer, the transformer forward pass, the KV cache, int8 quantization, the SIMD kernels and sampling are all written in about 1,000 lines. The only dependencies are `rayon` for threads and `serde_json` for reading the config and tokenizer.

```
$ kiln models/smollm2-1.7b --chat -p "Explain what a KV cache is in two sentences." -n 80 --temp 0 --q8
loaded models/smollm2-1.7b (24 layers, dim 2048, q8) in 2.35s
A KV cache is a type of cache that stores key-value pairs, ...
prefill: 20 tok in 0.93s (21.6 tok/s) | decode: 79 tok in 4.25s (18.6 tok/s)
```

## Results

Decode throughput, greedy, 128 tokens, Apple M4 (4P+6E cores, 16 GB), 4 threads, CPU only:

| model | PyTorch 2.15 f32 | kiln f32 | kiln q8 |
|---|---|---|---|
| SmolLM2-135M-Instruct | 57.7 tok/s | ~66 tok/s | **~148 tok/s** (2.6x) |
| SmolLM2-1.7B-Instruct | 7.7 tok/s | ~6.4 tok/s | **~15 tok/s** (2.0x) |

Correctness: `cargo test` checks kiln against Hugging Face's own outputs (`tests/reference.json`, made by `scripts/reference.py`):
- **Tokenizer:** token-for-token identical to HF `tokenizers` on 10 adversarial strings. These cover contractions, digits, runs of whitespace, tabs and CRLF, CJK, emoji, NBSP and chat special tokens.
- **f32 logits:** the top 5 match PyTorch within 1e-2.
- **q8:** picks the same top token as PyTorch, with logits within 0.5.

Decoding is memory-bandwidth bound: every token streams every weight through the CPU once. That is why int8 (about 4x fewer bytes than f32) is the biggest speedup.

## Quick start

```sh
scripts/download.sh HuggingFaceTB/SmolLM2-135M-Instruct models/smollm2-135m
cargo build --release
./target/release/kiln models/smollm2-135m --chat -p "What is a KV cache?" --q8
cargo test --release
```

Flags: `-p PROMPT`, `--chat` (applies the ChatML template), `-n MAX_NEW_TOKENS`, `--temp T` (0 means greedy), `--top-p P`, `--seed S`, `--ctx N`, `--q8`. Set `RAYON_NUM_THREADS` to choose the thread count; on Apple Silicon, using only the performance cores is usually fastest.

## How it works

| file | what's in it |
|---|---|
| `src/tokenizer.rs` | Byte-level BPE. Splits out special tokens, then pre-tokenizes with GPT-2's regex rules written by hand (Rust's `regex` crate has no lookahead). Maps bytes to printable chars and applies merges by rank. |
| `src/model.rs` | Parses the safetensors header itself and converts bf16 to f32 with a bit shift. Llama forward pass: RMSNorm, RoPE (rotate-half), grouped-query attention over a preallocated KV cache, SwiGLU MLP, tied LM head. |
| `src/kernels.rs` | The matvecs that dominate runtime. f32 uses NEON with 4 FMA accumulators. q8 quantizes weights and activations into 32-element blocks with an f32 scale and runs int8 dot products with the `SDOT` instruction through inline asm, since `vdotq_s32` is still unstable in Rust. Rows run in parallel with rayon. Portable fallbacks exist for non-ARM. |
| `src/main.rs` | CLI, streaming UTF-8-safe output, temperature and nucleus sampling (xorshift RNG). |

Optimizations so far:
- The q, k and v projections are stacked into one matrix at load time, and so are gate and up. That gives 4 parallel dispatches per layer instead of 7.
- The vocabulary projection is skipped for prompt tokens whose logits aren't needed.
- Scratch buffers and the KV cache are allocated once per session.

## Roadmap

- [ ] **Batched prefill.** Process the prompt as one GEMM instead of one matvec per token. Prefill is currently as slow as decode.
- [ ] **Q4 quantization** (4-bit blocks): half the bytes of q8.
- [ ] **Metal GPU backend** for matvec and attention.
- [ ] **OpenAI-compatible HTTP server** with SSE streaming.
- [ ] **Llama 3.x support:** `llama3` RoPE scaling and the Llama 3 pre-tokenizer regex.
- [ ] **Multi-turn chat** that reuses the KV cache across turns.
- [ ] **Benchmarks against llama.cpp** (same quantization level), plus perplexity on WikiText-2 for f32, q8 and q4.
- [ ] Shared embedding / LM-head storage in f32 mode, and mmap loading.
