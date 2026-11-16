"""PyTorch (CPU, f32) decode baseline to compare against kiln.

    python scripts/bench_pytorch.py models/smollm2-135m [n_tokens]
"""
import sys
import time

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

model_dir = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
ids = torch.tensor([tok.encode("Once upon a time", add_special_tokens=False).ids])

with torch.no_grad():
    out = model(ids, use_cache=True)  # prefill
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    t = time.perf_counter()
    for _ in range(n):
        out = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    dt = time.perf_counter() - t
print(f"pytorch {torch.__version__} f32 cpu ({torch.get_num_threads()} threads): {n / dt:.1f} tok/s decode")
