"""Generate tests/reference.json from Hugging Face's own tokenizer + PyTorch model.

    pip install torch transformers tokenizers
    python scripts/reference.py models/smollm2-135m
"""
import json
import sys

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/smollm2-135m"
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()

texts = [
    "Hello, world!",
    "The capital of France is",
    "I'm sure they'll say it's fine, isn't it? We've DONE it.",
    "Prices rose 12.5% to $1,234,567 in 2024.",
    "   leading spaces,  double  spaces,\ttabs\n\nand newlines\n",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "Unicode: café, naïve, Ελληνικά, 日本語のテキスト, emoji 🦀🔥!",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "trailing space ",
    "a b　c \r\n end",
]
prompts = [
    "The capital of France is",
    "<|im_start|>user\nWrite a haiku about compilers.<|im_end|>\n<|im_start|>assistant\n",
]

logits = []
for p in prompts:
    ids = tok.encode(p, add_special_tokens=False).ids
    with torch.no_grad():
        out = model(torch.tensor([ids])).logits[0, -1]
    top = torch.topk(out, 5)
    logits.append({"prompt": p, "top5": [[int(i), float(v)] for v, i in zip(top.values, top.indices)]})

ref = {
    "tokenize": [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts],
    "logits": logits,
}
with open("tests/reference.json", "w") as f:
    json.dump(ref, f, indent=1, ensure_ascii=False)
print("wrote tests/reference.json")
