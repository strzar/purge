"""utility_test.py
Evaluate a (fine‑tuned) causal‑LM checkpoint on the MMLU benchmark.

Usage
-----
python utility_test.py \
  --model_name_or_path /path/to/checkpoint \
  [--tokenizer_name_or_path /path/to/tokenizer] \
  [--precision fp16]  # choices: fp16 | bf16 | fp32

The script prints a JSON blob such as {"mmlu_accuracy": 0.682} to stdout, which
can be captured by CI pipelines.

python utility_test.py --model_name_or_path /home/stratis/projects/unlearning/models/Qwen-2.5-3B-Instruct-Forget/checkpoint-1000
# python utility_test.py --model_name_or_path Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

CHOICES: List[str] = ["A", "B", "C", "D"]
_logger = logging.getLogger(__name__)


def _format_example(question: str, options: List[str]) -> str:
    """Format a single multiple‑choice item into a prompt understood by the LLM."""
    prompt_lines = [question.strip()]
    prompt_lines += [f"{letter}. {opt}" for letter, opt in zip(CHOICES, options)]
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


def _generate(model, tokenizer, prompt: str, device: str = "cuda", *, max_new_tokens: int = 16) -> str:
    """Greedy generation (temperature=0) by default."""
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    output = model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    generated = tokenizer.decode(output[0][input_ids["input_ids"].shape[-1] :], skip_special_tokens=True)
    return generated.strip()


def _extract_choice(text: str) -> str | None:
    """Return the first occurrence of A/B/C/D in *text* (case‑insensitive)."""
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def evaluate_mmlu(model, tokenizer, *, split: str = "test", device: str = "cuda") -> float:
    """Return mean accuracy over the requested MMLU *split*."""
    ds = load_dataset("cais/mmlu", "all", split=split)
    correct = total = 0
    for sample in tqdm(ds, desc="MMLU"):
        prompt = _format_example(sample["question"], sample["choices"])
        gold = CHOICES[sample["answer"]]
        pred = _extract_choice(_generate(model, tokenizer, prompt, device=device))
        correct += int(pred == gold)
        total += 1
    accuracy = correct / total
    return accuracy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoint on MMLU benchmark")
    p.add_argument("--model_name_or_path", required=True, help="Path or Hub name of the model checkpoint")
    p.add_argument("--tokenizer_name_or_path", help="If omitted, defaults to `--model_name_or_path`.")
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16", help="Model dtype")
    p.add_argument("--device", default="cuda", help="Computation device (cuda, cpu, etc.)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.precision]

    _logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path, use_fast=True
    )
    _logger.info("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
    )

    _logger.info("Running MMLU evaluation …")
    acc = evaluate_mmlu(model, tokenizer, device=args.device)

    result = {"mmlu_accuracy": acc}
    print(json.dumps(result, indent=2))
    _logger.info("MMLU accuracy: %.2f %%", acc * 100)


if __name__ == "__main__":
    main()
