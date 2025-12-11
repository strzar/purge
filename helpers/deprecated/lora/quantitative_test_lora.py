"""Inference script for evaluating a base model **and** its LoRA‑finetuned adapters.

The script keeps your original evaluation logic (ROUGE metrics, hit‑count on
“forget words”, etc.) and adds the minimal changes required to load LoRA
adapters located in `MODEL_DIR` / `CHECKPOINT_DIR`.

Key changes
-----------
1.  `from peft import PeftModel` is imported.
2.  In the **after‑finetuning** section we:
    * load the *base* model (`HF_MODEL_NAME`),
    * wrap it with `PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)`,
      which injects the LoRA adapters, and
    * move the wrapped model to the correct `device`.
3.  The tokenizer for the LoRA run is loaded from the *base* model name.

If you ever want to *merge* the adapters into a standalone checkpoint,
replace the two lines annotated with `# (optional) merge` by the commented
`merge_and_unload()` call below.
"""

import os
import re
import json
from tqdm import tqdm

import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # NEW: import PEFT for LoRA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_DIR = "/mnt/ssd3/stratis/models/Meta-Llama-3-8B-Instruct-GRPO-LoRA"
CHECKPOINT_DIR = (
    os.path.join(MODEL_DIR, "checkpoint-500")
    if os.path.isdir(os.path.join(MODEL_DIR, "checkpoint-500"))
    else MODEL_DIR
)
FORGET_WORDS_FILE = "data/forget_words.json"
FORGET_DATASET_FILE = "data/test_forget_dataset.json"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 100

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
with open(FORGET_WORDS_FILE, "r") as f:
    forget_words = set(json.load(f))

FORGET_REFERENCE_STR = " ".join(sorted(forget_words))

def generate_responses_batch(model, tokenizer, prompts, *, max_new_tokens: int = MAX_NEW_TOKENS, batch_size: int = BATCH_SIZE):
    """Generate completions for a batch of prompts."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Llama tokenizer quirk

    responses = []
    model.eval()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt", truncation=True).to(model.device)
        prompt_lengths = inputs.attention_mask.sum(dim=1)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        for idx, output_ids in enumerate(generated):
            gen_ids = output_ids[prompt_lengths[idx] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(text)
    return responses

def test_model_responses(model, tokenizer, prompts, references, *, batch_size: int = BATCH_SIZE):
    """Compute hit‑rate & ROUGE‑based metrics for *one* model."""
    assert len(prompts) == len(references)

    hits = 0
    total = len(prompts)
    rougeL_scores, rouge1_scores = [], []

    scorerL = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scorer1 = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    for start in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch_prompts   = prompts[start : start + batch_size]
        batch_refs      = references[start : start + batch_size]
        batch_responses = generate_responses_batch(model, tokenizer, batch_prompts, batch_size=batch_size)

        for resp, ref in zip(batch_responses, batch_refs):
            resp_lc, ref_lc = resp.lower(), ref.lower()
            rougeL = scorerL.score(ref_lc, resp_lc)["rougeL"].recall
            rouge1 = scorer1.score(FORGET_REFERENCE_STR.lower(), resp_lc)["rouge1"].recall
            rougeL_scores.append(rougeL)
            rouge1_scores.append(rouge1)

            # Check if any forbidden word appears
            contains_forget_words = any(
                re.search(rf"\\b{re.escape(word)}\\b", resp_lc, re.IGNORECASE)
                for word in forget_words
            )
            if contains_forget_words:
                hits += 1

    avg_rougeL = sum(rougeL_scores) / total
    avg_rouge1 = sum(rouge1_scores) / total

    print(
        f"Hits: {hits}/{total}\n"
        f"ROUGE‑1 Recall on forget words: {avg_rouge1 * 100:.2f}%\n"
        f"ROUGE‑L Recall: {avg_rougeL * 100:.2f}%"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")

    random_seed = 42
    print(f"Seed: {random_seed}")

    # Load evaluation dataset
    with open(FORGET_DATASET_FILE, "r") as f:
        forget_data = json.load(f)
    forget_dataset = Dataset.from_list(forget_data)
    test_dataset = forget_dataset.train_test_split(test_size=0.2, shuffle=True, seed=random_seed)["test"]

    # -------------------------------------------------------------------
    # Baseline (pre‑finetuning) evaluation
    # -------------------------------------------------------------------
    print(f"------------------Before Finetuning — {HF_MODEL_NAME}-----------------")
    base_model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float16).to(device)
    base_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, padding_side="left")

    test_model_responses(base_model, base_tokenizer, test_dataset["question"], test_dataset["answer"])

    # -------------------------------------------------------------------
    # LoRA‑finetuned evaluation
    # -------------------------------------------------------------------
    print(f"------------------After Finetuning — {CHECKPOINT_DIR}-----------------")

    # 1️⃣  Load the *base* model again (weights frozen)
    lora_base = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float16)

    # 2️⃣  Inject LoRA adapters
    model = PeftModel.from_pretrained(lora_base, CHECKPOINT_DIR)

    # (optional) merge — uncomment *one* of the following two lines if you
    # prefer to bake the LoRA weights into a standalone checkpoint.
    # model = PeftModel.from_pretrained(lora_base, CHECKPOINT_DIR).merge_and_unload()

    model.to(device).eval()

    lora_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, padding_side="left")

    test_model_responses(model, lora_tokenizer, test_dataset["question"], test_dataset["answer"])


if __name__ == "__main__":
    main()
