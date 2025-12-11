import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from typing import List, Dict

import matplotlib.pyplot as plt  # NEW

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TARGET = "2_Confucius"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

METHOD = "grpo"  # Options: "npo_full", "grpo"
if METHOD == "grpo":
    CHECKPOINT = "1000"
    UNLEARNED_MODEL_DIR = f"/mnt/disk2/stratis/main/Phi-3-mini-4k-instruct-{TARGET}/checkpoint-{CHECKPOINT}"
else:
    UNLEARNED_MODEL_DIR = f"/mnt/disk2/stratis/saves/RWKU/Target/{TARGET}/npo_full/phi_3_mini_4k_instruct"

FORGET_WORDS_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/fts.json"
FORGET_DATASET_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/qa_pairs.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL = 20
MAX_CONTEXT_TOKENS = 128
MAX_NEW_TOKENS = 64
TOP_K_PLOT = 50  # how many tokens to show in the PDF plot

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def encode_prompt(example: Dict, tokenizer) -> torch.Tensor:
    """Get tokenized prompt tensor from a dataset example."""
    prompt = example.get("prompt") or example.get("input") or ""
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
    ).input_ids


def pretokenize_forbidden_phrases(
    tokenizer,
    forbidden_phrases: List[str]
) -> Dict[str, List[int]]:
    """Turn each forbidden phrase into its token ID sequence."""
    phrase_to_ids: Dict[str, List[int]] = {}
    for phrase in forbidden_phrases:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if token_ids:
            phrase_to_ids[phrase] = token_ids
    return phrase_to_ids


def compute_phrase_mass_for_one_prompt(
    step_probs: np.ndarray,          # [T, vocab_size]
    phrase_to_ids: Dict[str, List[int]]
) -> Dict[str, float]:
    """
    Given the per-step vocab distributions for one prompt, compute the
    (approximate) probability mass of each forbidden sequence appearing
    somewhere in the generated continuation.
    """
    T, vocab_size = step_probs.shape
    phrase_mass = {phrase: 0.0 for phrase in phrase_to_ids.keys()}

    for phrase, token_ids in phrase_to_ids.items():
        L = len(token_ids)
        if L == 0 or L > T:
            continue

        # Sliding window over generation steps
        for start in range(0, T - L + 1):
            p = 1.0
            for offset, tok in enumerate(token_ids):
                if tok >= vocab_size:
                    p = 0.0
                    break
                p *= float(step_probs[start + offset, tok])
            phrase_mass[phrase] += p

    return phrase_mass


@torch.no_grad()
def compute_forbidden_sequence_mass(
    model,
    tokenizer,
    dataset,
    forbidden_phrases: List[str],
    device: str = "cuda",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, object]:
    """
    For a given model, run autoregressive inference on each prompt.
    At each generation step:
      - save the next-token probability distribution.
      - sample (or greedy-pick) the next token and append it.

    After finishing generation for a prompt, compute the probability mass
    of each forbidden phrase appearing (approximate, via independent
    step-wise probs) and aggregate across prompts.

    Returns:
      {
        "per_phrase": { phrase: mass },
        "total": total_mass_over_all_phrases,
        "avg_probs": np.ndarray[vocab_size]   # average next-token probs
      }
    """
    model.eval()
    phrase_to_ids = pretokenize_forbidden_phrases(tokenizer, forbidden_phrases)
    agg_phrase_mass = {phrase: 0.0 for phrase in phrase_to_ids.keys()}

    vocab_size = model.config.vocab_size
    prob_sums = torch.zeros(vocab_size, dtype=torch.float64, device="cpu")
    total_steps = 0

    for ex in tqdm(dataset, desc="Running generation & accumulating phrase mass"):
        # 1) Encode prompt
        input_ids = encode_prompt(ex, tokenizer).to(device)

        # 2) Run autoregressive generation while saving probs
        step_probs_list = []

        generated = 0
        while generated < max_new_tokens:
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]            # [1, vocab]
            probs = torch.softmax(logits, dim=-1)[0]     # [vocab]
            probs_cpu = probs.cpu()

            step_probs_list.append(probs_cpu.numpy())

            # accumulate for global average over all steps/prompts
            prob_sums += probs_cpu.to(dtype=torch.float64)
            total_steps += 1

            # Choose the next token (greedy for determinism; you can change to sampling if desired)
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True)  # [1]
            token_id = next_token_id.item()

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            generated += 1

            # Optional early stop if EOS is emitted
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

        if not step_probs_list:
            continue

        step_probs = np.stack(step_probs_list, axis=0)  # [T, vocab]

        # 3) Phrase mass for this prompt
        prompt_phrase_mass = compute_phrase_mass_for_one_prompt(
            step_probs, phrase_to_ids
        )

        # 4) Aggregate over prompts
        for phrase, mass in prompt_phrase_mass.items():
            agg_phrase_mass[phrase] += mass

    total_mass = float(sum(agg_phrase_mass.values()))
    avg_probs = (prob_sums / max(total_steps, 1)).numpy()

    return {
        "per_phrase": agg_phrase_mass,
        "total": total_mass,
        "avg_probs": avg_probs,
    }

# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print("Loading models and tokenizer...")

    # Load tokenizer once (Phi-3 base & unlearned use the same one)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(DEVICE)
    unlearned_model = AutoModelForCausalLM.from_pretrained(UNLEARNED_MODEL_DIR).to(DEVICE)

    # Load forbidden phrases
    with open(FORGET_WORDS_FILE, "r") as f:
        forbidden_phrases = json.load(f)

    # Load dataset
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    dataset = dataset.select(range(min(NUM_EVAL, len(dataset))))

    # -----------------------------------------------------------------------
    # Compute forbidden sequence mass via generation
    # -----------------------------------------------------------------------
    print("\nComputing forbidden sequence mass for BASE model...")
    base_forbidden = compute_forbidden_sequence_mass(
        base_model, tokenizer, dataset, forbidden_phrases, device=DEVICE
    )

    print("\nComputing forbidden sequence mass for UNLEARNED model...")
    un_forbidden = compute_forbidden_sequence_mass(
        unlearned_model, tokenizer, dataset, forbidden_phrases, device=DEVICE
    )

    # -----------------------------------------------------------------------
    # Aggregate and save results
    # -----------------------------------------------------------------------
    base_total = base_forbidden["total"]
    un_total = un_forbidden["total"]
    reduction = base_total - un_total
    relative_reduction = 1.0 - un_total / max(base_total, 1e-12)

    results = {
        "target": TARGET,
        "base_total_forbidden_mass": base_total,
        "unlearned_total_forbidden_mass": un_total,
        "reduction": reduction,
        "relative_reduction": relative_reduction,
        "per_phrase_comparison": {
            p: {
                "base": base_forbidden["per_phrase"].get(p, 0.0),
                "unlearned": un_forbidden["per_phrase"].get(p, 0.0),
                "delta": base_forbidden["per_phrase"].get(p, 0.0)
                - un_forbidden["per_phrase"].get(p, 0.0),
            }
            for p in forbidden_phrases
        },
    }

    out_json = f"forbidden_sequence_mass_{TARGET}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved:\n  {out_json}")

    # -----------------------------------------------------------------------
    # Vocabulary probability distribution plot (BASE vs UNLEARNED)
    # -----------------------------------------------------------------------
    print("\nCreating vocabulary probability distribution plot...")

    base_avg = base_forbidden["avg_probs"]
    un_avg = un_forbidden["avg_probs"]

    diff = base_avg - un_avg
    abs_diff = np.abs(diff)

    vocab_size = len(base_avg)
    k = min(TOP_K_PLOT, vocab_size)
    top_indices = np.argsort(abs_diff)[::-1][:k]

    # Decode tokens for labels
    labels = []
    for idx in top_indices:
        tok_str = tokenizer.decode([int(idx)])
        tok_str = tok_str.replace("\n", "\\n")
        if tok_str.strip() == "":
            tok_str = f"<ws_{idx}>"
        labels.append(tok_str)

    x = np.arange(k)
    width = 0.4

    plt.figure(figsize=(max(8, k * 0.25), 6))
    plt.bar(x - width / 2, base_avg[top_indices], width, label="Base")
    plt.bar(x + width / 2, un_avg[top_indices], width, label="Unlearned")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Average next-token probability")
    plt.title(f"Top-{k} tokens with largest avg prob change\n(Base vs Unlearned)")
    plt.legend()
    plt.tight_layout()

    pdf_name = f"vocab_prob_diff_{TARGET}.pdf"
    plt.savefig(pdf_name)
    plt.close()

    print(f"Saved vocabulary probability plot as: {pdf_name}")

    # -----------------------------------------------------------------------
    # Summary Report
    # -----------------------------------------------------------------------
    print("\n==== Forbidden Sequence Mass Summary ====")
    print(f"Base total mass:      {base_total:.6e}")
    print(f"Unlearned total mass: {un_total:.6e}")
    print(f"Reduction:            {reduction:.6e}")
    print(f"({relative_reduction*100:.2f}% drop)")

    print("\nDone.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
