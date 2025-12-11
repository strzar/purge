import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from typing import List, Dict

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TARGET = "2_Confucius"
CHECKPOINT = "1000"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
UNLEARNED_MODEL_DIR = f"/mnt/disk2/stratis/main/Phi-3-mini-4k-instruct-{TARGET}/checkpoint-{CHECKPOINT}"
# UNLEARNED_MODEL_DIR = f"/mnt/disk2/stratis/saves/RWKU/Target/{TARGET}/npo_full/phi_3_mini_4k_instruct"
FORGET_WORDS_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/fts.json"
FORGET_DATASET_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/qa_pairs.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL = 100
MAX_CONTEXT_TOKENS = 128
MAX_NEW_TOKENS = 64  # you can tune this

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
) -> Dict[str, float]:
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
        "total": total_mass_over_all_phrases
      }
    """
    model.eval()
    phrase_to_ids = pretokenize_forbidden_phrases(tokenizer, forbidden_phrases)
    agg_phrase_mass = {phrase: 0.0 for phrase in phrase_to_ids.keys()}

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
            step_probs_list.append(probs.cpu().numpy())

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
    return {"per_phrase": agg_phrase_mass, "total": total_mass}


# ---------------------------------------------------------------------------
# MAIN EVALUATION + PLOT
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
    base_forbidden = compute_forbidden_sequence_mass(base_model, tokenizer, dataset, forbidden_phrases, device=DEVICE)

    print("\nComputing forbidden sequence mass for UNLEARNED model...")
    un_forbidden = compute_forbidden_sequence_mass(unlearned_model, tokenizer, dataset, forbidden_phrases, device=DEVICE)

    # -----------------------------------------------------------------------
    # Aggregate and save results
    # -----------------------------------------------------------------------
    results = {
        "target": TARGET,
        "base_total_forbidden_mass": base_forbidden["total"],
        "unlearned_total_forbidden_mass": un_forbidden["total"],
        "reduction": base_forbidden["total"] - un_forbidden["total"],
        "relative_reduction": (
            1.0 - un_forbidden["total"] / max(base_forbidden["total"], 1e-12)
        ),
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
    # Summary Report
    # -----------------------------------------------------------------------
    print("\n==== Forbidden Sequence Mass Summary ====")
    print(f"Base total mass:      {base_forbidden['total']:.6e}")
    print(f"Unlearned total mass: {un_forbidden['total']:.6e}")
    print(f"Reduction:            {results['reduction']:.6e} "
          f"({results['relative_reduction']*100:.2f}% drop)")

    # -----------------------------------------------------------------------
    # Plot per-phrase deltas
    # -----------------------------------------------------------------------
    print("\nGenerating per-phrase delta plot...")

    phrases = list(results["per_phrase_comparison"].keys())
    deltas = [results["per_phrase_comparison"][p]["delta"] for p in phrases]

    # Sort by delta for clarity
    sorted_pairs = sorted(zip(phrases, deltas), key=lambda x: x[1], reverse=True)
    sorted_phrases, sorted_deltas = zip(*sorted_pairs)

    plt.figure(figsize=(10, max(4, len(sorted_phrases) * 0.4)))
    plt.barh(sorted_phrases, sorted_deltas, color='steelblue')
    plt.xlabel("Δ Forbidden Phrase Mass (Base - Unlearned)")
    plt.title(f"Forbidden Phrase Delta Comparison — {TARGET}")
    plt.gca().invert_yaxis()  # largest on top
    plt.tight_layout()

    pdf_path = f"forbidden_phrase_deltas_{TARGET}.pdf"
    plt.savefig(pdf_path, format="pdf")
    plt.close()

    print(f"Plot saved to: {pdf_path}")
    print("\nDone.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
