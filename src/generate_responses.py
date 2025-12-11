#!/usr/bin/env python3
"""
Run Phi-3 QA-pair generation for every target listed in NAMES.
Assumes the directory layout:

  /home/stratis/unlearning/RWKU/Target/<NAME>/reject_phi.json
  /home/stratis/unlearning/PURGE/<NAME>/qa_pairs.json
"""

import os
import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------
HF_MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_BASE_DIR = "data/"
OUTPUT_BASE_DIR = "PURGE/"

NAMES = [
    '1_Stephen_King',
    '2_Confucius',
    '3_Bruce_Lee',
    '4_Warren_Buffett',
    '5_Christina_Aguilera',
    '6_Cindy_Crawford',
     '7_Marie_Osmond',
     '8_Paris_Hilton',
    '9_Justin_Bieber',
     '10_Prince_Harry,_Duke_of_Sussex',
     '11_Miley_Cyrus',
     '12_Genghis_Khan',
     '13_Liza_Minnelli',
     '14_Taylor_Swift',
     '15_Mark_Cuban',
     '16_Rhea_Perlman',
     '17_Mark_Hamill',
     '18_John_D._Rockefeller',
     '19_Alanis_Morissette',
     '20_Marlon_Brando',
]

# ---------------------------------------------
# 2. HELPER FUNCTIONS (same as your original)
# ---------------------------------------------
def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )


def test_model_responses(model, tokenizer, test_prompts):
    qa_pairs = []
    for prompt in tqdm(test_prompts, desc="   ↳ generating"):
        # if prompt == "": continue
        response = generate_response(model, tokenizer, prompt)
        qa_pairs.append({"prompt": prompt, "response": response})
    return qa_pairs


def process_single_target(model, tokenizer, name):
    """Load reject set → generate QA pairs → write output JSON."""
    dataset_path = os.path.join(TARGET_BASE_DIR, name, "reject.json")
    output_path  = os.path.join(OUTPUT_BASE_DIR, name, "qa_pairs.json")

    if not os.path.exists(dataset_path):
        print(f"⚠️  {dataset_path} not found – skipping.")
        return

    with open(dataset_path, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    print(f"• {name} – {len(dataset):,} prompts")
    qa_pairs = test_model_responses(model, tokenizer, dataset["instruction"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    print(f"   ✓ saved {len(qa_pairs):,} QA pairs → {output_path}\n")


# ---------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model/tokenizer only ONCE
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model.eval()

    for name in NAMES:
        process_single_target(model, tokenizer, name)

    # Optional: free GPU memory at the end
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()