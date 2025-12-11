import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import json
from tqdm import tqdm
from rouge_score import rouge_scorer

HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_DIR = "/mnt/ssd3/stratis/models/Phi-3-mini-4k-instruct-GRPO-v3"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoint-1000") if os.path.isdir(os.path.join(MODEL_DIR, "checkpoint-1000")) else MODEL_DIR
FORGET_WORDS_FILE = "data/forget_words_v3.json"
FORGET_DATASET_FILE = "data/qa_forget_og.json"

with open(FORGET_WORDS_FILE, "r") as f:
    forget_words = set(json.load(f))

# Build a single reference string for ROUGE
FORGET_REFERENCE_STR = " ".join(sorted(forget_words))

def generate_responses_batch(model, tokenizer, prompts, max_new_tokens: int = 100, batch_size: int = 16):
    if tokenizer.pad_token_id is None:
        # Llama‑style tokenizers often miss an explicit pad token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

        # slice off the prompt for each item and decode
        for idx, output_ids in enumerate(generated):
            gen_ids = output_ids[prompt_lengths[idx] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(text)
    return responses

def test_model_responses(model, tokenizer, prompts, references, batch_size: int = 16):
    assert len(prompts) == len(references)

    hits = 0
    total = len(prompts)

    rougeL_scores = []
    rouge1_scores = []

    scorerL = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scorer1 = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    for start in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch_prompts    = prompts[start : start + batch_size]
        batch_refs       = references[start : start + batch_size]
        batch_responses  = generate_responses_batch(model, tokenizer, batch_prompts, batch_size=batch_size)

        for resp, ref in zip(batch_responses, batch_refs):
            resp_lc, ref_lc = resp.lower(), ref.lower()
            rougeL = scorerL.score(ref_lc, resp_lc)["rougeL"].recall
            rougeL_scores.append(rougeL)
            rouge1 = scorer1.score(FORGET_REFERENCE_STR.lower(), resp_lc)["rouge1"].recall
            rouge1_scores.append(rouge1)

            contains_forget_words = False
            for word in forget_words:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                if pattern.search(resp_lc):
                    contains_forget_words = True

            if contains_forget_words:
                hits += 1
            
    avg_rougeL_score    = sum(rougeL_scores) / total
    avg_rouge1_score    = sum(rouge1_scores) / total

    print(f"Hits:{hits}/{total}\nROUGE-1 Recall on forget words:{avg_rouge1_score * 100:.2f}%\nROUGE-L Recall:{avg_rougeL_score * 100:.2f}%")

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # If using CUDA, print the device name
    if device.type == 'cuda':
        print(f'Device name: {torch.cuda.get_device_name(device)}')
    random_seed = 42
    print(f"Seed: {random_seed}")

    # Datasets
    with open(FORGET_DATASET_FILE, "r") as f:
        forget_data = json.load(f)
    forget_dataset = Dataset.from_list(forget_data)

    split_dataset = forget_dataset.train_test_split(test_size=0.2, shuffle=True, seed=random_seed)
    # train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    print(f"------------------Before-Finetuning-{HF_MODEL_NAME}-----------------")
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, padding_side='left')

    print("Test set:")
    test_model_responses(model, tokenizer, test_dataset["prompt"], test_dataset["response"])

    print(f"------------------After-Finetuning-{CHECKPOINT_DIR}-----------------")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, padding_side='left')

    print("Test set:")
    test_model_responses(model, tokenizer, test_dataset["prompt"], test_dataset["response"])


if __name__ == "__main__":
    main()
