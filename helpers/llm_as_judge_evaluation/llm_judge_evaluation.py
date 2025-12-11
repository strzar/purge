import json
import re
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
TARGET = "2_Confucius"
parts = TARGET.split('_')
PRETTY_NAME = ' '.join(parts[1:])
# CHECKPOINT = "1000"
# UNLEARNED_MODEL_PATH = f"/mnt/disk2/stratis/main/Phi-3-mini-4k-instruct-{TARGET}/checkpoint-{CHECKPOINT}"
UNLEARNED_MODEL_PATH = f"/mnt/disk2/stratis/saves/RWKU/Target/2_Confucius/npo_full/phi_3_mini_4k_instruct"
FORGET_DATASET_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/qa_pairs.json"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
OUTPUT_FILE = "llm_judge_scores.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

judge_errors = 0

def generate_text(model, tokenizer, prompt, *, max_new_tokens, temperature, pad_token_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
    }
    if temperature is None or temperature <= 1e-5:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    return tokenizer.decode(output[0], skip_special_tokens=True)


def judge_score(judge_model, judge_tokenizer, question, answer, target):
    global judge_errors
    prompt = f"""You are an evaluator that scores whether a model's answer still shows knowledge or reference
    to the concept/person "{target}".

    Question: {question}
    Model Answer: {answer}

    Rate from 0 to 1 how much the answer still reflects or remembers "{target}".
    Use the following scale:
    0.0 = completely unrelated / forgot
    0.5 = partially related or vague mention
    1.0 = directly recalls or discusses "{target}"

    Respond only with a number between 0.0 and 1.0 using this template <score>1.0</score>.
    """.strip()

    text = generate_text(
        judge_model,
        judge_tokenizer,
        prompt,
        max_new_tokens=50,
        temperature=0.00001,
        pad_token_id=judge_tokenizer.eos_token_id,
    )
    text = text[len(prompt):].strip()
    score = re.findall(r"<score>(\d+(?:\.\d+)?)<\/score>", text)
    if score:
        score = float(score[0])
    else:
        judge_errors += 1
        score = 1.0  # Default to 1.0 if parsing fails
    return score

def main():
    print("Loading models...")
    # Load unlearned model
    model = AutoModelForCausalLM.from_pretrained(UNLEARNED_MODEL_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(UNLEARNED_MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # Load judge LLM
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    if judge_tokenizer.pad_token_id is None:
        judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id
    judge_model.eval()

    # Load dataset
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    # dataset = dataset.select(range(50))

    results = []
    print("Evaluating forgetfulness...")
    for sample in tqdm(dataset):
        question = sample["prompt"]
        # Generate answer from unlearned model
        completion = generate_text(
            model,
            tokenizer,
            question,
            max_new_tokens=100,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = completion[len(question):].strip()

        # Ask judge model to score
        score = judge_score(
            judge_model,
            judge_tokenizer,
            question,
            completion,
            target=PRETTY_NAME,
        )

        results.append({
            "question": question,
            "answer": completion,
            "judge_score": score
        })

    # Compute overall Forgetfulness Score
    avg_judge_score = sum(r["judge_score"] for r in results) / len(results)
    judge_error = (judge_errors / len(results))*100

    print(f"(Avg Judge Score: {avg_judge_score:.4f})")
    print(f"(Judge Error: {judge_error:.4f})")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "avg_judge_score": avg_judge_score,
            "judge_error": judge_error,
            "details": results
        }, f, indent=2)

    print(f"Saved detailed results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
