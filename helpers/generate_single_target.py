import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm

# Configuration
HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
FORGET_DATASET_FILE = "/home/stratis/unlearning/RWKU/Target/1_Stephen_King/reject_phi.json"
OUTPUT_QA_FILE = "/home/stratis/unlearning/PURGE/1_Stephen_King/qa_pairs.json"


def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """Generates a model response for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    # Decode only the newly generated tokens
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def test_model_responses(model, tokenizer, test_prompts):
    """Runs the model on each prompt and collects prompt-response pairs."""
    qa_pairs = []
    for prompt in tqdm(test_prompts, desc="Generating responses"):
        response = generate_response(model, tokenizer, prompt)
        qa_pairs.append({
            "prompt": prompt,
            "response": response
        })
    return qa_pairs


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'Device name: {torch.cuda.get_device_name(device)}')

    # Load forget dataset
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model.eval()

    # Generate and collect QA pairs
    print("Generating QA pairs...")
    qa_pairs = test_model_responses(model, tokenizer, dataset['instruction'])

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_QA_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to JSON
    with open(OUTPUT_QA_FILE, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(qa_pairs)} prompt-response pairs to {OUTPUT_QA_FILE}")


if __name__ == '__main__':
    main()
