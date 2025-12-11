import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import json
from tqdm import tqdm

# Configuration
HF_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "/mnt/ssd3/stratis/models/Qwen2.5-3B-Instruct-Forget"
FORGET_WORDS_FILE = "./data/words_to_forget.json"
FORGET_DATASET_FILE = "./data/forget_dataset.json"
UTILITY_DATASET_FILE = "./data/utility_dataset.json"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint-800")

# Load forget words set
with open(FORGET_WORDS_FILE, 'r') as f:
    forget_words = set(json.load(f))

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
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def test_model_responses(model, tokenizer, test_prompts):
    hits = 0
    total = len(test_prompts)
    
    for prompt in tqdm(test_prompts):
        response = generate_response(model, tokenizer, prompt)
        # Check if response contains any forget words
        contains_forget_words = False
        found_words = []
        for word in forget_words:
            # Use word boundary regex pattern for exact matching
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            if pattern.search(response.lower()):
                contains_forget_words = True
                found_words.append(word)

        if contains_forget_words:
            hits += 1
        #     print(f"⚠️ Response contains forget words: {found_words}")
        # else:
        #     print("✅ Response doesn't contain any forget words")
            
    forget_accuracy = 1 - (hits/total)
    print(f"Forget Accuracy:{forget_accuracy*100:.2f}%")
    print("-------------------------------------------------------")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # If using CUDA, print the device name
    if device.type == 'cuda':
        print(f'Device name: {torch.cuda.get_device_name(device)}')
    random_seed = 42
    print(f"Seed:{random_seed}")

    # Load forget dataset
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    # Load utility dataset
    with open(UTILITY_DATASET_FILE, "r") as f:
        data = json.load(f)
    util_dataset = Dataset.from_list(data)
    
    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=random_seed)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']


    print("------------------Before-Finetuning------------------")
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    model.eval()
    print(f"Test set:")
    test_model_responses(model, tokenizer, test_dataset['prompt'])
    print(f"Utility set:")
    test_model_responses(model, tokenizer, util_dataset['prompt'])

    print("------------------After-Finetuning------------------")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    model.eval()
    print(f"Test set:")
    test_model_responses(model, tokenizer, test_dataset['prompt'])
    print(f"Utility set:")
    test_model_responses(model, tokenizer, util_dataset['prompt'])

if __name__ == '__main__':
    main()