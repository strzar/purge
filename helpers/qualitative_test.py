import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import json
from tqdm import tqdm

# Configuration
HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "/mnt/ssd3/stratis/models/Phi-3-mini-4k-instruct-GRPO-100"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint-1000") if os.path.isdir(os.path.join(OUTPUT_DIR, "checkpoint-1000")) else OUTPUT_DIR
FORGET_WORDS_FILE = "data/forget_words_v2.json"
FORGET_DATASET_FILE = "data/test_forget_dataset.json"
UTILITY_DATASET_FILE = "./data/utility_dataset.json"

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

def test_model_responses(model, fmodel, tokenizer, ftokenizer, test_prompts):
    for prompt in tqdm(test_prompts):
        response = generate_response(model, tokenizer, prompt)
        fresponse = generate_response(fmodel, ftokenizer, prompt)
        print(f"Prompt:{prompt}\nBase: {response}\nForget: {fresponse}")
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
    
    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=random_seed)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model.eval()

    fmodel = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR + "/checkpoint-1000", torch_dtype=torch.float16)
    ftokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR + "/checkpoint-1000")
    fmodel.eval()
    
    print(f"Test set:")
    test_model_responses(model, fmodel, tokenizer, ftokenizer, test_dataset['question'])


if __name__ == '__main__':
    main()