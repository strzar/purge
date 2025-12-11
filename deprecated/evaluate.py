import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

model_path = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "./models/Qwen2.5-7B-Instruct-Forget" 
# Load words that need to be forgotten
with open('./data/forget_set.json', 'r') as jf:
    forget_words = set(json.load(jf))
test_prompts = [
    "What is the most important ball in Quidditch according to Wood?",
    "Who does Lupin say is the most savage werewolf alive today?",
    "What name does Hagrid suggest Hermione allow Grawp to call her?"
]

util_prompts = [
    "How to play basketball?",
    "Can you suggest a book to read?",
    "What is the most important thing to do before you leave the house?"
]
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
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def test_model_responses(model, tokenizer, test_prompts):
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"Response: {response}")

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
            print(f"⚠️ Response contains forget words: {found_words}")
        else:
            print("✅ Response doesn't contain any forget words")
        print("-------------------------------------------------------")

model = AutoModelForCausalLM.from_pretrained(output_dir + "/checkpoint-1000", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(output_dir + "/checkpoint-1000")
print("------------------After-Finetuning------------------")
model.eval()
test_model_responses(model, tokenizer, test_prompts)
test_model_responses(model, tokenizer, util_prompts)