import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import os
import re
import json

model_path = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "./models/Qwen2.5-3B-Instruct-Forget" 
# Load words that need to be forgotten
with open('./data/forget_set.json', 'r') as jf:
    forget_words = set(json.load(jf))

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
    hits = 0
    total = len(test_prompts)
    
    for i, prompt in enumerate(test_prompts):
        # print(f"\nPrompt {i+1}: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        # print(f"Response: {response}")

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
            
        forget_accuracy = 1 - (hits/total)*100
        print(f"Forget Accuracy:{forget_accuracy}%")
        print("-------------------------------------------------------")
    
def reward_func(completions, **kwargs):
    """
    Returns 0.0 if a completion contains any word from the forget dataset,
    otherwise returns 1.0.
    """
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, forget_words)) + r')\b', re.IGNORECASE)
    return [0.0 if pattern.search(completion) else 1.0 for completion in completions]

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load forget dataset
    with open("./data/forget_dataset.json", "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    
    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    util_prompts = [
        "How to play basketball?",
        "Can you suggest a book to read?",
        "What is the most important thing to do before you leave the house?",
        "What's a quick and healthy breakfast idea?",
        "How do I reset my email password?",
        "What are the benefits of learning a second language?",
        "Can you explain photosynthesis in simple terms?",
        "How can I improve my public speaking skills?",
        "What should I pack for a weekend camping trip?",
        "What are some effective ways to manage stress?",
        "How do I start a small business?",
        "What is the capital of Canada?",
        "Can you recommend a good movie for a family night?",
        "How do I write a professional email?",
        "What are the signs of burnout?",
        "Can you help me plan a budget for the month?",
        "What are some beginner tips for learning guitar?",
        "How do I make a strong password?",
        "What's the best way to stay productive when working from home?",
        "What are common interview questions and how should I answer them?"
    ]

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    

    print("------------------Before-Finetuning------------------")
    model.eval()
    test_model_responses(model, tokenizer, test_dataset['prompt'])
    test_model_responses(model, tokenizer, util_prompts)

    training_args = GRPOConfig(
        output_dir=output_dir,
        logging_steps=10,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        num_generations=8
    )

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset
    )
    print("Started training...")
    model.train()
    trainer.train()
    print("Finished training.")

    model = AutoModelForCausalLM.from_pretrained(output_dir + "/checkpoint-1000", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(output_dir + "/checkpoint-1000")

    print("------------------After-Finetuning------------------")
    model.eval()
    test_model_responses(model, tokenizer, test_dataset['prompt'])
    test_model_responses(model, tokenizer, util_prompts)

if __name__ == '__main__':
    main()