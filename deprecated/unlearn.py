import os
import re
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# Configuration
HF_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./models/Qwen2.5-3B-Instruct-Forget"
FORGET_WORDS_FILE = "./data/forget_set.json"
FORGET_DATASET_FILE = "./data/forget_dataset.json"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint-1000")

# Load forget words set
with open(FORGET_WORDS_FILE, 'r') as f:
    FORGET_WORDS = set(json.load(f))


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


def test_model_responses(model, tokenizer, prompts, label):
    """Tests model responses against a list of prompts and logs results."""
    print(f"\n------------------ {label} ------------------")
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"Response: {response}")

        # Check for presence of forget words
        found_words = [
            word for word in FORGET_WORDS
            if re.search(rf'\b{re.escape(word)}\b', response, re.IGNORECASE)
        ]

        if found_words:
            print(f"⚠️ Response contains forget words: {found_words}")
        else:
            print("✅ Response doesn't contain any forget words")
        print("-------------------------------------------------------")


def reward_func(completions, **kwargs):
    """
    Reward function used during RL fine-tuning.
    Returns 0.0 if a completion contains any forbidden words, else 1.0.
    """
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, FORGET_WORDS)) + r')\b', re.IGNORECASE)
    return [0.0 if pattern.search(text) else 1.0 for text in completions]


def load_dataset(filepath):
    """Loads dataset from a JSON file and converts it to HuggingFace Dataset."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval()

    # Load datasets
    train_dataset = load_dataset(FORGET_DATASET_FILE)

    # Define evaluation prompts
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

    # Test responses before training
    test_model_responses(model, tokenizer, test_prompts, label="Before Fine-tuning (Test Prompts)")
    test_model_responses(model, tokenizer, util_prompts, label="Before Fine-tuning (Utility Prompts)")

    # Define training configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        logging_steps=10,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        num_generations=8
    )

    # Initialize and train
    trainer = GRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset
    )

    print("Starting training...")
    model.train()
    trainer.train()
    print("Training completed.")

    # Load fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    # Test responses after training
    test_model_responses(model, tokenizer, test_prompts, label="After Fine-tuning (Test Prompts)")
    test_model_responses(model, tokenizer, util_prompts, label="After Fine-tuning (Utility Prompts)")


if __name__ == '__main__':
    main()
