import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import os
import re
import json

# Configuration
TARGET = "21_50_Cent"
HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = f"/mnt/raid5/stratis/models/main/Phi-3-mini-4k-instruct-{TARGET}"
FORGET_WORDS_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/fts.json"
FORGET_DATASET_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/qa_pairs.json"

# Load forget words set
with open(FORGET_WORDS_FILE, 'r') as f:
    forget_words = set(json.load(f))

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
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    dataset = dataset.select(range(100))
    
    # split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=random_seed)
    # train_dataset = split_dataset['train']
    # test_dataset = split_dataset['test']

    training_args = GRPOConfig(
        output_dir = OUTPUT_DIR,
        num_train_epochs = 40,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        num_generations = 4,
        logging_steps = 10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
    )

    trainer = GRPOTrainer(
        model=HF_MODEL_NAME,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset
    )
    print("Started training...")
    trainer.train()
    print("Finished training.")

if __name__ == '__main__':
    main()