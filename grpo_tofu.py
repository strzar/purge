import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import math
import re
import json

# Configuration
# TARGET = "1_Stephen_King"
HF_MODEL_NAME = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
OUTPUT_DIR = f"/mnt/disk2/stratis/main/Llama-3.2-1B-Instruct_ft_tofu_forget10_v2"
FORGET_WORDS_FILE = f"/home/stratis/unlearning/TOFU/forget10_fts.json"
FORGET_DATASET_FILE = f"/home/stratis/unlearning/TOFU/forget10.json"

# Load forget words set
with open(FORGET_WORDS_FILE, 'r') as f:
    forget_words = set(json.load(f))

pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, forget_words)) + r')\b', re.IGNORECASE)

def reward_func(completions, **kwargs):
    return [0.0 if pattern.search(completion) else 1.0 for completion in completions]

# def reward_func(completions, **kwargs):
#     rewards = []
#     tau = 1.0  # Time constant parameter
#     for completion in completions:
#         # Count total forget-word matches
#         matches = pattern.findall(completion)
#         forget_count = len(matches)
#         # Exponential decay reward
#         reward = math.exp(-(forget_count/tau))
#         rewards.append(reward)
#     return rewards

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load forget dataset
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    # dataset = dataset.select(range(100)) # Total qa pairs are 300.
    
    # split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=random_seed)
    # train_dataset = split_dataset['train']
    # test_dataset = split_dataset['test']

    training_args = GRPOConfig(
        output_dir = OUTPUT_DIR,
        num_train_epochs = 20,
        # beta = 0.01,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        num_generations = 4,
        logging_steps = 10,
        save_strategy="steps",
        save_steps=200,
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