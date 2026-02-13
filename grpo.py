import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import math
import re
import json
import argparse

def build_reward_func(forget_words):
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, forget_words)) + r')\b', re.IGNORECASE)

    # def reward_func(completions, **kwargs):
    #     rewards = []
    #     tau = 1.0  # Time constant parameter
    #     for completion in completions:
    #         matches = pattern.findall(completion)
    #         forget_count = len(matches)
    #         reward = math.exp(-(forget_count / tau))
    #         rewards.append(reward)
    #     return rewards
    def reward_func(completions, **kwargs):
        return [0.0 if pattern.search(completion) else 1.0 for completion in completions]

    return reward_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Unlearning target name, e.g. '1_Stephen_King'",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=100,
        help="Max number of QA pairs to use from the dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    TARGET = args.target
    HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    OUTPUT_DIR = f"/mnt/raid5/stratis/models/binary/Phi-3-mini-4k-instruct-{TARGET}"
    FORGET_WORDS_FILE = f"data/PURGE/{TARGET}/fts.json"
    FORGET_DATASET_FILE = f"data/PURGE/{TARGET}/qa_pairs.json"

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"========== TARGET: {TARGET} | Device: {device} ==========")

    # Load forget words set
    with open(FORGET_WORDS_FILE, 'r') as f:
        forget_words = set(json.load(f))

    reward_func = build_reward_func(forget_words)

    # Load forget dataset
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    max_n = min(args.max_pairs, len(dataset))
    dataset = dataset.select(range(max_n))
    print(f"Using {max_n} QA pairs for {TARGET}")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=40,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_generations=4,
        logging_steps=10,
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

    print(f"Started training for {TARGET}...")
    trainer.train()
    print(f"Finished training for {TARGET}.")


if __name__ == '__main__':
    main()
