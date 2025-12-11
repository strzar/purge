import torch
import re
import json
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from trl import GRPOTrainer, GRPOConfig

# Configuration
HF_MODEL_NAME = "google/gemma-2-2b-it"
OUTPUT_DIR = "/mnt/ssd3/stratis/models/ablations/gemma-2-2b-it-Stephen-King"
FORGET_WORDS_FILE = "/home/stratis/unlearning/PURGE/1_Stephen_King/fts.json"
FORGET_DATASET_FILE = "/home/stratis/unlearning/PURGE/1_Stephen_King/qa_pairs.json"

# Optionally silence all Transformers logs below ERROR
hf_logging.set_verbosity_error()

# Load forget words set
with open(FORGET_WORDS_FILE, 'r') as f:
    forget_words = set(json.load(f))

def reward_func(completions, **kwargs):
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, forget_words)) + r')\b',
                         re.IGNORECASE)
    return [0.0 if pattern.search(c) else 1.0 for c in completions]

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare the model with ignore_logger_methods
    config = AutoConfig.from_pretrained(
        HF_MODEL_NAME,
        ignore_logger_methods=["warning_once", "warning"],
        attn_implementation='eager'
    )
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        config=config
    ).to(device)

    # (Optional) load tokenizer if you need it elsewhere
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    # Load and truncate your forget dataset
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data).select(range(100))

    # GRPO training args
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

    # Pass the actual model object (with our custom config) into GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset
    )

    print("🔄 Starting training …")
    trainer.train()
    print("✅ Finished training.")

if __name__ == "__main__":
    main()
