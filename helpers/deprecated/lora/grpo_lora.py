"""
Fine-tune Meta-Llama-3-8B-Instruct with GRPO + LoRA.

Requires:
  pip install "transformers>=4.41.0" "trl>=0.8.6" "peft>=0.11.0" bitsandbytes datasets
"""

import os, re, json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training   # optional, see note below

# --------------------------------------------------
# Paths & files
HF_MODEL_NAME          = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR             = "/mnt/ssd3/stratis/models/Meta-Llama-3-8B-Instruct-GRPO-LoRA"
FORGET_WORDS_FILE      = "data/forget_words.json"
FORGET_DATASET_FILE    = "data/forget_dataset.json"

# --------------------------------------------------
# 1.  Reward function (unchanged)
with open(FORGET_WORDS_FILE, "r") as f:
    forget_words = set(json.load(f))

def reward_func(completions, **kwargs):
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, forget_words)) + r')\b', re.IGNORECASE)
    return [0.0 if pattern.search(c) else 1.0 for c in completions]

# --------------------------------------------------
# 2.  Main training routine
def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load dataset ------------------------------------------------------
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    # ---- Load base model & tokenizer ---------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token        # -- avoid warnings

    # (optional) load in 4-bit to save VRAM; comment out if you prefer fp16
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        load_in_4bit=True,                           # bitsandbytes
        device_map="auto",
        trust_remote_code=True
    )
    # If you loaded in 4-bit, unfrozen LoRA layers need prep:
    model = prepare_model_for_kbit_training(model)

    # ---- Inject LoRA adapter ----------------------------------------------
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[           # Llama-3 projection layers
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()               # sanity check

    # ---- GRPO training config ---------------------------------------------
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=10,
        num_generations=2,
        gradient_accumulation_steps=2,
        logging_steps=10,
        gradient_checkpointing=True,                 # already helpful for 8 B
        save_strategy="epoch",
        save_total_limit=2
    )

    # ---- Trainer -----------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting GRPO-LoRA training...\n")
    trainer.train()
    print("\nFinished training. Adapter weights are in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
