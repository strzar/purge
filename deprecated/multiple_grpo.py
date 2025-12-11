import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: F401  (needed when you add custom code)
from trl import GRPOTrainer, GRPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unlearn Phi‑3‑mini for one target.")
    parser.add_argument(
        "--target",
        required=True,
        help="Folder / slug that identifies the target (e.g. 1_Stephen_King).",
    )
    parser.add_argument(
        "--hf_model_name",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base HF model hub name.",
    )
    parser.add_argument(
        "--output_root",
        default="/mnt/raid5/stratis/models/main",
        help="Directory in which per‑target checkpoints will be created.",
    )
    parser.add_argument(
        "--purge_root",
        default="/home/stratis/unlearning/PURGE",
        help="Root that contains <target>/fts.json and <target>/qa_pairs.json.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_forget_words(path: Path) -> set[str]:
    with path.open() as f:
        return set(json.load(f))


def reward_func_factory(forget_words: set[str]):
    """Return a closure usable by GRPOTrainer."""
    pattern = re.compile(
        r"\b(?:"
        + "|".join(re.escape(w) for w in sorted(forget_words, key=len, reverse=True))
        + r")\b",
        re.IGNORECASE,
    )

    def reward_func(completions, **kwargs):
        return [0.0 if pattern.search(c) else 1.0 for c in completions]

    return reward_func


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    target_dir = Path(args.purge_root) / args.target
    fts_path = target_dir / "fts.json"
    qa_path = target_dir / "qa_pairs.json"

    if not fts_path.exists() or not qa_path.exists():
        sys.exit(f"[ERROR] Could not find needed files in {target_dir}")

    forget_words = load_forget_words(fts_path)

    with qa_path.open() as f:
        dataset = Dataset.from_list(json.load(f))
        dataset = dataset.select(range(100))

    training_args = GRPOConfig(
        output_dir=os.path.join(
            args.output_root,
            f"{Path(args.hf_model_name).name}-{args.target}",
        ),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.generations,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
    )

    trainer = GRPOTrainer(
        model=args.hf_model_name,
        reward_funcs=reward_func_factory(forget_words),
        args=training_args,
        train_dataset=dataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"=== Starting training for {args.target} ===")
    trainer.train()
    print(f"=== Finished training for {args.target} ===")


if __name__ == "__main__":
    main()
