"""
GRPO training script for unlearning using modular reward functions.
Configured with Hydra for flexible hyperparameter management.

Usage:
    # Default run (Confucius with PageRank reward)
    python purge.py

    # Different entity
    python purge.py entity=taylor_swift

    # Different reward function
    python purge.py reward=binary

    # Fast testing config
    python purge.py training=fast

    # Override specific parameters
    python purge.py training.num_epochs=20 training.per_device_train_batch_size=4

    # Multi-run sweep
    python purge.py --multirun entity=confucius,taylor_swift,stephen_king
"""
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import json
import hydra
from omegaconf import DictConfig, OmegaConf

from rewards import RewardFunction, BinaryReward, PageRankWeightedReward, ExponentialDecayReward
from rewards.base import RewardConfig


def get_reward_class(reward_type: str) -> type[RewardFunction]:
    """
    Get the reward class based on the reward type string.
    
    Args:
        reward_type: Either "binary" or "pagerank"
        
    Returns:
        The corresponding RewardFunction subclass
    """
    reward_classes = {
        "binary": BinaryReward,
        "pagerank": PageRankWeightedReward,
        "exponential_decay": ExponentialDecayReward
    }
    
    if reward_type not in reward_classes:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available: {list(reward_classes.keys())}"
        )
    
    return reward_classes[reward_type]


def load_model_and_tokenizer(cfg: DictConfig):
    """Load the model and tokenizer from HuggingFace."""
    print(f"Loading model: {cfg.model.hf_model_id}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_model_id)
    return model, tokenizer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
    # Print resolved configuration
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load forget dataset
    print(f"Loading dataset from: {cfg.paths.forget_dataset_file}")
    with open(cfg.paths.forget_dataset_file, "r") as f:
        data = json.load(f)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)
    
    # Get reward class and create config
    reward_class = get_reward_class(cfg.reward.type)
    
    # Collect extra params from reward config (excluding 'type')
    extra_params = {k: v for k, v in cfg.reward.items() if k != "type"}
    
    reward_config = RewardConfig(
        target_entity=cfg.entity.name,
        forget_words_file=cfg.paths.forget_words_file,
        forget_dataset_file=cfg.paths.forget_dataset_file,
        model=model,
        tokenizer=tokenizer,
        extra_params=extra_params if extra_params else None,
    )
    
    # Preprocess reward function (e.g., compute PageRank weights)
    print(f"\n{'=' * 60}")
    print(f"Using reward function: {reward_class.__name__}")
    print(f"{'=' * 60}\n")
    
    reward_class.preprocess(reward_config)
    
    # Prepare dataset
    dataset = Dataset.from_list(data)
    if cfg.training.dataset_size is not None:
        dataset = dataset.select(range(min(cfg.training.dataset_size, len(dataset))))
    print(f"Dataset size: {len(dataset)} samples")

    # Training configuration
    training_args = GRPOConfig(
        output_dir=cfg.paths.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=cfg.training.num_generations,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
    )

    # Create trainer with the modular reward function
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_class.calc_reward,
        args=training_args,
        train_dataset=dataset
    )
    
    print("Started training...")
    trainer.train()
    print("Finished training.")


if __name__ == '__main__':
    main()
