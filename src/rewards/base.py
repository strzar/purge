"""
Abstract base class for reward functions used in GRPO unlearning.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Optional, Any, Dict
import json


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    target_entity: str
    forget_words_file: str
    forget_dataset_file: str
    model: Any = None
    tokenizer: Any = None
    # Additional reward-specific parameters
    extra_params: Optional[Dict[str, Any]] = None


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    
    Subclasses must implement:
    - preprocess(): Class method to compute any required preprocessing (e.g., PageRank)
    - calc_reward(): Static method that computes rewards for completions
    
    Usage with GRPOTrainer:
        reward_class = PageRankWeightedReward
        reward_class.preprocess(config)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_class.calc_reward,
            ...
        )
    """
    
    # Class-level state that gets set during preprocess()
    _forget_words: Set[str] = set()
    _forget_set: List[str] = []
    _config: Optional[RewardConfig] = None
    _preprocessed: bool = False
    
    @classmethod
    def load_forget_words(cls, forget_words_file: str) -> Set[str]:
        """Load forget words from JSON file."""
        with open(forget_words_file, 'r') as f:
            return set(json.load(f))
    
    @classmethod
    def load_forget_set(cls, forget_words_file: str, target_entity: str) -> List[str]:
        """Load forget set with target entity as first element."""
        with open(forget_words_file, 'r') as f:
            fts_list = json.load(f)
        # Remove target_entity if exists to avoid duplicates, then prepend it
        fts_list = [term for term in fts_list if term != target_entity]
        return [target_entity] + fts_list
    
    @classmethod
    @abstractmethod
    def preprocess(cls, config: RewardConfig) -> None:
        """
        Preprocess step that runs before training.
        
        This method should compute any required data structures (e.g., PageRank weights)
        and store them as class attributes for use in calc_reward().
        
        Args:
            config: RewardConfig with model, tokenizer, and file paths
        """
        pass
    
    @staticmethod
    @abstractmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        """
        Calculate rewards for a batch of completions.
        
        This method is passed directly to GRPOTrainer as reward_funcs.
        It should be a static method that can access class-level state
        set during preprocess().
        
        Args:
            completions: List of generated text completions
            **kwargs: Additional arguments passed by GRPOTrainer
            
        Returns:
            List of reward scores (typically in [0, 1] range)
        """
        pass
    
    @classmethod
    def get_reward_func(cls):
        """
        Returns the calc_reward method bound to the class state.
        Use this when you need to pass the reward function to GRPOTrainer.
        """
        if not cls._preprocessed:
            raise RuntimeError(
                f"{cls.__name__}.preprocess() must be called before get_reward_func()"
            )
        return cls.calc_reward
    
    @classmethod
    def reset(cls) -> None:
        """Reset class-level state."""
        cls._forget_words = set()
        cls._forget_set = []
        cls._config = None
        cls._preprocessed = False

