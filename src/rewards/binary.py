"""
Binary reward function for GRPO unlearning.
Returns 0.0 if completion contains forbidden words, 1.0 otherwise.
"""
import re
from typing import List

from .base import RewardFunction, RewardConfig


class BinaryReward(RewardFunction):
    """
    Simple binary reward function.
    
    - Returns 0.0 if a completion contains any word from the forget set
    - Returns 1.0 otherwise
    
    No preprocessing required beyond loading the forget words.
    """
    
    _forget_words: set = set()
    _pattern: re.Pattern = None
    _preprocessed: bool = False
    
    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        """
        Load forget words and compile regex pattern.
        
        Args:
            config: RewardConfig with forget_words_file path
        """
        cls._config = config
        cls._forget_words = cls.load_forget_words(config.forget_words_file)
        
        # Pre-compile regex pattern for efficiency
        cls._pattern = re.compile(
            r'\b(?:' + '|'.join(map(re.escape, cls._forget_words)) + r')\b',
            re.IGNORECASE
        )
        cls._preprocessed = True
        
        print(f"[BinaryReward] Loaded {len(cls._forget_words)} forget words")
    
    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        """
        Returns 0.0 if a completion contains any word from the forget dataset,
        otherwise returns 1.0.
        
        Args:
            completions: List of generated completions
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of binary rewards (0.0 or 1.0)
        """
        pattern = BinaryReward._pattern
        return [0.0 if pattern.search(completion) else 1.0 for completion in completions]

