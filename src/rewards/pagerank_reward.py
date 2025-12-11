"""
PageRank-weighted reward function for GRPO unlearning.
Uses personalized PageRank to weight penalties based on semantic importance.
"""
import re
from typing import List, Dict, Optional, Any

import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .base import RewardFunction, RewardConfig


class PageRankWeightedReward(RewardFunction):
    """
    PageRank-weighted reward function.
    
    - If NO forbidden words are found: reward = 1.0 (perfect)
    - If forbidden words are found: reward = 1.0 - sum(penalty_weights for matched words)
    - The penalty is proportional to the PageRank importance of the matched words
    - Result is clamped to [0, 1]
    
    Higher PageRank words (more semantically connected to the main entity) 
    cause larger penalties when generated.
    
    Requires preprocessing to compute embeddings and PageRank weights.
    
    Extra params (via config.extra_params):
        - similarity_threshold_quantile: float (default 0.75) - Keep top X% of similarities
    """
    
    _forget_set: List[str] = []
    _penalty_weights: Dict[str, float] = {}
    _preprocessed: bool = False
    _config: Optional[RewardConfig] = None
    _similarity_threshold_quantile: float = 0.75
    
    @classmethod
    def _get_embedding(cls, text: str, tokenizer: Any, model: Any) -> np.ndarray:
        """Get embedding for a single text."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Mean of last layer hidden states for sentence embedding
        return outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
    
    @classmethod
    def _get_embeddings(cls, texts: List[str], tokenizer: Any, model: Any) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = cls._get_embedding(text, tokenizer, model)
            embeddings.append(embedding)
        return embeddings
    
    @classmethod
    def _calculate_similarity_matrix(cls, embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate thresholded cosine similarity matrix."""
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Set diagonal to -1 (no self-similarity)
        similarity_matrix[np.diag_indices_from(similarity_matrix)] = -1
        
        # Threshold: keep only top (1 - quantile)% similarities
        threshold = np.quantile(similarity_matrix, cls._similarity_threshold_quantile)
        similarity_matrix[similarity_matrix < threshold] = -1
        
        # Normalize to [0, 1] and binarize
        similarity_matrix = (similarity_matrix + 1) / 2
        similarity_matrix[similarity_matrix != 0] = 1
        
        print(f"[PageRankWeightedReward] Similarity matrix shape: {similarity_matrix.shape}")
        print(f"[PageRankWeightedReward] Similarity threshold quantile: {cls._similarity_threshold_quantile}")
        return similarity_matrix
    
    @classmethod
    def _compute_pagerank_weights(cls, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute personalized PageRank weights.
        
        Uses the first element in forget_set as the personalization node.
        """
        G = nx.from_numpy_array(similarity_matrix)
        
        # Rename nodes to use actual term names
        mapping = {i: cls._forget_set[i] for i in range(len(cls._forget_set))}
        G = nx.relabel_nodes(G, mapping)
        
        # Personalized PageRank starting from the main entity (first element)
        start_node = cls._forget_set[0]
        
        print(f"[PageRankWeightedReward] FORGET_SET has {len(cls._forget_set)} terms")
        print(f"[PageRankWeightedReward] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"[PageRankWeightedReward] Personalization node: '{start_node}'")
        
        if start_node not in G.nodes():
            raise ValueError(f"Start node '{start_node}' not in graph: {list(G.nodes())[:10]}...")
        
        # Create personalization vector: only start_node gets probability 1.0
        personalization = {term: 0.0 for term in cls._forget_set}
        personalization[start_node] = 1.0
        
        # Run Personalized PageRank
        pagerank_scores = nx.pagerank(G, weight='weight', personalization=personalization)
        
        # Normalize to [0, 1] range
        max_score = max(pagerank_scores.values())
        min_score = min(pagerank_scores.values())
        
        penalty_weights = {
            term: (score - min_score) / (max_score - min_score) if max_score > min_score else 0.0
            for term, score in pagerank_scores.items()
        }
        
        return penalty_weights
    
    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        """
        Compute PageRank weights from forget set embeddings.
        
        This preprocessing step:
        1. Loads the forget set with target entity as first element
        2. Computes embeddings for all terms using the provided model
        3. Builds a similarity graph
        4. Computes personalized PageRank weights
        
        Args:
            config: RewardConfig with model, tokenizer, and file paths
        """
        cls._config = config
        
        if config.model is None or config.tokenizer is None:
            raise ValueError("PageRankWeightedReward requires model and tokenizer in config")
        
        # Load extra params from config
        if config.extra_params:
            cls._similarity_threshold_quantile = config.extra_params.get(
                "similarity_threshold_quantile", 0.75
            )
        
        # Load forget set with target entity first
        cls._forget_set = cls.load_forget_set(
            config.forget_words_file, 
            config.target_entity
        )
        
        print(f"[PageRankWeightedReward] Computing embeddings for {len(cls._forget_set)} terms...")
        embeddings = cls._get_embeddings(cls._forget_set, config.tokenizer, config.model)
        
        print("[PageRankWeightedReward] Building similarity matrix...")
        similarity_matrix = cls._calculate_similarity_matrix(embeddings)
        
        print("[PageRankWeightedReward] Computing PageRank weights...")
        cls._penalty_weights = cls._compute_pagerank_weights(similarity_matrix)
        
        cls._preprocessed = True
        print(f"[PageRankWeightedReward] Preprocessing complete. Top 5 penalty weights:")
        sorted_weights = sorted(cls._penalty_weights.items(), key=lambda x: x[1], reverse=True)
        for term, weight in sorted_weights[:5]:
            print(f"  {term}: {weight:.4f}")
    
    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        """
        PageRank-weighted reward function for GRPO unlearning.
        
        - If NO forbidden words are found: reward = 1.0 (perfect)
        - If forbidden words are found: reward = 1.0 - sum(penalty_weights for matched words)
        - Result is clamped to [0, 1]
        
        Args:
            completions: List of generated completions
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of reward scores in [0, 1]
        """
        rewards = []
        forget_set = PageRankWeightedReward._forget_set
        penalty_weights = PageRankWeightedReward._penalty_weights
        
        for completion in completions:
            total_penalty = 0.0
            
            # Check each term in forget_set
            for term in forget_set:
                # Case-insensitive word boundary match
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                if pattern.search(completion):
                    total_penalty += penalty_weights.get(term, 0.0)
            
            # Reward = 1.0 - penalty, clamped to [0, 1]
            reward = max(0.0, min(1.0, 1.0 - total_penalty))
            rewards.append(reward)
        
        return rewards

