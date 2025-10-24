# FILE: evaluation.py (NEW)
# PURPOSE: Evaluation utilities for collocation extraction

"""
Evaluation utilities for collocation systems.

Supports:
- Precision/Recall against gold standard
- Ranking metrics (MRR, MAP)
- Inter-annotator agreement utilities
"""

import logging
from typing import List, Set, Tuple, Dict
from .collocation import CollocationResult

logger = logging.getLogger(__name__)

def compute_precision_recall(predicted: List[CollocationResult],
                              gold_standard: Set[Tuple[str, ...]],
                              top_k: int = None) -> Dict[str, float]:
    """
    Compute precision and recall against gold standard.
    
    Args:
        predicted: List of predicted collocations (sorted by score)
        gold_standard: Set of gold standard collocations (word tuples)
        top_k: Evaluate only top K predictions (default: all)
    
    Returns:
        Dict with 'precision', 'recall', 'f1' keys
    """
    if top_k:
        predicted = predicted[:top_k]
    
    predicted_set = {coll.words for coll in predicted}
    
    true_positives = len(predicted_set & gold_standard)
    false_positives = len(predicted_set - gold_standard)
    false_negatives = len(gold_standard - predicted_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def compute_mean_reciprocal_rank(predicted: List[CollocationResult],
                                  gold_standard: Set[Tuple[str, ...]]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR = 1/|Q| * sum(1/rank_i)
    
    where rank_i is the rank of the first relevant item.
    
    Args:
        predicted: List of predicted collocations (sorted by score)
        gold_standard: Set of gold standard collocations
    
    Returns:
        MRR score (0-1)
    """
    for rank, coll in enumerate(predicted, start=1):
        if coll.words in gold_standard:
            return 1.0 / rank
    return 0.0

def compute_average_precision(predicted: List[CollocationResult],
                               gold_standard: Set[Tuple[str, ...]]) -> float:
    """
    Compute Average Precision (AP).
    
    AP = (1/|R|) * sum_{k=1..n} P(k) * rel(k)
    
    where R = relevant items, P(k) = precision at k, rel(k) = 1 if item k is relevant
    
    Args:
        predicted: List of predicted collocations (sorted by score)
        gold_standard: Set of gold standard collocations
    
    Returns:
        AP score (0-1)
    """
    if not gold_standard:
        return 0.0
    
    num_relevant = 0
    sum_precisions = 0.0
    
    for rank, coll in enumerate(predicted, start=1):
        if coll.words in gold_standard:
            num_relevant += 1
            precision_at_k = num_relevant / rank
            sum_precisions += precision_at_k
    
    return sum_precisions / len(gold_standard) if gold_standard else 0.0

def evaluate_ranking(predicted: List[CollocationResult],
                     gold_standard: Set[Tuple[str, ...]],
                     top_k_values: List[int] = [10, 20, 50]) -> Dict:
    """
    Comprehensive ranking evaluation.
    
    Args:
        predicted: List of predicted collocations
        gold_standard: Set of gold collocations
        top_k_values: List of K values for precision@K, recall@K
    
    Returns:
        Dict with multiple metrics
    """
    results = {
        'mrr': compute_mean_reciprocal_rank(predicted, gold_standard),
        'map': compute_average_precision(predicted, gold_standard),
    }
    
    for k in top_k_values:
        pr_metrics = compute_precision_recall(predicted, gold_standard, top_k=k)
        results[f'precision@{k}'] = pr_metrics['precision']
        results[f'recall@{k}'] = pr_metrics['recall']
        results[f'f1@{k}'] = pr_metrics['f1']
    
    return results

def load_gold_standard(file_path: str) -> Set[Tuple[str, ...]]:
    """
    Load gold standard collocations from file.
    
    Expected format: One collocation per line, words separated by space
    Example:
        noh k'khìm
        luum-na ci
    
    Args:
        file_path: Path to gold standard file
    
    Returns:
        Set of gold standard collocations (word tuples)
    """
    gold_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                words = tuple(line.split())
                gold_set.add(words)
    return gold_set