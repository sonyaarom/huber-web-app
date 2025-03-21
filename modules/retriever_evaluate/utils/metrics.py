from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_mrr(question_id: str, general_ids: List[str]) -> Tuple[int, float]:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        question_id: The ID of the correct document
        general_ids: List of retrieved document IDs, ordered by relevance
        
    Returns:
        Tuple of (rank, reciprocal_rank)
    """
    if question_id in general_ids:
        rank = general_ids.index(question_id) + 1
        reciprocal_rank = 1 / rank
    else:
        rank = 0
        reciprocal_rank = 0
    return rank, reciprocal_rank


def calculate_hit_at_k(question_id: str, general_ids: List[str], k: int) -> int:
    """
    Calculate Hit@K metric.
    
    Args:
        question_id: The ID of the correct document
        general_ids: List of retrieved document IDs, ordered by relevance
        k: The number of top results to consider
        
    Returns:
        1 if the correct document is in the top k results, 0 otherwise
    """
    return int(question_id in general_ids[:k]) 