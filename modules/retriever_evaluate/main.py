#!/usr/bin/env python3
"""
Run retriever evaluation with command-line arguments.
"""

import argparse
import logging
from .evaluate_retrievers import RetrieverEvaluator, DEFAULT_RERANKER_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
   
    
    # Create evaluator
    evaluator = RetrieverEvaluator(
        qa_pairs_path="qa_pairs_filtered.csv",
        top_k=10,
        wandb_project="retriever-evaluation",
        wandb_entity="konchakova-s-r-humboldt-universit-t-zu-berlin",
        specific_tables=["test_embeddings_recursive_512_text_embedding_3_large"],
        use_reranker=True,
        reranker_model=DEFAULT_RERANKER_MODEL,
        reranker_tables=["test_embeddings_recursive_512_text_embedding_3_large"],
        use_hybrid_search=True,
        hybrid_alpha=[0.2],  # Evaluate multiple alpha values: alpha * vector_score + (1-alpha) * bm25_score
        hybrid_tables=["test_embeddings_recursive_256_text_embedding_3_large", "test_embeddings_recursive_512_text_embedding_3_large"],
        # Note: BM25 search always uses the 'page_keywords' table and maps results to the target table
    )
    
    try:
        # Run evaluation
        evaluator.run_evaluation()
    finally:
        # Clean up resources
        evaluator.cleanup()
    
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main() 