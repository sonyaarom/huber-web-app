import logging
from .evaluate_retrievers import RetrieverEvaluator
from .config import settings


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():

    evaluator = RetrieverEvaluator(
        qa_pairs_path=settings.qa_pairs_path,
        top_k=settings.top_k,
        wandb_project="retriever-evaluation",
        wandb_entity=settings.wandb_entity,
        use_reranker=True,
        reranker_model=settings.reranker_model,
        use_hybrid_search=True,
        hybrid_alpha=[0.2, 0.5, 0.8],  
        threshold=settings.threshold
    )
    
    try:
        evaluator.run_evaluation()
    finally:
        evaluator.cleanup()
    
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main() 