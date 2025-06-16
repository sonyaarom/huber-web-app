import time
import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from .utils.text_processing import process_text

from hubert.config import settings
from hubert.common.utils.embedding_utils import EmbeddingGenerator
from hubert.db.postgres_storage import PostgresStorage

# Check if sentence-transformers is available for reranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Reranking will not be available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define sigmoid function for score normalization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class HybridRetriever:
    """
    Production-ready retriever that combines vector search with BM25 and applies reranking.
    
    This retriever is configured to use:
    - OpenAI's text-embedding-3-large model for embeddings
    - Recursive chunking with 512 token chunks
    - Hybrid search with alpha=0.5 (50% vector search, 50% BM25)
    - Cross-encoder reranking
    """
    
    def __init__(
        self,
        embedding_model: str = settings.embedding_model,
        embedding_method: str = settings.embedding_method,
        table_name: str = settings.table_name,
        top_k: int = settings.top_k,
        use_reranker: bool = settings.use_reranker,
        reranker_model: str = settings.reranker_model,
        use_hybrid_search: bool = settings.use_hybrid_search,
        hybrid_alpha: float = settings.hybrid_alpha,
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Model name for generating embeddings
            embedding_method: Method for generating embeddings ('openai' or 'local')
            table_name: PostgreSQL table name containing the embeddings
            top_k: Number of top results to retrieve
            use_reranker: Whether to use a reranker
            reranker_model: Model name for the reranker
            use_hybrid_search: Whether to use hybrid search (combining vector search with BM25)
            hybrid_alpha: Weight for vector search in hybrid search (1-alpha is weight for BM25)
        """
        self.embedding_model = embedding_model
        self.embedding_method = embedding_method
        self.table_name = table_name
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_alpha = hybrid_alpha
        
        self.storage = PostgresStorage()
        self.embedding_generator = EmbeddingGenerator(
            method=embedding_method,
            model_name=embedding_model,
        )
        
        self.reranker = None
        if self.use_reranker and CROSS_ENCODER_AVAILABLE:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.reranker = CrossEncoder(self.reranker_model, device=device)
                logger.info(f"Reranker initialized on device: {device}")
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}")
                self.use_reranker = False
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query text
            
        Returns:
            List of dictionaries containing document information and similarity scores
        """
        if not query or not query.strip():
            logger.warning("Received an empty query. Returning empty results.")
            return []
            
        start_time = time.time()
        
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
        retrieval_limit = self.top_k * 3 if self.use_hybrid_search else self.top_k * 2 if self.use_reranker else self.top_k

        vector_results = self.storage.vector_search(self.table_name, query_embedding, limit=retrieval_limit)
        
        if self.use_hybrid_search:
            bm25_query_text = process_text(query)
            bm25_results = self.storage.keyword_search(bm25_query_text, limit=retrieval_limit)
            
            # Combine and normalize scores
            all_docs: Dict[Any, Dict[str, Any]] = {}
            for res in vector_results:
                key = (res['url'], res['content'])
                if key not in all_docs:
                    all_docs[key] = {'vec_score': 0, 'bm25_score': 0, 'doc': res}
                all_docs[key]['vec_score'] = res.get('similarity', 0)

            for res in bm25_results:
                key = (res['url'], res['content'])
                if key not in all_docs:
                    all_docs[key] = {'vec_score': 0, 'bm25_score': 0, 'doc': res}
                all_docs[key]['bm25_score'] = res.get('rank', 0)

            combined_results = []
            for key, scores in all_docs.items():
                combined_score = (self.hybrid_alpha * scores['vec_score']) + ((1 - self.hybrid_alpha) * scores['bm25_score'])
                doc = scores['doc']
                doc['score'] = combined_score
                combined_results.append(doc)

            # Sort by combined score
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            final_results = combined_results
        else:
            final_results = vector_results
            for res in final_results:
                res['score'] = res.pop('similarity')

        if self.use_reranker and self.reranker:
            pairs = [[query, res['content']] for res in final_results]
            if pairs:
                scores = self.reranker.predict(pairs)
                for i, res in enumerate(final_results):
                    res['reranked_score'] = scores[i]
                final_results.sort(key=lambda x: x.get('reranked_score', 0), reverse=True)

        logger.info(f"Retrieval took {time.time() - start_time:.2f} seconds.")
        return final_results[:self.top_k]
    
    def cleanup(self):
        """Clean up resources."""
        self.storage.close()
        self.embedding_generator.cleanup()


def create_retriever() -> HybridRetriever:
    """Factory function to create a retriever with default settings."""
    return HybridRetriever()


if __name__ == "__main__":
    retriever = create_retriever()
    try:
        sample_query = "What are the admission requirements for the computer science program?"
        results = retriever.retrieve(sample_query)
        print(f"Query: {sample_query}")
        for res in results:
            print(f"Score: {res.get('reranked_score', res.get('score')):.4f} - Chunk: {res['chunk'][:100]}...")
    finally:
        retriever.cleanup()