import os
import time
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import psycopg2
import numpy as np

# Optional reranker import
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Reranking will not be available.")

# Local modules (adjust relative imports as needed)
from .config import settings
from .embedding_utils import EmbeddingGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = settings.reranker_model

class Retriever:
    """
    Production retriever for RAG systems.
    
    Supports vector search with optional hybrid BM25 search and cross-encoder reranking.
    """
    def __init__(
        self,
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        use_hybrid_search: bool = False,
        hybrid_alpha: Union[float, List[float]] = settings.alpha,
        embedding_provider: str = settings.embedding_provider,
        embedding_model: str = settings.embedding_model,
        query_type: str = "embeddings"
    ):

        self.top_k = top_k
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_alpha = float(hybrid_alpha) if isinstance(hybrid_alpha, (int, float)) else hybrid_alpha[0]
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.query_type = query_type
        # Initialize embedding generator
        self.embedding_generator = None
        
        # Initialize reranker if requested and available.
        if self.use_reranker:
            if not CROSS_ENCODER_AVAILABLE:
                logger.warning("Reranker requested but sentence-transformers not installed. Disabling reranking.")
                self.use_reranker = False
                self.reranker = None
            else:
                try:
                    logger.info("Initializing reranker with model: %s", self.reranker_model)
                    self.reranker = CrossEncoder(self.reranker_model)
                    logger.info("Reranker initialized successfully")
                except Exception as e:
                    logger.error("Failed to initialize reranker: %s", e)
                    self.use_reranker = False
                    self.reranker = None
        else:
            self.reranker = None
        
        
        # Database connection configuration
        self.db_config = {
            "dbname": settings.db_name,
            "user": settings.db_username,
            "password": settings.db_password,
            "host": settings.db_host,
            "port": settings.db_port
        }

    
    def _get_embedding_generator(self) -> EmbeddingGenerator:
        """
        Get or create an embedding generator for the given table.
        
        Args:
            table_name: Name of the table.
            
        Returns:
            An EmbeddingGenerator instance.
        """
        if self.embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator(
                method=self.embedding_provider,
                model_name=self.embedding_model,
                batch_size=8,
                api_key=None  # Uses settings or environment if needed
            )
            logger.info("Created embedding generator for %s using model %s", self.embedding_model, self.embedding_provider)
        return self.embedding_generator
    

    def _retrieve_text_from_page_keywords(self, doc_ids: List[str]) -> List[str]:
        """
        Retrieve text from the database for the given document IDs.
        """
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        query = f"""
            SELECT raw_text FROM {settings.page_content_table}
            WHERE id = ANY(%s);
        """
        cursor.execute(query, (doc_ids,))
        results = cursor.fetchall()
        return [row[0] for row in results]

    def _retrieve_text_from_page_embeddings(self, doc_ids: List[str]) -> List[str]:
        """
        Retrieve text from the database for the given document IDs.
        """
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()

        query = f"""
            SELECT chunk_text FROM {settings.page_embeddings_table}
            WHERE id = ANY(%s);
        """
        cursor.execute(query, (doc_ids,))
        results = cursor.fetchall()
        return [row[0] for row in results]
    
        
    def retrieve(
        self,
        question: str,
        table_name: str
    ) -> Tuple[List[str], List[float], float]:
        """
        Retrieve similar documents from a specified table.
        
        Args:
            question: The query string.
            table_name: PostgreSQL table to search in.
        
        Returns:
            A tuple of (document_ids, similarity_scores, query_time).
        """
        start_time = time.time()
        
        # Get embedding for the question.
        embedding_generator = self._get_embedding_generator()
        question_embedding = embedding_generator.generate_embeddings([question])[0]
        embedding_list = question_embedding.tolist() if hasattr(question_embedding, "tolist") else question_embedding
        
        # Connect to PostgreSQL.
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Adjust the retrieval limit based on optional reranking or hybrid search.
        retrieval_limit = self.top_k
        if self.use_reranker:
            retrieval_limit = self.top_k * 2
        if self.use_hybrid_search:
            retrieval_limit = self.top_k * 3
        
        logger.info(f"Using retrieval_limit={retrieval_limit}, top_k={self.top_k}, use_hybrid_search={self.use_hybrid_search}, use_reranker={self.use_reranker}")
        
        # Vector search query.
        vector_query = f"""
            SELECT id, chunk_text, 1 - (embedding <=> %s::vector) AS similarity
            FROM {table_name}
            ORDER BY similarity DESC
            LIMIT {retrieval_limit};
        """
        embedding_str = str(embedding_list)
        cursor.execute(vector_query, (embedding_str,))
        vector_results = cursor.fetchall()
        
        # Unpack initial vector search results.
        doc_ids = [row[0] for row in vector_results]
        scores = [row[2] for row in vector_results]
        contents_list = [row[1] for row in vector_results]
        
        logger.info(f"Vector search found {len(vector_results)} results")
        logger.info(f"Initial vector scores range: {min(scores) if scores else 0:.4f} to {max(scores) if scores else 0:.4f}")
        
        # Optional hybrid search: merge BM25 scores.
        if self.use_hybrid_search:
            bm25_query = """
                SELECT * FROM bm25_search(%s, %s, 1.2, 0.75, 'page_keywords');
            """
            cursor.execute(bm25_query, (question, retrieval_limit * 2))
            bm25_results = cursor.fetchall()
            
            logger.info(f"BM25 search found {len(bm25_results)} results")
            
            vector_ids = [row[0] for row in vector_results]
            vector_scores = {row[0]: row[2] for row in vector_results}
            bm25_scores = {}
            for bm25_row in bm25_results:
                bm25_id = bm25_row[0]
                bm25_score = bm25_row[1]
                if bm25_id in vector_ids:
                    bm25_scores[bm25_id] = bm25_score
            
            # If there is an overlap, combine scores.
            common_ids = set(vector_scores.keys()) & set(bm25_scores.keys())
            logger.info(f"Found {len(common_ids)} documents in common between vector and BM25 search")
            
            if common_ids:
                combined_scores = {}
                
                # Normalize vector scores to [0, 1] range
                max_vector = max(vector_scores.values()) if vector_scores else 1.0
                min_vector = min(vector_scores.values()) if vector_scores else 0.0
                vector_range = max_vector - min_vector if max_vector > min_vector else 1.0
                
                # Normalize BM25 scores to [0, 1] range
                max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
                min_bm25 = min(bm25_scores.values()) if bm25_scores else 0.0
                bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
                
                logger.info(f"Vector score range: [{min_vector:.4f}, {max_vector:.4f}]")
                logger.info(f"BM25 score range: [{min_bm25:.4f}, {max_bm25:.4f}]")
                
                for doc_id in common_ids:
                    # Min-max normalization for both scores
                    norm_vector = (vector_scores[doc_id] - min_vector) / vector_range if vector_range > 0 else 0.5
                    norm_bm25 = (bm25_scores[doc_id] - min_bm25) / bm25_range if bm25_range > 0 else 0.5
                    
                    # Combine normalized scores with alpha weighting
                    combined_scores[doc_id] = (
                        self.hybrid_alpha * norm_vector +
                        (1 - self.hybrid_alpha) * norm_bm25
                    )
                
                sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                doc_ids = [doc_id for doc_id, _ in sorted_results][:self.top_k]
                scores = [score for _, score in sorted_results][:self.top_k]
                
                # Log the normalized scores
                logger.info(f"Combined {len(common_ids)} documents with hybrid search (alpha={self.hybrid_alpha:.2f})")
                logger.info(f"Normalized hybrid scores range: {min(scores) if scores else 0:.4f} to {max(scores) if scores else 0:.4f}")
                
                if self.use_reranker:
                    # Retrieve document contents for reranking.
                    content_query = f"""
                        SELECT id, chunk_text FROM {table_name}
                        WHERE id = ANY(%s);
                    """
                    cursor.execute(content_query, (doc_ids,))
                    content_results = cursor.fetchall()
                    contents = {row[0]: row[1] for row in content_results}
                    contents_list = [contents.get(doc_id, "") for doc_id in doc_ids]
                    if self.query_type == "keywords":
                        contents_list = self._retrieve_text_from_page_keywords(doc_ids)
                    elif self.query_type == "embeddings":
                        contents_list = self._retrieve_text_from_page_embeddings(doc_ids)
            else:
                logger.warning("No overlap between BM25 and vector search results; using vector search only.")
                doc_ids = doc_ids[:self.top_k]
                scores = scores[:self.top_k]
                if self.query_type == "keywords":
                    contents_list = self._retrieve_text_from_page_keywords(doc_ids)
                elif self.query_type == "embeddings":
                    contents_list = self._retrieve_text_from_page_embeddings(doc_ids)
        
        cursor.close()
        conn.close()
        
        # Optional reranking with cross-encoder.
        if self.use_reranker and doc_ids:
            pairs = [[question, content] for content in contents_list]
            logger.info(f"Reranking {len(pairs)} results")
            
            # Store pre-reranking scores for comparison
            pre_rerank_scores = scores.copy()
            
            rerank_scores = self.reranker.predict(pairs)
            
            # Normalize reranker scores to [0,1] range
            if len(rerank_scores) > 0:
                min_rerank = min(rerank_scores)
                max_rerank = max(rerank_scores)
                rerank_range = max_rerank - min_rerank
                
                logger.info(f"Raw reranker scores range: [{min_rerank:.4f}, {max_rerank:.4f}]")
                
                # Apply min-max normalization to reranker scores
                if rerank_range > 0:
                    normalized_rerank_scores = [(score - min_rerank) / rerank_range for score in rerank_scores]
                else:
                    # If all scores are the same, set them all to 1.0 (perfect match)
                    normalized_rerank_scores = [1.0 for _ in rerank_scores]
                
                # Use normalized scores for ranking
                id_score_pairs = list(zip(doc_ids, normalized_rerank_scores))
            else:
                id_score_pairs = list(zip(doc_ids, rerank_scores))
                
            id_score_pairs.sort(key=lambda x: x[1], reverse=True)
            id_score_pairs = id_score_pairs[:self.top_k]
            doc_ids = [pair[0] for pair in id_score_pairs]
            scores = [pair[1] for pair in id_score_pairs]
            if self.query_type == "keywords":
                contents_list = self._retrieve_text_from_page_keywords(doc_ids)
            elif self.query_type == "embeddings":
                contents_list = self._retrieve_text_from_page_embeddings(doc_ids)
            
            logger.info(f"Pre-reranking scores range: {min(pre_rerank_scores) if pre_rerank_scores else 0:.4f} to {max(pre_rerank_scores) if pre_rerank_scores else 0:.4f}")
            logger.info(f"Post-reranking scores range: {min(scores) if scores else 0:.4f} to {max(scores) if scores else 0:.4f}")
        
        query_time = time.time() - start_time
        return doc_ids, scores, query_time, contents_list
    
    def cleanup(self):
        """Clean up embedding generator resources."""
        if self.embedding_generator:
            self.embedding_generator.cleanup()


# Example usage:
if __name__ == "__main__":
    retriever = Retriever(top_k=10, use_reranker=True, use_hybrid_search=True, hybrid_alpha=0.6)
    sample_question = "Who is Stefan Lessmann?"
    sample_table = "page_embeddings_alpha"
    
    doc_ids, scores, elapsed, contents_list = retriever.retrieve(sample_question, sample_table)
    logger.info("Retrieved Document IDs: %s", doc_ids)
    logger.info("Similarity Scores: %s", scores)
    logger.info("Contents List: %s", contents_list)
    logger.info("Query Time: %.4f seconds", elapsed)
    
    retriever.cleanup()
