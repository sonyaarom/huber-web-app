import time
import psycopg2
import logging
from typing import List, Dict, Tuple, Any, Optional
import torch
import numpy as np
from .utils.text_processing import process_text
from psycopg2.extensions import adapt

# Import local modules
from .config import settings
from .utils.embeddings_utils import EmbeddingGenerator

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
        threshold: float = settings.threshold,
        use_reranker: bool = settings.use_reranker,
        reranker_model: str = settings.reranker_model,
        use_hybrid_search: bool = settings.use_hybrid_search,
        hybrid_alpha: float = settings.hybrid_alpha,
        use_full_content: bool = False  # New parameter to control content source
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Model name for generating embeddings
            embedding_method: Method for generating embeddings ('openai' or 'local')
            table_name: PostgreSQL table name containing the embeddings
            top_k: Number of top results to retrieve
            threshold: Minimum similarity score threshold
            use_reranker: Whether to use a reranker
            reranker_model: Model name for the reranker
            use_hybrid_search: Whether to use hybrid search (combining vector search with BM25)
            hybrid_alpha: Weight for vector search in hybrid search (1-alpha is weight for BM25)
            use_full_content: Whether to return full page content instead of chunks
        """
        self.embedding_model = embedding_model
        self.embedding_method = embedding_method
        self.table_name = table_name
        self.top_k = top_k
        self.threshold = threshold
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_alpha = hybrid_alpha
        self.use_full_content = use_full_content
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            method=embedding_method,
            model_name=embedding_model,
            batch_size=1  # For production, process one query at a time
        )
        
        # Initialize reranker if needed
        self.reranker = None
        if self.use_reranker:
            if not CROSS_ENCODER_AVAILABLE:
                logger.warning("Reranker requested but sentence-transformers not installed. Reranking will be disabled.")
                self.use_reranker = False
            else:
                try:
                    logger.info(f"Initializing reranker with model: {self.reranker_model}")
                    self.reranker = CrossEncoder(self.reranker_model)
                    logger.info("Reranker initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize reranker: {e}")
                    self.use_reranker = False
        
        # Database connection parameters
        self.db_config = {
            "dbname": settings.db_name,
            "user": settings.db_username,
            "password": settings.db_password,
            "host": settings.db_host,
            "port": settings.db_port
        }
        
        logger.info(f"Initialized HybridRetriever with table: {self.table_name}")
        logger.info(f"Using embedding model: {self.embedding_model} with method: {self.embedding_method}")
        if self.use_reranker:
            logger.info(f"Using reranker: {self.reranker_model}")
        if self.use_hybrid_search:
            logger.info(f"Using hybrid search with alpha: {self.hybrid_alpha}")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query text
            
        Returns:
            List of dictionaries containing document information and similarity scores
        """
        start_time = time.time()
        
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
        # Convert embedding to PostgreSQL format
        if hasattr(query_embedding, 'tolist'):
            embedding_list = query_embedding.tolist()
        else:
            embedding_list = query_embedding  # Already a list
        
        # Prepare the embedding as a string
        embedding_str = str(embedding_list)
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # First, check if pgvector extension is installed
        try:
            check_query = "SELECT 1 FROM pg_extension WHERE extname = 'vector';"
            cursor.execute(check_query)
            if cursor.fetchone() is None:
                logger.warning("pgvector extension is not installed in the database")
        except Exception as e:
            logger.error(f"Error checking pgvector extension: {e}")
        
        # Check PostgreSQL version
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0]
        logger.info(f"PostgreSQL version: {pg_version}")

        # Check pgvector version
        try:
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            pgvector_version = cursor.fetchone()[0]
            logger.info(f"pgvector version: {pgvector_version}")
        except Exception as e:
            logger.error(f"Error checking pgvector version: {e}")
        
        # Check table structure
        try:
            cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{self.table_name}';")
            columns = cursor.fetchall()
            logger.info(f"Table structure for {self.table_name}: {columns}")
        except Exception as e:
            logger.error(f"Error checking table structure: {e}")
        
        # Determine the number of results to retrieve
        retrieval_limit = self.top_k
        # If reranking is used, we need more results to rerank
        if self.use_reranker:
            retrieval_limit = self.top_k * 2
        if self.use_hybrid_search:
            # For hybrid search, we need more results to merge
            retrieval_limit = self.top_k * 3
        
        # Try a simpler approach with L2 distance
        vector_query = f"""
            SELECT id, chunk_text, 
                   embedding <-> '{embedding_str}'::vector AS distance
            FROM {self.table_name}
            ORDER BY distance
            LIMIT {retrieval_limit};
        """
        
        cursor.execute(vector_query)
        
        vector_results = cursor.fetchall()
        
        if self.use_hybrid_search:
            # BM25 search using the existing function
            bm25_query = """
                SELECT * FROM bm25_search(%s, %s, 1.2, 0.75, 'page_keywords');
            """
            
            bm25_query_text = process_text(query)

            logger.info(f"BM25 query: {bm25_query_text}")
            cursor.execute(bm25_query, (bm25_query_text, retrieval_limit * 2))
            bm25_results = cursor.fetchall()
            
            # Get all IDs from vector results for mapping
            vector_ids = [row[0] for row in vector_results]
            
            # Create dictionaries for easier lookup
            vector_scores = {row[0]: row[2] for row in vector_results}
            vector_contents = {row[0]: row[1] for row in vector_results}
            
            # Create a mapping of BM25 scores by ID
            bm25_scores = {}
            
            # For each BM25 result, find the corresponding document in our table
            for bm25_row in bm25_results:
                bm25_id = bm25_row[0]  # ID from page_keywords
                bm25_score = bm25_row[1]  # Score from BM25
                
                # Check if this ID exists in our vector results
                if bm25_id in vector_ids:
                    bm25_scores[bm25_id] = bm25_score
            
            # Get all unique document IDs that have both vector and BM25 scores
            all_doc_ids = set(vector_scores.keys()) & set(bm25_scores.keys())
            
            # If we don't have any overlap, fall back to vector search only
            if not all_doc_ids:
                logger.warning(f"No overlap between BM25 and vector search results. Using vector search only.")
                doc_ids = [row[0] for row in vector_results]
                scores = [row[2] for row in vector_results]
                contents = [row[1] for row in vector_results]
            else:
                logger.info(f"Found {len(all_doc_ids)} overlapping document IDs between BM25 and vector search results")
                # Create combined scores
                combined_scores = {}
                for doc_id in all_doc_ids:
                    # Get scores
                    vector_score = vector_scores[doc_id]
                    bm25_score = bm25_scores[doc_id]
                    
                    # Normalize BM25 score to [0, 1] range if we have any scores
                    if bm25_scores:
                        max_bm25 = max(bm25_scores.values())
                        if max_bm25 > 0:
                            bm25_score = bm25_score / max_bm25
                    
                    # Combine scores using alpha parameter
                    combined_scores[doc_id] = (
                        self.hybrid_alpha * vector_score + 
                        (1 - self.hybrid_alpha) * float(bm25_score)
                    )
                
                # Sort by combined score
                sorted_results = sorted(
                    combined_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Get top results
                top_results = sorted_results[:retrieval_limit]
                
                # Get doc_ids, scores, and contents
                doc_ids = [doc_id for doc_id, _ in top_results]
                scores = [score for _, score in top_results]
                contents = [vector_contents.get(doc_id, "") for doc_id in doc_ids]
        else:
            # Use vector search results directly
            doc_ids = [row[0] for row in vector_results]
            scores = [row[2] for row in vector_results]
            contents = [row[1] for row in vector_results]
        
        # Get additional metadata for the documents
        if doc_ids:
            placeholders = ', '.join(['%s'] * len(doc_ids))
            
            if self.use_full_content:
                # Get full content from page_keywords
                metadata_query = f"""
                    SELECT pk.id, pc.url, pk.raw_text 
                    FROM page_keywords pk
                    JOIN page_content pc ON pk.id = pc.id
                    WHERE pk.id IN ({placeholders});
                """
            else:
                # Just get URL as before
                metadata_query = f"""
                    SELECT id, url
                    FROM page_content 
                    WHERE id IN ({placeholders});
                """
                
            cursor.execute(metadata_query, doc_ids)
            metadata_results = cursor.fetchall()
            
            # Create a dictionary mapping doc_id to metadata
            if self.use_full_content:
                metadata = {row[0]: {"url": row[1], "full_content": row[2]} for row in metadata_results}
            else:
                metadata = {row[0]: {"url": row[1]} for row in metadata_results}
        else:
            metadata = {}
        
        cursor.close()
        conn.close()
        
        # Apply reranking if enabled
        if self.use_reranker and doc_ids:
            # Prepare pairs for reranking
            pairs = [[query, content] for content in contents]
            
            # Rerank using cross-encoder
            logger.info(f"Reranking {len(pairs)} results")
            rerank_scores = self.reranker.predict(pairs)
            
            # Create (id, score, content) tuples and sort by reranker score
            id_score_content = list(zip(doc_ids, rerank_scores, contents))
            id_score_content.sort(key=lambda x: x[1], reverse=True)
            
            # Extract sorted doc_ids, scores, and contents
            doc_ids = [item[0] for item in id_score_content]
            scores = [item[1] for item in id_score_content]
            contents = [item[2] for item in id_score_content]
            
            # Normalize reranker scores to [0, 1] range
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = [(score - min_score) / (max_score - min_score) for score in scores]
                else:
                    scores = [1.0 for _ in scores]  # If all scores are the same
                
                # Alternative: use sigmoid normalization instead
                # scores = [sigmoid(score) for score in scores]
        
        # Apply threshold filtering if specified
        if self.threshold is not None:
            # Filter results by threshold
            filtered_results = [(doc_id, score, content) for doc_id, score, content in zip(doc_ids, scores, contents) if score > self.threshold]
            
            if filtered_results:
                # If we have results above threshold, use them
                doc_ids = [item[0] for item in filtered_results]
                scores = [item[1] for item in filtered_results]
                contents = [item[2] for item in filtered_results]
                logger.info(f"Filtered to {len(doc_ids)} results above threshold {self.threshold}")
            else:
                # If no results above threshold, log a warning but return the best result we have
                if doc_ids:
                    best_id = doc_ids[0]
                    best_score = scores[0]
                    best_content = contents[0]
                    doc_ids = [best_id]
                    scores = [best_score]
                    contents = [best_content]
                    logger.warning(f"No results above threshold {self.threshold}. Returning best result with score {best_score}")
                else:
                    logger.warning(f"No results found")
        
        # Limit to top_k
        doc_ids = doc_ids[:self.top_k]
        scores = scores[:self.top_k]
        contents = contents[:self.top_k]
        
        # Prepare the final results
        results = []
        for i, (doc_id, score, content) in enumerate(zip(doc_ids, scores, contents)):
            result = {
                "id": doc_id,
                "score": float(score),
                "rank": i + 1
            }
            
            # Add metadata if available
            if doc_id in metadata:
                result["url"] = metadata[doc_id]["url"]
                
                # Use full content if requested and available
                if self.use_full_content and "full_content" in metadata[doc_id]:
                    result["content"] = metadata[doc_id]["full_content"]
                else:
                    result["content"] = content
            else:
                result["content"] = content
            
            results.append(result)
        
        query_time = time.time() - start_time
        logger.info(f"Retrieved {len(results)} results in {query_time:.2f} seconds")
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'embedding_generator'):
            self.embedding_generator.cleanup()
        
        if self.use_reranker and hasattr(self, 'reranker'):
            # Free up GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# def create_retriever(use_full_content: bool = False) -> HybridRetriever:
#     """Factory function to create a retriever with default settings."""
#     return HybridRetriever(use_full_content=use_full_content) 


# if __name__ == "__main__":
#     retriever = create_retriever(use_full_content=False)
#     results = retriever.retrieve("Stefan")
#     print(results)
#     retriever.cleanup()