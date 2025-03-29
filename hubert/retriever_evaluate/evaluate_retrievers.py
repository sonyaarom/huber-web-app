import os
import time
import csv
import pandas as pd
import psycopg2
import wandb
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Reranking will not be available.")

# Import local modules
from .config import settings
from .utils.embeddings_utils import EmbeddingGenerator
from .utils.metrics import calculate_mrr, calculate_hit_at_k

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetrieverEvaluator:
    """Evaluates different retriever combinations stored in PostgreSQL tables."""
    
    def __init__(
        self, 
        qa_pairs_path: str = settings.qa_pairs_path,
        top_k: int = settings.top_k,
        wandb_project: str = "retriever-evaluation",
        wandb_entity: str = None,
        table_filter: Optional[str] = None,
        specific_tables: Optional[List[str]] = None,
        use_reranker: bool = False,
        reranker_model: str = settings.reranker_model,
        reranker_tables: Optional[List[str]] = None,
        use_hybrid_search: bool = False,
        hybrid_alpha: Union[float, List[float]] = settings.hybrid_alpha,
        hybrid_tables: Optional[List[str]] = None,
        threshold: Optional[float] = settings.threshold
    ):
        """
        Initialize the evaluator.
        
        Args:
            qa_pairs_path: Path to the CSV file containing QA pairs for evaluation
            top_k: Number of top results to retrieve
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity name
            table_filter: Optional filter for table names (e.g., 'recursive' to only evaluate recursive tables)
            specific_tables: Optional list of specific table names to evaluate. If provided, only these tables will be evaluated.
            use_reranker: Whether to use a reranker after initial retrieval
            reranker_model: Model name for the reranker (if use_reranker is True)
            reranker_tables: List of tables to apply reranking to. If None, reranking will be applied to all tables.
            use_hybrid_search: Whether to use hybrid search (combining vector search with BM25)
            hybrid_alpha: Weight(s) for vector search in hybrid search (1-alpha is weight for BM25).
                          Can be a single float or a list of floats to evaluate multiple alpha values.
            hybrid_tables: List of tables to apply hybrid search to. If None, hybrid search will be applied to all tables.
        """
        self.qa_pairs_path = Path(__file__).parent / qa_pairs_path
        self.top_k = top_k
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.table_filter = table_filter
        self.specific_tables = specific_tables
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker_tables = reranker_tables
        self.use_hybrid_search = use_hybrid_search
        self.threshold = threshold
        
        # Convert single alpha to list for consistent handling
        if isinstance(hybrid_alpha, (int, float)):
            self.hybrid_alphas = [float(hybrid_alpha)]
        else:
            self.hybrid_alphas = hybrid_alpha
            
        self.hybrid_tables = hybrid_tables
        
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
        
        # Create embedding generators on-demand based on table names
        self.embedding_generators = {}
        
        # Load QA pairs
        self.qa_pairs = self._load_qa_pairs()
        
        # Database connection parameters
        self.db_config = {
            "dbname": settings.db_name,
            "user": settings.db_username,
            "password": settings.db_password,
            "host": settings.db_host,
            "port": settings.db_port
        }
        
        logger.info(f"Initialized RetrieverEvaluator with {len(self.qa_pairs)} QA pairs")
        if self.table_filter:
            logger.info(f"Using table filter: {self.table_filter}")
        if self.specific_tables:
            logger.info(f"Will evaluate only specific tables: {', '.join(self.specific_tables)}")
        if self.use_reranker:
            if self.reranker_tables:
                logger.info(f"Will apply reranking to tables: {', '.join(self.reranker_tables)}")
            else:
                logger.info("Will apply reranking to all tables")
        if self.use_hybrid_search:
            logger.info(f"Using hybrid search with alphas: {', '.join(map(str, self.hybrid_alphas))}")
            if self.hybrid_tables:
                logger.info(f"Will apply hybrid search to tables: {', '.join(self.hybrid_tables)}")
            else:
                logger.info("Will apply hybrid search to all tables")
    
    def _load_qa_pairs(self) -> pd.DataFrame:
        """Load QA pairs from CSV file."""
        if not os.path.exists(self.qa_pairs_path):
            raise FileNotFoundError(f"QA pairs file not found at {self.qa_pairs_path}")
        
        # Use more robust CSV reading parameters to handle potential formatting issues
        try:
            # First try with standard parameters
            return pd.read_csv(self.qa_pairs_path)
        except pd.errors.ParserError as e:
            logger.warning(f"Error parsing CSV with standard parameters: {e}")
            logger.info("Trying with more robust parameters...")
            
            # Try with more robust parameters for newer pandas versions
            try:
                return pd.read_csv(
                    self.qa_pairs_path,
                    quotechar='"',       # Use double quotes as quote character
                    escapechar='\\',     # Use backslash as escape character
                    doublequote=True,    # Allow double quotes to escape quotes
                    on_bad_lines='warn'  # Warn about bad lines (pandas 1.3+)
                )
            except TypeError:
                # For older pandas versions
                return pd.read_csv(
                    self.qa_pairs_path,
                    quotechar='"',
                    escapechar='\\',
                    doublequote=True,
                    error_bad_lines=False,  # Skip bad lines (deprecated)
                    warn_bad_lines=True     # Warn about bad lines (deprecated)
                )
    
    def _get_test_tables(self) -> List[str]:
        """Get all tables starting with 'test_' from PostgreSQL."""
        logger.info("Getting test tables from PostgreSQL")
        # If specific tables are provided, use them directly
        if self.specific_tables:
            logger.info(f"Using specific tables: {', '.join(self.specific_tables)}")
            return self.specific_tables
            
        # Otherwise, query the database for tables
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Query to get all tables starting with 'test_'
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE 'test_%'
        ORDER BY table_name;
        """
        
        cursor.execute(query)
        tables = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        # Apply table filter if specified
        if self.table_filter:
            filtered_tables = [table for table in tables if self.table_filter in table]
            logger.info(f"Filtered {len(tables)} tables to {len(filtered_tables)} tables using filter '{self.table_filter}'")
            tables = filtered_tables
        
        logger.info(f"Found {len(tables)} test tables: {', '.join(tables)}")
        return tables
    
    def _parse_model_info_from_table_name(self, table_name: str) -> Dict[str, Any]:
        """
        Parse embedding model information from table name.
        
        Args:
            table_name: The table name to parse
            
        Returns:
            Dictionary with model information
        """
        model_info = {
            "method": "local",
            "model_name": "intfloat/multilingual-e5-large-instruct",  # Default model
            "chunk_size": 512,  # Default chunk size
            "is_recursive": False,
            "is_semantic": False
        }
        
        # Extract model type (multilingual_e5_large_instruct or text_embedding_3_*)
        if "multilingual_e5_large_instruct" in table_name:
            model_info["model_name"] = "intfloat/multilingual-e5-large-instruct"
            model_info["method"] = "local"
        elif "text_embedding_3_small" in table_name:
            model_info["model_name"] = "text-embedding-3-small"
            model_info["method"] = "openai"
        elif "text_embedding_3_large" in table_name:
            model_info["model_name"] = "text-embedding-3-large"
            model_info["method"] = "openai"
        
        # Extract chunk size
        chunk_size_match = re.search(r'_(\d+)_', table_name)
        if chunk_size_match:
            model_info["chunk_size"] = int(chunk_size_match.group(1))
        
        # Check if recursive
        model_info["is_recursive"] = "recursive" in table_name
        
        # Check if semantic
        model_info["is_semantic"] = "semantic" in table_name
        
        logger.info(f"Parsed model info from {table_name}: {model_info}")
        return model_info
    
    def _get_embedding_generator(self, table_name: str) -> EmbeddingGenerator:
        """
        Get or create an embedding generator for a specific table.
        
        Args:
            table_name: The table name to get an embedding generator for
            
        Returns:
            EmbeddingGenerator instance
        """
        if table_name not in self.embedding_generators:
            model_info = self._parse_model_info_from_table_name(table_name)
            
            self.embedding_generators[table_name] = EmbeddingGenerator(
                method=model_info["method"],
                model_name=model_info["model_name"],
                batch_size=8,
                api_key=None  # Will use from settings or environment
            )
            
            logger.info(f"Created embedding generator for {table_name} using {model_info['model_name']}")
        
        return self.embedding_generators[table_name]
    
    def query_similar_documents(
        self, 
        question: str, 
        table_name: str
    ) -> Tuple[List[str], List[float], float]:
        """
        Query similar documents from a specific table.
        
        Args:
            question: The question to search for
            table_name: The table to search in
            
        Returns:
            Tuple of (document_ids, similarity_scores, query_time)
        """
        start_time = time.time()
        
        # Check if we should apply hybrid search
        should_use_hybrid = self.use_hybrid_search and (
            self.hybrid_tables is None or table_name in self.hybrid_tables
        )
        
        # Get the appropriate embedding generator for this table
        embedding_generator = self._get_embedding_generator(table_name)
        
        # Generate embedding for the question
        question_embedding = embedding_generator.generate_embeddings([question])[0]
        
        # Convert embedding to PostgreSQL format
        # Check if question_embedding is already a list or needs tolist() conversion
        if hasattr(question_embedding, 'tolist'):
            embedding_list = question_embedding.tolist()
        else:
            embedding_list = question_embedding  # Already a list
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Determine the number of results to retrieve
        retrieval_limit = self.top_k
        # If reranking is used, we need more results to rerank
        if self.use_reranker:
            retrieval_limit = self.top_k * 2
        if should_use_hybrid:
            # For hybrid search, we need more results to merge
            retrieval_limit = self.top_k * 3
        
        # Vector search
        vector_query = f"""
            SELECT id, chunk_text, 1 - (embedding <=> %s::vector) AS similarity
            FROM {table_name}
            ORDER BY similarity DESC
            LIMIT {retrieval_limit};
        """
        
        # Convert the embedding list to a string in the format PostgreSQL expects
        embedding_str = str(embedding_list)
        
        cursor.execute(vector_query, (embedding_str,))
        vector_results = cursor.fetchall()
        
        if should_use_hybrid:
            # BM25 search using the existing function
            # Note: bm25_search always queries the page_keywords table
            bm25_query = """
                SELECT * FROM bm25_search(%s, %s, 1.2, 0.75, 'page_keywords');
            """
            
            cursor.execute(bm25_query, (question, retrieval_limit * 2))  # Get more results to ensure good coverage
            bm25_results = cursor.fetchall()
            
            # We need to map the BM25 results to our target table
            # First, get the mapping between page_keywords.id and our table's id
            # This assumes there's a relationship between the IDs in both tables
            # If the relationship is different, this mapping logic needs to be adjusted
            
            # Get all IDs from vector results for mapping
            vector_ids = [row[0] for row in vector_results]
            
            # Create dictionaries for easier lookup
            vector_scores = {row[0]: row[2] for row in vector_results}
            
            # Create a mapping of BM25 scores by ID
            bm25_scores = {}
            
            # For each BM25 result, find the corresponding document in our table
            for bm25_row in bm25_results:
                bm25_id = bm25_row[0]  # ID from page_keywords
                bm25_score = bm25_row[1]  # Score from BM25
                
                # Check if this ID exists in our vector results
                # If it does, use the same ID
                # If not, we'll skip it as we can't map it
                if bm25_id in vector_ids:
                    bm25_scores[bm25_id] = bm25_score
            
            # Get all unique document IDs that have both vector and BM25 scores
            all_doc_ids = set(vector_scores.keys()) & set(bm25_scores.keys())
            
            # If we don't have any overlap, fall back to vector search only
            if not all_doc_ids:
                logger.warning(f"No overlap between BM25 and vector search results for table {table_name}. Using vector search only.")
                doc_ids = [row[0] for row in vector_results]
                scores = [row[2] for row in vector_results]
                contents_list = [row[1] for row in vector_results]
            else:
                logger.info(f"Found {len(all_doc_ids)} overlapping document IDs between BM25 and vector search results for table {table_name}")
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
                        self.hybrid_alphas[0] * vector_score + 
                        (1 - self.hybrid_alphas[0]) * bm25_score
                    )
                
                # Sort by combined score
                sorted_results = sorted(
                    combined_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Get top_k results
                top_results = sorted_results[:retrieval_limit]
                
                # Get doc_ids and scores
                doc_ids = [doc_id for doc_id, _ in top_results]
                scores = [score for _, score in top_results]
                
                # Get the content for these documents for potential reranking
                if self.use_reranker:
                    # Query to get content for the top documents
                    content_query = f"""
                        SELECT id, chunk_text FROM {table_name}
                        WHERE id = ANY(%s);
                    """
                    cursor.execute(content_query, (doc_ids,))
                    content_results = cursor.fetchall()
                    
                    # Create a dictionary mapping doc_id to content
                    contents = {row[0]: row[1] for row in content_results}
                    
                    # Ensure we have content for all doc_ids in the same order
                    contents_list = [contents.get(doc_id, "") for doc_id in doc_ids]
                else:
                    # If no reranking, we don't need the contents
                    contents_list = []
        else:
            # Use vector search results directly
            doc_ids = [row[0] for row in vector_results]
            scores = [row[2] for row in vector_results]
            contents_list = [row[1] for row in vector_results]
        
        cursor.close()
        conn.close()
        
        # Check if we should apply reranking
        should_rerank = self.use_reranker and (
            self.reranker_tables is None or table_name in self.reranker_tables
        )
        
        if should_rerank and doc_ids:
            # Prepare pairs for reranking
            if should_use_hybrid:
                # For hybrid search, we need to ensure we have contents
                pairs = [[question, contents_list[i]] for i in range(len(doc_ids))]
            else:
                # For vector search, we already have contents
                pairs = [[question, content] for content in contents_list]
            
            # Rerank using cross-encoder
            logger.info(f"Reranking {len(pairs)} results for table {table_name}")
            rerank_scores = self.reranker.predict(pairs)
            
            # Create (id, score) pairs and sort by reranker score
            id_score_pairs = list(zip(doc_ids, rerank_scores))
            id_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Extract all doc_ids and scores
            doc_ids = [pair[0] for pair in id_score_pairs]
            scores = [pair[1] for pair in id_score_pairs]
        
        # Apply threshold filtering if specified
        if self.threshold is not None:
            # Filter results by threshold
            filtered_results = [(doc_id, score) for doc_id, score in zip(doc_ids, scores) if score > self.threshold]
            
            if filtered_results:
                # If we have results above threshold, use them
                doc_ids = [doc_id for doc_id, _ in filtered_results]
                scores = [score for _, score in filtered_results]
                logger.info(f"Filtered to {len(doc_ids)} results above threshold {self.threshold} for table {table_name}")
            else:
                # If no results above threshold, log a warning but return the best result we have
                # This ensures we always return at least one result
                if doc_ids:
                    best_id = doc_ids[0]
                    best_score = scores[0]
                    doc_ids = [best_id]
                    scores = [best_score]
                    logger.warning(f"No results above threshold {self.threshold} for table {table_name}. Returning best result with score {best_score}")
                else:
                    logger.warning(f"No results found for table {table_name}")
        
        # Limit to top_k
        doc_ids = doc_ids[:self.top_k]
        scores = scores[:self.top_k]
        
        query_time = time.time() - start_time
        
        return doc_ids, scores, query_time
    
    def evaluate_table(self, table_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific table using the QA pairs.
        
        Args:
            table_name: The table to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating table: {table_name}")
        
        # Parse model info from table name
        model_info = self._parse_model_info_from_table_name(table_name)
        
        # Check if reranking is applied to this table
        is_reranked = self.use_reranker and (self.reranker_tables is None or table_name in self.reranker_tables)
        
        # Check if hybrid search is applied to this table
        is_hybrid = self.use_hybrid_search and (self.hybrid_tables is None or table_name in self.hybrid_tables)
        
        results = {
            "table_name": table_name,
            "model_name": model_info["model_name"],
            "method": model_info["method"],
            "chunk_size": model_info["chunk_size"],
            "is_recursive": model_info["is_recursive"],
            "is_semantic": model_info["is_semantic"],
            "use_reranker": is_reranked,
            "reranker_model": self.reranker_model if is_reranked else None,
            "use_hybrid_search": is_hybrid,
            "hybrid_alpha": self.hybrid_alphas[0] if is_hybrid else None,
            "mrr": 0.0,
            "hit@1": 0.0,
            "hit@3": 0.0,
            "hit@5": 0.0,
            "avg_retrieval_time": 0.0,
            "total_queries": len(self.qa_pairs),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        reciprocal_ranks = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_5 = []
        retrieval_times = []
        
        # Process each QA pair
        for _, row in tqdm(self.qa_pairs.iterrows(), total=len(self.qa_pairs), desc=f"Evaluating {table_name}"):
            question = row['question']
            correct_id = row['id']
            
            # Query similar documents
            doc_ids, scores, query_time = self.query_similar_documents(question, table_name)
            
            # Calculate metrics
            _, reciprocal_rank = calculate_mrr(correct_id, doc_ids)
            hit_at_1 = calculate_hit_at_k(correct_id, doc_ids, 1)
            hit_at_3 = calculate_hit_at_k(correct_id, doc_ids, 3)
            hit_at_5 = calculate_hit_at_k(correct_id, doc_ids, 5)
            
            # Append to lists
            reciprocal_ranks.append(reciprocal_rank)
            hits_at_1.append(hit_at_1)
            hits_at_3.append(hit_at_3)
            hits_at_5.append(hit_at_5)
            retrieval_times.append(query_time)
        
        # Calculate average metrics
        results["mrr"] = np.mean(reciprocal_ranks)
        results["hit@1"] = np.mean(hits_at_1)
        results["hit@3"] = np.mean(hits_at_3)
        results["hit@5"] = np.mean(hits_at_5)
        results["avg_retrieval_time"] = np.mean(retrieval_times)
        
        logger.info(f"Evaluation results for {table_name}:")
        logger.info(f"  MRR: {results['mrr']:.4f}")
        logger.info(f"  Hit@1: {results['hit@1']:.4f}")
        logger.info(f"  Hit@3: {results['hit@3']:.4f}")
        logger.info(f"  Hit@5: {results['hit@5']:.4f}")
        logger.info(f"  Avg Retrieval Time: {results['avg_retrieval_time']:.4f}s")
        
        return results
    
    def run_evaluation(self):
        """Run evaluation on all test tables and log results to wandb."""
        # Get all test tables
        tables = self._get_test_tables()
        
        if not tables:
            logger.warning("No test tables found in the database")
            return
        
        all_results = []
        
        # Evaluate each table
        for table in tables:
            # Initialize wandb for this table (each table gets its own run)
            model_info = self._parse_model_info_from_table_name(table)
            
            # Check if reranking is applied to this table
            is_reranked = self.use_reranker and (self.reranker_tables is None or table in self.reranker_tables)
            
            # Check if hybrid search is applied to this table
            is_hybrid = self.use_hybrid_search and (self.hybrid_tables is None or table in self.hybrid_tables)
            
            # If using hybrid search with multiple alphas, evaluate each alpha value
            if is_hybrid and len(self.hybrid_alphas) > 1:
                for alpha in self.hybrid_alphas:
                    # Temporarily set the current alpha for evaluation
                    current_alpha = alpha
                    
                    # Create run name with reranker and hybrid search information
                    run_name = table
                    if is_reranked:
                        run_name = f"{table}_reranker_hybrid_a{current_alpha}"
                    else:
                        run_name = f"{table}_hybrid_a{current_alpha}"
                    
                    wandb.init(
                        project=self.wandb_project,
                        entity=self.wandb_entity,
                        name=run_name,
                        config={
                            "top_k": self.top_k,
                            "model_name": model_info["model_name"],
                            "method": model_info["method"],
                            "chunk_size": model_info["chunk_size"],
                            "is_recursive": model_info["is_recursive"],
                            "is_semantic": model_info["is_semantic"],
                            "qa_pairs_count": len(self.qa_pairs),
                            "use_reranker": is_reranked,
                            "reranker_model": self.reranker_model if is_reranked else None,
                            "use_hybrid_search": True,
                            "hybrid_alpha": current_alpha
                        },
                        reinit=True
                    )
                    
                    # Evaluate the table with the current alpha
                    # We need to temporarily modify the alpha value for this evaluation
                    original_alphas = self.hybrid_alphas
                    self.hybrid_alphas = [current_alpha]
                    
                    results = self.evaluate_table(table)
                    all_results.append(results)
                    
                    # Restore original alphas
                    self.hybrid_alphas = original_alphas
                    
                    # Log to wandb
                    wandb.log({
                        "mrr": results["mrr"],
                        "hit@1": results["hit@1"],
                        "hit@3": results["hit@3"],
                        "hit@5": results["hit@5"],
                        "avg_retrieval_time": results["avg_retrieval_time"]
                    })
                    
                    # Finish this wandb run
                    wandb.finish()
            else:
                # Standard evaluation for non-hybrid or single alpha hybrid
                # Create run name with reranker and hybrid search information if applicable
                run_name = table
                if is_reranked and is_hybrid:
                    run_name = f"{table}_reranker_hybrid_a{self.hybrid_alphas[0]}"
                elif is_reranked:
                    run_name = f"{table}_reranker"
                elif is_hybrid:
                    run_name = f"{table}_hybrid_a{self.hybrid_alphas[0]}"
                
                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=run_name,  # Use table name with reranker and hybrid info as run name
                    config={
                        "top_k": self.top_k,
                        "model_name": model_info["model_name"],
                        "method": model_info["method"],
                        "chunk_size": model_info["chunk_size"],
                        "is_recursive": model_info["is_recursive"],
                        "is_semantic": model_info["is_semantic"],
                        "qa_pairs_count": len(self.qa_pairs),
                        "use_reranker": is_reranked,
                        "reranker_model": self.reranker_model if is_reranked else None,
                        "use_hybrid_search": is_hybrid,
                        "hybrid_alpha": self.hybrid_alphas[0] if is_hybrid else None
                    },
                    reinit=True  # Allow reinitializing for each table
                )
                
                # Evaluate the table
                results = self.evaluate_table(table)
                all_results.append(results)
                
                # Log to wandb
                wandb.log({
                    "mrr": results["mrr"],
                    "hit@1": results["hit@1"],
                    "hit@3": results["hit@3"],
                    "hit@5": results["hit@5"],
                    "avg_retrieval_time": results["avg_retrieval_time"]
                })
                
                # Finish this wandb run
                wandb.finish()

        
    
    def cleanup(self):
        """Clean up resources."""
        for generator in self.embedding_generators.values():
            generator.cleanup()


if __name__ == "__main__":
    # Create evaluator
    evaluator = RetrieverEvaluator(
        top_k=10,
        wandb_project="retriever-evaluation"
    )
    
    try:
        # Run evaluation
        evaluator.run_evaluation()
    finally:
        # Clean up resources
        evaluator.cleanup() 