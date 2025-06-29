from functools import lru_cache
from .retriever import HybridRetriever
from .generator.main import together_generator as llm_main_func
from typing import Optional, Dict, Any, List
from .config import settings, reload_settings
import time
import sentry_sdk
from .common.monitoring import capture_rag_operation, capture_retrieval_metrics, capture_generation_metrics

def create_retriever() -> HybridRetriever:
    """
    Creates a new retriever instance based on the current settings.
    """
    # Reload settings to ensure the retriever uses the latest configuration
    current_settings = reload_settings()
    return HybridRetriever(
        embedding_model=current_settings.embedding_model,
        embedding_method=current_settings.embedding_method,
        table_name=current_settings.table_name,
        top_k=current_settings.top_k,
        use_reranker=current_settings.use_reranker,
        reranker_model=current_settings.reranker_model,
        use_hybrid_search=current_settings.use_hybrid_search,
        hybrid_alpha=current_settings.hybrid_alpha,
    )

@lru_cache(maxsize=None)
def get_retriever() -> HybridRetriever:
    """
    Factory function to create and cache a retriever instance.
    Using lru_cache ensures that the retriever is created only once.
    """
    return create_retriever()

def reinitialize_retriever():
    """
    Clears the cache for get_retriever, forcing re-initialization
    of the retriever with the latest settings on the next call.
    """
    get_retriever.cache_clear()

def retrieve_urls(question: str):
    """
    Function to retrieve URLs for a given question.
    """
    # Use the cached retriever
    retriever = get_retriever()
    retrieved_docs = retriever.retrieve(question)
    urls = [doc['url'] for doc in retrieved_docs if 'url' in doc]
    
    # Extract similarity scores for each URL
    url_scores = {}
    for doc in retrieved_docs:
        if 'url' in doc:
            # Get the best available score (reranked_score > score > similarity)
            score = doc.get('reranked_score') or doc.get('score') or doc.get('similarity') or 0.0
            # Convert numpy types to Python float
            url_scores[doc['url']] = float(score)
    
    return {"urls": list(set(urls)), "url_scores": url_scores}

def rag_main_func(question: str, ner_filters: Optional[Dict[str, List[str]]] = None):
    """
    Main RAG function that orchestrates retrieval and generation.
    """
    start_time = time.time()
    
<<<<<<< HEAD
    with sentry_sdk.start_transaction(name="rag_pipeline", op="rag"):
        try:
            # Capture the overall RAG operation
            capture_rag_operation("rag_pipeline", question)
            
            # RETRIEVAL PHASE
            retrieval_start = time.time()
            with sentry_sdk.start_span(op="retrieval", description="Document retrieval"):
                retriever = get_retriever()
                retrieved_docs = retriever.retrieve(question, filters=ner_filters)
            
            retrieval_duration = time.time() - retrieval_start
            
            # Capture retrieval metrics
            scores = []
            for doc in retrieved_docs:
                score = doc.get('reranked_score') or doc.get('score') or doc.get('similarity') or 0.0
                scores.append(float(score))
            
            capture_retrieval_metrics(len(retrieved_docs), settings.top_k, scores)
            sentry_sdk.set_measurement("retrieval_duration", retrieval_duration, "second")
            
            # CONTEXT PROCESSING
            with sentry_sdk.start_span(op="processing", description="Context processing"):
                context = " ".join([doc['content'] for doc in retrieved_docs])
            
            # GENERATION PHASE
            generation_start = time.time()
            with sentry_sdk.start_span(op="generation", description="LLM generation"):
                answer = llm_main_func(question, context)
            
            generation_duration = time.time() - generation_start
            
            # Capture generation metrics
            capture_generation_metrics(
                prompt_length=len(question) + len(context),
                response_length=len(answer),
                duration=generation_duration
            )
            
            # RESULT PROCESSING
            sources = [doc.get('url') for doc in retrieved_docs if doc.get('url')]
            
            # Extract similarity scores for each source URL
            source_scores = {}
            for doc in retrieved_docs:
                if doc.get('url'):
                    # Get the best available score (reranked_score > score > similarity)
                    score = doc.get('reranked_score') or doc.get('score') or doc.get('similarity') or 0.0
                    # Convert numpy types to Python float
                    source_scores[doc['url']] = float(score)
            
            # Capture overall timing
            total_duration = time.time() - start_time
            capture_rag_operation("rag_complete", question, total_duration)
            sentry_sdk.set_measurement("rag_total_duration", total_duration, "second")
            
            return {
                "answer": answer, 
                "sources": list(set(sources)),
                "source_scores": source_scores
            }
            
        except Exception as e:
            # Capture the error with context
            sentry_sdk.capture_exception(e)
            raise
=======
    # Process the retrieved documents
    context = " ".join([doc['content'] for doc in retrieved_docs])
    
    # Generate the answer using the LLM
    answer = llm_main_func(question, context)
    
    sources = [doc.get('url') for doc in retrieved_docs if doc.get('url')]
    
    # Extract similarity scores for each source URL
    source_scores = {}
    for doc in retrieved_docs:
        if doc.get('url'):
            # Get the best available score (reranked_score > score > similarity)
            score = doc.get('reranked_score') or doc.get('score') or doc.get('similarity') or 0.0
            # Convert numpy types to Python float
            source_scores[doc['url']] = float(score)
    
    return {
        "answer": answer, 
        "sources": list(set(sources)),
        "source_scores": source_scores
    }
>>>>>>> 7589e61585774d979e46db43f6a7e0a43545d4d2

if __name__ == '__main__':
    # Example usage
    test_question = "What is the role of Stefan Lessmann?"
    result = rag_main_func(test_question)
    print(result)

    # Example with NER filter
    ner_question = "Who is Stefan Lessmann?"
    ner_filters = {"PERSON": ["Stefan Lessmann"]}
    result_with_ner = rag_main_func(ner_question, ner_filters=ner_filters)
    print("\nWith NER filtering:")
    print(result_with_ner)