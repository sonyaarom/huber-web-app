from functools import lru_cache
from .retriever import HybridRetriever
from .generator.main import together_generator, main_generator
from typing import Optional, Dict, Any, List
from .config import settings, reload_settings
import time
import sentry_sdk
from .common.monitoring import capture_rag_operation, capture_retrieval_metrics, capture_generation_metrics
import logging

logger = logging.getLogger(__name__)

def get_generator_function():
    """
    Get the appropriate generator function based on the configuration.
    Always uses fresh settings to ensure dynamic switching works.
    """
    # Reload settings to ensure we get the latest configuration
    current_settings = reload_settings()
    generator_type = current_settings.generator_type.lower()
    
    if generator_type == 'together':
        return together_generator
    elif generator_type == 'openai':
        return lambda question, context: main_generator(question, context, model_type="openai")
    else:
        logger.warning(f"Unknown generator type '{generator_type}', defaulting to 'together'")
        return together_generator

def normalize_generator_response(response):
    """
    Normalize different generator response formats to a consistent string output.
    
    Args:
        response: Can be a string, dict with choices, or other format
        
    Returns:
        str: The extracted text response
    """
    if isinstance(response, str):
        # Already a string, return as-is
        return response
    elif isinstance(response, dict):
        # Handle OpenAI-style response format
        if 'choices' in response and isinstance(response['choices'], list) and len(response['choices']) > 0:
            choice = response['choices'][0]
            if isinstance(choice, dict) and 'text' in choice:
                return choice['text']
            elif isinstance(choice, dict) and 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
        # Handle other dict formats - try to extract text content
        if 'text' in response:
            return response['text']
        if 'content' in response:
            return response['content']
        # If no recognizable format, convert to string
        logger.warning(f"Unknown response format: {type(response)}, converting to string")
        return str(response)
    else:
        # For any other type, convert to string
        logger.warning(f"Unexpected response type: {type(response)}, converting to string")
        return str(response)

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
            logger.info(f"Overall retrieval phase took {retrieval_duration:.2f} seconds.")
            
            # Capture retrieval metrics
            scores = []
            for doc in retrieved_docs:
                score = doc.get('reranked_score') or doc.get('score') or doc.get('similarity') or 0.0
                scores.append(float(score))
            
            capture_retrieval_metrics(len(retrieved_docs), settings.top_k, scores)
            sentry_sdk.set_measurement("retrieval_duration", retrieval_duration, "second")
            
            # CONTEXT PROCESSING
            context_start = time.time()
            with sentry_sdk.start_span(op="processing", description="Context processing"):
                context = " ".join([doc['content'] for doc in retrieved_docs])
            context_duration = time.time() - context_start
            logger.info(f"Context processing took {context_duration:.3f} seconds.")
            
            # GENERATION PHASE
            generation_start = time.time()
            with sentry_sdk.start_span(op="generation", description="LLM generation"):
                raw_answer = get_generator_function()(question, context)
                answer = normalize_generator_response(raw_answer)
            
            generation_duration = time.time() - generation_start
            logger.info(f"Overall generation phase took {generation_duration:.2f} seconds.")
            
            # Debug logging for response format
            logger.debug(f"Raw generator response type: {type(raw_answer)}")
            logger.debug(f"Normalized answer type: {type(answer)}, length: {len(answer) if isinstance(answer, str) else 'N/A'}")
            
            # Capture generation metrics
            capture_generation_metrics(
                prompt_length=len(question) + len(context),
                response_length=len(answer) if isinstance(answer, str) else 0,
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