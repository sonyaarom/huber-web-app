from functools import lru_cache
from .retriever import HybridRetriever
from .generator.main import together_generator as llm_main_func
from typing import Optional, Dict, Any, List
from .config import settings, reload_settings

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
    return {"urls": list(set(urls))}

def rag_main_func(question: str, ner_filters: Optional[Dict[str, List[str]]] = None):
    """
    Main RAG function that orchestrates retrieval and generation.
    """
    # Use the cached retriever
    retriever = get_retriever()
    retrieved_docs = retriever.retrieve(question, filters=ner_filters)
    
    # Process the retrieved documents
    context = " ".join([doc['content'] for doc in retrieved_docs])
    
    # Generate the answer using the LLM
    answer = llm_main_func(question, context)
    
    sources = [doc.get('url') for doc in retrieved_docs if doc.get('url')]
    
    return {"answer": answer, "sources": list(set(sources))}

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