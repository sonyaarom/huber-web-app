from functools import lru_cache
from .retriever import HybridRetriever
from .generator.main import together_generator as llm_main_func
from typing import Optional, Dict, Any, List
from .config import settings, reload_settings

@lru_cache(maxsize=None)
def get_retriever(use_full_content: bool = False) -> HybridRetriever:
    """
    Factory function to create and cache a retriever instance.
    Using lru_cache ensures that the retriever is created only once.
    """
    return HybridRetriever(use_full_content=use_full_content)

# Initialize the retriever
retriever = get_retriever()

def reinitialize_retriever():
    """Reinitialize the retriever with new settings."""
    global retriever
    get_retriever.cache_clear()  # Clear the cache
    retriever = get_retriever()  # Create a new retriever with updated settings

def retrieve_urls(question: str, ner_filters: Optional[Dict[str, Any]] = None):
    """
    Only retrieve URLs based on a question, optionally filtered by NER.
    """
    retrieved_docs = retriever.retrieve(question, filters=ner_filters)
    return [doc.get('url') for doc in retrieved_docs]

def rag_main_func(question: str, ner_filters: Optional[Dict[str, List[str]]] = None):
    """
    Main RAG function that orchestrates retrieval and generation.
    """
    # Retrieve documents, passing NER filters if available
    # The HybridRetriever already handles reranking internally if enabled
    final_docs = retriever.retrieve(query=question, filters=ner_filters)

    context = "\n".join([doc['chunk'] for doc in final_docs])
    response = llm_main_func(question, context, final_docs)
    
    return {"answer": response, "sources": [doc.get('url') for doc in final_docs]}

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