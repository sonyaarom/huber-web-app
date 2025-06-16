from functools import lru_cache
from .retriever import HybridRetriever
from .generator.main import together_generator as llm_main_func
from typing import Optional, Dict, Any, List
from .config import settings, reload_settings

@lru_cache(maxsize=None)
def get_retriever() -> HybridRetriever:
    """
    Factory function to create and cache a retriever instance.
    Using lru_cache ensures that the retriever is created only once.
    """
    return create_retriever()

def retrieve_urls(question: str):
    """
    Function to retrieve URLs for a given question.
    """
    # Use the cached retriever
    retriever = get_retriever()
    retrieved_docs = retriever.retrieve(question)
    urls = [doc['url'] for doc in retrieved_docs if 'url' in doc]
    return {"urls": urls}

def rag_main_func(question: str, ner_filters: Optional[Dict[str, List[str]]] = None):
    """
    Main RAG function that orchestrates retrieval and generation.
    """
    # Use the cached retriever
    retriever = get_retriever()
    retrieved_docs = retriever.retrieve(question)
    
    # Process the retrieved documents
    context = " ".join([doc['content'] for doc in retrieved_docs])
    
    # Generate the answer using the LLM
    answer = llm_main_func(question, context)
    
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