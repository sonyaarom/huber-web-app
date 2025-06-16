from functools import lru_cache
from .retriever import HybridRetriever, create_retriever
from .generator.main import together_generator as llm_main_func

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

def rag_main_func(question: str):
    """
    Main function for the RAG model.
    """
    # Use the cached retriever
    retriever = get_retriever()
    retrieved_docs = retriever.retrieve(question)
    
    # Process the retrieved documents
    context = " ".join([doc['content'] for doc in retrieved_docs])
    
    # Generate the answer using the LLM
    answer = llm_main_func(question, context)
    
    # Optionally, you can include URLs in the response if needed
    urls = [doc['url'] for doc in retrieved_docs if 'url' in doc]
    
    return {"answer": answer, "urls": urls}

if __name__ == '__main__':
    # Example usage
    question = "What is the capital of Germany?"
    result = rag_main_func(question)
    print(result)