from .retriever.retriever import HybridRetriever
from .generator.main import main_generator

def retrieve_urls(question: str):
    """Retrieve URLs relevant to the question"""
    retriever = HybridRetriever()
    results = retriever.retrieve(question)
    
    # Extract URLs from the results
    urls = [chunk.get('url', '') for chunk in results if 'url' in chunk]
    
    # Remove duplicates while preserving order
    unique_urls = []
    for url in urls:
        if url and url not in unique_urls:
            unique_urls.append(url)
    
    return unique_urls

def rag_main_func(question: str):
    retriever = HybridRetriever()

    results = retriever.retrieve(question)
    # Join the 'content' field from all results with a space in between
    all_context = " ".join(chunk['content'] for chunk in results)

    response = main_generator(question, all_context)
    return response


if __name__ == "__main__":
    print(rag_main_func("Who is Stefan Lessmann?"))