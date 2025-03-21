from utils.vectorisation_utils import process_and_store_embeddings
from config import settings
import logging
import os

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set OpenAI API key as environment variable if available
if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    logger.info("Set OPENAI_API_KEY environment variable")

def main():
    embeddings = ['text-embedding-3-small']
    # Define a list of chunk sizes to loop through
    chunk_sizes = [128, 256, 512]
    # Include semantic in the list of chunking methods along with others
    chunking_methods = ['character']
    
    # Get API key from settings or environment
    api_key = getattr(settings, 'openai_api_key', os.environ.get("OPENAI_API_KEY"))
    if not api_key and 'text-embedding-3-small' in embeddings:
        logger.warning("No OpenAI API key found. OpenAI embeddings will not be available.")
        embeddings = [e for e in embeddings if not e.startswith('text-embedding')]
    
    for embedding in embeddings:
        logger.info(f"Processing with embedding model: {embedding}")
        for current_chunk_size in chunk_sizes:
            for chunking_method in chunking_methods:
                if chunking_method == 'semantic':
                    # For semantic chunking, override chunk_size and chunk_overlap values
                    cs = 0
                    co = 0
                    logger.info(f"Processing semantic chunking using {embedding} (ignoring chunk size).")
                else:
                    cs = current_chunk_size
                    co = 50
                    logger.info(f"Processing {chunking_method} chunking with size {cs} using {embedding}.")
                try:
                    process_and_store_embeddings(
                        chunk_size=cs,
                        chunk_overlap=co,
                        batch_size=1,
                        chunking_method=chunking_method,
                        model_name=embedding,
                        api_key=api_key,
                        prefer_openai=True
                    )
                except Exception as e:
                    if "maximum context length" in str(e):
                        logger.warning(
                            f"Skipping {chunking_method} chunking with size {cs} for {embedding} due to token limit error."
                        )
                    else:
                        raise

if __name__ == "__main__":
    main()