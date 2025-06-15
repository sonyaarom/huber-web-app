from ..utils.embedding_utils import process_and_store_embeddings
from hubert.data_ingestion.config import settings
import logging
import os
import sys
import argparse

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set OpenAI API key as environment variable if available
if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    logger.info("Set OPENAI_API_KEY environment variable")

def main():
    parser = argparse.ArgumentParser(description="Process and store text embeddings.")
    parser.add_argument('--model-name', type=str, default='text-embedding-3-small', help='Name of the embedding model to use.')
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size for text processing.')
    parser.add_argument('--chunking-method', type=str, default='recursive', help='Chunking method to use.')
    parser.add_argument('--table-name', type=str, required=True, help='Name of the database table to store embeddings in.')
    
    args = parser.parse_args()
    
    # Get API key from settings or environment
    api_key = getattr(settings, 'openai_api_key', os.environ.get("OPENAI_API_KEY"))
    if not api_key and 'text-embedding' in args.model_name:
        logger.error("No OpenAI API key found. Cannot generate embeddings.")
        sys.exit(1)
    
    logger.info(f"Processing with embedding model: {args.model_name}")
    logger.info(f"Using {args.chunking_method} chunking with size {args.chunk_size}")
    
    try:
        process_and_store_embeddings(
            chunk_size=args.chunk_size,
            chunk_overlap=50,
            batch_size=1,
            chunking_method=args.chunking_method,
            model_name=args.model_name,
            api_key=api_key,
            prefer_openai=True,
            table_name=args.table_name
        )
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()