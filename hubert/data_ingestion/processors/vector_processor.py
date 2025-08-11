from hubert.data_ingestion.processors.embedding_processor import process_and_store_embeddings
from hubert.config import settings
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
    parser.add_argument('--model-name', type=str, default='text-embedding-3-small', help='Name of the embedding model to use (for logging; configured via settings).')
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size for text processing.')
    parser.add_argument('--chunking-method', type=str, default='recursive', help='Chunking method to use (recursive|character|semantic).')
    parser.add_argument('--table-name', type=str, required=True, help='Name of the database table to store embeddings in.')
    parser.add_argument('--uids', nargs='+', help='Optional list of UIDs to process.')

    args = parser.parse_args()

    # Get API key from settings or environment
    api_key = getattr(settings, 'openai_api_key', os.environ.get("OPENAI_API_KEY"))
    if not api_key and 'text-embedding' in args.model_name:
        logger.error("No OpenAI API key found. Cannot generate embeddings.")
        sys.exit(1)

    logger.info(f"Processing with embedding model: {args.model_name}")
    logger.info(f"Using {args.chunking_method} chunking with size {args.chunk_size}")

    # Build DB URI from settings
    db_uri = f"postgresql://{settings.db_username}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"

    # Normalize chunk strategy name to match get_chunking_strategy
    strategy_name = args.chunking_method
    if strategy_name == 'recursive':
        strategy_name = 'recursive_chunk_text'
    elif strategy_name == 'character':
        strategy_name = 'character_chunk_text'
    elif strategy_name == 'semantic':
        strategy_name = 'semantic_chunk_text'

    chunk_options = {
        'chunk_size': args.chunk_size,
        'chunk_overlap': 50,
    }

    try:
        process_and_store_embeddings(
            db_uri=db_uri,
            table_name=args.table_name,
            chunk_strategy_name=strategy_name,
            chunk_options=chunk_options,
            uids=args.uids,
        )
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()