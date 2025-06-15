"""
Module for generating and storing embeddings from text data.
Supports multiple embedding methods and text chunking strategies.
"""

import sys
import os
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import time

# Third-party imports
import pandas as pd
import time
from tqdm import tqdm
from sqlalchemy import create_engine, inspect, text, Table, MetaData, Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, REAL


# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from hubert.config import settings
from hubert.common.utils.chunking_utils import recursive_chunk_text, character_chunk_text, semantic_chunk_text
from hubert.common.utils.db_utils import fetch_page_content
from hubert.common.utils.embedding_utils import EmbeddingGenerator, ensure_token_limit

# Configure logger
logger = logging.getLogger(__name__)


def store_embeddings(
    df: pd.DataFrame, 
    chunking_method: str, 
    chunk_size: int, 
    model_name: str,
    table_name: Optional[str] = None
) -> None:
    """
    Stores document embeddings and their metadata in a database table.
    
    This function processes a DataFrame of documents, generates embeddings for text chunks,
    and upserts them into a specified PostgreSQL table. It handles table creation,
    data insertion, and conflict resolution.
    
    Args:
        df: DataFrame with 'uid', 'url', and 'content' columns
        chunking_method: Strategy for splitting text ('recursive', 'character', 'semantic')
        chunk_size: The size of each text chunk
        model_name: The name of the embedding model used
        table_name: The name of the database table to store embeddings
        
    Raises:
        ValueError: If the table name is not specified or chunking method is invalid
    """
    if not table_name:
        raise ValueError("Table name must be specified.")

    # Determine embedding method from settings
    embedding_method = settings.embedding_method or "openai"
    
    # Initialize the embedding generator
    generator = EmbeddingGenerator(method=embedding_method, model_name=model_name)
    
    all_embeddings_data = []

    # Choose the chunking function
    chunking_functions = {
        'recursive': recursive_chunk_text,
        'character': character_chunk_text,
        'semantic': semantic_chunk_text
    }
    
    if chunking_method not in chunking_functions:
        raise ValueError(f"Invalid chunking method: {chunking_method}")
    
    chunk_function = chunking_functions[chunking_method]

    # Process each document
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing documents"):
        doc_id = row['uid']
        doc_url = row['url']
        content = row['content']
        
        if not content or not isinstance(content, str):
            logger.warning(f"Skipping document {doc_id} due to empty or invalid content.")
            continue

        # Split text into chunks
        chunks = chunk_function(content, chunk_size=chunk_size)
        
        # Generate embeddings for chunks
        if chunks:
            try:
                # Ensure each chunk respects the token limit before generating embeddings
                limited_chunks = []
                for chunk in chunks:
                    limited_chunks.extend(ensure_token_limit(chunk, model_name))
                
                chunk_embeddings = generator.generate_embeddings(limited_chunks)
                
                # Prepare data for storage
                for i, chunk_text in enumerate(limited_chunks):
                    embedding_vector = chunk_embeddings[i]
                    all_embeddings_data.append({
                        'uid': doc_id,
                        'url': doc_url,
                        'chunk_text': chunk_text,
                        'embedding': embedding_vector,
                        'model_name': model_name,
                        'chunking_method': chunking_method,
                        'chunk_size': chunk_size
                    })
            except Exception as e:
                logger.error(f"Failed to generate embeddings for document {doc_id}: {e}")

    # Store all collected data in the database
    if all_embeddings_data:
        try:
            db_url = f"postgresql://{settings.db_username}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
            engine = create_engine(db_url)
            
            metadata = MetaData()
            # Reflect the table to get its structure, especially the embedding dimension
            insp = inspect(engine)
            if not insp.has_table(table_name):
                # If table doesn't exist, create it with a default embedding dimension
                logger.info(f"Table '{table_name}' not found, creating it.")
                embedding_dim = len(all_embeddings_data[0]['embedding'])
                
                Table(
                    table_name,
                    metadata,
                    Column('id', Integer, primary_key=True, autoincrement=True),
                    Column('uid', String, nullable=False),
                    Column('url', String),
                    Column('chunk_text', Text),
                    Column('embedding', ARRAY(REAL, dimensions=1)),
                    Column('model_name', String),
                    Column('chunking_method', String),
                    Column('chunk_size', Integer)
                )
                metadata.create_all(engine)
            
            # Use pandas to_sql for efficient bulk upsert
            embeddings_df = pd.DataFrame(all_embeddings_data)
            with engine.connect() as connection:
                embeddings_df.to_sql(
                    table_name, 
                    connection, 
                    if_exists='append', 
                    index=False,
                    method='multi'  # Use multi-value insert for efficiency
                )
                logger.info(f"Successfully stored {len(all_embeddings_data)} embeddings in '{table_name}'.")

        except Exception as e:
            logger.error(f"Database operation failed: {e}")

    # Clean up the generator resources
    generator.cleanup()

def process_and_store_embeddings(
    chunk_size: int = 512, 
    chunk_overlap: int = 50, 
    batch_size: int = 8, 
    chunking_method: str = 'recursive',
    model_name: Optional[str] = None, 
    api_key: Optional[str] = None, 
    prefer_openai: bool = True, 
    table_name: Optional[str] = None
) -> None:
    """
    Main function to orchestrate the fetching, processing, and storing of text embeddings.
    
    This function ties together fetching page content, generating embeddings based on specified
    parameters, and storing them in the database. It serves as the primary entry point for
    the embedding generation pipeline.
    
    Args:
        chunk_size: The target size for text chunks.
        chunk_overlap: The overlap between consecutive chunks.
        batch_size: The number of items to process in a single batch.
        chunking_method: The method to use for splitting text.
        model_name: The name of the embedding model to use.
        api_key: The API key for the embedding service (if applicable).
        prefer_openai: Whether to prefer OpenAI for embeddings if available.
        table_name: The name of the table to store embeddings.
    """
    start_time = time.time()
    
    # Validate and get the table name from settings if not provided
    table_name = table_name or settings.table_name
    if not table_name:
        raise ValueError("Table name must be provided either as an argument or in settings.")
    
    # Determine embedding model from settings or arguments
    model_name = model_name or settings.embedding_model
    
    # Fetch page content from the database
    try:
        page_content_df = fetch_page_content()
        if page_content_df.empty:
            logger.info("No page content found to process.")
            return
    except Exception as e:
        logger.error(f"Failed to fetch page content: {e}")
        return

    # Store embeddings
    store_embeddings(
        df=page_content_df,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        model_name=model_name,
        table_name=table_name
    )
    
    end_time = time.time()
    logger.info(f"Embedding generation completed in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    # Example of how to run the embedding generation process
    logging.basicConfig(level=logging.INFO)
    
    # For demonstration, you might need to set up your environment variables
    # for database and API keys as defined in the Settings class.
    
    # Example run with specific parameters
    process_and_store_embeddings(
        chunk_size=1024,
        chunk_overlap=100,
        chunking_method='recursive',
        model_name='text-embedding-3-small', # Example model
        table_name='page_embeddings_demo' # Example table
    ) 