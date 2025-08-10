"""
Module for generating and storing embeddings from text data.
Supports multiple embedding methods and text chunking strategies.
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Tuple, Optional, Union, Any
import time

# Third-party imports
import pandas as pd
import time
from tqdm import tqdm
from sqlalchemy import create_engine, inspect, text, Table, MetaData, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY, REAL

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from hubert.config import settings
from hubert.common.utils.chunking_utils import get_chunking_strategy
from hubert.common.utils.db_utils import fetch_page_content, get_db_connection
from hubert.common.utils.embedding_utils import EmbeddingGenerator, ensure_token_limit
from hubert.db.models import PageEmbeddings, FailedJob
from hubert.db.postgres_storage import PostgresStorage

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_session(db_uri):
    """Creates and returns a new SQLAlchemy session."""
    engine = create_engine(db_uri)
    Session = sessionmaker(bind=engine)
    return Session()


def log_failed_job(db_uri:str, uid: str, job_type: str, error: Exception):
    """
    Logs a failed job to the failed_jobs table.
    """
    session = get_session(db_uri)
    try:
        failed_job = FailedJob(
            uid=uid,
            job_type=job_type,
            error_message=str(error)
        )
        session.add(failed_job)
        session.commit()
        logger.info(f"Successfully logged failed job for uid: {uid}")
    except Exception as e:
        logger.error(f"Failed to log failed job for uid {uid}: {e}")
        session.rollback()
    finally:
        session.close()


def process_and_store_embeddings(db_uri: str, table_name: str, chunk_strategy_name: str, chunk_options: dict, uids: Optional[List[str]] = None):
    """
    Fetches content, chunks it, generates embeddings, and stores them in the database.
    Includes a content_hash for data lineage.
    If uids are provided, it processes only those. Otherwise, it looks for content
    that hasn't been processed yet.
    """
    engine = create_engine(db_uri)

    if uids:
        logger.info(f"Processing specific UIDs: {uids}")
        query = text("SELECT id, url, extracted_content FROM page_content WHERE id = ANY(:uids)")
        df = pd.read_sql(query, engine, params={'uids': uids})
    else:
        logger.info("No UIDs provided. Fetching content without embeddings.")
        # This query assumes that if an id from page_content is not in the embeddings table, it needs processing.
        # This is a simplified version of the original logic.
        query = text(f"""
            SELECT pc.id, pc.url, pc.extracted_content
            FROM page_content pc
            LEFT JOIN {table_name} emb ON pc.id = emb.id
            WHERE emb.id IS NULL AND pc.extracted_content IS NOT NULL
        """)
        df = pd.read_sql(query, engine)
        uids_to_process = df['id'].tolist()
        if not uids_to_process:
            logger.info(f"No new content to process for embeddings in table {table_name}.")
            return
        logger.info(f"Found {len(uids_to_process)} new pages to process.")


    if df.empty:
        logger.warning(f"No content found for the given UIDs.")
        return

    chunking_strategy = get_chunking_strategy(chunk_strategy_name, **chunk_options)
    
    all_chunks = []
    from datetime import datetime
    now_timestamp = datetime.now()
    
    for _, row in df.iterrows():
        if not row['extracted_content']:
            continue
        try:
            chunks = chunking_strategy.split_text(row['extracted_content'])
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    'id': row['id'],  # Use id instead of uid
                    'split_id': i,   # Use split_id instead of chunk_id
                    'url': row['url'],  # Add url column
                    'chunk_text': chunk_text,
                    'last_scraped': now_timestamp  # Add last_scraped timestamp
                })
        except Exception as e:
            logger.error(f"Failed to chunk content for id {row['id']}: {e}")
            log_failed_job(db_uri, row['id'], 'chunking', e)


    if not all_chunks:
        logger.info(f"No chunks generated. Content might be empty or too small.")
        return

    chunks_df = pd.DataFrame(all_chunks)
    
    # Generate embeddings
    # Assuming EmbeddingGenerator is configured via settings or environment variables
    embedding_generator = EmbeddingGenerator(method=settings.embedding_method, model_name=settings.embedding_model)
    try:
        embeddings = embedding_generator.generate_embeddings(chunks_df['chunk_text'].tolist())
        chunks_df['embedding'] = embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        # Log failure for all UIDs in this batch
        for uid in df['id'].unique():
            log_failed_job(db_uri, uid, 'embedding', e)
        return


    # Store in DB
    with engine.begin() as connection:
        # First, delete old embeddings for these IDs to handle content updates
        ids_to_delete = chunks_df['id'].unique().tolist()
        if ids_to_delete:
            connection.execute(text(f"DELETE FROM {table_name} WHERE id = ANY(:ids)"), {'ids': ids_to_delete})
        
        # Then, insert new embeddings
        chunks_df.to_sql(table_name, connection, if_exists='append', index=False, method='multi')
        logger.info(f"Successfully stored {len(chunks_df)} chunks in {table_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and store embeddings for page content.")
    parser.add_argument('--uids', nargs='+', help='Optional list of UIDs to process.')
    parser.add_argument('--table-name', type=str, default=settings.table_name,
                        help='Name of the table to store embeddings.')
    parser.add_argument('--chunk-strategy', type=str, default='recursive_chunk_text',
                        help='Name of the chunking strategy to use (e.g., recursive_chunk_text).')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='The size of each text chunk.')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                        help='The overlap between chunks.')
    
    args = parser.parse_args()

    db_uri = f"postgresql://{settings.db_username}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    chunk_options = {
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap
        }
    
    # This is a bit of a hack to make the strategy name match the function name.
    # In a real scenario, get_chunking_strategy would be more robust.
    if args.chunk_strategy == 'recursive':
        args.chunk_strategy = 'recursive_chunk_text'


    process_and_store_embeddings(
        db_uri=db_uri,
        table_name=args.table_name,
        chunk_strategy_name=args.chunk_strategy,
        chunk_options=chunk_options,
        uids=args.uids
    ) 