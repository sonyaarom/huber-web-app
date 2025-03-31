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

# Optional dependencies with graceful fallbacks
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
except Exception as e:
    # Handle NumPy compatibility issues
    print(f"Error importing torch modules: {e}. Local embedding generation will be disabled.")
    TORCH_AVAILABLE = False

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from ..config import settings
from .chunking_utils import recursive_chunk_text, character_chunk_text, semantic_chunk_text
from .db_utils import fetch_page_content

# Configure logger
logger = logging.getLogger(__name__)


def ensure_token_limit(text: str, model_name: str, max_tokens: int = 8192) -> List[str]:
    """
    Recursively split text if it exceeds the maximum token limit.
    
    Args:
        text: The text to split
        model_name: The name of the model to get token encoding for
        max_tokens: Maximum allowed tokens per chunk
        
    Returns:
        List of text chunks each within the token limit
    """
    if not TIKTOKEN_AVAILABLE:
        logger.warning("tiktoken library not available. Skipping token limit check.")
        return [text]
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception as e:
        logger.warning(f"Could not get encoding for model {model_name}: {e}. Skipping token check.")
        return [text]
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    
    # Split the text at whitespace to avoid breaking words
    mid = len(text) // 2
    split_index = text.rfind(" ", 0, mid)
    if split_index == -1:
        split_index = mid
        
    first_part = text[:split_index].strip()
    second_part = text[split_index:].strip()
    
    return ensure_token_limit(first_part, model_name, max_tokens) + ensure_token_limit(second_part, model_name, max_tokens)


class EmbeddingGenerator:
    """
    Handles different embedding generation methods (local HuggingFace models or OpenAI API).
    
    Attributes:
        method: "local" for local models or "openai" for OpenAI API
        model_name: Model identifier
        batch_size: Number of texts to process at once
        device: The device to run the model on ("cuda" or "cpu")
    """
    
    def __init__(
        self, 
        method: str = "local", 
        model_name: Optional[str] = None, 
        batch_size: int = 8, 
        api_key: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            method: "local" for local models or "openai" for OpenAI API
            model_name: Model name to use
            batch_size: Batch size for local model processing
            api_key: OpenAI API key (required if method is "openai")
        
        Raises:
            ValueError: If the method is unknown or API key is missing
            ImportError: If required packages are not installed
        """
        self.method = method
        self.batch_size = batch_size
        self.model_name = model_name
        
        if method == "local":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch and/or transformers not available. Install with: pip install torch transformers")
            self._initialize_local_model()
        elif method == "openai":
            self._initialize_openai_client(api_key)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'local' or 'openai'")
    
    def _initialize_local_model(self):
        """Set up the local HuggingFace model."""
        # Default model if none specified
        if self.model_name is None:
            self.model_name = 'intfloat/multilingual-e5-large-instruct'
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Using local model: {self.model_name} on {self.device}")
    
    def _initialize_openai_client(self, api_key: Optional[str]):
        """Set up the OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        # Default model if none specified
        if self.model_name is None:
            self.model_name = "text-embedding-3-small"
        
        # Set API key
        client_params = {}
        if api_key:
            client_params["api_key"] = api_key
        elif hasattr(settings, 'openai_api_key') and settings.openai_api_key:
            client_params["api_key"] = settings.openai_api_key
        elif os.environ.get("OPENAI_API_KEY"):
            pass  # Client will pick it up automatically
        else:
            raise ValueError(
                "OpenAI API key must be provided via api_key parameter, "
                "settings.openai_api_key, or OPENAI_API_KEY environment variable"
            )
        
        self.client = OpenAI(**client_params)
        logger.info(f"Using OpenAI model: {self.model_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if self.method == "local":
            return self._generate_local_embeddings(texts)
        elif self.method == "openai":
            return self._generate_openai_embeddings(texts)
    
    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local HuggingFace model."""
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+self.batch_size]
            try:
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                
                embeddings.extend(batch_embeddings)
                
                # Periodically clear GPU memory
                if self.device == "cuda" and i % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning("CUDA out of memory. Reducing batch size and retrying.")
                    # Process one text at a time if batch processing fails
                    for text in batch_texts:
                        inputs = self.tokenizer(
                            [text], 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=512
                        ).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            single_embedding = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                        
                        embeddings.extend(single_embedding)
                        torch.cuda.empty_cache()
                else:
                    raise e
                    
        return embeddings
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating OpenAI embeddings"):
            batch_texts = texts[i:i+self.batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error with OpenAI API: {e}")
                raise
                
        return embeddings
    
    def cleanup(self):
        """Clean up resources."""
        if self.method == "local" and hasattr(self, 'model') and TORCH_AVAILABLE:
            if self.device == "cuda":
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache()
                logger.info("Model deleted and GPU memory cleared")


def store_embeddings(
    df: pd.DataFrame, 
    chunking_method: str, 
    chunk_size: int, 
    model_name: str,
    table_name: Optional[str] = None
) -> None:
    """
    Store embeddings in PostgreSQL database.
    
    Args:
        df: DataFrame containing embeddings and metadata
        chunking_method: Method used for chunking (recursive, character, semantic)
        chunk_size: Size of chunks used
        model_name: Name of the embedding model used
        table_name: Optional custom table name
        
    Raises:
        Exception: If database connection or write operations fail
    """
    # Create a sanitized table name if not provided
    if table_name is None:
        if '/' in model_name:
            model_identifier = model_name.split('/')[-1].replace('-', '_')
        else:
            model_identifier = model_name.replace('-', '_')
        # Add a timestamp suffix to avoid conflicts with existing tables
        timestamp = int(time.time())
        table_name = f"embeddings_{chunking_method}_{chunk_size}_{model_identifier}_{timestamp}"
    
    logger.info(f"Connecting to PostgreSQL database: {settings.db_name} on {settings.db_host}")
    
    try:
        # Import psycopg2 here to avoid issues if it's not used
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            PSYCOPG2_AVAILABLE = True
        except ImportError:
            logger.warning("psycopg2 not available, falling back to SQLAlchemy")
            PSYCOPG2_AVAILABLE = False
        
        # First, let's check types to avoid SQL errors
        # Ensure id is treated as string regardless of input type
        df['id'] = df['id'].astype(str)
        
        # Get vector dimensions for the first embedding
        vector_dim = len(df['embedding'].iloc[0])
        logger.info(f"Vector dimension: {vector_dim}")
        
        # Check if pgvector extension might be available
        use_vector_type = False
        
        if PSYCOPG2_AVAILABLE:
            # Use direct psycopg2 connection for better control
            conn = psycopg2.connect(
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_username,
                password=settings.db_password,
                dbname=settings.db_name
            )
            conn.autocommit = False
            cursor = conn.cursor()
            
            try:
                # Try to enable vector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                use_vector_type = True
                logger.info("Enabled pgvector extension")
            except Exception as e:
                logger.warning(f"Could not enable pgvector extension: {e}. Will use REAL[] instead.")
                conn.rollback()
                use_vector_type = False
            
            # Drop existing table if it exists
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
            
            # Create table
            if use_vector_type:
                create_table_sql = f"""
                CREATE TABLE {table_name} (
                    id VARCHAR(255),
                    split_id INTEGER,
                    url TEXT,
                    chunk_text TEXT,
                    embedding vector({vector_dim}),
                    last_scraped TIMESTAMP
                );"""
            else:
                create_table_sql = f"""
                CREATE TABLE {table_name} (
                    id VARCHAR(255),
                    split_id INTEGER,
                    url TEXT,
                    chunk_text TEXT,
                    embedding REAL[],
                    last_scraped TIMESTAMP
                );"""
                
            cursor.execute(create_table_sql)
            logger.info(f"Created table {table_name}")
            
            # Insert data in batches
            # Convert embeddings to proper format
            data_to_insert = []
            for _, row in df.iterrows():
                data_to_insert.append((
                    row['id'],
                    row['split_id'],
                    row['url'],
                    row['chunk_text'],
                    row['embedding'],
                    row['last_scraped']
                ))
            
            insert_sql = f"""
            INSERT INTO {table_name} (id, split_id, url, chunk_text, embedding, last_scraped)
            VALUES %s;
            """
            
            # FIX: Add the missing %s placeholder for last_scraped in the vector template
            if use_vector_type:
                # Changed from "(%s, %s, %s, %s, %s::vector)" to include the last_scraped placeholder
                template = "(%s, %s, %s, %s, %s::vector, %s)"
            else:
                template = "(%s, %s, %s, %s, %s::real[], %s)"
            
            execute_values(cursor, insert_sql, data_to_insert, template=template)
            logger.info(f"Inserted {len(data_to_insert)} rows")
            
            # Create indices
            cursor.execute(f"CREATE INDEX idx_{table_name}_id ON {table_name} (id);")
            
            if use_vector_type:
                try:
                    cursor.execute(f"""
                    CREATE INDEX idx_{table_name}_embedding ON {table_name}
                    USING ivfflat (embedding vector_cosine_ops);
                    """)
                    logger.info("Created vector index")
                except Exception as e:
                    logger.warning(f"Could not create vector index: {e}")
                    conn.rollback()
            
            # Commit all changes
            conn.commit()
            logger.info(f"Successfully stored {len(df)} embeddings in table: {table_name}")
            
            # Close connection
            cursor.close()
            conn.close()
            
        else:
            # Fallback to SQLAlchemy
            engine = create_engine(
                f"postgresql://{settings.db_username}:{settings.db_password}@"
                f"{settings.db_host}:{settings.db_port}/{settings.db_name}"
            )
            
            # Check for table existence
            inspector = inspect(engine)
            table_exists = table_name in inspector.get_table_names()
            
            with engine.begin() as conn:
                if table_exists:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    use_vector_type = True
                except Exception:
                    use_vector_type = False
                
                # Create table
                if use_vector_type:
                    conn.execute(text(f"""
                    CREATE TABLE {table_name} (
                        id VARCHAR(255),
                        split_id INTEGER,
                        url TEXT,
                        chunk_text TEXT,
                        embedding vector({vector_dim}),
                        last_scraped TIMESTAMP
                    )
                    """))
                else:
                    conn.execute(text(f"""
                    CREATE TABLE {table_name} (
                        id VARCHAR(255),
                        split_id INTEGER,
                        url TEXT,
                        chunk_text TEXT,
                        embedding REAL[],
                        last_scraped TIMESTAMP
                    )
                    """))
                
                # Insert data
                for _, row in df.iterrows():
                    embedding_str = str(row['embedding']).replace('[', '{').replace(']', '}')
                    
                    # FIX: Add a comma before last_scraped value
                    sql = f"""
                    INSERT INTO {table_name} (id, split_id, url, chunk_text, embedding, last_scraped)
                    VALUES ('{row['id']}', {row['split_id']}, 
                            '{row['url'].replace("'", "''")}', 
                            '{row['chunk_text'].replace("'", "''")}', 
                            '{embedding_str}'::{'vector' if use_vector_type else 'REAL[]'},
                            '{row['last_scraped']}'
                    )
                    """
                    conn.execute(text(sql))
                
                # Create indices
                conn.execute(text(f"CREATE INDEX idx_{table_name}_id ON {table_name} (id);"))
                
                if use_vector_type:
                    try:
                        conn.execute(text(f"""
                        CREATE INDEX idx_{table_name}_embedding ON {table_name}
                        USING ivfflat (embedding vector_cosine_ops);
                        """))
                    except Exception as e:
                        logger.warning(f"Could not create vector index: {e}")
            
            logger.info(f"Successfully stored {len(df)} embeddings in table: {table_name}")
            
    except Exception as e:
        logger.error(f"Error storing embeddings in PostgreSQL: {str(e)}")
        raise


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
    Process text data, generate embeddings, and store in database.
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        batch_size: Batch size for processing
        chunking_method: Method for chunking text ('recursive', 'character', 'semantic')
        model_name: Name of the model to use (defaults based on method)
        api_key: API key for OpenAI (if using 'openai' method)
        prefer_openai: Whether to try OpenAI first (if available)
        table_name: Optional custom table name
        
    Raises:
        RuntimeError: If no embedding generator can be initialized
        ValueError: If an unknown chunking method is specified
    """
    # Fetch the page content
    df = fetch_page_content()
    df = df.head(1)  # Process only first row for testing
    
    # Validate the id column - ensure it's always a string
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
        logger.info(f"ID column type in source data: {df['id'].dtype}")
        
    all_data = []
    
    # Define OpenAI model names
    openai_models = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
    
    # Adjust chunk size for OpenAI character chunking
    if chunking_method == 'character' and (model_name in openai_models):
        adjusted_chunk_size = min(chunk_size * 4, 7000)
        logger.info(f"Adjusting character chunk size to {adjusted_chunk_size} for OpenAI model")
        chunk_size = adjusted_chunk_size

    # Check if using an OpenAI model
    is_openai_model = model_name in openai_models if model_name else False
    
    # Set up embedding generation methods to try
    embedding_generator = None
    errors = []
    methods_to_try = []
    
    # Determine which methods to try and in what order
    if is_openai_model or (prefer_openai and model_name is None):
        methods_to_try.append(('openai', model_name))
        if not is_openai_model and TORCH_AVAILABLE:
            methods_to_try.append(('local', None))
    else:
        if TORCH_AVAILABLE:
            methods_to_try.append(('local', model_name))
        if prefer_openai:
            methods_to_try.append(('openai', None))
    
    # If torch is not available and 'local' was the only option, add a fallback to OpenAI
    if not TORCH_AVAILABLE and not any(method == 'openai' for method, _ in methods_to_try):
        logger.warning("PyTorch not available, falling back to OpenAI embedding")
        methods_to_try.append(('openai', None))
    
    # Try to initialize embedding generator with each method
    for method, method_model_name in methods_to_try:
        try:
            logger.info(
                f"Attempting to initialize {method} embedding generator with model: "
                f"{method_model_name or 'default'}"
            )
            
            current_api_key = None
            if method == 'openai':
                current_api_key = api_key or (hasattr(settings, 'openai_api_key') and settings.openai_api_key)
                
            embedding_generator = EmbeddingGenerator(
                method=method,
                model_name=method_model_name,
                batch_size=batch_size,
                api_key=current_api_key
            )
            
            logger.info(f"Successfully initialized {method} embedding generator")
            break
            
        except Exception as e:
            error_msg = f"Failed to initialize {method} embedding generator: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
    
    # If no generator could be initialized, raise an error
    if embedding_generator is None:
        raise RuntimeError(f"Failed to initialize any embedding generator. Errors: {'; '.join(errors)}")
    
    try:
        # Process each page
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Pages"):
            text = row['extracted_content']
            url = row['url']
            page_id = row['id']
            last_scraped = row['last_scraped']
            # Apply the selected chunking method
            if chunking_method == 'recursive':
                chunks = recursive_chunk_text(text, chunk_size, chunk_overlap)
            elif chunking_method == 'character':
                chunks = character_chunk_text(text, chunk_size, chunk_overlap)
            elif chunking_method == 'semantic':
                chunks = semantic_chunk_text(text)
            else:
                raise ValueError(f"Unknown chunking method: {chunking_method}")

            # For OpenAI models, ensure each chunk is within the token limit
            final_chunks = []
            if embedding_generator.method == "openai":
                for chunk in chunks:
                    sub_chunks = ensure_token_limit(chunk, embedding_generator.model_name, max_tokens=8192)
                    final_chunks.extend(sub_chunks)
            else:
                final_chunks = chunks

            # Generate embeddings if chunks exist
            if final_chunks:
                embeddings_list = embedding_generator.generate_embeddings(final_chunks)
                
                # Store results
                for i, (chunk, embedding) in enumerate(zip(final_chunks, embeddings_list)):
                    all_data.append({
                        "id": page_id,
                        "split_id": i + 1,
                        "url": url,
                        "chunk_text": chunk,
                        "embedding": embedding,
                        "last_scraped": last_scraped
                    })
            else:
                logger.warning(f"No valid chunks generated for page {page_id}")

        # Create DataFrame and store embeddings
        embeddings_df = pd.DataFrame(all_data)
        
        store_embeddings(
            embeddings_df, 
            chunking_method, 
            chunk_size, 
            embedding_generator.model_name,
            table_name
        )
        
        logger.info("Embeddings stored successfully in database.")
        
    finally:
        # Clean up resources
        if embedding_generator:
            embedding_generator.cleanup()



# if __name__ == "__main__":
# #     # First try the direct psycopg2 test to ensure connection works

#     print("Testing with OpenAI embeddings")
#     process_and_store_embeddings(
#         chunk_size=512,
#         chunk_overlap=50,
#         chunking_method='recursive',
#         prefer_openai=True,
#         model_name="text-embedding-3-small",
#         table_name="test_embeddings"
#         )