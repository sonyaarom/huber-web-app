import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings
# Import your other modules with absolute paths
from .splitting_utils import recursive_chunk_text, character_chunk_text, semantic_chunk_text
from .db_utils import fetch_page_content

# Optional imports for OpenAI
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import pandas as pd
import json
from sqlalchemy import create_engine, inspect, text
from tqdm import tqdm
from logging import getLogger
from transformers import AutoTokenizer, AutoModel
import torch

# Try importing tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Logger setup
logger = getLogger(__name__)

def ensure_token_limit(text, model_name, max_tokens=8192):
    """
    Recursively split the text if it exceeds the maximum token limit.
    Returns a list of text chunks each within the token limit.
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
    else:
        # Split the text roughly in half at a nearby whitespace to avoid breaking words.
        mid = len(text) // 2
        split_index = text.rfind(" ", 0, mid)
        if split_index == -1:
            split_index = mid
        first_part = text[:split_index].strip()
        second_part = text[split_index:].strip()
        return ensure_token_limit(first_part, model_name, max_tokens) + ensure_token_limit(second_part, model_name, max_tokens)

class EmbeddingGenerator:
    """Class to handle different embedding generation methods"""
    
    def __init__(self, method="local", model_name=None, batch_size=8, api_key=None):
        """
        Initialize the embedding generator.
        
        Args:
            method: "local" for local models or "openai" for OpenAI API
            model_name: Model name to use (e.g., 'intfloat/multilingual-e5-large-instruct')
            batch_size: Batch size for local model processing
            api_key: OpenAI API key (required if method is "openai")
        """
        self.method = method
        self.batch_size = batch_size
        self.model_name = model_name
        
        if method == "local":
            # Default model if none specified
            if model_name is None:
                self.model_name = 'intfloat/multilingual-e5-large-instruct'
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Check for GPU availability
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Using local model: {self.model_name} on {self.device}")
            
        elif method == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
            
            # Default model if none specified
            if model_name is None:
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
                raise ValueError("OpenAI API key must be provided via api_key parameter, settings.openai_api_key, or OPENAI_API_KEY environment variable")
            
            self.client = OpenAI(**client_params)
            logger.info(f"Using OpenAI model: {self.model_name}")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'local' or 'openai'")
    
    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts"""
        if self.method == "local":
            return self._generate_local_embeddings(texts)
        elif self.method == "openai":
            return self._generate_openai_embeddings(texts)
    
    def _generate_local_embeddings(self, texts):
        """Generate embeddings using local model"""
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+self.batch_size]
            try:
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, 
                                          truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                embeddings.extend(batch_embeddings)
                if self.device == "cuda" and i % 100 == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning("CUDA out of memory. Reducing batch size and retrying.")
                    for text in batch_texts:
                        inputs = self.tokenizer([text], return_tensors="pt", padding=True, 
                                                 truncation=True, max_length=512).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            single_embedding = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
                        embeddings.extend(single_embedding)
                        torch.cuda.empty_cache()
                else:
                    raise e
        return embeddings
    
    def _generate_openai_embeddings(self, texts):
        """Generate embeddings using OpenAI API"""
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
        """Clean up resources"""
        if self.method == "local" and hasattr(self, 'model'):
            if self.device == "cuda":
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache()
                logger.info("Model deleted and GPU memory cleared")

def store_embeddings(df: pd.DataFrame, chunking_method: str, chunk_size: int, model_name: str):
    """
    Store embeddings in PostgreSQL database.
    
    Args:
        df: DataFrame containing embeddings and metadata
        chunking_method: Method used for chunking (recursive, character, semantic)
        chunk_size: Size of chunks used
        model_name: Name of the embedding model used
    """
    if '/' in model_name:
        model_identifier = model_name.split('/')[-1].replace('-', '_')
    else:
        model_identifier = model_name.replace('-', '_')
    
    table_name = f"test_embeddings_{chunking_method}_{chunk_size}_{model_identifier}"
    logger.info(f"Connecting to PostgreSQL database: {settings.db_name} on {settings.db_host}")
    
    try:
        engine = create_engine(
            f"postgresql://{settings.db_username}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        )
        if isinstance(df['embedding'].iloc[0], list):
            df['embedding'] = df['embedding'].apply(json.dumps)
            logger.info("Converted embedding lists to JSON strings")
        inspector = inspect(engine)
        table_exists = table_name in inspector.get_table_names()
        with engine.begin() as conn:
            if table_exists:
                logger.info(f"Table {table_name} already exists. Appending data.")
                df.to_sql(table_name, conn, if_exists="append", index=False)
            else:
                logger.info(f"Creating new table: {table_name}")
                df.to_sql(table_name, conn, if_exists="replace", index=False)
        if not table_exists:
            with engine.begin() as conn:
                conn.execute(text(f"CREATE INDEX idx_{table_name}_id ON {table_name} (id);"))
                logger.info(f"Created index on id column for {table_name}")
        logger.info(f"Successfully stored {len(df)} embeddings in table: {table_name}")
    except Exception as e:
        logger.error(f"Error storing embeddings in PostgreSQL: {e}")
        raise

def process_and_store_embeddings(
    chunk_size=512, 
    chunk_overlap=50, 
    batch_size=8, 
    chunking_method='recursive',
    model_name=None,
    api_key=None,
    prefer_openai=True
):
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
    """
    df = fetch_page_content()
    all_data = []
    
    openai_models = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
    if chunking_method == 'character' and (model_name in openai_models):
        adjusted_chunk_size = min(chunk_size * 4, 7000)
        logger.info(f"Adjusting character chunk size to {adjusted_chunk_size} for OpenAI model")
        chunk_size = adjusted_chunk_size

    is_openai_model = model_name in openai_models if model_name else False
    
    embedding_generator = None
    errors = []
    methods_to_try = []
    
    if is_openai_model or (prefer_openai and model_name is None):
        methods_to_try.append(('openai', model_name))
        if not is_openai_model:
            methods_to_try.append(('local', None))
    else:
        methods_to_try.append(('local', model_name))
        if prefer_openai:
            methods_to_try.append(('openai', None))
    
    for method, method_model_name in methods_to_try:
        try:
            logger.info(f"Attempting to initialize {method} embedding generator with model: {method_model_name or 'default'}")
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
    
    if embedding_generator is None:
        raise RuntimeError(f"Failed to initialize any embedding generator. Errors: {'; '.join(errors)}")
    
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Pages"):
            text = row['extracted_content']
            url = row['url']
            page_id = row['id']

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

            if final_chunks:
                embeddings_list = embedding_generator.generate_embeddings(final_chunks)
                for i, (chunk, embedding) in enumerate(zip(final_chunks, embeddings_list)):
                    all_data.append({
                        "id": page_id,
                        "split_id": i + 1,
                        "url": url,
                        "chunk_text": chunk,
                        "embedding": embedding
                    })
            else:
                logger.warning(f"No valid chunks generated for page {page_id}")

        embeddings_df = pd.DataFrame(all_data)
        
        store_embeddings(
            embeddings_df, 
            chunking_method, 
            chunk_size, 
            embedding_generator.model_name
        )

        logger.info("Embeddings stored successfully in database.")
        
    finally:
        if embedding_generator:
            embedding_generator.cleanup()
