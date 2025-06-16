"""
Module for generating embeddings from text data.
Supports multiple embedding methods.
"""

import sys
import os
from pathlib import Path
import logging
from typing import List, Optional

# Third-party imports
from tqdm import tqdm

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
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from hubert.config import settings

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

            # Filter out empty or whitespace-only strings to avoid API errors
            cleaned_batch = [text.strip() for text in batch_texts if text and text.strip()]
            if not cleaned_batch:
                continue

            try:
                # Use the cleaned_batch instead of batch_texts
                response = self.client.embeddings.create(
                    input=cleaned_batch,
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

