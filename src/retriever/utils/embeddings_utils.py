import torch
import os
import logging
from ..config import settings

logger = logging.getLogger(__name__)

# Check if OpenAI package is available
try:
    from openai import OpenAI
    OPENAI_PACKAGE_AVAILABLE = True
except ImportError:
    OPENAI_PACKAGE_AVAILABLE = False
    logger.warning("OpenAI package not installed. Only local embedding methods will be available.")

class EmbeddingGenerator:
    """Class to handle different embedding generation methods"""
    
    def __init__(self, method="openai", model_name=None, batch_size=1, api_key=None):
        """
        Initialize the embedding generator.
        
        Args:
            method: "local" for local models or "openai" for OpenAI API
            model_name: Model name to use
            batch_size: Batch size for local model processing
            api_key: OpenAI API key (required if method is "openai")
        """
        self.method = method
        self.batch_size = batch_size
        self.model_name = model_name
        
        if method == "local":
            try:
                from transformers import AutoTokenizer, AutoModel
                # Default model if none specified
                if model_name is None:
                    self.model_name = 'intfloat/multilingual-e5-large-instruct'
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                
                # Check for GPU availability
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                logger.info(f"Using local model: {self.model_name} on {self.device}")
            except ImportError:
                raise ImportError("transformers package not installed. Install with: pip install transformers")
            
        elif method == "openai":
            if not OPENAI_PACKAGE_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
            
            # Default model if none specified
            if model_name is None:
                self.model_name = "text-embedding-3-large"
            
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
        for i in range(0, len(texts), self.batch_size):
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
        for i in range(0, len(texts), self.batch_size):
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