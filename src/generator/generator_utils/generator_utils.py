from src.config import settings
from src.generator.prompt_utils.prompt_templates import PromptFactory
import torch
import time
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from src.generator.generator_utils.embedding_utils import EmbeddingGenerator
import logging
from typing import Tuple, Optional, Any, Dict

logger = logging.getLogger(__name__)

RERANKER_MODEL = settings.reranker_model
EMBEDDING_MODEL = settings.embedding_model  



def initialize_models(model_type: str = "llama"):
    """Initialize and return the LLM and embedding models"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. GPU is required for this application.")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu_devices

    # Initialize embedding generator for OpenAI embeddings
    print("\nInitializing embedding generator...")
    embedding_generator = EmbeddingGenerator(
        method=settings.embedding_provider,
        model_name=EMBEDDING_MODEL,
        batch_size=8
    )
    
    # Initialize reranker model (this is a local model)
    print("\nInitializing reranker model...")
    reranker_model = SentenceTransformer(RERANKER_MODEL)
    
    if model_type == "llama":
        from llama_cpp import Llama
        # Load the LLM
        print("\nLoading LLM...")
        load_start = time.time()
        llm = Llama(
            model_path=settings.model_path,
            n_gpu_layers=settings.n_gpu_layers,
            n_ctx=settings.context_length,
            n_batch=settings.n_batch,
            verbose=True,
            use_mmap=settings.use_mmap,
            use_mlock=settings.use_mlock,
            offload_kqv=settings.offload_kqv,
            seed=settings.seed
        )
        load_time = time.time() - load_start
        print(f"\nModel loading took {load_time:.2f} seconds ({load_time/60:.2f} minutes)")
    elif model_type == "openai":
        llm = OpenAI(api_key=settings.openai_api_key)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    
    return llm, embedding_generator, reranker_model


def generate_answer(
    llm,
    prompt_text: str,
    generation_type: str,
    max_tokens: int = 256,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """Generate an answer using the LLM."""
    
    # Generate response based on model type
    if generation_type == "llama":
        response = llm(
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            echo=False
        )
        
        # Extract the generated text from the response
        if isinstance(response, dict) and "choices" in response:
            generated_text = response["choices"][0]["text"]
        else:
            generated_text = response.get("choices", [{}])[0].get("text", str(response))
            
        return {
            "choices": [{"text": generated_text}],
            "context": context,
            "prompt": prompt_text
        }
    
    elif generation_type == "openai":
        response = llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        generated_text = response.choices[0].message.content
        
        return {
            "choices": [{"text": generated_text}],
            #"prompt": prompt_text
        }
    
    else:
        raise ValueError(f"Unsupported generation type: {generation_type}")