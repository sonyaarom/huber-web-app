

from src.evaluators.retriever import Retriever
from src.config import settings
from .config import settings as evaluator_settings
from src.prompts.prompt_templates import PromptFactory
import torch
import time
import os
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from openai import OpenAI
from src.evaluators.embedding_utils import EmbeddingGenerator



RERANKER_MODEL = settings.RERANKER_MODEL
EMBEDDING_MODEL = settings.EMBEDDING_MODEL  



def initialize_models(model_type: str = "llama"):
    """Initialize and return the LLM and embedding models"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for this application.")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    os.environ['CUDA_VISIBLE_DEVICES'] = settings.GPU_DEVICES

    # Initialize embedding generator for OpenAI embeddings
    print("\nInitializing embedding generator...")
    embedding_generator = EmbeddingGenerator(
        method=evaluator_settings.embedding_provider,
        model_name=EMBEDDING_MODEL,
        batch_size=8
    )
    
    # Initialize reranker model (this is a local model)
    print("\nInitializing reranker model...")
    reranker_model = SentenceTransformer(RERANKER_MODEL)
    
    if model_type == "llama":
        # Load the LLM
        print("\nLoading LLM...")
        load_start = time.time()
        llm = Llama(
            model_path=settings.MODEL_PATH,
            n_gpu_layers=settings.N_GPU_LAYERS,
            n_ctx=settings.CONTEXT_LENGTH,
            n_batch=settings.N_BATCH,
            verbose=True,
            use_mmap=settings.USE_MMAP,
            use_mlock=settings.USE_MLOCK,
            offload_kqv=settings.OFFLOAD_KQV,
            seed=settings.SEED
        )
        load_time = time.time() - load_start
        print(f"\nModel loading took {load_time:.2f} seconds ({load_time/60:.2f} minutes)")
    elif model_type == "openai":
        llm = OpenAI(api_key=settings.OPENAI_API_KEY)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    
    return llm, embedding_generator, reranker_model
