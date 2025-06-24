from hubert.config import settings
import torch
import time
import os
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

def initialise_llama(model_path: str):
    """Initializes and returns the Llama model."""
    from llama_cpp import Llama
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Running on CPU may be slow.")

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu_devices

    logger.info("Loading Llama model...")
    load_start = time.time()
    llm = Llama(
        model_path=model_path,
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
    logger.info(f"Model loading took {load_time:.2f} seconds ({load_time/60:.2f} minutes)")
    return llm

def initialise_openai(api_key: str):
    """Initializes and returns the OpenAI client."""
    if not api_key:
        raise ValueError("OpenAI API key is required.")
    return OpenAI(api_key=api_key) 