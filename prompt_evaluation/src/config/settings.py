from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Optional
import os

class Settings(BaseSettings):
    # Required settings with defaults for testing
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_HOST: Optional[str] = "https://cloud.langfuse.com"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = "us-west1-gcp"
    PINECONE_INDEX: Optional[str] = "default-index"
    LANGCHAIN_API_KEY: Optional[str] = None
    MODEL_PATH: str = "/home/RDC/konchaks/prompt-evaluation/Meta-Llama-3.1-70B-Instruct-Q2_K.gguf"
    GPU_DEVICES: str = "0"
    N_BATCH: int = 1
    USE_MMAP: bool = False
    USE_MLOCK: bool = False
    OFFLOAD_KQV: bool = False
    SEED: int = 42
    N_GPU_LAYERS: int = 80
    CONTEXT_LENGTH: int = 4096
    SEPARATOR: str = "\n\n"
    OPENAI_API_KEY: Optional[str] 
    OPENAI_MODEL: Optional[str] = "gpt-3.5-turbo"
    BM25_VALUES: str = "/home/RDC/konchaks/prompt-evaluation/assets/json/bm25_values.json"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Provide default values for database settings
    db_host: str = Field("localhost", env='DB_HOST')
    db_port: int = Field(5432, env='DB_PORT')
    db_name: str = Field("huber_db", env='DB_NAME')
    db_username: str = Field("huber", env='DB_USERNAME')
    db_password: str = Field(env='DB_PASSWORD')

    EMBEDDING_MODEL: str = "text-embedding-3-large"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    #Model configuration
    BM25_ALPHA: float = 0.6

    #Evaluation configuration
    WANDB_PROJECT: str = "prompt-evaluation"
    
    # Langfuse prompt IDs configuration
    langfuse_prompt_ids: Dict[str, str] = {
        "main-task-v2": "main-task-v2",
        "main-task-v1": "main-task-v1", 
        "quality-check-v1": "quality-check-v1",
        "shot-examples-v1": "shot-examples-v1",
        "style-guidelines-v1": "style-guidelines-v1",
        "role-base-v2": "role-base-v2",
        "role-basic-v1": "role-basic-v1",
        "role-university-advisor": "role-university-advisor",
        "rag-basic-v1": "rag-basic-v1",
        "context-v1": "context-v1",
        "question-v1": "question-v1",
        "starting_meta_tags": "starting_meta_tags",
        "ending_meta_tags": "ending_meta_tags"
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        ignore_extra = True

settings = Settings()

