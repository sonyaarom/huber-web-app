from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Optional
import os

class Settings(BaseSettings):
    # Required settings with defaults for testing
    LANGFUSE_SECRET_KEY: Optional[str] = Field(env='LANGFUSE_SECRET_KEY', default=None)
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(env='LANGFUSE_PUBLIC_KEY', default=None)
    LANGFUSE_HOST: Optional[str] = Field(env='LANGFUSE_HOST', default="https://cloud.langfuse.com")
    DEFAULT_PROMPT_TYPE: Optional[str] = Field(env='DEFAULT_PROMPT_TYPE', default="advanced")
    OPENAI_API_KEY: Optional[str] = Field(env='OPENAI_API_KEY', default=None)
    EMBEDDING_MODEL: Optional[str] = Field(env='EMBEDDING_MODEL', default="text-embedding-3-large")
    RERANKER_MODEL: Optional[str] = Field(env='RERANKER_MODEL', default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    GPU_DEVICES: Optional[str] = Field(env='GPU_DEVICES', default="0")
    N_GPU_LAYERS: Optional[int] = Field(env='N_GPU_LAYERS', default=80)
    CONTEXT_LENGTH: Optional[int] = Field(env='CONTEXT_LENGTH', default=4096)
    SEPARATOR: Optional[str] = Field(env='SEPARATOR', default="\n\n")
    EMBEDDING_PROVIDER: Optional[str] = Field(env='EMBEDDING_PROVIDER', default="openai")
    
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
        env_file = ".venv"
        env_file_encoding = "utf-8"
        ignore_extra = True
        #allow extra
        extra = "allow"

settings = Settings()

