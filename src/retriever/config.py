from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database settings
    db_host: str = Field(env='DB_HOST')
    db_port: int = Field(env='DB_PORT')
    db_name: str = Field(env='DB_NAME')
    db_username: str = Field(env='DB_USERNAME')
    db_password: str = Field(env='DB_PASSWORD')
    
    # OpenAI API settings
    openai_api_key: str = Field(env='OPENAI_API_KEY')
    
    # Retriever settings
    embedding_model: str = Field(env='EMBEDDING_MODEL', default='text-embedding-3-large')
    embedding_method: str = Field(env='EMBEDDING_METHOD', default='openai')
    table_name: str = Field(env='TABLE_NAME', default='page_embeddings_alpha')
    top_k: int = Field(env='TOP_K', default=10)
    threshold: Optional[float] = Field(env='THRESHOLD', default=0.5)
    
    # Reranker settings
    use_reranker: bool = Field(env='USE_RERANKER', default=True)
    reranker_model: str = Field(env='RERANKER_MODEL', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Hybrid search settings
    use_hybrid_search: bool = Field(env='USE_HYBRID_SEARCH', default=True)
    hybrid_alpha: float = Field(env='HYBRID_ALPHA', default=0.5)
    
    class Config:
        env_file = '.venv'
        extra = 'ignore'
        case_sensitive = False

settings = Settings() 