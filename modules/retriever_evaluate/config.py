from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List, Union

class Settings(BaseSettings):

    db_host: str = Field(env='DB_HOST')
    db_port: int = Field(env='DB_PORT')
    db_name: str = Field(env='DB_NAME')
    db_username: str = Field(env='DB_USERNAME')
    db_password: str = Field(env='DB_PASSWORD')
    openai_api_key: str = Field(env='OPENAI_API_KEY')

    wandb_entity: str = Field(env='WANDB_ENTITY')

    qa_pairs_path: str = Field(env='QA_PAIRS_PATH')
    top_k: int = Field(env='TOP_K', default=10)
    threshold: Optional[float] = Field(env='THRESHOLD', default=None)
    
    hybrid_alpha: Union[float, List[float]] = Field(env='HYBRID_ALPHA', default=0.5)

    reranker_model: str = Field(env='RERANKER_MODEL', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    class Config:
        extra = 'ignore'
        case_sensitive = False  # Make environment variable names case-insensitive

settings = Settings()

print("Settings loaded successfully!")
print(f"DB Host: {settings.db_host}")
print(f"DB Port: {settings.db_port}")

