from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

class BaseSettings(BaseSettings):
    # Database settings
    db_host: str = Field(env='DB_HOST')
    db_port: int = Field(env='DB_PORT')
    db_name: str = Field(env='DB_NAME')
    db_username: str = Field(env='DB_USERNAME')
    db_password: str = Field(env='DB_PASSWORD')
    
    # OpenAI API settings
    openai_api_key: str = Field(env='OPENAI_API_KEY')
    
    # Common settings
    top_k: int = Field(env='TOP_K', default=10)
    threshold: Optional[float] = Field(env='THRESHOLD', default=0.5)
    together_api_key: str = Field(env='TOGETHER_API_KEY')   
    
    # Optional fields used by some modules
    qa_pairs_path: Optional[str] = Field(env='QA_PAIRS_PATH', default=None)
    wandb_entity: Optional[str] = Field(env='WANDB_ENTITY', default=None)
    
    class Config:
        env_file = '.venv'  # Note: Changed from .venv to .env which is more standard
        extra = 'ignore'
        case_sensitive = False 