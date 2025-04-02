from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    alpha: float = 0.5
    openai_api_key: str 
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str 
    db_username: str = "huber"
    db_password: str 
    embedding_model: str = "text-embedding-3-large"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_provider: str = "openai"
    page_content_table: str = "page_content"
    page_embeddings_table: str = "page_embeddings_alpha"
    page_keywords_table: str = "page_keywords"

    
    class Config:
        ignore_extra = True

settings = Settings()