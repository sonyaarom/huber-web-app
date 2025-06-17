import os
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings
from typing import Optional, List, Union, Dict
from dotenv import load_dotenv

class Settings(BaseSettings):
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # --------------------------------------------------------------------------
    # Database Settings
    # --------------------------------------------------------------------------
    db_host: str = Field(env='DB_HOST', default='localhost')
    db_port: int = Field(env='DB_PORT', default=5432)
    db_name: str = Field(env='DB_NAME', default='huber_db')
    db_username: str = Field(env='DB_USERNAME', default='postgres')
    db_password: str = Field(env='DB_PASSWORD', default='password')

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """
        Constructs the full database URL from individual components.
        """
        return f"postgresql+psycopg2://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # --------------------------------------------------------------------------
    # API Keys
    # --------------------------------------------------------------------------
    openai_api_key: str = Field(env='OPENAI_API_KEY')
    together_api_key: str = Field(env='TOGETHER_API_KEY')
    langfuse_secret_key: Optional[str] = Field(env='LANGFUSE_SECRET_KEY', default=None)
    langfuse_public_key: Optional[str] = Field(env='LANGFUSE_PUBLIC_KEY', default=None)
    pinecone_api_key: Optional[str] = Field(env='PINECONE_API_KEY', default=None)
    langchain_api_key: Optional[str] = Field(env='LANGCHAIN_API_KEY', default=None)
    secret_key: str = Field(env='SECRET_KEY', default='a-super-secret-key-for-development')
    admin_password: str = Field(env='ADMIN_PASSWORD', default='password')
    
    # --------------------------------------------------------------------------
    # Langfuse Settings
    # --------------------------------------------------------------------------
    langfuse_host: Optional[str] = Field(env='LANGFUSE_HOST', default="https://cloud.langfuse.com")
    default_prompt_type: Optional[str] = Field(env='DEFAULT_PROMPT_TYPE', default="advanced")
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

    # --------------------------------------------------------------------------
    # Retriever Settings
    # --------------------------------------------------------------------------
    embedding_model: str = Field(env='EMBEDDING_MODEL', default='text-embedding-3-large')
    embedding_method: str = Field(env='EMBEDDING_METHOD', default='openai')
    table_name: str = Field(env='TABLE_NAME', default='page_embeddings_a')
    top_k: int = Field(env='TOP_K', default=10)
    threshold: Optional[float] = Field(env='THRESHOLD', default=0.5)

    # --------------------------------------------------------------------------
    # Reranker Settings
    # --------------------------------------------------------------------------
    use_reranker: bool = Field(env='USE_RERANKER', default=True)
    reranker_model: str = Field(env='RERANKER_MODEL', default='cross-encoder/ms-marco-MiniLM-L-6-v2')

    # --------------------------------------------------------------------------
    # Hybrid Search Settings
    # --------------------------------------------------------------------------
    use_hybrid_search: bool = Field(env='USE_HYBRID_SEARCH', default=True)
    hybrid_alpha: float = Field(env='HYBRID_ALPHA', default=0.5)
    use_ner: bool = Field(env='USE_NER', default=False)

    # --------------------------------------------------------------------------
    # Data Ingestion Settings
    # --------------------------------------------------------------------------
    url: Optional[str] = Field('https://www.wiwi.hu-berlin.de/sitemap.xml.gz')
    pattern: Optional[str] = Field(r'<url>\\s*<loc>(.*?)</loc>\\s*<lastmod>(.*?)</lastmod>')
    exclude_extensions: Optional[List[str]] = Field(['.jpg', '.pdf', '.jpeg', '.png'])
    exclude_patterns: Optional[List[str]] = Field(['view'])
    include_patterns: Optional[List[str]] = Field(['/en/'])
    allowed_base_url: Optional[str] = Field('https://www.wiwi.hu-berlin.de')
    
    # --------------------------------------------------------------------------
    # Evaluation Settings
    # --------------------------------------------------------------------------
    qa_pairs_path: Optional[str] = Field(env='QA_PAIRS_PATH', default=None)
    wandb_entity: Optional[str] = Field(env='WANDB_ENTITY', default=None)
    wandb_project: str = Field(env='WANDB_PROJECT', default="prompt-evaluation")
    alpha: float = Field(env='ALPHA', default=0.5)
    page_content_table: str = Field(env='PAGE_CONTENT_TABLE', default="page_content")
    page_embeddings_table: str = Field(env='PAGE_EMBEDDINGS_TABLE', default="page_embeddings_alpha")
    page_keywords_table: str = Field(env='PAGE_KEYWORDS_TABLE', default="page_keywords")
    pinecone_environment: Optional[str] = Field(env='PINECONE_ENVIRONMENT', default="us-west1-gcp")
    pinecone_index: Optional[str] = Field(env='PINECONE_INDEX', default="default-index")
    model_path: str = Field(env='MODEL_PATH', default="/home/RDC/konchaks/prompt-evaluation/Meta-Llama-3.1-70B-Instruct-Q2_K.gguf")
    n_batch: int = Field(env='N_BATCH', default=1)
    use_mmap: bool = Field(env='USE_MMAP', default=False)
    use_mlock: bool = Field(env='USE_MLOCK', default=False)
    offload_kqv: bool = Field(env='OFFLOAD_KQV', default=False)
    seed: int = Field(env='SEED', default=42)
    bm25_values: str = Field(env='BM25_VALUES', default="/home/RDC/konchaks/prompt-evaluation/assets/json/bm25_values.json")
    bm25_alpha: float = Field(env='BM25_ALPHA', default=0.6)

    # --------------------------------------------------------------------------
    # Generator Settings
    # --------------------------------------------------------------------------
    gpu_devices: Optional[str] = Field(env='GPU_DEVICES', default="0")
    n_gpu_layers: Optional[int] = Field(env='N_GPU_LAYERS', default=80)
    context_length: Optional[int] = Field(env='CONTEXT_LENGTH', default=4096)
    separator: Optional[str] = Field(env='SEPARATOR', default="\\n\\n")
    embedding_provider: Optional[str] = Field(env='EMBEDDING_PROVIDER', default="openai")
    openai_model: Optional[str] = Field(env='OPENAI_MODEL', default="gpt-3.5-turbo")
    
    class Config:
        env_file = '.venv'
        extra = 'ignore'
        case_sensitive = False

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> any:
            if field_name in ['use_ner', 'use_reranker', 'use_hybrid_search']:
                return raw_val.lower() == 'true'
            return raw_val

# Initialize settings
settings = Settings()

def reload_settings():
    """Reload settings from the .venv file."""
    global settings
    dotenv_path = os.path.join(settings.base_dir, '.venv')
    load_dotenv(dotenv_path=dotenv_path, override=True)
    settings = Settings()
    return settings

DB_PARAMS = {
    "host": settings.db_host,
    "port": settings.db_port,
    "dbname": settings.db_name,
    "user": settings.db_username,
    "password": settings.db_password,
} 