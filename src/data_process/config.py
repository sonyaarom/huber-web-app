import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List

# Get the absolute path to the .env file
current_dir = Path(__file__).parent
env_file_path = current_dir / '.env'

class Settings(BaseSettings):
    url: Optional[str] = Field('https://www.wiwi.hu-berlin.de/sitemap.xml.gz')
    pattern: Optional[str] = Field(r'<url>\s*<loc>(.*?)</loc>\s*<lastmod>(.*?)</lastmod>')
    exclude_extensions: Optional[List[str]] = Field(['.jpg', '.pdf', '.jpeg', '.png'])
    exclude_patterns: Optional[List[str]] = Field(['view'])
    include_patterns: Optional[List[str]] = Field(['/en/'])
    allowed_base_url: Optional[str] = Field('https://www.wiwi.hu-berlin.de')
    
    # Provide default values for database settings
    db_host: str = Field("localhost", env='DB_HOST')
    db_port: int = Field(5433, env='DB_PORT')
    db_name: str = Field("huber_db", env='DB_NAME')
    db_username: str = Field("huber", env='DB_USERNAME')
    db_password: str = Field("ruhe2805", env='DB_PASSWORD')

    class Config:
        env_file = str(env_file_path)
        env_file_encoding = 'utf-8'
        extra = 'ignore'
        case_sensitive = False  # Make environment variable names case-insensitive

# Print debug information
print(f"Looking for .env file at: {env_file_path}")
print(f"File exists: {env_file_path.exists()}")
if env_file_path.exists():
    with open(env_file_path, 'r') as f:
        print(f"File content preview: {f.read()[:100]}...")

settings = Settings()
print("Settings loaded successfully!")
print(f"DB Host: {settings.db_host}")
print(f"DB Port: {settings.db_port}")

