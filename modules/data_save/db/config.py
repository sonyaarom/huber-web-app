from pydantic import  Field
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    url: Optional[str] = Field('https://www.wiwi.hu-berlin.de/sitemap.xml.gz')
    pattern: Optional[str] = Field(r'<url>\s*<loc>(.*?)</loc>\s*<lastmod>(.*?)</lastmod>')
    exclude_extensions: Optional[List[str]] = Field(['.jpg', '.pdf', '.jpeg', '.png'])
    exclude_patterns: Optional[List[str]] = Field(['view'])
    include_patterns: Optional[List[str]] = Field(['/en/'])
    allowed_base_url: Optional[str] = Field('https://www.wiwi.hu-berlin.de')
    db_host: Optional[str] 
    db_port: Optional[int] 
    db_name: Optional[str]
    db_username: Optional[str] 
    db_password: Optional[str] 

    class Config:
        env_file = '/Users/s.konchakova/data-engineering-huber/.venv'
        extra = 'ignore'
        
settings = Settings()

