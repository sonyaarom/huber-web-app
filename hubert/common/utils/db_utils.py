import psycopg2
import os
from psycopg2.extras import RealDictCursor
import pandas as pd
from hubert.config import settings

# PostgreSQL Connection Function
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using environment variables."""
    return psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_username,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port
    )

# Fetch Data from page_content Table
def fetch_page_content(limit=None):
    if limit is not None:
        query = f"SELECT id, url, extracted_content, last_scraped FROM page_content WHERE extracted_content IS NOT NULL LIMIT {limit};"
    else:
        query = f"SELECT id, url, extracted_content, last_scraped FROM page_content WHERE extracted_content IS NOT NULL ;"
    
    with get_db_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df