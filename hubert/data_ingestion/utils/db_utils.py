import psycopg2
import os
from psycopg2.extras import RealDictCursor
import pandas as pd

# PostgreSQL Connection Function
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using environment variables."""
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USERNAME"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT", 5432)  # Default port for PostgreSQL
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