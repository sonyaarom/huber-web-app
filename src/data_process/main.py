import spacy
from config import settings
import pandas as pd
from sqlalchemy import create_engine, text
import string
from datetime import datetime
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Installing en_core_web_sm...")
    # Optionally, add installation logic here if needed

def remove_extra_spaces(text):
    return ' '.join(text.split())

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_no_stops = [token.lemma_ for token in doc if not token.is_stop]
    # Remove punctuation tokens
    lemmatized_no_stops = [token for token in lemmatized_no_stops if token not in string.punctuation]
    return ' '.join(lemmatized_no_stops)

def process_text(text):
    text = remove_extra_spaces(text)
    text = lemmatize_text(text)
    text = text.lower()
    return text

if __name__ == "__main__":
    # Connect to the PostgreSQL database
    db_url = (
        f"postgresql://{settings.db_username}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )
    engine = create_engine(db_url)

    # Create the page_keywords table with tokenized_text as tsvector if it does not exist
    create_table_query = text("""
        CREATE TABLE IF NOT EXISTS page_keywords (
            id CHAR(32) PRIMARY KEY,
            url TEXT,
            last_modified TIMESTAMP,
            tokenized_text tsvector,
            raw_text TEXT
        )
    """)
    
    with engine.begin() as connection:
        connection.execute(create_table_query)
    
    # Read the page_content table and limit to the first 5 rows
    df = pd.read_sql_table('page_content', engine)
    # Define the upsert SQL query for page_keywords table.
    # Note: The tokenized_text is converted to a tsvector using to_tsvector in the VALUES clause.


    upsert_query = text("""
        INSERT INTO page_keywords (id, url, last_modified, tokenized_text, raw_text)
        VALUES (:id, :url, :last_modified, to_tsvector('simple', :tokenized_text), :raw_text)
        ON CONFLICT (id) DO UPDATE
        SET url = EXCLUDED.url,
            last_modified = EXCLUDED.last_modified,
            tokenized_text = EXCLUDED.tokenized_text,
            raw_text = EXCLUDED.raw_text
    """)
    
    # Upsert the first 5 rows into page_keywords
    with engine.begin() as connection:
        for idx, row in df.iterrows():
            tokenized_text = process_text(row['extracted_content'])
            params = {
                'id': row['id'],
                'url': row['url'],
                'last_modified': datetime.now(),
                'tokenized_text': tokenized_text,
                'raw_text': row['extracted_content']
            }
            connection.execute(upsert_query, params)
    
    print("Upsert completed for the first 5 rows into page_keywords.")
