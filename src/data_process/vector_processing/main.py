import psycopg2
import pandas as pd
import json
from tqdm import tqdm
from logging import getLogger
from langchain_openai.embeddings import OpenAIEmbeddings
from sqlalchemy import create_engine
from src.data_process.config import settings
from src.data_process.vector_processing.splitting_utils import recursive_chunk_text, character_chunk_text, semantic_chunk_text

# PostgreSQL Connection Details
DB_NAME = settings.DB_NAME
DB_USER = settings.DB_USER
DB_PASSWORD = settings.DB_PASSWORD
DB_HOST = settings.DB_HOST
DB_PORT = settings.DB_PORT

# Logger setup
logger = getLogger(__name__)

# Initialize Embedding Model
embedding_model = OpenAIEmbeddings()

# PostgreSQL Connection Function
def get_pg_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

# Fetch Data from page_content Table
def fetch_page_content():
    query = "SELECT id, url, text FROM page_content WHERE text IS NOT NULL;"
    
    with get_pg_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

# Store Embeddings into PostgreSQL
def store_embeddings(df: pd.DataFrame):
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    with engine.connect() as conn:
        df.to_sql("page_embeddings", conn, if_exists="replace", index=False)

# Process Data and Generate Embeddings
def process_and_store_embeddings(chunking_method: str, chunk_size: int = 512, chunk_overlap: int = 50):
    df = fetch_page_content()
    
    all_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Pages"):
        text = row['text']
        url = row['url']
        page_id = row['id']

        # Apply chosen chunking method
        if chunking_method == "recursive":
            chunks = recursive_chunk_text(text, chunk_size, chunk_overlap)
        elif chunking_method == "character":
            chunks = character_chunk_text(text, chunk_size, chunk_overlap)
        elif chunking_method == "semantic":
            chunks = semantic_chunk_text(text)
        else:
            raise ValueError("Invalid chunking method. Choose from 'recursive', 'character', or 'semantic'.")

        # Embed each chunk
        embeddings = embedding_model.embed_documents(chunks)

        # Prepare data for storage
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            all_data.append({
                "id": page_id,  # Keep original page_id
                "split_id": i + 1,  # Ordered chunk number
                "url": url,
                "chunk_text": chunk,
                "embedding": json.dumps(embedding)  # Store embedding as JSON
            })

    # Convert to DataFrame
    embeddings_df = pd.DataFrame(all_data)

    # Store in PostgreSQL
    store_embeddings(embeddings_df)

    logger.info("Embeddings stored successfully in page_embeddings table.")

# Run the function with recursive chunking
process_and_store_embeddings(chunking_method="recursive", chunk_size=512, chunk_overlap=50)
