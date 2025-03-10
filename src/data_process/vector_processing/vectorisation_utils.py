import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now you can import using the 'src' prefix
from ..config import settings
import psycopg2
import pandas as pd
import json
import torch
from tqdm import tqdm
from logging import getLogger
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModel
from src.data_process.vector_processing.splitting_utils import recursive_chunk_text



# Logger setup
logger = getLogger(__name__)

# Load the Linq-Embed-Mistral Model
MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# PostgreSQL Connection Function
def get_pg_connection():
    return psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_username,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port
    )

# Fetch Data from page_content Table
def fetch_page_content():
    query = "SELECT id, url, extracted_content FROM page_content WHERE extracted_content IS NOT NULL LIMIT 5;"
    
    with get_pg_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

# Generate Embeddings
def generate_embeddings(texts: list) -> list:
    """
    Generate embeddings using Linq-Embed-Mistral model.
    """
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()  # Mean pooling
        embeddings.append(embedding)
    
    return embeddings

# # Store Embeddings into PostgreSQL
# def store_embeddings(df: pd.DataFrame):
#     engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    
#     with engine.connect() as conn:
#         df.to_sql("page_embeddings", conn, if_exists="replace", index=False)

# Process Data and Generate Embeddings
def process_and_store_embeddings(chunk_size: int = 512, chunk_overlap: int = 50):
    df = fetch_page_content()
    all_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Pages"):
        text = row['extracted_content']
        url = row['url']
        page_id = row['id']

        # Chunk text
        chunks = recursive_chunk_text(text, chunk_size, chunk_overlap)

        # Generate embeddings
        embeddings = generate_embeddings(chunks)

        # Store results
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
    print(embeddings_df)
    # Store in PostgreSQL
    # store_embeddings(embeddings_df)

    logger.info("Embeddings stored successfully in page_embeddings table.")

# Run the function
process_and_store_embeddings(chunk_size=512, chunk_overlap=50)
