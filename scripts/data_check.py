# scripts/data_check.py

import json
import os
import pandas as pd
from sqlalchemy import create_engine, text
from hubert.config import settings

def check_for_updates():
    """
    Checks for new or updated content in `page_content` that needs processing.
    The check is based on missing records in the processing tables.
    """
    db_uri = settings.DATABASE_URL
    engine = create_engine(db_uri)

    with engine.connect() as connection:
        # First, let's check what tables exist in the database
        tables_query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'page_%'
            ORDER BY table_name;
        """)
        
        tables_result = pd.read_sql(tables_query, connection)
        print("Available page_* tables:")
        print(tables_result)
        
        # Check schema of page_keywords table
        schema_query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'page_keywords' 
            ORDER BY ordinal_position;
        """)

        schema_result = pd.read_sql(schema_query, connection)
        print("\nPage_keywords table schema:")
        print(schema_result)
        
        # Query for keyword processing - find page_content records that don't have corresponding page_keywords
        keyword_query = text("""
            SELECT pc.id
            FROM page_content pc
            WHERE pc.is_active = TRUE
            AND NOT EXISTS (
                SELECT 1 FROM page_keywords pk 
                WHERE pk.id = pc.id OR pk.url = pc.url
            );
        """)

        uids_for_keywords = pd.read_sql(keyword_query, connection)['id'].tolist()
        
        # Check for embedding tables and process them if they exist
        uids_for_embeddings = {}
        
        # Check if page_embeddings_alpha exists
        if 'page_embeddings_alpha' in tables_result['table_name'].values:
            embedding_query_alpha = text("""
                SELECT pc.id
                FROM page_content pc
                WHERE pc.is_active = TRUE
                AND NOT EXISTS (
                    SELECT 1 FROM page_embeddings_alpha pe 
                    WHERE pe.id = pc.id
                );
            """)
            uids_for_embeddings['page_embeddings_alpha'] = pd.read_sql(embedding_query_alpha, connection)['id'].tolist()
        
        # Check if page_embeddings_a exists
        if 'page_embeddings_a' in tables_result['table_name'].values:
            embedding_query_a = text("""
                SELECT pc.id
                FROM page_content pc
                WHERE pc.is_active = TRUE
                AND NOT EXISTS (
                    SELECT 1 FROM page_embeddings_a pe 
                    WHERE pe.id = pc.id
                );
            """)
            uids_for_embeddings['page_embeddings_a'] = pd.read_sql(embedding_query_a, connection)['id'].tolist()
        
        # Check if page_embeddings exists
        if 'page_embeddings' in tables_result['table_name'].values:
            embedding_query_generic = text("""
                SELECT pc.id
                FROM page_content pc
                WHERE pc.is_active = TRUE
                AND NOT EXISTS (
                    SELECT 1 FROM page_embeddings pe 
                    WHERE pe.id = pc.id
                );
            """)
            uids_for_embeddings['page_embeddings'] = pd.read_sql(embedding_query_generic, connection)['id'].tolist()

    # Consolidate all UIDs that need any form of processing
    all_uids_to_process = set(uids_for_keywords)
    for embedding_uids in uids_for_embeddings.values():
        all_uids_to_process.update(embedding_uids)
    
    processing_needed = bool(all_uids_to_process)

    results = {
        "processing_needed": processing_needed,
        "keyword_processing_uids": uids_for_keywords,
        "embedding_processing_uids": uids_for_embeddings,
        "total_uids_to_process": len(all_uids_to_process)
    }

    # Output results for GitHub Actions
    if os.getenv('GITHUB_ACTIONS') == 'true':
        # Ensure the output file exists and is writable
        output_file = os.getenv('GITHUB_OUTPUT')
        if output_file and os.access(os.path.dirname(output_file), os.W_OK):
             with open(output_file, 'a') as f:
                f.write(f"processing_needed={str(processing_needed).lower()}\n")
                f.write(f"keyword_uids={json.dumps(uids_for_keywords)}\n")
                f.write(f"embedding_uids={json.dumps(uids_for_embeddings)}\n")


    return results

if __name__ == "__main__":
    check_results = check_for_updates()
    print(json.dumps(check_results, indent=4))