# scripts/data_check.py

import json
import os
import pandas as pd
from sqlalchemy import create_engine, text
from hubert.config import settings

def check_for_updates():
    """
    Checks for new or updated content in `page_content` that needs processing.
    The check is based on missing records or mismatched content hashes.
    """
    db_uri = settings.DATABASE_URL
    engine = create_engine(db_uri)

    # --- Corrected Query Logic ---

    # Query for keyword processing.
    # Finds UIDs that are not in page_keywords OR where the content hash has changed.
    keyword_query = text("""
        SELECT pc.uid
        FROM page_content pc
        LEFT JOIN page_keywords pk ON pc.uid = pk.uid
        WHERE pc.is_active = TRUE
        AND (pk.uid IS NULL OR md5(pc.extracted_content) != pk.content_hash);
    """)

    # Query for embedding processing (example for 'page_embeddings_alpha').
    # Finds UIDs that are not in the table OR where the content hash has changed.
    embedding_query_alpha = text("""
        SELECT pc.uid
        FROM page_content pc
        LEFT JOIN page_embeddings_alpha pe ON pc.uid = pe.uid
        WHERE pc.is_active = TRUE
        AND (pe.uid IS NULL OR md5(pc.extracted_content) != pe.content_hash)
        GROUP BY pc.uid;
    """)

    # --- End of Corrected Query Logic ---

    with engine.connect() as connection:
        uids_for_keywords = pd.read_sql(keyword_query, connection)['uid'].tolist()
        uids_for_embeddings_alpha = pd.read_sql(embedding_query_alpha, connection)['uid'].tolist()

    # Consolidate all UIDs that need any form of processing
    all_uids_to_process = set(uids_for_keywords) | set(uids_for_embeddings_alpha)
    
    processing_needed = bool(all_uids_to_process)

    results = {
        "processing_needed": processing_needed,
        "keyword_processing_uids": uids_for_keywords,
        "embedding_processing_uids": {
            "page_embeddings_alpha": uids_for_embeddings_alpha
        },
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
                f.write(f"embedding_uids_alpha={json.dumps(uids_for_embeddings_alpha)}\n")


    return results

if __name__ == "__main__":
    check_results = check_for_updates()
    print(json.dumps(check_results, indent=4))