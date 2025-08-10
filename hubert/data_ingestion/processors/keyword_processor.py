import hashlib
import logging
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from hubert.common.utils.text_utils import process_text_for_keyword_search
from hubert.config import settings
from hubert.db.postgres_storage import PostgresStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_and_store_keywords(db_uri, uids):
    """
    Processes text content to extract keywords and stores them in the database.
    Now includes a content_hash for data lineage.
    """
    if not uids:
        logger.info("No new UIDs to process for keywords.")
        return

    engine = create_engine(db_uri)
    
    # Fetch content and compute hash
    query = text("SELECT uid, extracted_content FROM page_content WHERE uid = ANY(:uids)")
    df = pd.read_sql(query, engine, params={'uids': uids})

    if df.empty:
        logger.info("No content found for the given UIDs.")
        return

    # --- Start of new implementation ---
    # Compute content hash
    df['content_hash'] = df['extracted_content'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest() if x else None)
    # --- End of new implementation ---
    
    # Process text for keywords
    df['processed_text'] = df['extracted_content'].apply(process_text_for_keyword_search)

    # Prepare data for insertion
    df_to_insert = df[['uid', 'processed_text', 'content_hash']].copy()
    df_to_insert.rename(columns={'processed_text': 'keywords_tsvector'}, inplace=True)

    with engine.connect() as connection:
        # Use INSERT ... ON CONFLICT to update existing records
        insert_query = """
        INSERT INTO page_keywords (uid, keywords_tsvector, content_hash)
        VALUES (:uid, to_tsvector('german', :keywords_tsvector), :content_hash)
        ON CONFLICT (uid) DO UPDATE
        SET keywords_tsvector = EXCLUDED.keywords_tsvector,
            content_hash = EXCLUDED.content_hash,
            last_updated = NOW();
        """
        connection.execute(text(insert_query), df_to_insert.to_dict(orient='records'))
        logger.info(f"Successfully processed and stored keywords for {len(df_to_insert)} pages.")

if __name__ == "__main__":
    print("Starting keyword processing job.")
    storage = PostgresStorage()
    try:
        # 1. Fetch all records that need processing
        records = storage.get_content_to_process_for_keywords()

        if not records:
            print("No records require keyword processing. Exiting.")
        else:
            print(f"Found {len(records)} records to process for keywords.")
            # 2. Process the records in memory
            processed_data = []
            for page_content_id, raw_content in records:
                # Skip processing if content is empty
                if not raw_content:
                    continue
                
                processed_text = process_text_for_keyword_search(raw_content)
                processed_data.append({
                    "page_content_id": page_content_id,
                    "content": processed_text
                })

            # 3. Perform a single bulk write operation to the database
            if processed_data:
                storage.upsert_keywords(processed_data)
                print(f"Successfully bulk upserted keywords for {len(processed_data)} records.")

        print("Keyword processing job finished successfully.")

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred in the keyword processing job: {e}")
        # Exit with a non-zero status code to indicate failure,
        # which can be picked up by orchestration tools like GitHub Actions.
        sys.exit(1)
    finally:
        storage.close()
