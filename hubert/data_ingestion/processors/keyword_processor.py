import os
import sys
import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now that the path is set, we can use absolute imports
from hubert.data_ingestion.utils.db_utils import get_db_connection
from hubert.data_ingestion.utils.text_utils import remove_extra_spaces, lemmatize_text

def process_text_for_keywords(text: str) -> str:
    """
    Processes the given text by lemmatizing, lowercasing, and removing extra spaces.
    Handles None input gracefully.
    """
    if text is None:
        return ""
    # The spaCy model for lemmatization should be pre-loaded if possible,
    # but for this script, we assume lemmatize_text handles it.
    processed_text = lemmatize_text(text)
    processed_text = remove_extra_spaces(processed_text)
    return processed_text.lower()

def get_records_to_process(conn):
    """
    Fetches records from page_content that need keyword processing.
    This includes new pages or pages that have been updated more recently
    than their corresponding keyword record.
    """
    query = """
        SELECT pc.uid, pc.content
        FROM page_content pc
        LEFT JOIN page_keywords pk ON pc.uid = pk.uid
        WHERE pk.uid IS NULL OR pc.last_updated > pk.last_scraped;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        records = cur.fetchall()
    print(f"Found {len(records)} records to process for keywords.")
    return records

def bulk_upsert_keywords(conn, data_to_upsert: list):
    """
    Performs a bulk UPSERT operation to the page_keywords table.

    Args:
        conn: The database connection object.
        data_to_upsert: A list of tuples, where each tuple contains
                        (uid, processed_text, raw_content, last_scraped_timestamp).
    """
    if not data_to_upsert:
        print("No new data to upsert.")
        return

    # Note: The to_tsvector function is called within the query itself
    upsert_query = """
        INSERT INTO page_keywords (uid, content, tsvector, last_scraped)
        VALUES %s
        ON CONFLICT (uid) DO UPDATE SET
            content = EXCLUDED.content,
            tsvector = EXCLUDED.tsvector,
            last_scraped = EXCLUDED.last_scraped;
    """
    with conn.cursor() as cur:
        try:
            # psycopg2.extras.execute_values is highly efficient for bulk operations
            psycopg2.extras.execute_values(
                cur,
                upsert_query,
                # The template ensures the tsvector function is applied correctly
                # to the second value in each tuple.
                [(item[0], item[1], psycopg2.extras.Json(None), item[2]) for item in data_to_upsert],
                template='(%s, %s, to_tsvector(\'simple\', %s), %s)',
                page_size=500  # Adjust page_size based on record size and memory
            )
            conn.commit()
            print(f"Successfully bulk upserted {len(data_to_upsert)} records into page_keywords.")
        except Exception as e:
            print(f"An error occurred during bulk upsert: {e}")
            conn.rollback()


if __name__ == "__main__":
    print("Starting keyword processing job.")
    
    # It's good practice to wrap the main logic in a try/except block
    try:
        # Establish a single database connection for the entire job
        with get_db_connection() as conn:
            
            # 1. Fetch all records that need processing
            records = get_records_to_process(conn)

            if not records:
                print("No records require keyword processing. Exiting.")
            else:
                # 2. Process the records in memory
                processed_data = []
                now_timestamp = datetime.now()
                for uid, raw_content in records:
                    # Skip processing if content is empty
                    if not raw_content:
                        continue
                    
                    processed_text = process_text_for_keywords(raw_content)
                    processed_data.append((uid, processed_text, now_timestamp))

                # 3. Perform a single bulk write operation to the database
                bulk_upsert_keywords(conn, processed_data)

        print("Keyword processing job finished successfully.")

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred in the keyword processing job: {e}")
        # Exit with a non-zero status code to indicate failure,
        # which can be picked up by orchestration tools like GitHub Actions.
        sys.exit(1)
