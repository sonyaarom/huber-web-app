import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import requests
import psycopg2
import psycopg2.extras
import trafilatura

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now that the path is set, we can use absolute imports
from hubert.common.utils.db_utils import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_html_content(url: str, timeout: int = 10) -> Optional[str]:
    """
    Retrieve HTML content from the given URL.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None

def extract_info(html_content: str) -> Dict[str, str]:
    """
    Extracts the title and main content from the given HTML using trafilatura.
    """
    if not html_content:
        return {"title": "Title not found", "content": "Main content not found"}
    
    # trafilatura's extract function returns the main text content.
    # It can also extract metadata like title, but we'll get it from the main extract call for simplicity.
    content_text = trafilatura.extract(html_content, include_comments=False, include_tables=False)
    
    # For the title, we can attempt to extract it separately if needed,
    # or rely on a simpler method as trafilatura focuses on main content.
    # A simple approach is to extract the <title> tag text.
    # Note: This part is a simplification. For robust title extraction, 
    # you might need a different approach or to parse the HTML again.
    # For now, we will use a basic title extraction.
    
    # A more direct way to get title with trafilatura is not straightforward from the main `extract` function.
    # We will parse it manually or leave it as "Title not found" if content is extracted.
    
    title_text = "Title not found" # Placeholder
    if content_text:
        # Try to find title in the original HTML as a fallback
        start_title = html_content.find('<title>')
        end_title = html_content.find('</title>')
        if start_title != -1 and end_title != -1:
            title_text = html_content[start_title + len('<title>'):end_title].strip()

    if not content_text:
        content_text = "Main content not found"

    return {"title": title_text, "content": content_text}

def get_records_to_process(conn) -> List[Tuple]:
    """
    Fetches records from page_raw that need content extraction.
    This includes new pages or pages that have been updated more recently
    than their corresponding content record.
    """
    query = """
        SELECT pr.id, pr.url, pr.last_updated
        FROM page_raw pr
        LEFT JOIN page_content pc ON pr.id = pc.id
        WHERE pr.is_active = TRUE AND (pc.id IS NULL OR pr.last_updated > pc.last_scraped);
    """
    with conn.cursor() as cur:
        cur.execute(query)
        records = cur.fetchall()
    logger.info(f"Found {len(records)} active records to process for content extraction.")
    return records

def bulk_upsert_content(conn, data_to_upsert: list):
    """
    Performs a bulk UPSERT operation to the page_content table using psycopg2.
    """
    if not data_to_upsert:
        logger.info("No new content to upsert.")
        return

    upsert_query = """
        INSERT INTO page_content (id, url, html_content, title, content, last_updated, is_active, last_scraped)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            url = EXCLUDED.url,
            html_content = EXCLUDED.html_content,
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            last_updated = EXCLUDED.last_updated,
            is_active = EXCLUDED.is_active,
            last_scraped = EXCLUDED.last_scraped;
    """
    with conn.cursor() as cur:
        try:
            psycopg2.extras.execute_values(
                cur,
                upsert_query,
                data_to_upsert,
                page_size=100  # Batch in groups of 100 to manage memory
            )
            conn.commit()
            logger.info(f"Successfully bulk upserted {len(data_to_upsert)} records into page_content.")
        except Exception as e:
            logger.error(f"An error occurred during bulk content upsert: {e}")
            conn.rollback()

if __name__ == "__main__":
    logger.info("Starting content extraction job.")
    try:
        with get_db_connection() as conn:
            # 1. Fetch all records that need processing in one go
            records = get_records_to_process(conn)
            
            if not records:
                logger.info("No records require content extraction. Exiting.")
            else:
                # 2. Process records in memory: Fetch HTML and extract content
                processed_data = []
                now_timestamp = datetime.now()
                for id, url, last_updated in records:
                    logger.info(f"Processing URL: {url}")
                    html_content = get_html_content(url)
                    if not html_content:
                        logger.warning(f"Skipping URL due to fetch failure: {url}")
                        continue
                    
                    extracted = extract_info(html_content)

                    # Clean NUL characters which cause database errors
                    clean_html = html_content.replace('\\x00', '')
                    clean_title = extracted['title'].replace('\\x00', '')
                    clean_content = extracted['content'].replace('\\x00', '') if extracted['content'] else ''
                    
                    processed_data.append((
                        id,
                        url,
                        clean_html,
                        clean_title,
                        clean_content,
                        last_updated,
                        True, # is_active
                        now_timestamp
                    ))
                
                # 3. Perform a single bulk write operation to the database
                bulk_upsert_content(conn, processed_data)

        logger.info("Content extraction job finished successfully.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in the content extraction job: {e}", exc_info=True)
        sys.exit(1)
