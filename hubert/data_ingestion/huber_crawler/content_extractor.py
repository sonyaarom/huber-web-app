import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import requests
import trafilatura

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now that the path is set, we can use absolute imports
from hubert.db.postgres_storage import PostgresStorage
from hubert.db.base_storage import BaseStorage

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

if __name__ == "__main__":
    logger.info("Starting content extraction job.")
    storage = PostgresStorage()
    try:
        records_to_process = storage.get_urls_to_process()

        if not records_to_process:
            logger.info("No records require content extraction. Exiting.")
        else:
            processed_data = []
            now_timestamp = datetime.now()
            
            for id, uid, url, last_updated in records_to_process:
                logger.info(f"Processing URL: {url}")
                try:
                    html_content = get_html_content(url)
                    if not html_content:
                        logger.warning(f"Skipping URL due to fetch failure: {url}")
                        storage.log_failed_job(uid, 'content_extraction', "HTML content fetch failed")
                        continue
                    
                    extracted = extract_info(html_content)

                    clean_html = html_content.replace('\x00', '')
                    clean_title = extracted['title'].replace('\x00', '')
                    clean_content = extracted['content'].replace('\x00', '') if extracted['content'] else ''
                    
                    processed_data.append({
                        "id": id,
                        "url": url,
                        "html_content": clean_html,
                        "title": clean_title,
                        "content": clean_content,
                        "last_updated": last_updated or datetime.now(),
                        "is_active": True,
                        "last_scraped": now_timestamp
                    })
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    storage.log_failed_job(uid, 'content_extraction', str(e))
            
            if processed_data:
                storage.upsert_page_content(processed_data)
                logger.info(f"Successfully processed and upserted content for {len(processed_data)} pages.")

    except Exception as e:
        logger.critical(f"An unhandled error occurred in the content extraction job: {e}", exc_info=True)
        sys.exit(1)
    finally:
        storage.close()
        logger.info("Content extraction job finished.")
