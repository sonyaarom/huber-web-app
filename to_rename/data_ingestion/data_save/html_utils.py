import logging
from datetime import datetime
from typing import Optional, Dict

import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, MetaData, select, insert, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def remove_extra_spaces(text):
    return ' '.join(text.split())


def get_html_content(url: str, timeout: int = 10) -> Optional[str]:
    """
    Retrieve HTML content from the given URL with improved error handling.
    
    Args:
        url (str): The URL to fetch
        timeout (int, optional): Request timeout in seconds. Defaults to 10.
    
    Returns:
        Optional[str]: HTML content if successful, None otherwise
    """
    try:
        # Add headers to mimic a browser request
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
    if not html_content:
        return {"title": "Title not found", "content": "Main content not found"}
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    title_candidates = [
        soup.find('h2', class_='documentFirstHeading'),
        soup.find('h1'),
        soup.find('title'),
        soup.find('h2')
    ]
    title_tag = next((tag for tag in title_candidates if tag), None)
    title_text = title_tag.get_text(strip=True) if title_tag else "Title not found"
    if title_tag:
        title_text = remove_extra_spaces(title_text)
    main_content = soup.find('main') or soup.find('article') or soup.body
    if main_content:
        for tag in main_content.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        
        # Ensure line breaks are preserved
        for br in main_content.find_all("br"):
            br.replace_with("\n")

        content_text = ' '.join(main_content.stripped_strings)
        content_text = remove_extra_spaces(content_text)
    else:
        content_text = "Main content not found"
    
    return {"title": title_text, "content": f"{title_text} {content_text}"}


def upsert_page_content(conn, page_content, record_id, url, html, extracted, now):
    """
    Perform an upsert operation for page content using PostgreSQL-specific insert.
    
    Args:
        conn: SQLAlchemy connection
        page_content: SQLAlchemy table object
        record_id: Unique identifier for the record
        url: URL of the page
        html: HTML content
        extracted: Extracted information dictionary
        now: Current timestamp
    """
    insert_stmt = pg_insert(page_content).values(
        id=record_id,
        url=url,
        html_content=html,
        extracted_title=extracted['title'],
        extracted_content=extracted['content'],
        fetched_at=now
    )
    
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=['id'],
        set_={
            'url': insert_stmt.excluded.url,
            'html_content': insert_stmt.excluded.html_content,
            'extracted_title': insert_stmt.excluded.extracted_title,
            'extracted_content': insert_stmt.excluded.extracted_content,
            'fetched_at': insert_stmt.excluded.fetched_at
        }
    )
    
    conn.execute(upsert_stmt)

def main():
    """
    Main function to scrape and store web page content.
    Includes improved error handling and logging.
    """
    try:
        # Construct database URL more securely
        db_url = (
            f"postgresql://{settings.db_username}:{settings.db_password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        )
        engine = create_engine(db_url)
        metadata = MetaData()
        metadata.reflect(bind=engine)

        # Validate required tables exist
        required_tables = ['page_raw', 'page_content']
        for table_name in required_tables:
            if table_name not in metadata.tables:
                logger.error(f"{table_name} table not found! Ensure it's defined in Terraform.")
                return

        page_raw = metadata.tables['page_raw']
        page_content = metadata.tables['page_content']

        # Fetch records with more robust error handling
        with engine.connect() as conn:
            try:
                stmt = select(page_raw)
                result = conn.execute(stmt)
                records = result.fetchall()
            except SQLAlchemyError as e:
                logger.error(f"Database query error: {e}")
                return

        logger.info(f"Fetched {len(records)} rows from page_raw.")

        # Process each record
        for record in records:
            # More flexible record access
            record_id = record['id'] if isinstance(record, dict) else record[0]
            url = record['url'] if isinstance(record, dict) else record[1]
            
            logger.info(f"Processing URL: {url}")
            
            try:
                html = get_html_content(url)
                if html is None:
                    logger.warning(f"Skipping URL due to fetch failure: {url}")
                    continue
                
                extracted = extract_info(html)
                
                now = datetime.utcnow()
                
                # Transactional insert with conflict handling
                with engine.begin() as conn:
                    upsert_page_content(conn, page_content, record_id, url, html, extracted, now)
                
                logger.info(f"Inserted/updated content for URL: {url}")
            
            except Exception as e:
                logger.error(f"Error processing record {record_id} with URL {url}: {e}", exc_info=True)
                # Continue with next record to prevent total script failure

    except Exception as e:
        logger.critical(f"Unhandled error in main function: {e}", exc_info=True)

if __name__ == "__main__":
    main()