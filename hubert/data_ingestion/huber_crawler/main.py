#!/usr/bin/env python3
import os
import sys
import logging
import time
import json
from datetime import datetime
from typing import Dict, List

import psycopg2
import psycopg2.extras

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now that the path is set, we can use absolute imports
from hubert.data_ingestion.config import settings
from hubert.data_ingestion.huber_crawler.sitemap import process_sitemap
from hubert.data_ingestion.huber_crawler.content_extractor import content_extractor
from hubert.data_ingestion.utils.db_utils import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrawlerMetrics:
    """Class to track and report crawler metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.sitemap_processing_time = 0
        self.database_update_time = 0
        self.total_urls_found = 0
        self.new_urls = 0
        self.updated_urls = 0
        self.removed_urls = 0
        self.unchanged_urls = 0
        self.errors = 0
        
    def to_dict(self):
        """Convert metrics to dictionary for JSON serialization"""
        elapsed = time.time() - self.start_time
        return {
            "timestamp": datetime.now().isoformat(),
            "total_runtime_seconds": elapsed,
            "sitemap_processing_time_seconds": self.sitemap_processing_time,
            "database_update_time_seconds": self.database_update_time,
            "total_urls_found": self.total_urls_found,
            "new_urls": self.new_urls,
            "updated_urls": self.updated_urls,
            "removed_urls": self.removed_urls,
            "unchanged_urls": self.unchanged_urls,
            "errors": self.errors
        }
        
    def save_to_file(self, filename="crawler_metrics.json"):
        """Save metrics to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {filename}")

def bulk_upsert_raw_pages(conn, records_to_upsert: List, metrics: CrawlerMetrics):
    """
    Performs a bulk UPSERT to the page_raw table using psycopg2.extras.execute_values.
    """
    if not records_to_upsert:
        logger.info("No records to upsert.")
        return

    upsert_query = """
        INSERT INTO page_raw (uid, url, last_updated, last_scraped, is_active)
        VALUES %s
        ON CONFLICT (uid) DO UPDATE SET
            url = EXCLUDED.url,
            last_updated = EXCLUDED.last_updated,
            last_scraped = EXCLUDED.last_scraped,
            is_active = EXCLUDED.is_active;
    """
    
    with conn.cursor() as cur:
        try:
            psycopg2.extras.execute_values(
                cur,
                upsert_query,
                records_to_upsert,
                page_size=500
            )
            # Track metrics based on the operation
            # Note: A more complex approach would be needed to get exact new/updated counts
            # from a bulk operation. This is a simplified metric tracking.
            metrics.new_urls = cur.rowcount # This is an approximation
            logger.info(f"Successfully bulk upserted {cur.rowcount} records into page_raw.")

        except Exception as e:
            logger.error(f"Error during bulk upsert of raw pages: {e}")
            metrics.errors += len(records_to_upsert)
            conn.rollback()
            raise

def process_page_raw_records(conn, records: Dict, metrics: CrawlerMetrics):
    """
    Prepares data for bulk insertion and marks old records as inactive.
    """
    start_time = time.time()
    
    if not records:
        logger.warning("No records from sitemap to process!")
        return
        
    new_ids = set(records.keys())
    metrics.total_urls_found = len(new_ids)

    # 1. Prepare the list of records for bulk upsert
    records_to_upsert = []
    now_timestamp = datetime.now()
    for uid, record_data in records.items():
        records_to_upsert.append((
            uid,
            record_data['url'],
            record_data['last_updated'],
            now_timestamp,
            True # is_active
        ))

    # 2. Perform the bulk upsert
    bulk_upsert_raw_pages(conn, records_to_upsert, metrics)

    # 3. Mark pages no longer in the sitemap as inactive
    with conn.cursor() as cur:
        try:
            # Create a temporary table to hold the new IDs
            cur.execute("CREATE TEMPORARY TABLE temp_new_ids (id CHAR(32) PRIMARY KEY);")
            psycopg2.extras.execute_values(cur, "INSERT INTO temp_new_ids (id) VALUES %s", [(uid,) for uid in new_ids])

            # Update page_raw, setting is_active to FALSE for any IDs not in our temp table
            update_inactive_query = """
                UPDATE page_raw
                SET is_active = FALSE
                WHERE uid NOT IN (SELECT id FROM temp_new_ids);
            """
            cur.execute(update_inactive_query)
            metrics.removed_urls = cur.rowcount
            logger.info(f"Marked {cur.rowcount} old records as inactive.")
            
            # The temporary table is automatically dropped at the end of the session
        except Exception as e:
            logger.error(f"Error marking missing pages as inactive: {e}")
            metrics.errors += 1
            conn.rollback()
            raise
    
    metrics.database_update_time = time.time() - start_time
    logger.info(f"Database update for page_raw completed in {metrics.database_update_time:.2f} sec")


def main():
    """
    Main function to process the sitemap and update the database.
    Now with efficient bulk operations.
    """
    metrics = CrawlerMetrics()
    logger.info("🚀 Starting sitemap processing...")
    
    sitemap_start_time = time.time()
    records = {}
    try:
        # Assuming settings are configured correctly
        records = process_sitemap(
            settings.url,
            settings.pattern,
            settings.exclude_extensions,
            settings.exclude_patterns,
            settings.include_patterns,
            settings.allowed_base_url
        )
        metrics.sitemap_processing_time = time.time() - sitemap_start_time
        logger.info(f"Sitemap processed in {metrics.sitemap_processing_time:.2f} sec, {len(records)} records found.")
    except Exception as e:
        logger.error(f"Fatal error processing sitemap: {e}", exc_info=True)
        metrics.errors += 1
        metrics.save_to_file()
        sys.exit(1)

    # Use a single connection for all database operations in this run
    try:
        with get_db_connection() as conn:
            # Process page_raw records
            process_page_raw_records(conn, records, metrics)
            conn.commit()

            # Now, run the content extractor which will use the same connection pattern
            logger.info("Starting content extraction...")
            content_extractor()
            logger.info("Content extraction completed successfully.")

    except Exception as e:
        logger.error(f"A database error occurred during the main process: {e}", exc_info=True)
        metrics.errors += 1
    
    # Save the final metrics and print summary
    metrics.save_to_file()
    print("\n" + "="*50)
    print("CRAWLER METRICS SUMMARY")
    print("="*50)
    print(f"Total runtime: {metrics.to_dict()['total_runtime_seconds']:.2f} seconds")
    print(f"Total URLs found: {metrics.total_urls_found}")
    print(f"New URLs (approximated): {metrics.new_urls}")
    print(f"Updated URLs: Not tracked in bulk mode")
    print(f"Removed URLs: {metrics.removed_urls}")
    print(f"Errors: {metrics.errors}")
    print("="*50)

if __name__ == "__main__":
    main()
