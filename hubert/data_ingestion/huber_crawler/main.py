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
from hubert.config import settings
from hubert.data_ingestion.huber_crawler.sitemap import process_sitemap
from hubert.db.postgres_storage import PostgresStorage

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

def process_page_raw_records(storage: PostgresStorage, records: Dict, metrics: CrawlerMetrics) -> List[str]:
    """
    Prepares data for bulk insertion and marks old records as inactive.
    
    Returns:
        List of UIDs that were deactivated
    """
    start_time = time.time()
    
    if not records:
        logger.warning("No records from sitemap to process!")
        return []
        
    current_uids = list(records.keys())
    metrics.total_urls_found = len(current_uids)

    # 1. Prepare the list of records for bulk upsert
    records_to_upsert = []
    now_timestamp = datetime.now()
    for uid, record_data in records.items():
        records_to_upsert.append({
            'uid': uid,
            'url': record_data['url'],
            'last_updated': record_data['last_updated'],
        })

    # 2. Perform the bulk upsert using the storage layer
    try:
        storage.upsert_raw_pages(records_to_upsert)
        # Note: We can't easily get the number of new vs. updated from the storage layer
        # without a more complex return value. This is a simplification.
        logger.info(f"Successfully bulk upserted {len(records_to_upsert)} records into page_raw.")
    except Exception as e:
        logger.error(f"Error during bulk upsert of raw pages: {e}")
        metrics.errors += len(records_to_upsert)
        raise

    # 3. Mark pages no longer in the sitemap as inactive and get the deactivated UIDs
    deactivated_uids = []
    try:
        deactivated_uids = storage.deactivate_old_urls(current_uids)
        metrics.removed_urls = len(deactivated_uids)
        if deactivated_uids:
            logger.info(f"Deactivated {len(deactivated_uids)} URLs no longer present in the sitemap.")
        else:
            logger.info("No URLs needed to be deactivated.")
    except Exception as e:
        logger.error(f"Error marking missing pages as inactive: {e}")
        metrics.errors += 1
        raise
    
    metrics.database_update_time = time.time() - start_time
    logger.info(f"Database update for page_raw completed in {metrics.database_update_time:.2f} sec")
    
    return deactivated_uids

def main():
    """
    Main function to process the sitemap and update the database.
    Now using the abstract storage layer.
    """
    # Initialize Sentry for this process
    from hubert.common.monitoring import init_sentry, capture_crawler_metrics
    import sentry_sdk
    
    init_sentry()
    
    with sentry_sdk.start_transaction(name="crawler_main", op="crawler"):
        metrics = CrawlerMetrics()
        logger.info("🚀 Starting sitemap processing...")
        
        sitemap_start_time = time.time()
        records = {}
        try:
            with sentry_sdk.start_span(op="sitemap", description="Process sitemap"):
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
            
            # Capture sitemap processing metrics
            sentry_sdk.set_measurement("sitemap_processing_time", metrics.sitemap_processing_time, "second")
            sentry_sdk.set_measurement("sitemap_records_found", len(records))
            
        except Exception as e:
            logger.error(f"Fatal error processing sitemap: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)
            metrics.errors += 1
            metrics.save_to_file()
            sys.exit(1)

        storage = PostgresStorage()
        try:
            with sentry_sdk.start_span(op="database", description="Process page records"):
                deactivated_uids = process_page_raw_records(storage, records, metrics)
            
            # Immediately purge the deactivated records from all related tables
            if deactivated_uids:
                logger.info(f"Starting immediate garbage collection for {len(deactivated_uids)} deactivated UIDs...")
                purge_start_time = time.time()
                try:
                    with sentry_sdk.start_span(op="database", description="Garbage collection"):
                        total_deleted = storage.purge_specific_inactive_records(deactivated_uids)
                    purge_time = time.time() - purge_start_time
                    logger.info(f"Garbage collection completed in {purge_time:.2f} sec. Total records deleted: {total_deleted}")
                    sentry_sdk.set_measurement("purge_time", purge_time, "second")
                    sentry_sdk.set_measurement("records_purged", total_deleted)
                except Exception as e:
                    logger.error(f"Error during immediate garbage collection: {e}", exc_info=True)
                    sentry_sdk.capture_exception(e)
                    metrics.errors += 1
            else:
                logger.info("No records to purge - skipping garbage collection.")
                
        except Exception as e:
            logger.error(f"A database error occurred during the main process: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)
            metrics.errors += 1
        finally:
            storage.close()
        
        # Capture final crawler metrics
        total_runtime = time.time() - metrics.start_time
        capture_crawler_metrics(
            pages_processed=metrics.total_urls_found, 
            errors=metrics.errors, 
            duration=total_runtime
        )
        
        metrics.save_to_file()
        print("\n" + "="*50)
        logger.info(f"Crawler run finished in {total_runtime:.2f} seconds.")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
