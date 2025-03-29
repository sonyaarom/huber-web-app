#!/usr/bin/env python3
import logging
import time
import json
from datetime import datetime
from sqlalchemy import create_engine, MetaData, func
from sqlalchemy.dialects.postgresql import insert
from ..config import settings
from .sitemap import process_sitemap

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

def insert_page_raw_records(records, metrics):
    """
    Inserts or updates records in the existing page_raw table.
    Pages missing from the new sitemap are marked as inactive (is_active = FALSE).
    Now also tracks metrics.
    """
    start_time = time.time()

    db_url = (
        f"postgresql://{settings.db_username}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )

    # Add timeout to prevent long-running connections
    engine = create_engine(db_url, connect_args={"connect_timeout": 10})

    metadata = MetaData()
    metadata.reflect(bind=engine)
    page_raw = metadata.tables['page_raw']

    new_ids = set(records.keys())
    metrics.total_urls_found = len(new_ids)

    with engine.begin() as conn:
        # First, get existing records to track what's changed
        existing_query = f"SELECT id, url, last_updated, is_active FROM page_raw;"
        existing_records = {}
        for row in conn.execute(existing_query):
            existing_records[row[0]] = {
                "url": row[1],
                "last_updated": row[2],
                "is_active": row[3]
            }
        
        existing_ids = set(existing_records.keys())
        metrics.removed_urls = len(existing_ids - new_ids)
        
        logger.info(f"Starting database updates: {len(new_ids)} records to process.")

        for record_id, record in records.items():
            try:
                last_updated_dt = datetime.strptime(record['last_updated'], "%Y-%m-%d")

                base_stmt = insert(page_raw).values(
                    id=record_id,
                    url=record['url'],
                    last_updated=last_updated_dt,
                    is_active=True
                )

                stmt = base_stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_={
                        'url': record['url'],
                        'last_updated': func.greatest(page_raw.c.last_updated, base_stmt.excluded.last_updated),
                        'is_active': True
                    }
                )

                # Track changes for metrics
                if record_id not in existing_ids:
                    metrics.new_urls += 1
                elif existing_records[record_id]["last_updated"] < last_updated_dt:
                    metrics.updated_urls += 1
                else:
                    metrics.unchanged_urls += 1

                start_query = time.time()
                conn.execute(stmt)
                end_query = time.time()

                logger.info(f"Record {record_id} updated in {end_query - start_query:.4f} sec")

            except Exception as e:
                logger.error(f"Error updating record {record_id}: {e}")
                metrics.errors += 1

        # Mark missing pages as inactive
        try:
            start_query = time.time()
            stmt_mark_inactive = page_raw.update().where(
                ~page_raw.c.id.in_(new_ids)
            ).values(is_active=False)

            conn.execute(stmt_mark_inactive)
            end_query = time.time()

            logger.info(f"Marked missing pages as inactive in {end_query - start_query:.4f} sec")

        except Exception as e:
            logger.error(f"Error marking missing pages inactive: {e}")
            metrics.errors += 1

    metrics.database_update_time = time.time() - start_time
    logger.info(f"Database update completed in {metrics.database_update_time:.2f} sec")


def main():
    """
    Main function to process the sitemap and update the database.
    Now with metrics tracking.
    """
    metrics = CrawlerMetrics()
    logger.info("🚀 Starting sitemap processing...")

    sitemap_start_time = time.time()
    
    try:
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
        logger.error(f"Error processing sitemap: {e}")
        metrics.errors += 1
        
        # Save metrics even if we encounter errors
        metrics.save_to_file()
        return

    if not records:
        logger.info("⚠ No records to insert.")
        metrics.save_to_file()
        return

    # Insert or update records in the database
    insert_page_raw_records(records, metrics)
    
    # Save the final metrics
    metrics.save_to_file()
    
    # Print summary to standard output for GitHub Actions log
    print("\n" + "="*50)
    print("CRAWLER METRICS SUMMARY")
    print("="*50)
    print(f"Total runtime: {metrics.to_dict()['total_runtime_seconds']:.2f} seconds")
    print(f"Total URLs found: {metrics.total_urls_found}")
    print(f"New URLs: {metrics.new_urls}")
    print(f"Updated URLs: {metrics.updated_urls}")
    print(f"Removed URLs: {metrics.removed_urls}")
    print(f"Unchanged URLs: {metrics.unchanged_urls}")
    print(f"Errors: {metrics.errors}")
    print("="*50)

if __name__ == "__main__":
    main()