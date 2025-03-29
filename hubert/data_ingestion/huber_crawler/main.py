#!/usr/bin/env python3

import logging
import time
from datetime import datetime

from sqlalchemy import create_engine, MetaData, func
from sqlalchemy.dialects.postgresql import insert
from ..config import settings
from .sitemap import process_sitemap


#TODO: clean up the code
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def insert_page_raw_records(records):
    """
    Inserts or updates records in the existing page_raw table.
    Pages missing from the new sitemap are marked as inactive (is_active = FALSE).
    """
    start_time = time.time()

    db_url = (
        f"postgresql://{settings.db_username}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )

    #  Add timeout to prevent long-running connections
    engine = create_engine(db_url, connect_args={"connect_timeout": 10})

    metadata = MetaData()
    metadata.reflect(bind=engine)
    page_raw = metadata.tables['page_raw']

    new_ids = set(records.keys())

    with engine.begin() as conn:
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

                start_query = time.time()
                conn.execute(stmt)
                end_query = time.time()

                logger.info(f"Record {record_id} updated in {end_query - start_query:.4f} sec")

            except Exception as e:
                logger.error(f"Error updating record {record_id}: {e}")

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

    logger.info(f"Database update completed in {time.time() - start_time:.2f} sec")


def main():
    """
    Main function to process the sitemap and update the database.
    """
    logger.info("🚀 Starting sitemap processing...")

    start_time = time.time()
    
    try:
        records = process_sitemap(
            settings.url,
            settings.pattern,
            settings.exclude_extensions,
            settings.exclude_patterns,
            settings.include_patterns,
            settings.allowed_base_url
        )

        logger.info(f"Sitemap processed in {time.time() - start_time:.2f} sec, {len(records)} records found.")

    except Exception as e:
        logger.error(f"Error processing sitemap: {e}")
        return

    if not records:
        logger.info("⚠ No records to insert.")
        return

    #  Insert or update records in the database
    insert_page_raw_records(records)

if __name__ == "__main__":
    main()
