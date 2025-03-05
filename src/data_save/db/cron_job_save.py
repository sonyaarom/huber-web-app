#!/usr/bin/env python3

import logging
from datetime import datetime

from sqlalchemy import create_engine, MetaData, func
from sqlalchemy.dialects.postgresql import insert
from config import settings
from sitemap_utils import process_sitemap

logger = logging.getLogger(__name__)

def insert_page_raw_records(records):
    """
    Inserts or updates records in the existing page_raw table.
    Expects records to be a dictionary where the key is the hash (id)
    and the value is a dict with keys 'url' and 'last_updated'.

    For the same id (hash), only the last_updated timestamp is updated
    if the new value is later, and the url is updated.
    """
    # Build the DB URL from your settings.
    db_url = (
        f"postgresql://{settings.db_username}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )

    # Create the SQLAlchemy engine.
    engine = create_engine(db_url)

    # Reflect the existing table structure.
    metadata = MetaData()
    metadata.reflect(bind=engine)
    page_raw = metadata.tables['page_raw']  # Table must have: id (text primary key), url, last_updated

    # Iterate over the dictionary of records
    with engine.begin() as conn:
        for record_id, record in records.items():
            # Convert the "last_updated" string (YYYY-MM-DD) to a Python datetime.
            last_updated_dt = datetime.strptime(record['last_updated'], "%Y-%m-%d")
            
            # Build a base INSERT statement.
            base_stmt = insert(page_raw).values(
                id=record_id,
                url=record['url'],
                last_updated=last_updated_dt
            )
            # Create an upsert statement using ON CONFLICT.
            stmt = base_stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={
                    'url': record['url'],
                    'last_updated': func.greatest(page_raw.c.last_updated, base_stmt.excluded.last_updated)
                }
            )
            conn.execute(stmt)

    logger.info("Records inserted/updated successfully.")

def main():
    # Retrieve the records by processing your sitemap.
    records = process_sitemap(
        settings.url,
        settings.pattern,
        settings.exclude_extensions,
        settings.exclude_patterns,
        settings.include_patterns,
        settings.allowed_base_url
    )

    if not records:
        logger.info("No records to insert.")
        return

    # Insert or update records in the existing table.
    insert_page_raw_records(records)

if __name__ == "__main__":
    main()
