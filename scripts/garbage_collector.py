import psycopg2
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection parameters from environment variables
db_config = {
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USERNAME"),
    "password": os.environ.get("DB_PASSWORD"),
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT", 5432)
}

def get_embedding_tables(cursor):
    """Get all table names starting with 'page_embeddings_'."""
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name LIKE 'page_embeddings_%';
    """)
    return [row[0] for row in cursor.fetchall()]

def run_garbage_collection():
    """
    Deletes records from page_content, page_keywords, and page_embeddings_* tables
    for which the corresponding entry in page_raw has is_active = FALSE.
    """
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        conn.autocommit = False  # Start a transaction
        cursor = conn.cursor()

        logger.info("Identifying inactive UIDs from page_raw...")
        cursor.execute("SELECT id FROM page_raw WHERE is_active = FALSE;")
        inactive_uids = [row[0] for row in cursor.fetchall()]

        if not inactive_uids:
            logger.info("No inactive records to delete. Exiting.")
            return

        logger.info(f"Found {len(inactive_uids)} inactive UIDs to purge.")

        # 1. Delete from page_content
        logger.info("Deleting from page_content...")
        cursor.execute("DELETE FROM page_content WHERE id = ANY(%s);", (inactive_uids,))
        logger.info(f"Deleted {cursor.rowcount} records from page_content.")

        # 2. Delete from page_keywords
        logger.info("Deleting from page_keywords...")
        cursor.execute("DELETE FROM page_keywords WHERE id = ANY(%s);", (inactive_uids,))
        logger.info(f"Deleted {cursor.rowcount} records from page_keywords.")

        # 3. Delete from all embedding tables
        embedding_tables = get_embedding_tables(cursor)
        logger.info(f"Found embedding tables: {embedding_tables}")
        for table in embedding_tables:
            logger.info(f"Deleting from {table}...")
            # The 'id' column in embedding tables corresponds to 'uid' in page_raw.
            cursor.execute(f"DELETE FROM {table} WHERE id = ANY(%s);", (inactive_uids,))
            logger.info(f"Deleted {cursor.rowcount} records from {table}.")

        conn.commit()
        logger.info("Garbage collection completed successfully.")

    except psycopg2.Error as e:
        logger.error(f"Database error during garbage collection: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run_garbage_collection()