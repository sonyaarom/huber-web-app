import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any, Tuple
from hubert.db.base_storage import BaseStorage
from hubert.config import DB_PARAMS # Assuming DB_PARAMS is in your config

class PostgresStorage(BaseStorage):
    """Concrete implementation of the BaseStorage interface for PostgreSQL."""

    def __init__(self, db_params: Dict[str, Any] = DB_PARAMS):
        self.db_params = db_params
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**self.db_params)
            self.cursor = self.conn.cursor()
            # Register pgvector extension
            register_vector(self.conn)

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def get_urls_to_process(self, limit: int = None) -> List[Tuple[str, str, Any]]:
        """Fetch URLs that need their content to be scraped."""
        self.connect()
        query = """
            SELECT id, url, last_updated FROM page_raw 
            WHERE is_active = TRUE AND id NOT IN (SELECT id FROM page_content)
        """
        if limit:
            query += f" LIMIT {limit}"
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        self.conn.commit()
        return results

    def upsert_raw_pages(self, records: List[Dict[str, Any]]):
        """Move the logic from hubert.data_ingestion.huber_crawler.main.py here."""
        self.connect()
        query = """
            INSERT INTO page_raw (id, url, last_updated)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                url = EXCLUDED.url,
                last_updated = EXCLUDED.last_updated,
                is_active = TRUE;
        """
        values = [(r['id'], r['url'], r['last_updated']) for r in records]
        execute_values(self.cursor, query, values)
        self.conn.commit()

    def deactivate_old_urls(self, current_ids: List[str]) -> List[str]:
        """Mark URLs no longer present in the sitemap as inactive and return deactivated UIDs."""
        self.connect()
        
        # First, find which IDs will be deactivated
        select_query = "SELECT id FROM page_raw WHERE is_active = TRUE AND id NOT IN %s;"
        self.cursor.execute(select_query, (tuple(current_ids),))
        deactivated_uids = [row[0] for row in self.cursor.fetchall()]
        
        if deactivated_uids:
            # Then update them to inactive
            update_query = "UPDATE page_raw SET is_active = FALSE WHERE id = ANY(%s);"
            self.cursor.execute(update_query, (deactivated_uids,))
            self.conn.commit()
            
        return deactivated_uids

    def upsert_page_content(self, content_records: List[Dict[str, Any]]):
        """Upsert extracted HTML content for pages."""
        self.connect()
        upsert_query = """
            INSERT INTO page_content (id, url, html_content, extracted_title, extracted_content, is_active, last_updated, last_scraped)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                url = EXCLUDED.url,
                html_content = EXCLUDED.html_content,
                extracted_title = EXCLUDED.extracted_title,
                extracted_content = EXCLUDED.extracted_content,
                is_active = EXCLUDED.is_active,
                last_updated = EXCLUDED.last_updated,
                last_scraped = EXCLUDED.last_scraped;
        """
        values = [
            (
                r['id'],
                r['url'],
                r['html_content'],
                r['extracted_title'],
                r['extracted_content'],
                r.get('is_active', True),
                r['last_updated'],
                r['last_scraped'],
            )
            for r in content_records
        ]
        execute_values(self.cursor, upsert_query, values, page_size=100)
        self.conn.commit()
    
    def get_content_to_process_for_keywords(self) -> List[Tuple[str, str]]:
        """Fetch content that needs keyword processing."""
        self.connect()
        query = """
            SELECT pc.id, pc.extracted_content FROM page_content pc
            LEFT JOIN page_keywords pk ON pc.id = pk.uid
            WHERE pk.id IS NULL AND pc.extracted_content IS NOT NULL;
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_content_to_process_for_embeddings(self, table_name: str) -> List[Tuple[str, str]]:
        """Move logic from scripts/data_check.py or embedding_processor.py"""
        self.connect()
        query = f"""
            SELECT pc.id, pc.extracted_content FROM page_content pc
            LEFT JOIN {table_name} pe ON pc.url = pe.url
            WHERE pe.id IS NULL AND pc.extracted_content IS NOT NULL;
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def upsert_embeddings(self, table_name: str, embedding_records: List[Dict[str, Any]]):
        """Move logic from hubert.data_ingestion.processors.embedding_processor.py"""
        self.connect()
        
        if table_name == 'page_embeddings_alpha':
            query = f"""
                INSERT INTO {table_name} (id, split_id, url, chunk_text, embedding, last_scraped) 
                VALUES %s
            """
            values = [(r['id'], r['split_id'], r['url'], r['chunk_text'], r['embedding'], r.get('last_scraped')) for r in embedding_records]
        else:
            query = f"""
                INSERT INTO {table_name} (split_id, url, chunk_text, embedding, last_scraped) 
                VALUES %s
            """
            values = [(r['split_id'], r['url'], r['chunk_text'], r['embedding'], r.get('last_scraped')) for r in embedding_records]
        
        execute_values(self.cursor, query, values)
        self.conn.commit()

    def vector_search(self, table_name: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Move logic from hubert.retriever.retriever.py"""
        self.connect()
        query = f"""
            SELECT chunk, 1 - (embedding <-> %s) AS similarity
            FROM {table_name}
            ORDER BY similarity DESC
            LIMIT %s;
        """
        self.cursor.execute(query, (np.array(query_embedding), limit))
        results = self.cursor.fetchall()
        return [{"chunk": row[0], "similarity": row[1]} for row in results]
    
    def upsert_keywords(self, keyword_records: List[Dict[str, Any]]):
        """Insert or update tsvector keywords for content."""
        self.connect()

        if not keyword_records:
            return

        with self.conn.cursor() as cursor:
            cursor.execute("CREATE TEMP TABLE temp_keywords (uid TEXT, content TEXT) ON COMMIT DROP")

            values = [(r['uid'], r['content']) for r in keyword_records]
            execute_values(cursor, "INSERT INTO temp_keywords (uid, content) VALUES %s", values)

            # First, delete old entries
            cursor.execute("DELETE FROM page_keywords pk USING temp_keywords tk WHERE pk.uid = tk.uid")

            # Then, insert new ones
            cursor.execute("""
                INSERT INTO page_keywords (id, uid, tokenized_text, raw_text, last_modified, last_scraped)
                SELECT 
                    gen_random_uuid()::text,
                    uid, 
                    to_tsvector('simple', content), 
                    content,
                    NOW(),
                    NOW()
                FROM temp_keywords
            """)
        
        self.conn.commit()

    def keyword_search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform a full-text search."""
        self.connect()
        query = """
            SELECT
                pc.extracted_title,
                pc.extracted_content,
                pc.url,
                ts_rank(pk.tokenized_text, plainto_tsquery('simple', %s)) AS rank
            FROM
                page_content pc
            JOIN
                page_keywords pk ON pc.id = pk.uid
            WHERE
                pk.tokenized_text @@ plainto_tsquery('simple', %s)
            ORDER BY
                rank DESC
            LIMIT %s;
        """
        self.cursor.execute(query, (query_text, query_text, limit))
        results = self.cursor.fetchall()
        
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in results]

    def purge_inactive_records(self):
        """Delete all records associated with inactive URLs."""
        self.connect()
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT id FROM page_raw WHERE is_active = FALSE")
            inactive_page_ids = [row[0] for row in cursor.fetchall()]

            if not inactive_page_ids:
                print("No inactive records to purge.")
                return

            print(f"Found {len(inactive_page_ids)} inactive records to purge.")

            # Get URLs for these inactive pages
            cursor.execute("SELECT url FROM page_raw WHERE id = ANY(%s)", (inactive_page_ids,))
            inactive_urls = [row[0] for row in cursor.fetchall()]

            # Delete from embedding tables
            embedding_tables = ['page_embeddings_a', 'page_embeddings_alpha', 'page_embeddings'] 
            for table in embedding_tables:
                cursor.execute(
                    f"DELETE FROM {table} WHERE url = ANY(%s)",
                    (inactive_urls,)
                )
                print(f"Deleted corresponding records from {table}.")

            # Delete from keywords table
            cursor.execute(
                "DELETE FROM page_keywords WHERE uid = ANY(%s)",
                (inactive_page_ids,)
            )
            print(f"Deleted corresponding records from page_keywords.")

            # Delete from content table
            cursor.execute(
                "DELETE FROM page_content WHERE id = ANY(%s)",
                (inactive_page_ids,)
            )
            print(f"Deleted corresponding records from page_content.")

            # Delete from raw table
            cursor.execute(
                "DELETE FROM page_raw WHERE id = ANY(%s)",
                (inactive_page_ids,)
            )
            print(f"Deleted corresponding records from page_raw.")

        self.conn.commit()
        print("Purge of inactive records complete.")

    def log_failed_job(self, uid: str, job_type: str, error: str = ""):
        """Logs a failed job to the failed_jobs table."""
        self.connect()
        insert_query = """
            INSERT INTO failed_jobs (uid, job_type)
            VALUES (%s, %s);
        """
        self.cursor.execute(insert_query, (uid, job_type))
        self.conn.commit()

    def purge_specific_inactive_records(self, inactive_uids: List[str]) -> int:
        """Immediately purge specific inactive UIDs from all related tables.
        
        Args:
            inactive_uids: List of UIDs to purge from all tables
            
        Returns:
            Total number of records deleted across all tables
        """
        if not inactive_uids:
            return 0
            
        self.connect()
        total_deleted = 0
        
        with self.conn.cursor() as cursor:
            # Get URLs for these inactive pages (needed for embedding tables)
            cursor.execute("SELECT url FROM page_raw WHERE id = ANY(%s)", (inactive_uids,))
            inactive_urls = [row[0] for row in cursor.fetchall()]
            
            if not inactive_urls:
                return 0
            
            # Delete from embedding tables (these use URLs as keys)
            embedding_tables = ['page_embeddings_a', 'page_embeddings_alpha', 'page_embeddings']
            for table in embedding_tables:
                try:
                    cursor.execute(f"DELETE FROM {table} WHERE url = ANY(%s)", (inactive_urls,))
                    deleted_count = cursor.rowcount
                    total_deleted += deleted_count
                    print(f"Deleted {deleted_count} records from {table}")
                except Exception as e:
                    print(f"Warning: Could not delete from {table}: {e}")
            
            # Delete from keywords table (uses UIDs)
            cursor.execute("DELETE FROM page_keywords WHERE uid = ANY(%s)", (inactive_uids,))
            deleted_count = cursor.rowcount
            total_deleted += deleted_count
            print(f"Deleted {deleted_count} records from page_keywords")
            
            # Delete from content table (uses UIDs)
            cursor.execute("DELETE FROM page_content WHERE id = ANY(%s)", (inactive_uids,))
            deleted_count = cursor.rowcount
            total_deleted += deleted_count
            print(f"Deleted {deleted_count} records from page_content")
            
            # Finally delete from raw table (uses UIDs)
            cursor.execute("DELETE FROM page_raw WHERE id = ANY(%s)", (inactive_uids,))
            deleted_count = cursor.rowcount
            total_deleted += deleted_count
            print(f"Deleted {deleted_count} records from page_raw")
        
        self.conn.commit()
        return total_deleted

    # ... and so on for all other methods defined in BaseStorage ...
    # You will need to implement:
    # - get_content_to_process_for_keywords
    # - keyword_search
    # - purge_inactive_records 