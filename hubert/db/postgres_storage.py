import psycopg2
import numpy as np
from psycopg2.extras import execute_values, Json
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any, Tuple
from hubert.db.base_storage import BaseStorage
from hubert.config import DB_PARAMS
from psycopg2 import pool

class PostgresStorage(BaseStorage):
    """Concrete implementation of the BaseStorage interface for PostgreSQL."""

    def __init__(self, db_params: Dict[str, Any] = DB_PARAMS, min_conn: int = 1, max_conn: int = 10):
        self.db_params = db_params
        self.pool = pool.SimpleConnectionPool(min_conn, max_conn, **db_params)
        # Register pgvector with a connection from the pool
        conn = self.pool.getconn()
        try:
            register_vector(conn)
        finally:
            self.pool.putconn(conn)

    def close(self):
        """Close the database connection pool."""
        if self.pool:
            self.pool.closeall()

    def _execute_query(self, query: str, params: Tuple = None, fetch: str = None):
        """Helper function to execute a query using a connection from the pool."""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                if fetch == 'one':
                    return cursor.fetchone()
                if fetch == 'all':
                    return cursor.fetchall()
        except psycopg2.Error as e:
            # In case of an error, rollback and re-raise
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                self.pool.putconn(conn)

    def get_urls_to_process(self, limit: int = None) -> List[Tuple[str, str, Any]]:
        """Fetch URLs that need their content to be scraped."""
        query = """
            SELECT id, url, last_updated FROM page_raw 
            WHERE is_active = TRUE AND id NOT IN (SELECT id FROM page_content)
        """
        if limit:
            query += f" LIMIT {limit}"
        return self._execute_query(query, fetch='all')

    def upsert_raw_pages(self, records: List[Dict[str, Any]]):
        """Move the logic from hubert.data_ingestion.huber_crawler.main.py here."""
        query = """
            INSERT INTO page_raw (id, url, last_updated)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                url = EXCLUDED.url,
                last_updated = EXCLUDED.last_updated,
                is_active = TRUE;
        """
        values = [(r['id'], r['url'], r['last_updated']) for r in records]
        
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                execute_values(cursor, query, values)
            conn.commit()
        finally:
            if conn:
                self.pool.putconn(conn)

    def deactivate_old_urls(self, current_ids: List[str]) -> List[str]:
        """Mark URLs no longer present in the sitemap as inactive and return deactivated UIDs."""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                select_query = "SELECT id FROM page_raw WHERE is_active = TRUE AND id NOT IN %s;"
                cursor.execute(select_query, (tuple(current_ids),))
                deactivated_uids = [row[0] for row in cursor.fetchall()]
                
                if deactivated_uids:
                    update_query = "UPDATE page_raw SET is_active = FALSE WHERE id = ANY(%s);"
                    cursor.execute(update_query, (deactivated_uids,))
                
            conn.commit()
            return deactivated_uids
        finally:
            if conn:
                self.pool.putconn(conn)

    def upsert_page_content(self, content_records: List[Dict[str, Any]]):
        """Upsert extracted HTML content for pages."""
        upsert_query = """
            INSERT INTO page_content (id, url, html_content, extracted_title, extracted_content, entities, is_active, last_updated, last_scraped)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                url = EXCLUDED.url,
                html_content = EXCLUDED.html_content,
                extracted_title = EXCLUDED.extracted_title,
                extracted_content = EXCLUDED.extracted_content,
                entities = EXCLUDED.entities,
                is_active = EXCLUDED.is_active,
                last_updated = EXCLUDED.last_updated,
                last_scraped = EXCLUDED.last_scraped;
        """
        values = [
            (
                r['id'], r['url'], r['html_content'], r['extracted_title'],
                r['extracted_content'], Json(r.get('entities', {})),
                r.get('is_active', True), r['last_updated'], r['last_scraped'],
            ) for r in content_records
        ]

        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                execute_values(cursor, upsert_query, values, page_size=100)
            conn.commit()
        finally:
            if conn:
                self.pool.putconn(conn)

    def get_content_to_process_for_keywords(self) -> List[Tuple[str, str]]:
        """Fetch content that needs keyword processing."""
        query = """
            SELECT pc.id, pc.extracted_content FROM page_content pc
            LEFT JOIN page_keywords pk ON pc.id = pk.id
            LEFT JOIN page_keywords pk ON pc.id = pk.id
            WHERE pk.id IS NULL AND pc.extracted_content IS NOT NULL;
        """
        return self._execute_query(query, fetch='all')

    def get_content_to_process_for_embeddings(self, table_name: str) -> List[Tuple[str, str]]:
        """Move logic from scripts/data_check.py or embedding_processor.py"""
        query = f"""
            SELECT pc.id, pc.extracted_content FROM page_content pc
            LEFT JOIN {table_name} pe ON pc.url = pe.url
            WHERE pe.id IS NULL AND pc.extracted_content IS NOT NULL;
        """
        return self._execute_query(query, fetch='all')

    def upsert_embeddings(self, table_name: str, embedding_records: List[Dict[str, Any]]):
        """Move logic from hubert.data_ingestion.processors.embedding_processor.py"""
        if table_name == 'page_embeddings_alpha':
            query = f"INSERT INTO {table_name} (id, split_id, url, chunk_text, embedding, last_scraped) VALUES %s"
            values = [(r['id'], r['split_id'], r['url'], r['chunk_text'], r['embedding'], r.get('last_scraped')) for r in embedding_records]
        else:
            query = f"INSERT INTO {table_name} (split_id, url, chunk_text, embedding, last_scraped) VALUES %s"
            values = [(r['split_id'], r['url'], r['chunk_text'], r['embedding'], r.get('last_scraped')) for r in embedding_records]
        
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                execute_values(cursor, query, values)
            conn.commit()
        finally:
            if conn:
                self.pool.putconn(conn)

    def vector_search(self, table_name: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Perform a vector search."""
        self.connect()
        query = f"""
            SELECT url, chunk_text as content, 1 - (embedding <-> %s) AS similarity
            FROM {table_name}
            ORDER BY similarity DESC
            LIMIT %s;
        """
        self.cursor.execute(query, (np.array(query_embedding), limit))
        results = self.cursor.fetchall()
        return [{"url": row[0], "content": row[1], "similarity": row[2]} for row in results]
    
    def upsert_keywords(self, keyword_records: List[Dict[str, Any]]):
        """Insert or update tsvector keywords for content."""
        if not keyword_records:
            return
        
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("CREATE TEMP TABLE temp_keywords (uid TEXT, content TEXT) ON COMMIT DROP")
                values = [(r['uid'], r['content']) for r in keyword_records]
                execute_values(cursor, "INSERT INTO temp_keywords (uid, content) VALUES %s", values)
                cursor.execute("DELETE FROM page_keywords pk USING temp_keywords tk WHERE pk.uid = tk.uid")
                cursor.execute("""
                    INSERT INTO page_keywords (id, uid, tokenized_text, raw_text, last_modified, last_scraped)
                    SELECT gen_random_uuid()::text, uid, to_tsvector('simple', content), content, NOW(), NOW()
                    FROM temp_keywords
                """)
            conn.commit()
        finally:
            if conn:
                self.pool.putconn(conn)

    def keyword_search(self, query_text: str, limit: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform a full-text search."""
        params = [query_text, query_text]
        query = """
            SELECT
                pc.url,
                pc.extracted_content as content,
                ts_rank(pk.tokenized_text, plainto_tsquery('simple', %s)) AS rank
            FROM
                page_content pc
            JOIN
                page_keywords pk ON pc.id = pk.id
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
        with self.pool.getconn() as conn, conn.cursor() as cursor:
            # First, get all inactive URLs
            cursor.execute("SELECT id FROM page_raw WHERE is_active = FALSE")
            inactive_uids = [row[0] for row in cursor.fetchall()]

            if not inactive_uids:
                return

            # Delete from all related tables
            tables_to_purge = [
                'page_embeddings_alpha', 'page_keywords', 'page_content', 'page_raw'
            ]
            for table in tables_to_purge:
                # Use a placeholder for the list of UIDs
                delete_query = f"DELETE FROM {table} WHERE id = ANY(%s)"
                cursor.execute(delete_query, (inactive_uids,))
            
            conn.commit()

    def log_failed_job(self, uid: str, job_type: str, error: str = ""):
        """Log a failed job attempt."""
        query = "INSERT INTO failed_jobs (uid, job_type, error_message) VALUES (%s, %s, %s)"
        self._execute_query(query, (uid, job_type, error))

    def purge_specific_inactive_records(self, inactive_uids: List[str]) -> int:
        """Delete records for a specific list of inactive UIDs."""
        if not inactive_uids:
            return 0
        
        total_deleted = 0
        with self.pool.getconn() as conn, conn.cursor() as cursor:
            tables_to_purge = [
                'page_embeddings_alpha', 'page_keywords', 'page_content', 'page_raw'
            ]
            for table in tables_to_purge:
                delete_query = f"DELETE FROM {table} WHERE id = ANY(%s) RETURNING *"
                cursor.execute(delete_query, (inactive_uids,))
                total_deleted += cursor.rowcount
            conn.commit()
            
        return total_deleted
