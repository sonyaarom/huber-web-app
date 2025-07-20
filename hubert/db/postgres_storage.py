import psycopg2
import numpy as np
from psycopg2.extras import execute_values, Json
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import quote_plus
from hubert.db.base_storage import BaseStorage
from hubert.config import settings
from psycopg2 import pool
from werkzeug.security import generate_password_hash, check_password_hash
from hubert.db.models import User
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

class PostgresStorage(BaseStorage):
    """Concrete implementation of the BaseStorage interface for PostgreSQL."""

    def __init__(self, db_params: Dict[str, Any] = None, min_conn: int = 1, max_conn: int = 10):
        if db_params is None:
            db_params = {
                "host": settings.db_host,
                "port": settings.db_port,
                "dbname": settings.db_name,
                "user": settings.db_username,
                "password": settings.db_password,
            }

        dsn = self._build_dsn(db_params)
        self.pool = pool.SimpleConnectionPool(min_conn, max_conn, dsn)
        
        # Register pgvector with a connection from the pool
        conn = self.pool.getconn()
        try:
            register_vector(conn)
        finally:
            self.pool.putconn(conn)

    @staticmethod
    def _build_dsn(db_params: Dict[str, Any]) -> str:
        """Build the data source name (DSN) for PostgreSQL connection."""
        encoded_password = quote_plus(str(db_params['password']))
        return (
            f"host={db_params['host']} "
            f"port={db_params['port']} "
            f"dbname={db_params['dbname']} "
            f"user={db_params['user']} "
            f"password={encoded_password}"
        )

    def connect(self):
        """Establish a connection to the data store."""
        # Connection is managed by the pool, so this is a no-op
        pass

    def close(self):
        """Close the connection to the data store."""
        self.close_pool()

    def close_pool(self):
        """Close all connections in the pool."""
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()

    def __del__(self):
        """Cleanup method to close connections when the object is destroyed."""
        self.close_pool()

    def _execute_query(self, query: str, params: Tuple = None, fetch: str = None, commit=False):
        """Helper function to execute a query using a connection from the pool."""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                result = None
                if fetch == 'one':
                    result = cursor.fetchone()
                elif fetch == 'all':
                    result = cursor.fetchall()
                
                if commit:
                    conn.commit()
                
                return result
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

    def upsert_raw_pages(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Insert or update raw page metadata (URL, last_modified).
        
        Returns:
            Dict with counts of new_records, updated_records, unchanged_records
        """
        if not records:
            return {"new_records": 0, "updated_records": 0, "unchanged_records": 0}
            
        # Create a temporary table to hold our new data
        temp_table_query = """
        CREATE TEMP TABLE temp_page_raw (
            id TEXT,
            url TEXT,
            last_updated TIMESTAMPTZ,
            is_active BOOLEAN DEFAULT TRUE
        ) ON COMMIT DROP;
        """
        
        # Insert data into temp table
        insert_temp_query = """
        INSERT INTO temp_page_raw (id, url, last_updated, is_active)
        VALUES %s;
        """
        
        # Analysis query to determine new vs updated vs unchanged
        analysis_query = """
        WITH record_analysis AS (
            SELECT 
                t.id,
                t.url,
                t.last_updated,
                t.is_active,
                CASE 
                    WHEN p.id IS NULL THEN 'new'
                    WHEN p.url != t.url OR p.last_updated != t.last_updated OR p.is_active != t.is_active THEN 'updated'
                    ELSE 'unchanged'
                END as record_status
            FROM temp_page_raw t
            LEFT JOIN page_raw p ON t.id = p.id
        )
        SELECT 
            record_status,
            COUNT(*) as count
        FROM record_analysis
        GROUP BY record_status;
        """
        
        # Actual upsert query
        upsert_query = """
        INSERT INTO page_raw (id, url, last_updated, is_active, last_scraped) 
        SELECT id, url, last_updated, is_active, NOW()
        FROM temp_page_raw
        ON CONFLICT (id) DO UPDATE SET
            url = EXCLUDED.url,
            last_updated = EXCLUDED.last_updated,
            is_active = EXCLUDED.is_active,
            last_scraped = EXCLUDED.last_scraped;
        """

        values = [(r['uid'], r['url'], r['last_updated'], r.get('is_active', True)) for r in records]

        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                # Create temp table
                cursor.execute(temp_table_query)
                
                # Insert into temp table
                execute_values(cursor, insert_temp_query, values, page_size=100)
                
                # Analyze changes
                cursor.execute(analysis_query)
                analysis_results = cursor.fetchall()
                
                # Perform the actual upsert
                cursor.execute(upsert_query)
                
            conn.commit()
            
            # Process analysis results
            stats = {"new_records": 0, "updated_records": 0, "unchanged_records": 0}
            for row in analysis_results:
                status, count = row
                if status == 'new':
                    stats["new_records"] = count
                elif status == 'updated':
                    stats["updated_records"] = count
                elif status == 'unchanged':
                    stats["unchanged_records"] = count
                    
            return stats
            
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
        if table_name == 'page_embeddings_a':
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

    def vector_search(self, table_name: str, query_embedding: List[float], limit: int = 5, filters: Dict[str, Any] = None, threshold: float = None) -> List[Dict[str, Any]]:
        """Perform a vector search."""
        params = [np.array(query_embedding)]
        
        query_parts = [
            f"SELECT t.url, t.chunk_text as content, 1 - (t.embedding <-> %s) AS similarity",
            f"FROM {table_name} t"
        ]
        
        where_clauses = []
        
        if filters:
            # Join with page_content to filter by entities
            query_parts.append("JOIN page_content pc ON t.url = pc.url")
            for key, values in filters.items():
                if values:
                    json_filter = {key: values}
                    where_clauses.append("pc.entities @> %s::jsonb")
                    params.append(Json(json_filter))

        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        query_parts.append("ORDER BY similarity DESC")
        query_parts.append("LIMIT %s")
        params.append(limit)
        
        final_query = "\n".join(query_parts)
        results = self._execute_query(final_query, tuple(params), fetch='all')
        
        # Convert to list of dictionaries
        result_dicts = [{"url": row[0], "content": row[1], "similarity": row[2]} for row in results]
        return result_dicts
    
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
        query_parts = [
            """
            SELECT
                pc.url,
                pc.extracted_content as content,
                ts_rank(pk.tokenized_text, plainto_tsquery('simple', %s)) AS rank
            FROM
                page_content pc
            JOIN
                page_keywords pk ON pc.id = pk.id
            """
        ]
        
        where_clauses = ["pk.tokenized_text @@ plainto_tsquery('simple', %s)"]

        if filters:
            for key, values in filters.items():
                if values:
                    json_filter = {key: values}
                    where_clauses.append("pc.entities @> %s::jsonb")
                    params.append(Json(json_filter))
        
        query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        query_parts.append("ORDER BY rank DESC")
        query_parts.append("LIMIT %s")
        params.append(limit)
        
        query = "\n".join(query_parts)
        
        results = self._execute_query(query, tuple(params), fetch='all')
        
        columns = ['url', 'content', 'rank']
        result_dicts = [dict(zip(columns, row)) for row in results]
        return result_dicts

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
                'page_embeddings_a', 'page_keywords', 'page_content', 'page_raw'
            ]
            for table in tables_to_purge:
                # Use a placeholder for the list of UIDs
                delete_query = f"DELETE FROM {table} WHERE id = ANY(%s)"
                cursor.execute(delete_query, (inactive_uids,))
            
            conn.commit()

    def log_failed_job(self, uid: str, job_type: str, error: str = ""):
        """Log a failed job attempt."""
        query = "INSERT INTO failed_jobs (uid, job_type, error_message) VALUES (%s, %s, %s)"
        self._execute_query(query, (uid, job_type, error), commit=True)

    def purge_specific_inactive_records(self, inactive_uids: List[str]) -> int:
        """Delete records for a specific list of inactive UIDs."""
        if not inactive_uids:
            return 0
        
        total_deleted = 0
        with self.pool.getconn() as conn, conn.cursor() as cursor:
            tables_to_purge = [
                'page_embeddings_a', 'page_keywords', 'page_content', 'page_raw'
            ]
            for table in tables_to_purge:
                delete_query = f"DELETE FROM {table} WHERE id = ANY(%s) RETURNING *"
                cursor.execute(delete_query, (inactive_uids,))
                total_deleted += cursor.rowcount
            conn.commit()
            
        return total_deleted

    def get_user_by_username(self, username):
        """Fetches a user by their username."""
        query = "SELECT id, username, password_hash, role FROM users WHERE username = %s"
        result = self._execute_query(query, (username,), fetch='one')
        if result:
            return User(id=result[0], username=result[1], password_hash=result[2], role=result[3])
        return None

    def get_user_by_id(self, user_id):
        """Fetches a user by their ID."""
        query = "SELECT id, username, password_hash, role FROM users WHERE id = %s"
        result = self._execute_query(query, (user_id,), fetch='one')
        if result:
            return User(id=result[0], username=result[1], password_hash=result[2], role=result[3])
        return None

    def add_user(self, user):
        """Add a user to the database."""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) RETURNING id", 
                             (user.username, user.password_hash, user.role))
                user_id = cursor.fetchone()[0]
            conn.commit()
            return user_id
        finally:
            if conn:
                self.pool.putconn(conn)

    def get_embedding_tables(self) -> List[str]:
        """Get all table names starting with 'page_embeddings'."""
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name LIKE 'page_embeddings%'
            ORDER BY table_name;
        """
        result = self._execute_query(query, fetch='all')
        return [row[0] for row in result] if result else []

    def deactivate_old_urls(self, current_ids: List[str]) -> List[str]:
        """Mark URLs no longer present in the sitemap as inactive."""
        if not current_ids:
            return []
            
        query = """
            UPDATE page_raw 
            SET is_active = FALSE 
            WHERE id NOT IN %s AND is_active = TRUE
            RETURNING id;
        """
        
        result = self._execute_query(query, (tuple(current_ids),), fetch='all', commit=True)
        return [row[0] for row in result] if result else []

    def upsert_keywords(self, keyword_records: List[Dict[str, Any]]):
        """Insert or update tsvector keywords for content."""
        upsert_query = """
            INSERT INTO page_keywords (id, uid, last_modified, tokenized_text, raw_text) 
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                uid = EXCLUDED.uid,
                last_modified = EXCLUDED.last_modified,
                tokenized_text = EXCLUDED.tokenized_text,
                raw_text = EXCLUDED.raw_text,
                last_scraped = EXCLUDED.last_scraped;
        """
        
        values = []
        for r in keyword_records:
            values.append((
                r['id'], r['uid'], r['last_modified'], 
                r['tokenized_text'], r['raw_text']
            ))

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
            WHERE pk.id IS NULL AND pc.extracted_content IS NOT NULL;
        """
        return self._execute_query(query, fetch='all')

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
                'page_embeddings_a', 'page_keywords', 'page_content', 'page_raw'
            ]
            for table in tables_to_purge:
                # Use a placeholder for the list of UIDs
                delete_query = f"DELETE FROM {table} WHERE id = ANY(%s)"
                cursor.execute(delete_query, (inactive_uids,))
            
            conn.commit()

    def log_failed_job(self, uid: str, job_type: str, error: str = ""):
        """Log a failed job attempt."""
        query = "INSERT INTO failed_jobs (uid, job_type, error_message) VALUES (%s, %s, %s)"
        self._execute_query(query, (uid, job_type, error), commit=True)

    def purge_specific_inactive_records(self, inactive_uids: List[str]) -> int:
        """Delete records for a specific list of inactive UIDs."""
        if not inactive_uids:
            return 0
        
        total_deleted = 0
        with self.pool.getconn() as conn, conn.cursor() as cursor:
            tables_to_purge = [
                'page_embeddings_a', 'page_keywords', 'page_content', 'page_raw'
            ]
            for table in tables_to_purge:
                delete_query = f"DELETE FROM {table} WHERE id = ANY(%s) RETURNING *"
                cursor.execute(delete_query, (inactive_uids,))
                total_deleted += cursor.rowcount
            conn.commit()
            
        return total_deleted

    # New methods for feedback and analytics
    
    def store_user_feedback(self, feedback_data: Dict[str, Any]) -> int:
        """Store user feedback in the database."""
        query = """
            INSERT INTO user_feedback (
                session_id, user_id, query, generated_answer, prompt_used, 
                retrieval_method, sources_urls, rating, feedback_comment, response_time_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        values = (
            feedback_data.get('session_id'),
            feedback_data.get('user_id'),
            feedback_data.get('query'),
            feedback_data.get('generated_answer'),
            feedback_data.get('prompt_used'),
            feedback_data.get('retrieval_method'),
            Json(feedback_data.get('sources_urls', [])),
            feedback_data.get('rating'),
            feedback_data.get('feedback_comment'),
            feedback_data.get('response_time_ms')
        )
        result = self._execute_query(query, values, fetch='one', commit=True)
        return result[0] if result else None

    def store_query_analytics(self, query_data: Dict[str, Any]) -> int:
        """Store query analytics in the database."""
        query = """
            INSERT INTO query_analytics (
                session_id, user_id, query, query_tokens, query_length, 
                has_answer, response_time_ms, retrieval_method, num_sources_found
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        
        # Tokenize query for analytics
        query_tokens = query_data.get('query', '').lower().split() if query_data.get('query') else []
        
        values = (
            query_data.get('session_id'),
            query_data.get('user_id'),
            query_data.get('query'),
            query_tokens,
            len(query_data.get('query', '')),
            query_data.get('has_answer', True),
            query_data.get('response_time_ms'),
            query_data.get('retrieval_method'),
            query_data.get('num_sources_found')
        )
        result = self._execute_query(query, values, fetch='one', commit=True)
        return result[0] if result else None

    def store_retrieval_analytics(self, query_analytics_id: int, retrieved_results: List[Dict[str, Any]]) -> None:
        """Store retrieval results for MRR calculation."""
        if not retrieved_results:
            return
            
        query = """
            INSERT INTO retrieval_analytics (
                query_analytics_id, retrieved_url, rank_position, similarity_score, is_relevant
            ) VALUES %s;
        """
        
        values = [
            (
                query_analytics_id,
                result.get('url'),
                result.get('rank_position', idx + 1),
                result.get('similarity_score'),
                result.get('is_relevant')  # This can be set later for training data
            )
            for idx, result in enumerate(retrieved_results)
        ]
        
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                execute_values(cursor, query, values, page_size=100)
            conn.commit()
        finally:
            if conn:
                self.pool.putconn(conn)

    def get_feedback_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback metrics for the dashboard."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        queries = [
            # Total feedback count
            ("total_feedback", """
                SELECT COUNT(*) as total_feedback
                FROM user_feedback 
                WHERE timestamp >= %s
            """),
            
            # Positive vs negative feedback
            ("feedback_distribution", """
                SELECT rating, COUNT(*) as count
                FROM user_feedback 
                WHERE timestamp >= %s
                GROUP BY rating
            """),
            
            # Feedback over time (daily)
            ("feedback_over_time", """
                SELECT DATE(timestamp) as date, rating, COUNT(*) as count
                FROM user_feedback 
                WHERE timestamp >= %s
                GROUP BY DATE(timestamp), rating
                ORDER BY date
            """),
            
            # Average response time by rating
            ("response_time_by_rating", """
                SELECT rating, AVG(response_time_ms) as avg_response_time
                FROM user_feedback 
                WHERE timestamp >= %s AND response_time_ms IS NOT NULL
                GROUP BY rating
            """)
        ]
        
        metrics = {}
        for metric_name, query in queries:
            result = self._execute_query(query, (cutoff_date,), fetch='all')
            metrics[metric_name] = result
            
        return metrics

    def get_query_analytics_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get query analytics for time series and word cloud."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        queries = [
            # Queries per day
            ("queries_per_day", """
                SELECT DATE(timestamp) as date, COUNT(*) as query_count
                FROM query_analytics 
                WHERE timestamp >= %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """),
            
            # Popular search terms (word cloud)
            ("popular_terms", """
                SELECT unnest(query_tokens) as term, COUNT(*) as frequency
                FROM query_analytics 
                WHERE timestamp >= %s AND query_tokens IS NOT NULL
                GROUP BY term
                HAVING COUNT(*) > 1
                ORDER BY frequency DESC
                LIMIT 100
            """),
            
            # Average query length over time
            ("avg_query_length", """
                SELECT DATE(timestamp) as date, AVG(query_length) as avg_length
                FROM query_analytics 
                WHERE timestamp >= %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """),
            
            # Retrieval method distribution
            ("retrieval_methods", """
                SELECT retrieval_method, COUNT(*) as count
                FROM query_analytics 
                WHERE timestamp >= %s AND retrieval_method IS NOT NULL
                GROUP BY retrieval_method
            """)
        ]
        
        metrics = {}
        for metric_name, query in queries:
            result = self._execute_query(query, (cutoff_date,), fetch='all')
            metrics[metric_name] = result
            
        return metrics

    def calculate_mrr_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Calculate Mean Reciprocal Rank (MRR) metrics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get MRR calculation - for now we'll use click position as relevance indicator
        # In the future, this can be enhanced with explicit relevance judgments
        query = """
            WITH ranked_results AS (
                SELECT 
                    qa.id,
                    qa.query,
                    qa.timestamp::date as date,
                    ra.rank_position,
                    ra.similarity_score,
                    -- For now, we assume top 3 results are relevant, this can be improved
                    CASE WHEN ra.rank_position <= 3 THEN 1.0 / ra.rank_position ELSE 0 END as reciprocal_rank
                FROM query_analytics qa
                JOIN retrieval_analytics ra ON qa.id = ra.query_analytics_id
                WHERE qa.timestamp >= %s
            ),
            mrr_by_query AS (
                SELECT 
                    id,
                    query,
                    date,
                    MAX(reciprocal_rank) as max_reciprocal_rank
                FROM ranked_results
                GROUP BY id, query, date
            )
            SELECT 
                date,
                AVG(max_reciprocal_rank) as mrr,
                COUNT(*) as query_count
            FROM mrr_by_query
            GROUP BY date
            ORDER BY date;
        """
        
        mrr_results = self._execute_query(query, (cutoff_date,), fetch='all')
        
        # Overall MRR
        overall_mrr_query = """
            WITH ranked_results AS (
                SELECT 
                    qa.id,
                    CASE WHEN ra.rank_position <= 3 THEN 1.0 / ra.rank_position ELSE 0 END as reciprocal_rank
                FROM query_analytics qa
                JOIN retrieval_analytics ra ON qa.id = ra.query_analytics_id
                WHERE qa.timestamp >= %s
            ),
            mrr_by_query AS (
                SELECT 
                    id,
                    MAX(reciprocal_rank) as max_reciprocal_rank
                FROM ranked_results
                GROUP BY id
            )
            SELECT AVG(max_reciprocal_rank) as overall_mrr
            FROM mrr_by_query;
        """
        
        overall_mrr = self._execute_query(overall_mrr_query, (cutoff_date,), fetch='one')
        
        return {
            'mrr_over_time': mrr_results,
            'overall_mrr': overall_mrr[0] if overall_mrr and overall_mrr[0] else 0
        }

    def get_preference_dataset(self) -> List[Dict]:
        """Export preference dataset for RLHF/DPO training."""
        query = """
        SELECT 
            uf.query,
            uf.generated_answer,
            uf.rating,
            uf.feedback_comment,
            uf.sources_urls,
            uf.prompt_used,
            uf.retrieval_method,
            uf.response_time_ms,
            uf.timestamp
        FROM user_feedback uf
        ORDER BY uf.timestamp DESC
        """
        
        results = self._execute_query(query, fetch='all')
        
        preference_data = []
        for row in results:
            preference_data.append({
                'query': row[0],
                'generated_answer': row[1],
                'rating': row[2],
                'feedback_comment': row[3],
                'sources_urls': row[4],
                'prompt_used': row[5],
                'retrieval_method': row[6],
                'response_time_ms': row[7],
                'timestamp': row[8].isoformat() if row[8] else None
            })
        
        return preference_data

    def update_retrieval_relevance(self, query_analytics_id: int, url: str, rank_position: int, is_relevant: bool):
        """Update relevance information for a specific retrieval result."""
        query = """
        UPDATE retrieval_analytics 
        SET is_relevant = %s, timestamp = CURRENT_TIMESTAMP
        WHERE query_analytics_id = %s AND retrieved_url = %s AND rank_position = %s
        """
        
        result = self._execute_query(query, (is_relevant, query_analytics_id, url, rank_position), commit=True)
        logger.info(f"Updated relevance for URL {url} at rank {rank_position}: {is_relevant}")
        
        # Let's also return the number of rows affected for debugging
        return result

    def get_query_analytics_by_session_query(self, session_id: str, query: str) -> Optional[int]:
        """Find existing query analytics ID by session and query."""
        sql_query = """
        SELECT id FROM query_analytics 
        WHERE session_id = %s AND query = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        
        result = self._execute_query(sql_query, (session_id, query), fetch='one')
        return result[0] if result else None

    def get_recent_query_analytics_by_user_query(self, user_id: int, query: str, hours: int = 1) -> Optional[int]:
        """Find recent query analytics ID by user and query within the last N hours."""
        sql_query = """
        SELECT id FROM query_analytics 
        WHERE user_id = %s AND query = %s 
        AND timestamp >= NOW() - INTERVAL '%s hours'
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        
        result = self._execute_query(sql_query, (user_id, query, hours), fetch='one')
        return result[0] if result else None

    def get_comprehensive_analytics_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics metrics for the dashboard."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        metrics = {}
        
        # 1. Requests per day
        requests_query = """
        SELECT DATE(timestamp) as date, COUNT(*) as request_count
        FROM query_analytics 
        WHERE timestamp >= %s
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        raw_requests = self._execute_query(requests_query, (cutoff_date,), fetch='all')
        metrics['requests_per_day'] = [
            {'date': row[0].strftime('%Y-%m-%d'), 'count': int(row[1])}
            for row in raw_requests
        ] if raw_requests else []
        
        # 2. Average response time per day
        response_time_query = """
        SELECT DATE(timestamp) as date, 
               AVG(response_time_ms) as avg_response_time,
               COUNT(*) as request_count
        FROM query_analytics 
        WHERE timestamp >= %s AND response_time_ms IS NOT NULL
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        raw_response_times = self._execute_query(response_time_query, (cutoff_date,), fetch='all')
        metrics['avg_response_time_per_day'] = [
            {
                'date': row[0].strftime('%Y-%m-%d'), 
                'avg_response_time': round(float(row[1]) / 1000, 2) if row[1] else 0,  # Convert ms to seconds
                'request_count': int(row[2])
            }
            for row in raw_response_times
        ] if raw_response_times else []
        
        # 3. Most searched phrases/queries
        popular_queries_query = """
        SELECT query, COUNT(*) as search_count,
               AVG(response_time_ms) as avg_response_time,
               AVG(num_sources_found) as avg_sources
        FROM query_analytics 
        WHERE timestamp >= %s
        GROUP BY query
        ORDER BY search_count DESC
        LIMIT 20
        """
        raw_popular_queries = self._execute_query(popular_queries_query, (cutoff_date,), fetch='all')
        metrics['popular_queries'] = [
            {
                'query': row[0],
                'search_count': int(row[1]),
                'avg_response_time': round(float(row[2]) / 1000, 2) if row[2] else 0,  # Convert ms to seconds
                'avg_sources': float(row[3]) if row[3] else 0
            }
            for row in raw_popular_queries
        ] if raw_popular_queries else []
        
        # 4. Popular search terms (word cloud data)
        popular_terms_query = """
        SELECT unnest(query_tokens) as term, COUNT(*) as frequency
        FROM query_analytics 
        WHERE timestamp >= %s AND query_tokens IS NOT NULL
        GROUP BY term
        HAVING COUNT(*) > 1
        ORDER BY frequency DESC
        LIMIT 50
        """
        raw_popular_terms = self._execute_query(popular_terms_query, (cutoff_date,), fetch='all')
        metrics['popular_terms'] = [
            {'term': row[0], 'frequency': int(row[1])}
            for row in raw_popular_terms
        ] if raw_popular_terms else []
        
        # 5. Endpoint usage distribution (search vs chat)
        endpoint_usage_query = """
        SELECT 
            CASE 
                WHEN retrieval_method = 'search' THEN 'search'
                ELSE 'chat'
            END as endpoint_type,
            COUNT(*) as count,
            AVG(response_time_ms) as avg_response_time
        FROM query_analytics 
        WHERE timestamp >= %s AND retrieval_method IS NOT NULL
        GROUP BY endpoint_type
        ORDER BY count DESC
        """
        raw_endpoint_usage = self._execute_query(endpoint_usage_query, (cutoff_date,), fetch='all')
        metrics['endpoint_usage'] = [
            {
                'endpoint': row[0],
                'count': int(row[1]),
                'avg_response_time': round(float(row[2]) / 1000, 2) if row[2] else 0  # Convert ms to seconds
            }
            for row in raw_endpoint_usage
        ] if raw_endpoint_usage else []
        
        # 6. MRR Calculation (based on actual user feedback only)
        mrr_query = """
        WITH ranked_results AS (
            SELECT 
                qa.id,
                qa.query,
                qa.timestamp::date as date,
                ra.rank_position,
                ra.similarity_score,
                ra.is_relevant,
                -- Only use actual relevance feedback (no assumptions)
                CASE 
                    WHEN ra.is_relevant = true THEN 1.0 / ra.rank_position
                    WHEN ra.is_relevant = false THEN 0
                    ELSE NULL  -- No feedback = exclude from MRR calculation
                END as reciprocal_rank
            FROM query_analytics qa
            JOIN retrieval_analytics ra ON qa.id = ra.query_analytics_id
            WHERE qa.timestamp >= %s AND ra.is_relevant IS NOT NULL  -- Only queries with feedback
        ),
        mrr_by_query AS (
            SELECT 
                id, query, date,
                MAX(reciprocal_rank) as max_reciprocal_rank
            FROM ranked_results
            GROUP BY id, query, date
        )
        SELECT 
            date,
            AVG(max_reciprocal_rank) as mrr,
            COUNT(*) as query_count
        FROM mrr_by_query
        GROUP BY date
        ORDER BY date
        """
        mrr_over_time = self._execute_query(mrr_query, (cutoff_date,), fetch='all')
        
        # Overall MRR
        overall_mrr_query = """
        WITH ranked_results AS (
            SELECT 
                qa.id,
                CASE 
                    WHEN ra.is_relevant = true THEN 1.0 / ra.rank_position
                    WHEN ra.is_relevant = false THEN 0
                    ELSE NULL  -- No feedback = exclude from MRR calculation
                END as reciprocal_rank
            FROM query_analytics qa
            JOIN retrieval_analytics ra ON qa.id = ra.query_analytics_id
            WHERE qa.timestamp >= %s AND ra.is_relevant IS NOT NULL  -- Only queries with feedback
        ),
        mrr_by_query AS (
            SELECT id, MAX(reciprocal_rank) as max_reciprocal_rank
            FROM ranked_results
            GROUP BY id
        )
        SELECT AVG(max_reciprocal_rank) as overall_mrr
        FROM mrr_by_query
        """
        overall_mrr_result = self._execute_query(overall_mrr_query, (cutoff_date,), fetch='one')
        
        # Convert MRR data to JSON-serializable format
        mrr_over_time_formatted = [
            {
                'date': row[0].strftime('%Y-%m-%d'),
                'mrr': float(row[1]) if row[1] else 0,
                'query_count': int(row[2])
            }
            for row in mrr_over_time
        ] if mrr_over_time else []
        
        metrics['mrr_metrics'] = {
            'mrr_over_time': mrr_over_time_formatted,
            'overall_mrr': float(overall_mrr_result[0]) if overall_mrr_result and overall_mrr_result[0] else 0
        }
        
        # 7. Top performing URLs
        top_urls_query = """
        SELECT ra.retrieved_url,
               COUNT(*) as shown_count,
               COUNT(CASE WHEN ra.is_relevant = true THEN 1 END) as relevant_count,
               COUNT(CASE WHEN ra.is_relevant = false THEN 1 END) as not_relevant_count,
               AVG(ra.rank_position) as avg_rank,
               AVG(ra.similarity_score) as avg_similarity
        FROM retrieval_analytics ra
        WHERE ra.timestamp >= %s
        GROUP BY ra.retrieved_url
        HAVING COUNT(*) >= 2  -- Only URLs shown at least twice
        ORDER BY relevant_count DESC, shown_count DESC
        LIMIT 20
        """
        raw_top_urls = self._execute_query(top_urls_query, (cutoff_date,), fetch='all')
        metrics['top_urls'] = [
            {
                'url': row[0],
                'shown_count': int(row[1]),
                'relevant_count': int(row[2]),
                'not_relevant_count': int(row[3]),
                'avg_rank': float(row[4]) if row[4] else 0,
                'avg_similarity': float(row[5]) if row[5] else 0,
                'precision': float(row[2]) / float(row[1]) if row[1] > 0 else 0  # relevant/shown
            }
            for row in raw_top_urls
        ] if raw_top_urls else []
        
        # 8. Precision metrics
        precision_query = """
        SELECT DATE(qa.timestamp) as date,
               COUNT(DISTINCT qa.id) as total_queries,
               COUNT(ra.id) as total_results_shown,
               COUNT(CASE WHEN ra.is_relevant = true THEN 1 END) as relevant_results,
               COUNT(CASE WHEN ra.is_relevant IS NOT NULL THEN 1 END) as results_with_feedback,
               CASE 
                   WHEN COUNT(CASE WHEN ra.is_relevant IS NOT NULL THEN 1 END) > 0 
                   THEN COUNT(CASE WHEN ra.is_relevant = true THEN 1 END)::float / 
                        COUNT(CASE WHEN ra.is_relevant IS NOT NULL THEN 1 END)
                   ELSE NULL 
               END as precision
        FROM query_analytics qa
        LEFT JOIN retrieval_analytics ra ON qa.id = ra.query_analytics_id
        WHERE qa.timestamp >= %s
        GROUP BY DATE(qa.timestamp)
        ORDER BY date
        """
        raw_precision_metrics = self._execute_query(precision_query, (cutoff_date,), fetch='all')
        metrics['precision_metrics'] = [
            {
                'date': row[0].strftime('%Y-%m-%d'),
                'total_queries': int(row[1]),
                'total_results_shown': int(row[2]),
                'relevant_results': int(row[3]),
                'results_with_feedback': int(row[4]),
                'precision': float(row[5]) if row[5] is not None else 0
            }
            for row in raw_precision_metrics
        ] if raw_precision_metrics else []
        
        # 9. Summary statistics
        summary_query = """
        SELECT 
            COUNT(DISTINCT qa.id) as total_requests,
            COUNT(DISTINCT qa.query) as total_unique_queries,
            AVG(qa.response_time_ms) as avg_response_time,
            COUNT(CASE WHEN ra.is_relevant IS NOT NULL THEN 1 END) as total_feedback_given,
            COUNT(CASE WHEN ra.is_relevant = true THEN 1 END) as positive_feedback,
            COUNT(CASE WHEN ra.is_relevant = false THEN 1 END) as negative_feedback
        FROM query_analytics qa
        LEFT JOIN retrieval_analytics ra ON qa.id = ra.query_analytics_id
        WHERE qa.timestamp >= %s
        """
        summary_result = self._execute_query(summary_query, (cutoff_date,), fetch='one')
        
        if summary_result:
            metrics['summary_stats'] = {
                'total_requests': int(summary_result[0]) if summary_result[0] else 0,
                'total_unique_queries': int(summary_result[1]) if summary_result[1] else 0,
                'avg_response_time': round(float(summary_result[2]) / 1000 if summary_result[2] else 0, 2),  # Convert ms to seconds
                'total_feedback_given': int(summary_result[3]) if summary_result[3] else 0,
                'positive_feedback': int(summary_result[4]) if summary_result[4] else 0,
                'negative_feedback': int(summary_result[5]) if summary_result[5] else 0
            }
        else:
            metrics['summary_stats'] = {
                'total_requests': 0,
                'total_unique_queries': 0,
                'avg_response_time': 0,
                'total_feedback_given': 0,
                'positive_feedback': 0,
                'negative_feedback': 0
            }
        
        return metrics