"""
Tests for PostgresStorage class functionality.

This module tests the PostgresStorage database abstraction layer, including:
- Connection pool management
- CRUD operations for pages, content, and embeddings
- Vector and keyword search functionality
- User management operations
- Analytics and feedback operations
- Error handling and edge cases
"""

import pytest
import psycopg2
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from hubert.db.postgres_storage import PostgresStorage
from hubert.db.models import User, UserFeedback, QueryAnalytics


class TestPostgresStorageUnit:
    """Unit tests for PostgresStorage with mocked database operations."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock connection pool."""
        pool = Mock()
        conn = Mock()
        pool.getconn.return_value = conn
        pool.putconn = Mock()
        pool.closeall = Mock()
        return pool, conn
    
    @pytest.fixture
    def storage_with_mock_pool(self, mock_pool):
        """PostgresStorage instance with mocked connection pool."""
        pool, conn = mock_pool
        
        with patch('hubert.db.postgres_storage.pool.SimpleConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = pool
            with patch('hubert.db.postgres_storage.register_vector'):
                storage = PostgresStorage()
                storage.pool = pool
                return storage, conn
    
    def test_build_dsn(self):
        """Test DSN building functionality."""
        db_params = {
            "host": "localhost",
            "port": "5432",
            "dbname": "test_db",
            "user": "test_user",
            "password": "test@pass#word"
        }
        
        dsn = PostgresStorage._build_dsn(db_params)
        
        expected = "host=localhost port=5432 dbname=test_db user=test_user password=test%40pass%23word"
        assert dsn == expected
    
    def test_initialization(self):
        """Test PostgresStorage initialization."""
        mock_db_params = {
            "host": "test_host",
            "port": "5432",
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_password"
        }
        
        with patch('hubert.db.postgres_storage.pool.SimpleConnectionPool') as mock_pool:
            with patch('hubert.db.postgres_storage.register_vector'):
                storage = PostgresStorage(db_params=mock_db_params, min_conn=2, max_conn=5)
                
                mock_pool.assert_called_once()
                args, kwargs = mock_pool.call_args
                assert args[0] == 2  # min_conn
                assert args[1] == 5  # max_conn
    
    def test_execute_query_fetch_one(self, storage_with_mock_pool):
        """Test _execute_query with fetch='one'."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        mock_cursor.fetchone.return_value = ('result1', 'result2')
        
        result = storage._execute_query("SELECT * FROM test", fetch='one')
        
        assert result == ('result1', 'result2')
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test", None)
        mock_cursor.fetchone.assert_called_once()
    
    def test_execute_query_fetch_all(self, storage_with_mock_pool):
        """Test _execute_query with fetch='all'."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        mock_cursor.fetchall.return_value = [('result1', 'result2'), ('result3', 'result4')]
        
        result = storage._execute_query("SELECT * FROM test", fetch='all')
        
        assert result == [('result1', 'result2'), ('result3', 'result4')]
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test", None)
        mock_cursor.fetchall.assert_called_once()
    
    def test_execute_query_with_commit(self, storage_with_mock_pool):
        """Test _execute_query with commit=True."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        result = storage._execute_query("INSERT INTO test VALUES (1)", commit=True)
        
        mock_conn.commit.assert_called_once()
    
    def test_upsert_raw_pages(self, storage_with_mock_pool):
        """Test upsert_raw_pages functionality."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        records = [
            {'id': 'page1', 'url': 'https://example.com/page1', 'last_updated': '2024-01-01'},
            {'id': 'page2', 'url': 'https://example.com/page2', 'last_updated': '2024-01-02'}
        ]
        
        with patch('hubert.db.postgres_storage.execute_values') as mock_execute_values:
            storage.upsert_raw_pages(records)
            
            mock_execute_values.assert_called_once()
            args, kwargs = mock_execute_values.call_args
            assert args[0] == mock_cursor
            assert 'INSERT INTO page_raw' in args[1]
            # The actual implementation has 4 values per record: id, url, last_updated, is_active
            assert args[2] == [('page1', 'https://example.com/page1', '2024-01-01', True),
                              ('page2', 'https://example.com/page2', '2024-01-02', True)]
    
    def test_deactivate_old_urls(self, storage_with_mock_pool):
        """Test deactivate_old_urls functionality."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        # Mock the select query result
        mock_cursor.fetchall.return_value = [('old_page1',), ('old_page2',)]
        
        current_ids = ['page1', 'page2', 'page3']
        result = storage.deactivate_old_urls(current_ids)
        
        # Should have called execute: SELECT and possibly UPDATE if there are results
        assert mock_cursor.execute.call_count >= 1
        assert result == ['old_page1', 'old_page2']
        mock_conn.commit.assert_called_once()
    
    def test_vector_search(self, storage_with_mock_pool):
        """Test vector_search functionality."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        # Mock search results - format: (url, content, similarity)
        mock_cursor.fetchall.return_value = [
            ('https://example.com/page1', 'content1', 0.95),
            ('https://example.com/page2', 'content2', 0.87)
        ]
        
        query_embedding = [0.1] * 1536
        results = storage.vector_search('page_embeddings', query_embedding, limit=10)
        
        assert len(results) == 2
        assert results[0]['url'] == 'https://example.com/page1'
        assert results[0]['content'] == 'content1'
        assert results[0]['similarity'] == 0.95
        assert results[1]['url'] == 'https://example.com/page2'
        assert results[1]['similarity'] == 0.87
    
    def test_keyword_search(self, storage_with_mock_pool):
        """Test keyword_search functionality."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        # Mock search results - format: (url, content, rank)
        mock_cursor.fetchall.return_value = [
            ('https://example.com/page1', 'content1', 0.95),
            ('https://example.com/page2', 'content2', 0.87)
        ]
        
        results = storage.keyword_search('test query', limit=10)
        
        assert len(results) == 2
        assert results[0]['url'] == 'https://example.com/page1'
        assert results[0]['content'] == 'content1'
        assert results[0]['rank'] == 0.95
    
    def test_user_operations(self, storage_with_mock_pool):
        """Test user management operations."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        # Test get_user_by_username
        mock_cursor.fetchone.return_value = (1, 'testuser', 'hashed_password', 'user')
        user = storage.get_user_by_username('testuser')
        
        assert user.id == 1
        assert user.username == 'testuser'
        
        # Test get_user_by_id
        mock_cursor.fetchone.return_value = (1, 'testuser', 'hashed_password', 'user')
        user = storage.get_user_by_id(1)
        
        assert user.id == 1
        assert user.username == 'testuser'
    
    def test_feedback_operations(self, storage_with_mock_pool):
        """Test user feedback storage operations."""
        storage, mock_conn = storage_with_mock_pool
        
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock()
        
        # Mock the _execute_query method to return a mock result
        with patch.object(storage, '_execute_query') as mock_execute_query:
            mock_execute_query.return_value = (123,)  # Mock returned ID
            
            feedback_data = {
                'session_id': 'session123',
                'user_id': 1,
                'query': 'test query',
                'generated_answer': 'test answer',
                'rating': 'positive',
                'feedback_comment': 'Great answer!',
                'response_time_ms': 1500
            }
            
            result = storage.store_user_feedback(feedback_data)
            
            # Verify _execute_query was called with correct parameters
            mock_execute_query.assert_called_once()
            assert result == 123
    
    def test_close_operations(self, storage_with_mock_pool):
        """Test cleanup operations."""
        storage, mock_conn = storage_with_mock_pool
        
        storage.close()
        storage.pool.closeall.assert_called_once()


class TestPostgresStorageIntegration:
    """Integration tests for PostgresStorage with real database operations."""
    
    @pytest.fixture
    def test_db_params(self):
        """Test database parameters - would need to be configured for real testing."""
        return {
            "host": "localhost",
            "port": "5432",
            "dbname": "hubert_test",
            "user": "test_user",
            "password": "test_password"
        }
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires test database setup")
    def test_real_database_connection(self, test_db_params):
        """Test real database connection - skipped unless test DB is configured."""
        storage = PostgresStorage(db_params=test_db_params)
        
        try:
            # Test basic connection
            result = storage._execute_query("SELECT 1", fetch='one')
            assert result == (1,)
        finally:
            storage.close()
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires test database setup")
    def test_full_page_workflow(self, test_db_params):
        """Test complete page workflow with real database."""
        storage = PostgresStorage(db_params=test_db_params)
        
        try:
            # Create test records
            test_records = [
                {
                    'id': 'test_page_1',
                    'url': 'https://test.example.com/page1',
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }
            ]
            
            # Test upsert
            storage.upsert_raw_pages(test_records)
            
            # Test retrieval
            urls_to_process = storage.get_urls_to_process(limit=10)
            assert len(urls_to_process) >= 1
            
            # Test deactivation
            deactivated = storage.deactivate_old_urls(['test_page_1'])
            assert isinstance(deactivated, list)
            
        finally:
            # Cleanup
            storage._execute_query(
                "DELETE FROM page_raw WHERE id = %s", 
                ('test_page_1',), 
                commit=True
            )
            storage.close()


class TestPostgresStorageEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_records_handling(self):
        """Test handling of empty record lists."""
        with patch('hubert.db.postgres_storage.pool.SimpleConnectionPool'):
            with patch('hubert.db.postgres_storage.register_vector'):
                storage = PostgresStorage()
                
                # Mock the connection pool and cursor properly
                with patch.object(storage, 'pool') as mock_pool:
                    mock_conn = Mock()
                    mock_cursor = Mock()
                    mock_pool.getconn.return_value = mock_conn
                    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
                    mock_conn.cursor.return_value.__exit__ = Mock()
                    
                    # Should handle empty lists gracefully without errors
                    try:
                        storage.upsert_raw_pages([])
                        storage.deactivate_old_urls([])
                    except Exception as e:
                        pytest.fail(f"Empty record handling failed: {e}")
    
    def test_invalid_embedding_dimensions(self, mock_storage):
        """Test handling of invalid embedding dimensions."""
        storage = mock_storage
        
        # Test with wrong embedding dimensions
        invalid_embedding = [0.1] * 512  # Should be 1536 for text-embedding-3-large
        
        with pytest.raises(Exception):
            storage.vector_search('page_embeddings', invalid_embedding)
    
    def test_connection_pool_exhaustion(self):
        """Test behavior when connection pool is exhausted."""
        with patch('hubert.db.postgres_storage.pool.SimpleConnectionPool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool
            
            with patch('hubert.db.postgres_storage.register_vector'):
                # First create storage successfully
                storage = PostgresStorage()
                
                # Then simulate pool exhaustion during query
                storage.pool.getconn.side_effect = Exception("Pool exhausted")
                
                with pytest.raises(Exception):
                    storage._execute_query("SELECT 1")
    
    def test_malformed_query_handling(self, mock_storage):
        """Test handling of malformed SQL queries."""
        storage = mock_storage
        
        with patch.object(storage, '_execute_query') as mock_execute:
            mock_execute.side_effect = psycopg2.Error("Syntax error in SQL")
            
            with pytest.raises(psycopg2.Error):
                storage.vector_search('invalid_table', [0.1] * 1536)


@pytest.fixture
def mock_storage():
    """Fixture providing a mocked PostgresStorage instance."""
    with patch('hubert.db.postgres_storage.pool.SimpleConnectionPool'):
        with patch('hubert.db.postgres_storage.register_vector'):
            storage = PostgresStorage()
            storage._execute_query = Mock()
            return storage


def test_postgres_storage_factory():
    """Test that PostgresStorage can be instantiated with default settings."""
    with patch('hubert.db.postgres_storage.pool.SimpleConnectionPool'):
        with patch('hubert.db.postgres_storage.register_vector'):
            with patch('hubert.config.settings') as mock_settings:
                mock_settings.db_host = 'localhost'
                mock_settings.db_port = '5432'
                mock_settings.db_name = 'test_db'
                mock_settings.db_username = 'test_user'
                mock_settings.db_password = 'test_password'
                
                storage = PostgresStorage()
                assert storage is not None 