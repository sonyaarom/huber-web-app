"""
Tests for sitemap parsing functionality.

This module tests the sitemap parsing logic, including:
- Parsing different sitemap formats
- Handling sitemap changes (new/updated/removed URLs)
- Database integration for tracking changes
- Error handling and edge cases
"""

import pytest
import io
import gzip
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, List

from hubert.data_ingestion.huber_crawler.sitemap import (
    parse_sitemap,
    filter_sitemap_entries,
    security_check_urls,
    create_matches_dict,
    process_sitemap,
    download_sitemap_file
)
from hubert.data_ingestion.huber_crawler.main import process_page_raw_records, CrawlerMetrics
from hubert.db.postgres_storage import PostgresStorage


class TestSitemapParsing:
    """Test cases for basic sitemap parsing functionality."""
    
    def test_parse_xml_sitemap(self):
        """Test parsing of standard XML sitemap."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
                <lastmod>2024-01-16T11:00:00Z</lastmod>
            </url>
        </urlset>"""
        
        entries = parse_sitemap(xml_content)
        
        assert len(entries) == 2
        assert entries[0]['url'] == 'https://example.com/page1'
        assert entries[0]['lastmod'] == '2024-01-15T10:30:00Z'
        assert entries[1]['url'] == 'https://example.com/page2'
        assert entries[1]['lastmod'] == '2024-01-16T11:00:00Z'
    
    def test_parse_malformed_xml_fallback_to_regex(self):
        """Test that malformed XML falls back to regex parsing."""
        malformed_content = """
        <url>
            <loc>https://example.com/page1</loc>
            <lastmod>2024-01-15T10:30:00Z</lastmod>
        </url>
        <url>
            <loc>https://example.com/page2</loc>
            <lastmod>2024-01-16T11:00:00Z</lastmod>
        </url>
        """
        
        entries = parse_sitemap(malformed_content)
        
        assert len(entries) == 2
        assert entries[0]['url'] == 'https://example.com/page1'
        assert entries[0]['lastmod'] == '2024-01-15T10:30:00Z'
    
    def test_filter_sitemap_entries(self):
        """Test filtering of sitemap entries."""
        entries = [
            {'url': 'https://example.com/page1.html', 'lastmod': '2024-01-15T10:30:00Z'},
            {'url': 'https://example.com/image.jpg', 'lastmod': '2024-01-15T10:30:00Z'},
            {'url': 'https://example.com/en/page2.html', 'lastmod': '2024-01-16T11:00:00Z'},
            {'url': 'https://example.com/view/page3.html', 'lastmod': '2024-01-17T12:00:00Z'},
            {'url': 'https://example.com/de/page4.html', 'lastmod': '2024-01-18T13:00:00Z'},
        ]
        
        filtered = filter_sitemap_entries(
            entries,
            exclude_extensions=['.jpg', '.pdf'],
            exclude_patterns=['view'],
            include_patterns=['/en/']
        )
        
        assert len(filtered) == 1
        assert filtered[0]['url'] == 'https://example.com/en/page2.html'
    
    def test_security_check_urls(self):
        """Test security validation of URLs."""
        entries = [
            {'url': 'https://example.com/safe-page', 'lastmod': '2024-01-15T10:30:00Z'},
            {'url': 'https://malicious.com/page', 'lastmod': '2024-01-16T11:00:00Z'},
            {'url': 'https://example.com/another-safe', 'lastmod': '2024-01-17T12:00:00Z'},
        ]
        
        safe, unsafe = security_check_urls(entries, 'https://example.com')
        
        assert len(safe) == 2
        assert len(unsafe) == 1
        assert safe[0]['url'] == 'https://example.com/safe-page'
        assert unsafe[0]['url'] == 'https://malicious.com/page'


class TestSitemapChanges:
    """Test cases for detecting and handling sitemap changes."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage instance for testing."""
        storage = Mock(spec=PostgresStorage)
        storage.upsert_raw_pages = Mock()
        storage.deactivate_old_urls = Mock()
        storage.close = Mock()
        return storage
    
    @pytest.fixture
    def sample_sitemap_v1(self):
        """Sample sitemap - version 1."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://example.com/page3</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
    
    @pytest.fixture
    def sample_sitemap_v2(self):
        """Sample sitemap - version 2 with changes."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
                <lastmod>2024-01-16T11:00:00Z</lastmod>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://example.com/page4</loc>
                <lastmod>2024-01-16T12:00:00Z</lastmod>
            </url>
        </urlset>"""
    
    def test_detect_sitemap_changes(self, sample_sitemap_v1, sample_sitemap_v2):
        """Test detection of changes between sitemap versions."""
        # Parse both versions
        entries_v1 = parse_sitemap(sample_sitemap_v1)
        entries_v2 = parse_sitemap(sample_sitemap_v2)
        
        # Create matches dictionaries
        matches_v1 = create_matches_dict(entries_v1)
        matches_v2 = create_matches_dict(entries_v2)
        
        # Get URL sets for comparison
        urls_v1 = set(data['url'] for data in matches_v1.values())
        urls_v2 = set(data['url'] for data in matches_v2.values())
        
        # Detect changes
        new_urls = urls_v2 - urls_v1
        removed_urls = urls_v1 - urls_v2
        common_urls = urls_v1 & urls_v2
        
        # Verify changes
        assert new_urls == {'https://example.com/page4'}
        assert removed_urls == {'https://example.com/page3'}
        assert len(common_urls) == 2
        assert 'https://example.com/page1' in common_urls
        assert 'https://example.com/page2' in common_urls
    
    @patch('hubert.data_ingestion.huber_crawler.sitemap.download_sitemap_file')
    def test_process_sitemap_changes(self, mock_download, mock_storage, sample_sitemap_v2):
        """Test processing sitemap changes through the main pipeline."""
        # Mock sitemap download
        mock_download.return_value = sample_sitemap_v2.encode('utf-8')
        
        # Process sitemap
        records = process_sitemap(
            'https://example.com/sitemap.xml',
            exclude_extensions=['.jpg', '.pdf'],
            include_patterns=['/'],
            allowed_base_url='https://example.com'
        )
        
        # Verify records structure
        assert len(records) == 3
        assert all('url' in data and 'last_updated' in data for data in records.values())
        
        # Test database operations
        metrics = CrawlerMetrics()
        process_page_raw_records(mock_storage, records, metrics)
        
        # Verify storage calls
        mock_storage.upsert_raw_pages.assert_called_once()
        mock_storage.deactivate_old_urls.assert_called_once()
        
        # Verify upsert data
        upsert_call_args = mock_storage.upsert_raw_pages.call_args[0][0]
        assert len(upsert_call_args) == 3
        urls_in_upsert = {record['url'] for record in upsert_call_args}
        expected_urls = {
            'https://example.com/page1',
            'https://example.com/page2', 
            'https://example.com/page4'
        }
        assert urls_in_upsert == expected_urls


class TestSitemapFormats:
    """Test cases for different sitemap formats and edge cases."""
    
    def test_gzipped_sitemap(self):
        """Test handling of gzipped sitemaps."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Create gzipped content
        gzipped_content = io.BytesIO()
        with gzip.GzipFile(fileobj=gzipped_content, mode='wb') as gz:
            gz.write(xml_content.encode('utf-8'))
        gzipped_bytes = gzipped_content.getvalue()
        
        # Mock the download function to return gzipped content
        with patch('hubert.data_ingestion.huber_crawler.sitemap.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = gzipped_bytes
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            content = download_sitemap_file('https://example.com/sitemap.xml.gz')
            entries = parse_sitemap(content)
            
            assert len(entries) == 1
            assert entries[0]['url'] == 'https://example.com/page1'
    
    def test_empty_sitemap(self):
        """Test handling of empty sitemaps."""
        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        </urlset>"""
        
        entries = parse_sitemap(empty_xml)
        assert len(entries) == 0
    
    def test_sitemap_without_lastmod(self):
        """Test handling of sitemaps without lastmod dates."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        entries = parse_sitemap(xml_content)
        
        assert len(entries) == 2
        assert entries[0]['url'] == 'https://example.com/page1'
        assert entries[0]['lastmod'] is None
        assert entries[1]['lastmod'] == '2024-01-15T10:30:00Z'


class TestSitemapIntegration:
    """Integration tests for sitemap processing with database operations."""
    
    @pytest.fixture
    def real_storage(self):
        """Create a real storage instance for integration tests."""
        # This would use a test database
        storage = PostgresStorage({
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        })
        return storage
    
    @pytest.mark.integration
    @patch('hubert.data_ingestion.huber_crawler.sitemap.download_sitemap_file')
    def test_full_sitemap_processing_pipeline(self, mock_download):
        """Test the complete sitemap processing pipeline."""
        # Create test sitemap data
        initial_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        updated_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
                <lastmod>2024-01-16T11:00:00Z</lastmod>
            </url>
            <url>
                <loc>https://example.com/page3</loc>
                <lastmod>2024-01-16T12:00:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Mock storage for this test
        mock_storage = Mock(spec=PostgresStorage)
        mock_storage.upsert_raw_pages = Mock()
        mock_storage.deactivate_old_urls = Mock()
        mock_storage.close = Mock()
        
        # Process initial sitemap
        mock_download.return_value = initial_sitemap.encode('utf-8')
        initial_records = process_sitemap(
            'https://example.com/sitemap.xml',
            allowed_base_url='https://example.com'
        )
        
        metrics_initial = CrawlerMetrics()
        process_page_raw_records(mock_storage, initial_records, metrics_initial)
        
        # Verify initial processing
        assert len(initial_records) == 2
        mock_storage.upsert_raw_pages.assert_called()
        
        # Process updated sitemap
        mock_download.return_value = updated_sitemap.encode('utf-8')
        updated_records = process_sitemap(
            'https://example.com/sitemap.xml',
            allowed_base_url='https://example.com'
        )
        
        metrics_updated = CrawlerMetrics()
        process_page_raw_records(mock_storage, updated_records, metrics_updated)
        
        # Verify updated processing
        assert len(updated_records) == 2
        
        # Verify that the storage was called with the correct data
        calls = mock_storage.upsert_raw_pages.call_args_list
        assert len(calls) == 2  # Two separate calls
        
        # Check the URLs in the second call (updated sitemap)
        second_call_records = calls[1][0][0]
        urls_in_second_call = {record['url'] for record in second_call_records}
        expected_urls = {'https://example.com/page1', 'https://example.com/page3'}
        assert urls_in_second_call == expected_urls


class TestSitemapErrorHandling:
    """Test cases for error handling in sitemap processing."""
    
    def test_network_error_handling(self):
        """Test handling of network errors during sitemap download."""
        with patch('hubert.data_ingestion.huber_crawler.sitemap.requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            with pytest.raises(Exception, match="Network error"):
                download_sitemap_file('https://example.com/sitemap.xml')
    
    def test_invalid_xml_handling(self):
        """Test handling of completely invalid XML."""
        invalid_xml = "This is not XML at all!"
        
        # Should not raise an exception, should fall back to regex
        entries = parse_sitemap(invalid_xml)
        assert isinstance(entries, list)
    
    def test_database_error_handling(self):
        """Test handling of database errors during processing."""
        mock_storage = Mock(spec=PostgresStorage)
        mock_storage.upsert_raw_pages.side_effect = Exception("Database error")
        
        test_records = {
            'test_id': {
                'url': 'https://example.com/page1',
                'last_updated': datetime.now(timezone.utc)
            }
        }
        
        metrics = CrawlerMetrics()
        
        with pytest.raises(Exception, match="Database error"):
            process_page_raw_records(mock_storage, test_records, metrics)
        
        # Verify metrics were updated
        assert metrics.errors > 0


# Utility functions for creating mock sitemaps
def create_mock_sitemap(urls_with_dates: List[tuple]) -> str:
    """
    Helper function to create mock XML sitemaps for testing.
    
    Args:
        urls_with_dates: List of tuples (url, lastmod_date)
    
    Returns:
        XML sitemap string
    """
    urls_xml = ""
    for url, lastmod in urls_with_dates:
        urls_xml += f"""
            <url>
                <loc>{url}</loc>
                <lastmod>{lastmod}</lastmod>
            </url>"""
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        {urls_xml}
    </urlset>"""


def create_gzipped_sitemap(xml_content: str) -> bytes:
    """
    Helper function to create gzipped sitemap content.
    
    Args:
        xml_content: XML sitemap content as string
    
    Returns:
        Gzipped content as bytes
    """
    gzipped_content = io.BytesIO()
    with gzip.GzipFile(fileobj=gzipped_content, mode='wb') as gz:
        gz.write(xml_content.encode('utf-8'))
    return gzipped_content.getvalue()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 