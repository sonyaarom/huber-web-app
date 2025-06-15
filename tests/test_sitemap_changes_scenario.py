"""
Scenario-based tests for sitemap changes.

This module contains practical tests that simulate real-world scenarios
where a sitemap changes and we need to verify the parsing works correctly.
"""

import pytest
import tempfile
import os
import gzip
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from hubert.data_ingestion.huber_crawler.sitemap import process_sitemap
from hubert.data_ingestion.huber_crawler.main import process_page_raw_records, CrawlerMetrics
from hubert.db.postgres_storage import PostgresStorage


class TestSitemapChangesScenarios:
    """Test realistic sitemap change scenarios."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_files = []
        self.mock_storage = Mock(spec=PostgresStorage)
        self.mock_storage.upsert_raw_pages = Mock()
        self.mock_storage.deactivate_old_urls = Mock(return_value=[])  # Return empty list by default
        self.mock_storage.close = Mock()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def create_temp_sitemap(self, content: str, gzipped: bool = False) -> str:
        """Create a temporary sitemap file."""
        if gzipped:
            # Create gzipped file
            fd, path = tempfile.mkstemp(suffix='.xml.gz')
            with os.fdopen(fd, 'wb') as temp_file:
                with gzip.GzipFile(fileobj=temp_file, mode='wb') as gz:
                    gz.write(content.encode('utf-8'))
        else:
            # Create regular file
            fd, path = tempfile.mkstemp(suffix='.xml')
            with os.fdopen(fd, 'w', encoding='utf-8') as temp_file:
                temp_file.write(content)
        
        self.temp_files.append(path)
        return path
    
    def test_new_pages_added_to_sitemap(self):
        """
        Test Scenario: New pages are added to the sitemap.
        Expected: New URLs should be added to the database.
        """
        # Initial sitemap with 2 pages
        initial_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Updated sitemap with 3 pages (1 new)
        updated_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page3</loc>
                <lastmod>2024-01-16T12:00:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Create temporary files
        initial_path = self.create_temp_sitemap(initial_sitemap)
        updated_path = self.create_temp_sitemap(updated_sitemap)
        
        # Process initial sitemap
        initial_records = process_sitemap(
            initial_path,
            exclude_extensions=['.jpg', '.pdf'],
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        # Process updated sitemap
        updated_records = process_sitemap(
            updated_path,
            exclude_extensions=['.jpg', '.pdf'],
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        # Verify results
        assert len(initial_records) == 2
        assert len(updated_records) == 3
        
        # Check that the new URL is present
        initial_urls = {data['url'] for data in initial_records.values()}
        updated_urls = {data['url'] for data in updated_records.values()}
        new_urls = updated_urls - initial_urls
        
        assert len(new_urls) == 1
        assert 'https://www.wiwi.hu-berlin.de/en/page3' in new_urls
        
        # Test database operations
        metrics = CrawlerMetrics()
        deactivated_uids = process_page_raw_records(self.mock_storage, updated_records, metrics)
        
        # Verify storage was called
        self.mock_storage.upsert_raw_pages.assert_called_once()
        upserted_urls = {record['url'] for record in self.mock_storage.upsert_raw_pages.call_args[0][0]}
        assert len(upserted_urls) == 3
        assert 'https://www.wiwi.hu-berlin.de/en/page3' in upserted_urls
        
        # Verify no URLs were deactivated (since this is adding new pages)
        assert deactivated_uids == []
        assert metrics.removed_urls == 0
    
    def test_pages_removed_from_sitemap(self):
        """
        Test Scenario: Pages are removed from the sitemap.
        Expected: Removed URLs should be marked as inactive.
        """
        # Initial sitemap with 3 pages
        initial_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page3</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Updated sitemap with 2 pages (1 removed)
        updated_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Create temporary files
        initial_path = self.create_temp_sitemap(initial_sitemap)
        updated_path = self.create_temp_sitemap(updated_sitemap)
        
        # Process both sitemaps
        initial_records = process_sitemap(
            initial_path,
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        updated_records = process_sitemap(
            updated_path,
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        # Verify results
        assert len(initial_records) == 3
        assert len(updated_records) == 2
        
        # Check that page3 was removed
        initial_urls = {data['url'] for data in initial_records.values()}
        updated_urls = {data['url'] for data in updated_records.values()}
        removed_urls = initial_urls - updated_urls
        
        assert len(removed_urls) == 1
        assert 'https://www.wiwi.hu-berlin.de/en/page3' in removed_urls
        
        # Test database operations - the deactivate function should be called
        metrics = CrawlerMetrics()
        
        # Mock that some UIDs will be deactivated
        expected_deactivated_uids = ['uid-page3']
        self.mock_storage.deactivate_old_urls.return_value = expected_deactivated_uids
        
        deactivated_uids = process_page_raw_records(self.mock_storage, updated_records, metrics)
        
        # Verify that deactivated UIDs are returned
        assert deactivated_uids == expected_deactivated_uids
        assert metrics.removed_urls == 1
        
        self.mock_storage.deactivate_old_urls.assert_called_once()
        # Verify that the remaining URLs are in the active list
        active_ids = self.mock_storage.deactivate_old_urls.call_args[0][0]
        assert len(active_ids) == 2
    
    def test_pages_updated_in_sitemap(self):
        """
        Test Scenario: Existing pages have updated lastmod dates.
        Expected: URLs should be updated with new timestamps.
        """
        # Initial sitemap
        initial_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/research</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/teaching</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Updated sitemap with newer timestamps
        updated_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/research</loc>
                <lastmod>2024-02-01T14:20:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/teaching</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Create temporary files
        initial_path = self.create_temp_sitemap(initial_sitemap)
        updated_path = self.create_temp_sitemap(updated_sitemap)
        
        # Process both sitemaps
        initial_records = process_sitemap(
            initial_path,
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        updated_records = process_sitemap(
            updated_path,
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        # Verify same number of URLs
        assert len(initial_records) == 2
        assert len(updated_records) == 2
        
        # Check that research page has updated timestamp
        research_id = None
        for id_hash, data in updated_records.items():
            if 'research' in data['url']:
                research_id = id_hash
                break
        
        assert research_id is not None
        updated_timestamp = updated_records[research_id]['last_updated']
        
        # The timestamp should reflect the newer date
        # (This depends on your date conversion logic)
        assert updated_timestamp is not None
    
    def test_mixed_changes_in_sitemap(self):
        """
        Test Scenario: Sitemap has new pages, removed pages, and updated pages.
        Expected: All changes should be correctly processed.
        """
        # Initial sitemap
        initial_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page1</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/old-page</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Updated sitemap with mixed changes
        updated_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page1</loc>
                <lastmod>2024-02-01T15:45:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/page2</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/new-page</loc>
                <lastmod>2024-02-01T16:00:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Create temporary files
        initial_path = self.create_temp_sitemap(initial_sitemap)
        updated_path = self.create_temp_sitemap(updated_sitemap)
        
        # Process both sitemaps
        initial_records = process_sitemap(
            initial_path,
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        updated_records = process_sitemap(
            updated_path,
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        # Verify counts
        assert len(initial_records) == 3
        assert len(updated_records) == 3
        
        # Analyze changes
        initial_urls = {data['url'] for data in initial_records.values()}
        updated_urls = {data['url'] for data in updated_records.values()}
        
        new_urls = updated_urls - initial_urls
        removed_urls = initial_urls - updated_urls
        common_urls = initial_urls & updated_urls
        
        # Verify changes
        assert new_urls == {'https://www.wiwi.hu-berlin.de/en/new-page'}
        assert removed_urls == {'https://www.wiwi.hu-berlin.de/en/old-page'}
        assert len(common_urls) == 2
        
        # Test database operations
        metrics = CrawlerMetrics()
        
        # Mock that the old-page UID will be deactivated
        expected_deactivated_uids = ['uid-old-page']
        self.mock_storage.deactivate_old_urls.return_value = expected_deactivated_uids
        
        deactivated_uids = process_page_raw_records(self.mock_storage, updated_records, metrics)
        
        # Verify both operations were called
        self.mock_storage.upsert_raw_pages.assert_called_once()
        self.mock_storage.deactivate_old_urls.assert_called_once()
        
        # Verify deactivated UIDs are returned
        assert deactivated_uids == expected_deactivated_uids
        assert metrics.removed_urls == 1
    
    def test_gzipped_sitemap_changes(self):
        """
        Test Scenario: Working with gzipped sitemaps.
        Expected: Gzipped content should be correctly processed.
        """
        sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/compressed-page</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Create gzipped sitemap
        gzipped_path = self.create_temp_sitemap(sitemap_content, gzipped=True)
        
        # Process gzipped sitemap
        records = process_sitemap(
            gzipped_path,
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        # Verify processing worked
        assert len(records) == 1
        assert any('compressed-page' in data['url'] for data in records.values())
    
    def test_filtered_sitemap_changes(self):
        """
        Test Scenario: Sitemap contains URLs that should be filtered out.
        Expected: Only URLs matching filters should be processed.
        """
        sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/valid-page</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/de/deutsch-seite</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/document.pdf</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/view/admin-page</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        sitemap_path = self.create_temp_sitemap(sitemap_content)
        
        # Process with filters
        records = process_sitemap(
            sitemap_path,
            exclude_extensions=['.pdf', '.jpg'],
            exclude_patterns=['view'],
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        # Verify filtering worked
        assert len(records) == 1
        urls = [data['url'] for data in records.values()]
        assert 'https://www.wiwi.hu-berlin.de/en/valid-page' in urls[0]
    
    @patch('hubert.data_ingestion.huber_crawler.sitemap.requests.get')
    def test_remote_sitemap_changes(self, mock_get):
        """
        Test Scenario: Working with remote sitemaps.
        Expected: Remote sitemaps should be downloaded and processed correctly.
        """
        sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.wiwi.hu-berlin.de/en/remote-page</loc>
                <lastmod>2024-01-15T10:30:00Z</lastmod>
            </url>
        </urlset>"""
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = sitemap_content.encode('utf-8')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Process remote sitemap
        records = process_sitemap(
            'https://www.wiwi.hu-berlin.de/sitemap.xml',
            include_patterns=['/en/'],
            allowed_base_url='https://www.wiwi.hu-berlin.de'
        )
        
        # Verify processing worked
        assert len(records) == 1
        mock_get.assert_called_once_with('https://www.wiwi.hu-berlin.de/sitemap.xml', timeout=30)


class TestSitemapMetrics:
    """Test that metrics are correctly tracked during sitemap changes."""
    
    def test_metrics_tracking(self):
        """
        Test that CrawlerMetrics correctly tracks sitemap processing.
        Expected: Metrics should reflect the processing results.
        """
        # Create mock storage
        mock_storage = Mock(spec=PostgresStorage)
        mock_storage.upsert_raw_pages = Mock()
        mock_storage.deactivate_old_urls = Mock(return_value=[])  # Return empty list
        mock_storage.close = Mock()
        
        # Create test records
        test_records = {
            'id1': {'url': 'https://example.com/page1', 'last_updated': datetime.now(timezone.utc)},
            'id2': {'url': 'https://example.com/page2', 'last_updated': datetime.now(timezone.utc)},
        }
        
        # Process records
        metrics = CrawlerMetrics()
        deactivated_uids = process_page_raw_records(mock_storage, test_records, metrics)
        
        # Check metrics
        assert metrics.total_urls_found == 2
        assert metrics.database_update_time > 0
        assert metrics.errors == 0
        assert deactivated_uids == []  # No UIDs should be deactivated in this test
        
        # Convert to dict to verify serialization
        metrics_dict = metrics.to_dict()
        assert 'total_urls_found' in metrics_dict
        assert 'timestamp' in metrics_dict
        assert metrics_dict['total_urls_found'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 