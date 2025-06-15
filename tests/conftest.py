"""
Pytest configuration and common fixtures for sitemap testing.
"""

import pytest
import os
import sys
from unittest.mock import Mock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hubert.db.postgres_storage import PostgresStorage


@pytest.fixture
def mock_storage():
    """Fixture providing a mocked PostgresStorage instance."""
    storage = Mock(spec=PostgresStorage)
    storage.upsert_raw_pages = Mock()
    storage.deactivate_old_urls = Mock()
    storage.close = Mock()
    storage.connect = Mock()
    return storage


@pytest.fixture
def sample_xml_sitemap():
    """Fixture providing a sample XML sitemap."""
    return """<?xml version="1.0" encoding="UTF-8"?>
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


@pytest.fixture
def test_settings():
    """Fixture providing test settings for sitemap processing."""
    class TestSettings:
        url = 'https://www.wiwi.hu-berlin.de/sitemap.xml.gz'
        pattern = r'<url>\s*<loc>(.*?)</loc>\s*<lastmod>(.*?)</lastmod>'
        exclude_extensions = ['.jpg', '.pdf', '.jpeg', '.png']
        exclude_patterns = ['view']
        include_patterns = ['/en/']
        allowed_base_url = 'https://www.wiwi.hu-berlin.de'
    
    return TestSettings()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    ) 