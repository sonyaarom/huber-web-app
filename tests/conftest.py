"""
Pytest configuration and common fixtures for sitemap testing.
"""

import pytest
import os
import sys
from unittest.mock import Mock, MagicMock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hubert.db.postgres_storage import PostgresStorage
from ui.app import create_app
from hubert.db.models import User


@pytest.fixture(scope='module')
def app():
    """Create and configure a new app instance for each test module."""
    app = create_app()
    app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,  # Disable CSRF for testing forms
        "SECRET_KEY": "test-secret-key",
    })
    yield app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def mock_storage():
    """Fixture providing a mocked PostgresStorage instance."""
    storage = Mock(spec=PostgresStorage)
    storage.upsert_raw_pages = Mock()
    storage.deactivate_old_urls = Mock()
    storage.close = Mock()
    storage.connect = Mock()
    storage.get_user_by_username = Mock()
    storage.get_user_by_id = Mock()
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


@pytest.fixture
def mock_user():
    """Fixture for a standard mock user."""
    user = MagicMock(spec=User)
    user.id = 1
    user.username = 'testuser'
    user.role = 'user'
    user.is_authenticated = True
    user.is_active = True
    user.is_anonymous = False
    user.is_admin = False
    user.get_id.return_value = '1'
    return user


@pytest.fixture
def mock_admin():
    """Fixture for a mock admin user."""
    admin = MagicMock(spec=User)
    admin.id = 2
    admin.username = 'admin'
    admin.role = 'admin'
    admin.is_authenticated = True
    admin.is_active = True
    admin.is_anonymous = False
    admin.is_admin = True
    admin.get_id.return_value = '2'
    return admin


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    ) 