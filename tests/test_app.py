import pytest
from flask import url_for
from unittest.mock import patch, MagicMock

def test_landing_page(client):
    """Test that the landing page is accessible without login."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome to HUBer' in response.data

def test_login_page(client):
    """Test that the login page loads correctly."""
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Sign In' in response.data

def test_register_page(client):
    """Test that the register page loads correctly."""
    response = client.get('/register')
    assert response.status_code == 200
    assert b'Register' in response.data

@pytest.mark.parametrize('page', ['/chat', '/search', '/evaluation', '/config'])
def test_protected_pages_unauthorized(client, page):
    """Test that protected pages redirect to login when not logged in."""
    response = client.get(page)
    assert response.status_code == 302
    assert '/login' in response.location

