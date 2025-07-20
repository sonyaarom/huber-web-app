"""
Utility functions for crawling operations.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


def with_progress(func: Callable) -> Callable:
    """
    Decorator to add progress logging to functions.
    
    This decorator logs when a function starts and finishes,
    providing basic progress tracking for long-running operations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Starting {func_name}...")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completed {func_name}")
            return result
        except Exception as e:
            logger.error(f"Error in {func_name}: {e}")
            raise
    
    return wrapper


def convert_to_date(date_string: Optional[str]) -> Optional[datetime]:
    """
    Convert a date string to a datetime object.
    
    Args:
        date_string: ISO format date string (e.g., '2024-01-15T10:30:00Z')
    
    Returns:
        datetime object or None if conversion fails
    """
    if not date_string:
        return None
    
    try:
        # Clean up date strings with double timezone info
        date_string = re.sub(r'([+-]\d{2}:\d{2})\+00:00$', r'\1', date_string)

        # Handle various ISO date formats
        # Remove timezone info if present and add UTC
        if date_string.endswith('Z'):
            date_string = date_string[:-1] + '+00:00'
        
        # Parse the date
        dt = datetime.fromisoformat(date_string)
        
        # Ensure timezone awareness
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        return dt
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert date string '{date_string}': {e}")
        return None


def create_id_for_url(url: str) -> str:
    """
    Create a unique ID for a URL using SHA-256 hash.
    
    Args:
        url: The URL to create an ID for
    
    Returns:
        Hexadecimal hash string (first 16 characters)
    """
    if not url:
        raise ValueError("URL cannot be empty")
    
    # Create SHA-256 hash of the URL
    hash_object = hashlib.sha256(url.encode('utf-8'))
    
    # Return first 16 characters of the hex digest for shorter IDs
    return hash_object.hexdigest()[:16]


def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing trailing slashes and query parameters.
    
    Args:
        url: URL to normalize
    
    Returns:
        Normalized URL
    """
    if not url:
        return url
    
    # Remove trailing slash
    if url.endswith('/') and len(url) > 1:
        url = url[:-1]
    
    # Remove fragment identifier
    if '#' in url:
        url = url.split('#')[0]
    
    return url.strip()


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and safe.
    
    Args:
        url: URL to validate
    
    Returns:
        True if URL is valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # Basic URL validation
    if not (url.startswith('http://') or url.startswith('https://')):
        return False
    
    # Check for dangerous protocols
    dangerous_protocols = ['javascript:', 'data:', 'file:', 'ftp:']
    for protocol in dangerous_protocols:
        if url.lower().startswith(protocol):
            return False
    
    return True