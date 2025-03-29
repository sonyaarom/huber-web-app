import logging
import time
from datetime import datetime
import hashlib
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def with_progress(func):
    """Decorator to add progress reporting to functions."""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"Completed {func_name} in {elapsed:.2f} seconds")
        return result
    return wrapper


def convert_to_date(datetime_string, format=None) -> str:
    """
    Converts the datetime string to a date string.

    Args:
        datetime_string: The datetime string to convert
        format: Optional format string for datetime parsing
        
    Returns:
        str: The formatted date string or None if conversion failed
    """
    logger = logging.getLogger(__name__)
    
    if datetime_string is None:
        return None
    
    # Try using dateutil if available for more robust parsing
    try:
        from dateutil import parser
        try:
            return parser.parse(datetime_string).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            logger.warning(f"dateutil failed to parse date '{datetime_string}'")
            # Continue with manual parsing
    except ImportError:
        # dateutil not available, continue with manual parsing
        pass
        
    # Try different formats if none is specified
    if format is None:
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",  # ISO format with timezone
            "%Y-%m-%dT%H:%M:%S+%z",  # ISO format with + timezone
            "%Y-%m-%dT%H:%M:%S-%z",  # ISO format with - timezone
            "%Y-%m-%dT%H:%M:%S",     # ISO format without timezone
            "%Y-%m-%d",              # Just date
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(datetime_string, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If all formats fail, return the original string
        logger.warning(f"Could not parse date '{datetime_string}'")
        return datetime_string
    else:
        try:
            return datetime.strptime(datetime_string, format).strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"Could not parse date '{datetime_string}' with format '{format}'")
            return datetime_string
        


def create_id_for_url(url, algorithm='md5') -> str:
    """Create a hash ID for a URL with configurable algorithm."""
    if algorithm == 'md5':
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(url.encode('utf-8')).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(url.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
