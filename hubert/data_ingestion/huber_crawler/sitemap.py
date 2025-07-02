import requests
import logging
import io
import gzip
import re
import xml.etree.ElementTree as ET
import sys
from pathlib import Path


from ..utils.crawl_utils import with_progress, convert_to_date, create_id_for_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle settings import safelys
try:
    from ..config import settings
    logger.info("Successfully imported settings from package")
except ImportError:
    logger.warning("Could not import settings from package, attempting alternative imports")
    try:
        # Try importing from parent directory
        sys.path.append(str(Path(__file__).parent.parent))
        from config import settings
        logger.info("Successfully imported settings from parent directory")
    except ImportError:
        logger.warning("Could not import settings, creating default settings")
        # Create a minimal settings object as fallback
        class DefaultSettings:
            url = 'https://www.example.com/sitemap.xml'
            pattern = r'<url>\s*<loc>(.*?)</loc>\s*<lastmod>(.*?)</lastmod>'
            exclude_extensions = ['.jpg', '.pdf', '.jpeg', '.png']
            exclude_patterns = ['view']
            include_patterns = ['/en/']
            allowed_base_url = 'https://www.example.com'
        
        settings = DefaultSettings()
        logger.info("Using default settings")

@with_progress
def download_sitemap_file(file_path):
    """
    Downloads a sitemap file from the given URL and saves it to a temporary location.

    Args:
        file_path (str): The path to the sitemap file to download.

    Returns:
        bytes: The unzipped content of the file.
    
    Raises:
        requests.exceptions.RequestException: If there's an issue with the HTTP request
        IOError: If there's an issue reading the file
    """
    logger = logging.getLogger(__name__)
    
    if file_path.startswith('https://') or file_path.startswith('http://'):
        try:
            response = requests.get(file_path, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully downloaded sitemap from {file_path}")
            
            # Check if the file is gzipped
            if file_path.endswith('.gz'):
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gzipfile:
                        content = gzipfile.read()
                    logger.info("Unzipped gzipped content")
                except IOError as e:
                    logger.error(f"Failed to unzip content: {e}")
                    raise
            else:
                content = response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download sitemap from {file_path}: {e}")
            raise
    else:
        # Local file
        try:
            logger.info(f"Reading sitemap from local file: {file_path}")
            # Check if the file is gzipped
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as gzipfile:
                    content = gzipfile.read()
                logger.info("Unzipped gzipped content")
            else:
                with open(file_path, 'rb') as file:
                    content = file.read()
        except IOError as e:
            logger.error(f"Failed to read local file {file_path}: {e}")
            raise
            
    return content


@with_progress
def parse_sitemap(content, pattern=None):
    """
    Parses the sitemap content using XML parsing or the provided pattern.

    This function takes the sitemap content and optionally a pattern,
    and returns a list of dictionaries with the URL and last modified date.
    """
    if isinstance(content, bytes):
        content_str = content.decode('utf-8')
    else:
        content_str = content
    
    # Try XML parsing first (more reliable)
    try:
        root = ET.fromstring(content_str)
        # Define namespace if present in the XML
        ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        entries = []
        for url in root.findall('.//sm:url', ns):
            loc = url.find('sm:loc', ns)
            lastmod = url.find('sm:lastmod', ns)
            
            if loc is not None:
                entry = {'url': loc.text.strip()}
                if lastmod is not None:
                    entry['lastmod'] = lastmod.text.strip()
                else:
                    entry['lastmod'] = None
                entries.append(entry)
        
        # If we found entries, return them
        if entries:
            return entries
    except Exception as e:
        print(f"XML parsing failed: {e}. Falling back to regex pattern.")
    
    # Fallback to regex pattern if XML parsing fails or no entries found
    if pattern:
        matches = re.findall(pattern, content_str, re.DOTALL)
        return [{'url': url.strip(), 'lastmod': lastmod.strip()} for url, lastmod in matches]
    else:
        # Default pattern for standard sitemaps
        default_pattern = r'<loc>(.*?)</loc>.*?<lastmod>(.*?)</lastmod>'
        matches = re.findall(default_pattern, content_str, re.DOTALL)
        return [{'url': url.strip(), 'lastmod': lastmod.strip()} for url, lastmod in matches]


@with_progress
def filter_sitemap_entries(entries, exclude_extensions=None, exclude_patterns=None, include_patterns=None):
    """Filters sitemap entries in a single pass."""
    result = []
    
    for entry in entries:
        url = entry['url']
        
        # Check exclusions first (fail fast)
        if exclude_extensions and any(url.lower().endswith(ext.lower()) for ext in exclude_extensions):
            continue
            
        if exclude_patterns and any(pattern in url for pattern in exclude_patterns):
            continue
            
        # Check inclusions (must match at least one if provided)
        if include_patterns and not any(pattern in url for pattern in include_patterns):
            continue
            
        # Entry passed all filters
        result.append(entry)
        
    return result

@with_progress
def security_check_urls(entries, allowed_base_url):
    """Enhanced security check for URLs."""
    safe = []
    unsafe = []
    
    for entry in entries:
        url = entry['url']
        # Check base URL
        is_safe = url.startswith(allowed_base_url)
        
        #TODO: Add additional checks
        # Additional checks could be added:
        # - Check for javascript: protocol
        # - Validate URL format
        # - Check for suspicious patterns
        
        if is_safe:
            safe.append(entry)
        else:
            unsafe.append(entry)
            
    return safe, unsafe


def create_matches_dict(entries):
    """
    Creates a dictionary with the URL and last modified date.

    This function takes a list of sitemap entries,
    and returns a dictionary with the URL and last modified date.
    """
    result = {}
    for entry in entries:
        if 'url' in entry:
            url_hash = create_id_for_url(entry['url'])
            result[url_hash] = {
                'url': entry['url'],
                'last_updated': convert_to_date(entry.get('lastmod'))
            }
    return result

@with_progress
def process_sitemap(url, pattern=None, exclude_extensions=None, exclude_patterns=None, 
                    include_patterns=None, allowed_base_url=None):
    """
    Processes the sitemap and returns a dictionary with the URL and last modified date.
    """
    logger.info(f"Starting to process sitemap: {url}")
    
    try:
        content = download_sitemap_file(url)
        logger.info(f"Downloaded sitemap content: {len(content)} bytes")
        
        sitemap_entries = parse_sitemap(content, pattern)
        logger.info(f"Parsed sitemap entries: {len(sitemap_entries)}")
        
        filtered_entries = filter_sitemap_entries(sitemap_entries, exclude_extensions, exclude_patterns, include_patterns)
        logger.info(f"Filtered sitemap entries: {len(filtered_entries)}")
        
        if allowed_base_url:
            safe, unsafe = security_check_urls(filtered_entries, allowed_base_url)
            logger.info(f"Safe URLs: {len(safe)}")
            logger.info(f"Unsafe URLs: {len(unsafe)}")
            matches_dict = create_matches_dict(safe)
        else:
            matches_dict = create_matches_dict(filtered_entries)
        
        logger.info(f"Final URL count: {len(matches_dict)}")
        return matches_dict
    
    except Exception as e:
        logger.error(f"Error in process_sitemap: {e}", exc_info=True)
        raise