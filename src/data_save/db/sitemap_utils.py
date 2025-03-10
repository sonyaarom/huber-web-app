import requests
import io
import gzip
import re
import hashlib
from datetime import datetime
import pandas as pd 
import xml.etree.ElementTree as ET
from config import settings

def download_sitemap_file(file_path):
    """
    Downloads a sitemap file from the given URL and saves it to a temporary location.

    Args:
        file_path (str): The path to the sitemap file to download.

    Returns:
        bytes: The unzipped content of the file.
    """
    if file_path.startswith('https://'):
        response = requests.get(file_path)
        # Check if the file is gzipped
        if file_path.endswith('.gz'):
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gzipfile:
                content = gzipfile.read()
        else:
            content = response.content
    else:
        # Check if the file is gzipped
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rb') as gzipfile:
                content = gzipfile.read()
        else:
            with open(file_path, 'rb') as file:
                content = file.read()
    return content

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


def filter_sitemap_entries(entries, exclude_extensions=None, exclude_patterns=None, include_patterns=None):
    """
    Filters the sitemap entries based on the provided criteria.

    This function takes a list of sitemap entries and optional filters,
    and returns a filtered list of entries.
    """
    filtered = entries
    if exclude_extensions:
        filtered = [entry for entry in filtered 
                    if not any(entry['url'].lower().endswith(ext.lower()) for ext in exclude_extensions)]
    if exclude_patterns:
        filtered = [entry for entry in filtered 
                    if not any(pattern in entry['url'] for pattern in exclude_patterns)]
    if include_patterns:
        filtered = [entry for entry in filtered 
                    if any(pattern in entry['url'] for pattern in include_patterns)]
    return filtered


def security_check_urls(entries, allowed_base_url):
    """
    Checks the security of the URLs.

    This function takes a list of sitemap entries and an allowed base URL,
    and returns two lists: safe and unsafe URLs.
    """
    safe = [entry for entry in entries if entry['url'].startswith(allowed_base_url)]
    unsafe = [entry for entry in entries if not entry['url'].startswith(allowed_base_url)]
    return safe, unsafe

def convert_to_date(datetime_string, format=None):
    """
    Converts the datetime string to a date string.

    This function takes a datetime string and an optional format,
    and returns a date string.
    """
    if datetime_string is None:
        return None
    
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
        print(f"Warning: Could not parse date '{datetime_string}'")
        return datetime_string
    else:
        try:
            return datetime.strptime(datetime_string, format).strftime("%Y-%m-%d")
        except ValueError:
            print(f"Warning: Could not parse date '{datetime_string}' with format '{format}'")
            return datetime_string

def create_matches_dict(entries):
    """
    Creates a dictionary with the URL and last modified date.

    This function takes a list of sitemap entries,
    and returns a dictionary with the URL and last modified date.
    """
    result = {}
    for entry in entries:
        if 'url' in entry:
            url_hash = hashlib.md5(entry['url'].encode('utf-8')).hexdigest()
            result[url_hash] = {
                'url': entry['url'],
                'last_updated': convert_to_date(entry.get('lastmod'))
            }
    return result

def process_sitemap(url, pattern=None, exclude_extensions=None, exclude_patterns=None, include_patterns=None, allowed_base_url=None):
    """
    Processes the sitemap and returns a dictionary with the URL and last modified date.
    """
    content = download_sitemap_file(url)
    print(f"Downloaded sitemap content: {len(content)} bytes")
    
    sitemap_entries = parse_sitemap(content, pattern)
    print(f"Parsed sitemap entries: {len(sitemap_entries)}")
    
    filtered_entries = filter_sitemap_entries(sitemap_entries, exclude_extensions, exclude_patterns, include_patterns)
    print(f"Filtered sitemap entries: {len(filtered_entries)}")
    
    if allowed_base_url:
        safe, unsafe = security_check_urls(filtered_entries, allowed_base_url)
        print(f"Safe URLs: {len(safe)}")
        print(f"Unsafe URLs: {len(unsafe)}")
        matches_dict = create_matches_dict(safe)
    else:
        matches_dict = create_matches_dict(filtered_entries)
    
    return matches_dict



# if __name__ == "__main__":
#     url = settings.url
#     pattern = getattr(settings, 'pattern', None)
#     exclude_extensions = getattr(settings, 'exclude_extensions', None)
#     exclude_patterns = getattr(settings, 'exclude_patterns', None)
#     include_patterns = getattr(settings, 'include_patterns', None)
#     allowed_base_url = getattr(settings, 'allowed_base_url', None)
    
#     matches_dict = process_sitemap(url, pattern, exclude_extensions, exclude_patterns, include_patterns, allowed_base_url)
#     #print first 10 entries
#     print(list(matches_dict.values())[:10])