import requests
import io
import gzip
import re
import hashlib
from datetime import datetime
import pandas as pd 

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
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gzipfile:
            content = gzipfile.read()
    else:
        with gzip.open(file_path, 'rb') as gzipfile:
            content = gzipfile.read()
    return content

def parse_sitemap(content, pattern):
    """
    Parses the sitemap content using the provided pattern.

    This function takes the sitemap content and a pattern,
    and returns a list of dictionaries with the URL and last modified date.
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    matches = re.findall(pattern, content, re.DOTALL)
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

def convert_to_date(datetime_string, format="%Y-%m-%dT%H:%M:%S%z"):
    """
    Converts the datetime string to a date string.

    This function takes a datetime string and a format,
    and returns a date string.
    """
    return datetime.strptime(datetime_string, format).strftime("%Y-%m-%d")

def create_matches_dict(entries):
    """
    Creates a dictionary with the URL and last modified date.

    This function takes a list of sitemap entries,
    and returns a dictionary with the URL and last modified date.
    """
    return {
        hashlib.md5(entry['url'].encode('utf-8')).hexdigest(): {
            'url': entry['url'],
            'last_updated': convert_to_date(entry['lastmod'])
        }
        for entry in entries
    }

def process_sitemap(url, pattern, exclude_extensions, exclude_patterns, include_patterns, allowed_base_url):
    """
    Processes the sitemap and returns a dictionary with the URL and last modified date.
    """
    content = download_sitemap_file(url)
    sitemap_entries = parse_sitemap(content, pattern)
    filtered_entries = filter_sitemap_entries(sitemap_entries, exclude_extensions, exclude_patterns, include_patterns)
    safe, unsafe = security_check_urls(filtered_entries, allowed_base_url)
    matches_dict = create_matches_dict(safe)
    return matches_dict
