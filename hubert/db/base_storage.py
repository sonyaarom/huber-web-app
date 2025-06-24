# hubert/db/base_storage.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseStorage(ABC):
    """
    Abstract base class defining the interface for all storage operations.
    This acts as a contract for any concrete storage implementation.
    """

    @abstractmethod
    def connect(self):
        """Establish a connection to the data store."""
        pass

    @abstractmethod
    def close(self):
        """Close the connection to the data store."""
        pass

    @abstractmethod
    def get_urls_to_process(self, limit: int = None) -> List[Tuple[str, str, str]]:
        """Fetch URLs that need their content to be scraped."""
        pass

    @abstractmethod
    def upsert_raw_pages(self, records: List[Dict[str, Any]]):
        """Insert or update raw page metadata (URL, last_modified)."""
        pass

    @abstractmethod
    def deactivate_old_urls(self, current_ids: List[str]) -> List[str]:
        """Mark URLs no longer present in the sitemap as inactive.
        
        Args:
            current_ids: List of UIDs that are currently active in the sitemap
            
        Returns:
            List of UIDs that were deactivated
        """
        pass

    @abstractmethod
    def upsert_page_content(self, content_records: List[Dict[str, Any]]):
        """Insert or update extracted HTML content for pages."""
        pass

    @abstractmethod
    def get_content_to_process_for_keywords(self) -> List[Tuple[str, str]]:
        """Fetch content that needs keyword processing."""
        pass

    @abstractmethod
    def upsert_keywords(self, keyword_records: List[Dict[str, Any]]):
        """Insert or update tsvector keywords for content."""
        pass
    
    @abstractmethod
    def get_content_to_process_for_embeddings(self, table_name: str) -> List[Tuple[str, str]]:
        """Fetch content that needs embedding."""
        pass

    @abstractmethod
    def upsert_embeddings(self, table_name: str, embedding_records: List[Dict[str, Any]]):
        """Insert or update text chunks and their vector embeddings."""
        pass

    @abstractmethod
    def keyword_search(self, query_text: str, limit: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform a full-text search."""
        pass

    @abstractmethod
    def vector_search(self, table_name: str, query_embedding: List[float], limit: int = 5, filters: Dict[str, Any] = None, threshold: float = None) -> List[Dict[str, Any]]:
        """Perform a vector similarity search."""
        pass
    
    @abstractmethod
    def purge_inactive_records(self):
        """Delete all records associated with inactive URLs."""
        pass

    @abstractmethod
    def purge_specific_inactive_records(self, inactive_uids: List[str]) -> int:
        """Immediately purge specific inactive UIDs from all related tables.
        
        Args:
            inactive_uids: List of UIDs to purge from all tables
            
        Returns:
            Total number of records deleted across all tables
        """
        pass

    @abstractmethod
    def log_failed_job(self, uid: str, job_type: str, error: str = ""):
        """Logs a failed job to the failed_jobs table."""
        pass

    @abstractmethod
    def get_embedding_tables(self) -> List[str]:
        """Get all embedding table names."""
        pass 