"""
Tests for HybridRetriever functionality.

This module tests the HybridRetriever class logic in isolation, including:
- Vector search retrieval
- Hybrid search (vector + BM25) functionality  
- Cross-encoder reranking
- Score normalization and combination
- Error handling and edge cases
- Configuration parameter handling
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

from hubert.retriever.retriever import HybridRetriever, create_retriever


class TestHybridRetrieverUnit:
    """Unit tests for HybridRetriever with mocked dependencies."""
    
    @pytest.fixture
    def mock_storage(self):
        """Mock PostgresStorage."""
        storage = Mock()
        storage.vector_search = Mock()
        storage.keyword_search = Mock()
        storage.close = Mock()
        return storage
    
    @pytest.fixture  
    def mock_embedding_generator(self):
        """Mock EmbeddingGenerator."""
        generator = Mock()
        generator.generate_embeddings = Mock()
        generator.cleanup = Mock()
        return generator
    
    @pytest.fixture
    def mock_reranker(self):
        """Mock CrossEncoder reranker."""
        reranker = Mock()
        reranker.predict = Mock()
        return reranker
    
    @pytest.fixture
    def retriever_with_mocks(self, mock_storage, mock_embedding_generator):
        """HybridRetriever instance with mocked dependencies."""
        with patch('hubert.retriever.retriever.PostgresStorage') as mock_storage_class:
            with patch('hubert.retriever.retriever.EmbeddingGenerator') as mock_generator_class:
                with patch('hubert.retriever.retriever.CROSS_ENCODER_AVAILABLE', True):
                    mock_storage_class.return_value = mock_storage
                    mock_generator_class.return_value = mock_embedding_generator
                    
                    retriever = HybridRetriever(
                        embedding_model='test-model',
                        embedding_method='openai',
                        table_name='test_table',
                        top_k=5,
                        use_reranker=False,
                        use_hybrid_search=False,
                        hybrid_alpha=0.5
                    )
                    
                    return retriever, mock_storage, mock_embedding_generator
    
    def test_initialization_basic(self):
        """Test basic HybridRetriever initialization."""
        with patch('hubert.retriever.retriever.PostgresStorage'):
            with patch('hubert.retriever.retriever.EmbeddingGenerator'):
                with patch('hubert.retriever.retriever.CROSS_ENCODER_AVAILABLE', False):
                    retriever = HybridRetriever(
                        embedding_model='test-model',
                        top_k=10,
                        use_reranker=False
                    )
                    
                    assert retriever.embedding_model == 'test-model'
                    assert retriever.top_k == 10
                    assert retriever.use_reranker == False
                    assert retriever.reranker is None
    
    def test_initialization_with_reranker(self):
        """Test HybridRetriever initialization with reranker enabled."""
        with patch('hubert.retriever.retriever.PostgresStorage'):
            with patch('hubert.retriever.retriever.EmbeddingGenerator'):
                with patch('hubert.retriever.retriever.CROSS_ENCODER_AVAILABLE', True):
                    with patch('hubert.retriever.retriever.CrossEncoder') as mock_cross_encoder:
                        with patch('hubert.retriever.retriever.torch') as mock_torch:
                            mock_torch.cuda.is_available.return_value = False
                            
                            retriever = HybridRetriever(
                                use_reranker=True,
                                reranker_model='test-reranker'
                            )
                            
                            assert retriever.use_reranker == True
                            mock_cross_encoder.assert_called_once_with('test-reranker', device='cpu')
    
    def test_initialization_reranker_error_handling(self):
        """Test reranker initialization error handling."""
        with patch('hubert.retriever.retriever.PostgresStorage'):
            with patch('hubert.retriever.retriever.EmbeddingGenerator'):
                with patch('hubert.retriever.retriever.CROSS_ENCODER_AVAILABLE', True):
                    with patch('hubert.retriever.retriever.CrossEncoder') as mock_cross_encoder:
                        # Simulate reranker initialization error
                        mock_cross_encoder.side_effect = Exception("Model loading failed")
                        
                        retriever = HybridRetriever(use_reranker=True)
                        
                        # Should gracefully handle error and disable reranker
                        assert retriever.use_reranker == False
                        assert retriever.reranker is None
    
    def test_retrieve_empty_query(self, retriever_with_mocks):
        """Test retrieval with empty query."""
        retriever, mock_storage, mock_embedding_generator = retriever_with_mocks
        
        # Test empty string
        result = retriever.retrieve("")
        assert result == []
        
        # Test whitespace only
        result = retriever.retrieve("   ")
        assert result == []
        
        # Test None
        result = retriever.retrieve(None)
        assert result == []
    
    def test_retrieve_vector_only(self, retriever_with_mocks):
        """Test basic vector search retrieval without hybrid or reranking."""
        retriever, mock_storage, mock_embedding_generator = retriever_with_mocks
        
        # Mock embedding generation
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
        
        # Mock vector search results
        mock_vector_results = [
            {'id': 'doc1', 'url': 'https://example.com/1', 'content': 'Test content 1', 'similarity': 0.95},
            {'id': 'doc2', 'url': 'https://example.com/2', 'content': 'Test content 2', 'similarity': 0.87},
            {'id': 'doc3', 'url': 'https://example.com/3', 'content': 'Test content 3', 'similarity': 0.75}
        ]
        mock_storage.vector_search.return_value = mock_vector_results
        
        result = retriever.retrieve("test query")
        
        # Verify calls
        mock_embedding_generator.generate_embeddings.assert_called_once_with(["test query"])
        mock_storage.vector_search.assert_called_once_with(
            'test_table', [0.1] * 1536, limit=5, filters=None
        )
        
        # Verify results
        assert len(result) == 3
        assert result[0]['id'] == 'doc1'
        assert result[0]['score'] == 0.95  # similarity renamed to score
        assert result[1]['score'] == 0.87
        assert result[2]['score'] == 0.75
    
    def test_retrieve_with_filters(self, retriever_with_mocks):
        """Test retrieval with filters."""
        retriever, mock_storage, mock_embedding_generator = retriever_with_mocks
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
        mock_storage.vector_search.return_value = []
        
        filters = {'category': 'academic', 'year': 2024}
        retriever.retrieve("test query", filters=filters)
        
        mock_storage.vector_search.assert_called_once_with(
            'test_table', [0.1] * 1536, limit=5, filters=filters
        )
    
    def test_retrieve_hybrid_search(self):
        """Test hybrid search (vector + BM25) functionality."""
        with patch('hubert.retriever.retriever.PostgresStorage') as mock_storage_class:
            with patch('hubert.retriever.retriever.EmbeddingGenerator') as mock_generator_class:
                with patch('hubert.retriever.retriever.process_text') as mock_process_text:
                    mock_storage = Mock()
                    mock_embedding_generator = Mock()
                    mock_storage_class.return_value = mock_storage
                    mock_generator_class.return_value = mock_embedding_generator
                    
                    retriever = HybridRetriever(
                        use_hybrid_search=True,
                        hybrid_alpha=0.6,
                        top_k=3
                    )
                    
                    # Mock embedding and text processing
                    mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
                    mock_process_text.return_value = "processed query text"
                    
                    # Mock vector search results
                    mock_vector_results = [
                        {'url': 'https://example.com/1', 'content': 'content1', 'similarity': 0.9},
                        {'url': 'https://example.com/2', 'content': 'content2', 'similarity': 0.8}
                    ]
                    mock_storage.vector_search.return_value = mock_vector_results
                    
                    # Mock BM25 search results
                    mock_bm25_results = [
                        {'url': 'https://example.com/1', 'content': 'content1', 'rank': 0.7},
                        {'url': 'https://example.com/3', 'content': 'content3', 'rank': 0.6}
                    ]
                    mock_storage.keyword_search.return_value = mock_bm25_results
                    
                    result = retriever.retrieve("test query")
                    
                    # Verify both searches were called
                    mock_storage.vector_search.assert_called_once()
                    mock_storage.keyword_search.assert_called_once_with(
                        "processed query text", limit=9, filters=None  # 3 * 3 for hybrid
                    )
                    
                    # Verify hybrid scoring
                    assert len(result) >= 1
                    # Document 1 appears in both searches, should have combined score
                    doc1_result = next((r for r in result if r['url'] == 'https://example.com/1'), None)
                    assert doc1_result is not None
                    # Combined score should be: 0.6 * 0.9 + 0.4 * 0.7 = 0.54 + 0.28 = 0.82
                    assert abs(doc1_result['score'] - 0.82) < 0.01
    
    def test_retrieve_with_reranking(self):
        """Test retrieval with cross-encoder reranking."""
        with patch('hubert.retriever.retriever.PostgresStorage') as mock_storage_class:
            with patch('hubert.retriever.retriever.EmbeddingGenerator') as mock_generator_class:
                with patch('hubert.retriever.retriever.CROSS_ENCODER_AVAILABLE', True):
                    with patch('hubert.retriever.retriever.CrossEncoder') as mock_cross_encoder_class:
                        mock_storage = Mock()
                        mock_embedding_generator = Mock()
                        mock_reranker = Mock()
                        
                        mock_storage_class.return_value = mock_storage
                        mock_generator_class.return_value = mock_embedding_generator
                        mock_cross_encoder_class.return_value = mock_reranker
                        
                        retriever = HybridRetriever(
                            use_reranker=True,
                            top_k=2
                        )
                        
                        # Mock embedding generation
                        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
                        
                        # Mock vector search results
                        mock_vector_results = [
                            {'id': 'doc1', 'content': 'Test content 1', 'similarity': 0.8},
                            {'id': 'doc2', 'content': 'Test content 2', 'similarity': 0.7}
                        ]
                        mock_storage.vector_search.return_value = mock_vector_results
                        
                        # Mock reranker scores (reverse order to test reranking)
                        mock_reranker.predict.return_value = [0.6, 0.9]  # doc2 should rank higher
                        
                        result = retriever.retrieve("test query")
                        
                        # Verify reranker was called with correct pairs
                        mock_reranker.predict.assert_called_once_with([
                            ["test query", "Test content 1"],
                            ["test query", "Test content 2"]
                        ])
                        
                        # Verify results are reranked
                        assert len(result) == 2
                        assert result[0]['id'] == 'doc2'  # Higher reranked score
                        assert result[0]['reranked_score'] == 0.9
                        assert result[1]['id'] == 'doc1'  # Lower reranked score
                        assert result[1]['reranked_score'] == 0.6
    
    def test_retrieval_limits(self):
        """Test retrieval limit calculations for different configurations."""
        test_cases = [
            # (use_hybrid, use_reranker, top_k, expected_limit)
            (False, False, 5, 5),      # Basic: top_k
            (False, True, 5, 10),      # Reranker only: top_k * 2  
            (True, False, 5, 15),      # Hybrid only: top_k * 3
            (True, True, 5, 15),       # Both: top_k * 3
        ]
        
        for use_hybrid, use_reranker, top_k, expected_limit in test_cases:
            with patch('hubert.retriever.retriever.PostgresStorage') as mock_storage_class:
                with patch('hubert.retriever.retriever.EmbeddingGenerator') as mock_generator_class:
                    mock_storage = Mock()
                    mock_embedding_generator = Mock()
                    mock_storage_class.return_value = mock_storage
                    mock_generator_class.return_value = mock_embedding_generator
                    
                    retriever = HybridRetriever(
                        use_hybrid_search=use_hybrid,
                        use_reranker=use_reranker,
                        top_k=top_k
                    )
                    
                    mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
                    mock_storage.vector_search.return_value = []
                    
                    retriever.retrieve("test")
                    
                    # Check the limit passed to vector_search
                    call_args = mock_storage.vector_search.call_args
                    assert call_args[1]['limit'] == expected_limit
    
    def test_cleanup(self, retriever_with_mocks):
        """Test cleanup functionality."""
        retriever, mock_storage, mock_embedding_generator = retriever_with_mocks
        
        retriever.cleanup()
        
        mock_storage.close.assert_called_once()
        mock_embedding_generator.cleanup.assert_called_once()
    
    def test_retrieval_timing_logging(self, retriever_with_mocks):
        """Test that retrieval timing is logged."""
        retriever, mock_storage, mock_embedding_generator = retriever_with_mocks
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
        mock_storage.vector_search.return_value = []
        
        with patch('hubert.retriever.retriever.logger') as mock_logger:
            retriever.retrieve("test query")
            
            # Should log timing information
            mock_logger.info.assert_called()
            logged_message = mock_logger.info.call_args[0][0]
            assert "Retrieval took" in logged_message
            assert "seconds" in logged_message


class TestHybridRetrieverEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_reranker_not_available(self):
        """Test behavior when sentence-transformers is not available."""
        with patch('hubert.retriever.retriever.PostgresStorage'):
            with patch('hubert.retriever.retriever.EmbeddingGenerator'):
                with patch('hubert.retriever.retriever.CROSS_ENCODER_AVAILABLE', False):
                    retriever = HybridRetriever(use_reranker=True)
                    
                    # Should disable reranker when not available
                    assert retriever.use_reranker == True  # Config remains true
                    assert retriever.reranker is None      # But reranker is None
    
    def test_empty_search_results(self, retriever_with_mocks):
        """Test handling of empty search results."""
        retriever, mock_storage, mock_embedding_generator = retriever_with_mocks
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
        mock_storage.vector_search.return_value = []
        
        result = retriever.retrieve("test query")
        
        assert result == []
    
    def test_reranker_with_empty_results(self):
        """Test reranker behavior with empty results."""
        with patch('hubert.retriever.retriever.PostgresStorage') as mock_storage_class:
            with patch('hubert.retriever.retriever.EmbeddingGenerator') as mock_generator_class:
                with patch('hubert.retriever.retriever.CROSS_ENCODER_AVAILABLE', True):
                    with patch('hubert.retriever.retriever.CrossEncoder') as mock_cross_encoder_class:
                        mock_storage = Mock()
                        mock_embedding_generator = Mock()
                        mock_reranker = Mock()
                        
                        mock_storage_class.return_value = mock_storage
                        mock_generator_class.return_value = mock_embedding_generator
                        mock_cross_encoder_class.return_value = mock_reranker
                        
                        retriever = HybridRetriever(use_reranker=True)
                        
                        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
                        mock_storage.vector_search.return_value = []
                        
                        result = retriever.retrieve("test query")
                        
                        # Reranker should not be called with empty results
                        mock_reranker.predict.assert_not_called()
                        assert result == []
    
    def test_score_normalization_edge_cases(self):
        """Test score normalization with edge cases."""
        with patch('hubert.retriever.retriever.PostgresStorage') as mock_storage_class:
            with patch('hubert.retriever.retriever.EmbeddingGenerator') as mock_generator_class:
                with patch('hubert.retriever.retriever.process_text') as mock_process_text:
                    mock_storage = Mock()
                    mock_embedding_generator = Mock()
                    mock_storage_class.return_value = mock_storage
                    mock_generator_class.return_value = mock_embedding_generator
                    mock_process_text.return_value = "processed"
                    
                    retriever = HybridRetriever(
                        use_hybrid_search=True,
                        hybrid_alpha=0.5
                    )
                    
                    mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 1536]
                    
                    # Test with zero scores
                    mock_storage.vector_search.return_value = [
                        {'url': 'url1', 'content': 'content1', 'similarity': 0.0}
                    ]
                    mock_storage.keyword_search.return_value = [
                        {'url': 'url1', 'content': 'content1', 'rank': 0.0}
                    ]
                    
                    result = retriever.retrieve("test")
                    
                    assert len(result) == 1
                    assert result[0]['score'] == 0.0  # 0.5 * 0.0 + 0.5 * 0.0


class TestHybridRetrieverFactory:
    """Test factory function and configuration."""
    
    def test_create_retriever_factory(self):
        """Test the create_retriever factory function."""
        with patch('hubert.retriever.retriever.HybridRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value = mock_retriever
            
            result = create_retriever(use_full_content=True)
            
            mock_retriever_class.assert_called_once_with(use_full_content=True)
            assert result == mock_retriever
    
    def test_parameter_validation(self):
        """Test parameter validation and defaults."""
        with patch('hubert.retriever.retriever.PostgresStorage'):
            with patch('hubert.retriever.retriever.EmbeddingGenerator'):
                retriever = HybridRetriever(
                    hybrid_alpha=0.7,
                    top_k=15
                )
                
                assert retriever.hybrid_alpha == 0.7
                assert retriever.top_k == 15


# Integration-style test with minimal mocking
class TestHybridRetrieverIntegration:
    """Integration tests with minimal mocking to test component interaction."""
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires configured database and embedding service")
    def test_end_to_end_retrieval(self):
        """End-to-end test with real components (requires proper setup)."""
        # This would test with real PostgresStorage, EmbeddingGenerator, etc.
        # Skipped for unit testing but shows how integration tests would work
        retriever = HybridRetriever()
        try:
            results = retriever.retrieve("What are the admission requirements?")
            assert isinstance(results, list)
        finally:
            retriever.cleanup() 