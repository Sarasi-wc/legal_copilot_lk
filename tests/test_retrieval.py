"""
Unit tests for retrieval components.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class TestBM25Retriever:
    """Test BM25 retrieval."""
    
    def test_build_index(self):
        """Test index building."""
        retriever = BM25Retriever()
        passages = [
            {
                'passage_id': 'p1',
                'text': 'The Penal Code defines theft as...',
                'metadata': {}
            },
            {
                'passage_id': 'p2',
                'text': 'Murder is defined in Section 296...',
                'metadata': {}
            }
        ]
        retriever.build_index(passages)
        assert retriever.bm25 is not None
    
    def test_retrieve(self):
        """Test retrieval."""
        retriever = BM25Retriever()
        passages = [
            {
                'passage_id': 'p1',
                'text': 'The Penal Code defines theft as...',
                'metadata': {}
            }
        ]
        retriever.build_index(passages)
        results = retriever.retrieve('theft', top_k=1)
        assert len(results) > 0
        assert results[0][0] == 'p1'


class TestDenseRetriever:
    """Test dense retrieval."""
    
    def test_build_index(self):
        """Test index building."""
        retriever = DenseRetriever()
        passages = [
            {
                'passage_id': 'p1',
                'text': 'The Penal Code defines theft...',
                'metadata': {}
            }
        ]
        retriever.build_index(passages)
        assert retriever.index is not None


class TestHybridRetriever:
    """Test hybrid retrieval."""
    
    def test_initialization(self):
        """Test retriever initialization."""
        retriever = HybridRetriever()
        assert retriever.bm25_retriever is not None
        assert retriever.dense_retriever is not None
        assert retriever.reranker is not None
    
    def test_retrieve_methods(self):
        """Test different retrieval methods."""
        retriever = HybridRetriever()
        passages = [
            {
                'passage_id': 'p1',
                'text': 'Theft is defined in Section 365...',
                'metadata': {}
            }
        ]
        retriever.build_indices(passages)
        
        # Test BM25
        results = retriever.retrieve('theft', top_k=1, method='bm25')
        assert len(results) > 0
        
        # Test dense
        results = retriever.retrieve('theft', top_k=1, method='dense')
        assert len(results) > 0
        
        # Test hybrid
        results = retriever.retrieve('theft', top_k=1, method='hybrid')
        assert len(results) > 0
