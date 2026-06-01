"""
Tests for evaluation metrics.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.retrieval_metrics import RetrievalMetrics


class TestRetrievalMetrics:
    """Test retrieval evaluation metrics."""
    
    def test_recall_at_k(self):
        """Test Recall@k calculation."""
        retrieved = ['p1', 'p2', 'p3', 'p4', 'p5']
        relevant = {'p1', 'p3', 'p5', 'p6'}
        
        recall_5 = RetrievalMetrics.recall_at_k(retrieved, relevant, k=5)
        assert 0 <= recall_5 <= 1
        assert recall_5 == 3/4  # 3 out of 4 relevant found
    
    def test_precision_at_k(self):
        """Test Precision@k calculation."""
        retrieved = ['p1', 'p2', 'p3']
        relevant = {'p1', 'p3', 'p4'}
        
        precision_3 = RetrievalMetrics.precision_at_k(retrieved, relevant, k=3)
        assert 0 <= precision_3 <= 1
        assert precision_3 == 2/3  # 2 out of 3 retrieved are relevant
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        retrieved = ['p1', 'p2', 'p3']
        relevant = {'p2', 'p4'}
        
        mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant)
        assert 0 <= mrr <= 1
        assert mrr == 0.5  # First relevant at position 2, so 1/2
