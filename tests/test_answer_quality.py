"""
Tests for answer quality evaluation metrics.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.answer_quality_metrics import AnswerQualityMetrics, AnswerQualityScore


class TestAnswerQualityMetrics:
    """Test answer quality evaluation metrics."""
    
    def test_create_evaluation_rubric(self):
        """Test rubric creation."""
        rubric = AnswerQualityMetrics.create_evaluation_rubric()
        
        assert 'correctness' in rubric
        assert 'completeness' in rubric
        assert 'clarity' in rubric
        
        # Check correctness scale
        assert 0 in rubric['correctness']
        assert 5 in rubric['correctness']
        assert rubric['correctness'][5] == 'Completely correct, no errors'
        
        # Check completeness scale
        assert 0 in rubric['completeness']
        assert 5 in rubric['completeness']
        
        # Check clarity scale
        assert 0 in rubric['clarity']
        assert 5 in rubric['clarity']
    
    def test_evaluate_answer_quality(self):
        """Test answer quality evaluation."""
        # Test perfect answer
        score = AnswerQualityMetrics.evaluate_answer_quality(
            correctness_score=5,
            completeness_score=5,
            clarity_score=5
        )
        
        assert isinstance(score, AnswerQualityScore)
        assert score.correctness == 5.0
        assert score.completeness == 5.0
        assert score.clarity == 5.0
        assert score.overall == 5.0
        
        # Test average answer
        score = AnswerQualityMetrics.evaluate_answer_quality(
            correctness_score=3,
            completeness_score=3,
            clarity_score=3
        )
        
        assert score.correctness == 3.0
        assert score.completeness == 3.0
        assert score.clarity == 3.0
        assert score.overall == 3.0
        
        # Test mixed scores
        score = AnswerQualityMetrics.evaluate_answer_quality(
            correctness_score=4,
            completeness_score=3,
            clarity_score=5
        )
        
        assert score.correctness == 4.0
        assert score.completeness == 3.0
        assert score.clarity == 5.0
        assert score.overall == 4.0  # (4+3+5)/3
    
    def test_score_bounds(self):
        """Test that scores are within valid bounds."""
        # Test minimum scores
        score = AnswerQualityMetrics.evaluate_answer_quality(0, 0, 0)
        assert score.correctness == 0.0
        assert score.completeness == 0.0
        assert score.clarity == 0.0
        assert score.overall == 0.0
        
        # Test maximum scores
        score = AnswerQualityMetrics.evaluate_answer_quality(5, 5, 5)
        assert score.correctness == 5.0
        assert score.completeness == 5.0
        assert score.clarity == 5.0
        assert score.overall == 5.0
