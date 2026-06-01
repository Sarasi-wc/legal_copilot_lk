"""
Answer quality evaluation metrics.
Measures correctness, completeness, and clarity through automated and manual evaluation.
"""

from typing import Dict, List
from dataclasses import dataclass

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnswerQualityScore:
    """Answer quality evaluation scores."""
    correctness: float  # 0-5 scale
    completeness: float  # 0-5 scale
    clarity: float  # 0-5 scale
    overall: float  # Average


class AnswerQualityMetrics:
    """
    Evaluate answer quality using structured rubrics.
    Implements evaluation framework from RO5.
    """

    @staticmethod
    def create_evaluation_rubric() -> Dict:
        """
        Create structured evaluation rubric for answer quality.

        Returns:
            Rubric dictionary
        """
        rubric = {
            'correctness': {
                5: 'Completely correct, no errors',
                4: 'Mostly correct, minor inaccuracies',
                3: 'Partially correct, some errors',
                2: 'Mostly incorrect',
                1: 'Completely incorrect',
                0: 'No answer provided'
            },
            'completeness': {
                5: 'Fully addresses all aspects of question',
                4: 'Addresses most aspects, minor gaps',
                3: 'Addresses main points, notable gaps',
                2: 'Incomplete, major gaps',
                1: 'Minimal information',
                0: 'No relevant information'
            },
            'clarity': {
                5: 'Very clear and well-structured',
                4: 'Clear with minor issues',
                3: 'Somewhat clear, organization issues',
                2: 'Unclear or confusing',
                1: 'Very unclear',
                0: 'Incomprehensible'
            }
        }

        return rubric

    @staticmethod
    def evaluate_answer_quality(
        correctness_score: int,
        completeness_score: int,
        clarity_score: int
    ) -> AnswerQualityScore:
        """
        Create answer quality score from rubric ratings.

        Args:
            correctness_score: Score for correctness (0-5)
            completeness_score: Score for completeness (0-5)
            clarity_score: Score for clarity (0-5)

        Returns:
            AnswerQualityScore object
        """
        overall = (correctness_score + completeness_score + clarity_score) / 3.0

        return AnswerQualityScore(
            correctness=float(correctness_score),
            completeness=float(completeness_score),
            clarity=float(clarity_score),
            overall=overall
        )

    @staticmethod
    def compute_citation_metrics(
        answer: str,
        citations_valid: bool,
        num_citations: int,
        num_passages_used: int
    ) -> Dict:
        """
        Compute citation-related metrics.

        Args:
            answer: Answer text
            citations_valid: Whether all citations are valid
            num_citations: Number of citations in answer
            num_passages_used: Number of evidence passages used

        Returns:
            Citation metrics dictionary
        """
        metrics = {
            'num_citations': num_citations,
            'num_passages_used': num_passages_used,
            'citations_valid': citations_valid,
            'citation_passage_ratio': (
                num_citations / num_passages_used
                if num_passages_used > 0 else 0.0
            )
        }

        return metrics

    @staticmethod
    def aggregate_quality_scores(scores: List[AnswerQualityScore]) -> Dict:
        """
        Aggregate answer quality scores across multiple examples.

        Args:
            scores: List of AnswerQualityScore objects

        Returns:
            Aggregated metrics
        """
        if not scores:
            return {}

        import numpy as np

        correctness_scores = [s.correctness for s in scores]
        completeness_scores = [s.completeness for s in scores]
        clarity_scores = [s.clarity for s in scores]
        overall_scores = [s.overall for s in scores]

        return {
            'correctness_mean': np.mean(correctness_scores),
            'correctness_std': np.std(correctness_scores),
            'completeness_mean': np.mean(completeness_scores),
            'completeness_std': np.std(completeness_scores),
            'clarity_mean': np.mean(clarity_scores),
            'clarity_std': np.std(clarity_scores),
            'overall_mean': np.mean(overall_scores),
            'overall_std': np.std(overall_scores)
        }
