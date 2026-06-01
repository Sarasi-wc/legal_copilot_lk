"""
Query Classifier
Classifies legal queries into types: procedural, factual, interpretive, advisory.
"""

import re
from typing import Dict, Tuple
from enum import Enum

from src.utils import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of legal queries."""
    PROCEDURAL = "procedural"      # How to do something
    FACTUAL = "factual"            # What is something
    INTERPRETIVE = "interpretive"  # Legal interpretation
    ADVISORY = "advisory"          # What should I do
    GENERAL = "general"            # General query


class QueryClassifier:
    """
    Classify legal queries into types.
    Uses pattern matching and keyword analysis.
    """

    def __init__(self):
        """Initialize query classifier with patterns."""
        self.procedural_patterns = [
            r'\bhow\s+to\b',
            r'\bhow\s+do\s+i\b',
            r'\bhow\s+can\s+i\b',
            r'\bwhat\s+is\s+the\s+procedure\b',
            r'\bwhat\s+are\s+the\s+steps\b',
            r'\bwhat\s+process\b',
            r'\bfile\s+a\b',
            r'\bapply\s+for\b',
            r'\bsubmit\b'
        ]

        self.factual_patterns = [
            r'\bwhat\s+is\b',
            r'\bwhat\s+are\b',
            r'\bwhat\s+does\b',
            r'\bdefine\b',
            r'\bdefinition\s+of\b',
            r'\bmeaning\s+of\b',
            r'\bexplain\b',
            r'\bwho\s+is\b',
            r'\bwhen\s+is\b',
            r'\bwhere\s+is\b'
        ]

        self.interpretive_patterns = [
            r'\bdoes\s+this\b',
            r'\bis\s+this\b',
            r'\bwould\s+this\b',
            r'\bcan\s+this\s+be\s+considered\b',
            r'\bqualify\s+as\b',
            r'\bcount\s+as\b',
            r'\bapply\s+to\b',
            r'\bunder\s+which\b',
            r'\bwhich\s+law\b'
        ]

        self.advisory_patterns = [
            r'\bshould\s+i\b',
            r'\bwhat\s+should\b',
            r'\bcan\s+i\b',
            r'\bmay\s+i\b',
            r'\bcould\s+i\b',
            r'\bwhat\s+are\s+my\s+options\b',
            r'\bwhat\s+can\s+i\s+do\b',
            r'\badvice\b',
            r'\brecommend\b',
            r'\bwhat\s+happens\s+if\b'
        ]

        self.procedural_keywords = [
            'procedure', 'process', 'steps', 'file', 'submit', 'apply',
            'register', 'lodge', 'appeal', 'petition', 'plaint'
        ]

        self.factual_keywords = [
            'definition', 'meaning', 'is', 'are', 'what', 'explain',
            'describe', 'penalty', 'punishment', 'provision'
        ]

        self.interpretive_keywords = [
            'interpret', 'applicable', 'apply', 'qualify', 'consider',
            'constitute', 'classify', 'categorize'
        ]

        self.advisory_keywords = [
            'should', 'advice', 'recommend', 'options', 'rights',
            'remedy', 'recourse', 'action', 'do'
        ]

    def classify(self, query: str, normalized_query: str) -> Tuple[str, float]:
        """
        Classify query type.

        Args:
            query: Original query
            normalized_query: Normalized query

        Returns:
            Tuple of (query_type, confidence)
        """
        logger.info("Classifying query type")

        text = normalized_query.lower()

        # Score each type
        scores = {
            'procedural': self._score_type(text, self.procedural_patterns, self.procedural_keywords),
            'factual': self._score_type(text, self.factual_patterns, self.factual_keywords),
            'interpretive': self._score_type(text, self.interpretive_patterns, self.interpretive_keywords),
            'advisory': self._score_type(text, self.advisory_patterns, self.advisory_keywords)
        }

        # Get highest scoring type
        if max(scores.values()) > 0:
            query_type = max(scores, key=scores.get)
            total_score = sum(scores.values())
            confidence = scores[query_type] / total_score if total_score > 0 else 0.5

            logger.info(f"Classified as: {query_type} (confidence: {confidence:.2f})")
            return query_type, confidence
        else:
            logger.info("Classified as: general (default)")
            return 'general', 0.3

    def _score_type(self, text: str, patterns: list, keywords: list) -> float:
        """Score a query type based on patterns and keywords."""
        score = 0.0

        # Pattern matching (higher weight)
        for pattern in patterns:
            if re.search(pattern, text):
                score += 2.0

        # Keyword matching (lower weight)
        for keyword in keywords:
            if keyword in text:
                score += 0.5

        return score

    def get_query_characteristics(self, query: str) -> Dict:
        """
        Get additional characteristics of the query.

        Returns:
            Dictionary with query characteristics
        """
        text = query.lower()

        return {
            'is_question': query.strip().endswith('?'),
            'has_negation': any(word in text for word in ['not', 'no', 'never', "don't", "doesn't"]),
            'has_conditional': any(word in text for word in ['if', 'when', 'provided', 'unless']),
            'has_comparison': any(word in text for word in ['or', 'versus', 'vs', 'between', 'differ']),
            'has_temporal': any(word in text for word in ['before', 'after', 'during', 'within', 'deadline']),
            'has_quantifier': any(word in text for word in ['how much', 'how many', 'amount', 'number']),
            'complexity_score': self._calculate_complexity(query)
        }

    def _calculate_complexity(self, query: str) -> str:
        """Calculate query complexity based on length and structure."""
        word_count = len(query.split())

        if word_count < 10:
            return 'simple'
        elif word_count < 25:
            return 'medium'
        else:
            return 'complex'
