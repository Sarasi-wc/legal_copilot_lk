"""
Query Processing Module
Handles parsing, classification, and understanding of legal queries.
Implements Query Processing component from Diagram 1.
"""

from .query_parser import QueryParser
from .domain_identifier import DomainIdentifier
from .statute_extractor import StatuteExtractor
from .query_classifier import QueryClassifier

__all__ = [
    'QueryParser',
    'DomainIdentifier',
    'StatuteExtractor',
    'QueryClassifier'
]
