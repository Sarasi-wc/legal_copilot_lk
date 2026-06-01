"""
Query Parser
Parses natural language legal queries and extracts structured information.
"""

import re
from typing import Dict, List, Optional
import spacy

from src.utils import get_logger

logger = get_logger(__name__)


class QueryParser:
    """
    Parse natural language legal queries.
    Extracts entities, normalizes text, and identifies key elements.
    """

    def __init__(self):
        """Initialize query parser with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def parse(self, query: str) -> Dict:
        """
        Parse legal query and extract structured information.

        Args:
            query: User query string

        Returns:
            Dictionary with parsed query information
        """
        logger.info(f"Parsing query: {query[:100]}...")

        # Normalize query
        normalized = self._normalize_text(query)

        # Extract entities
        entities = self._extract_entities(query)

        # Extract facts (simple sentence segmentation)
        facts = self._extract_facts(query)

        # Identify question type
        question_type = self._identify_question_type(query)

        return {
            'original_query': query,
            'normalized_query': normalized,
            'entities': entities,
            'facts': facts,
            'question_type': question_type,
            'is_question': query.strip().endswith('?')
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize query text."""
        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Expand common abbreviations
        abbreviations = {
            ' pc ': ' penal code ',
            ' cpc ': ' civil procedure code ',
            ' eo ': ' evidence ordinance ',
            's.': 'section ',
            'sec.': 'section ',
            'art.': 'article '
        }

        for abbr, full in abbreviations.items():
            normalized = normalized.replace(abbr, full)

        return normalized

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        doc = self.nlp(text)

        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'laws': []
        }

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['money'].append(ent.text)
            elif ent.label_ == 'LAW':
                entities['laws'].append(ent.text)

        return entities

    def _extract_facts(self, text: str) -> List[str]:
        """Extract fact statements from query."""
        doc = self.nlp(text)

        facts = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Skip questions
            if not sent_text.endswith('?'):
                facts.append(sent_text)

        return facts

    def _identify_question_type(self, text: str) -> str:
        """Identify the type of question being asked."""
        text_lower = text.lower()

        # What questions - factual
        if text_lower.startswith(('what is', 'what are', 'what does')):
            return 'factual'

        # How questions - procedural
        if text_lower.startswith(('how to', 'how do i', 'how can')):
            return 'procedural'

        # Why questions - interpretive
        if text_lower.startswith(('why', 'when', 'where')):
            return 'interpretive'

        # Can/Could/May questions - advisory
        if text_lower.startswith(('can i', 'could i', 'may i', 'should i', 'would')):
            return 'advisory'

        # Is/Are questions - factual
        if text_lower.startswith(('is', 'are', 'was', 'were')):
            return 'factual'

        # Default
        return 'general'
