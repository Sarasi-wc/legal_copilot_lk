"""
Statute Extractor
Extracts statute and section references from queries.
"""

import re
from typing import List, Dict

from src.utils import get_logger

logger = get_logger(__name__)


class StatuteExtractor:
    """
    Extract statute and section references from legal queries.
    Uses regex patterns to identify references.
    """

    def __init__(self):
        """Initialize statute extractor with patterns."""
        # Common Sri Lankan Acts and their abbreviations
        self.statute_mappings = {
            'penal code': 'Penal Code',
            'pc': 'Penal Code',
            'civil procedure code': 'Civil Procedure Code',
            'cpc': 'Civil Procedure Code',
            'evidence ordinance': 'Evidence Ordinance',
            'eo': 'Evidence Ordinance',
            'constitution': 'Constitution of Sri Lanka',
            'criminal procedure code': 'Criminal Procedure Code',
            'crpc': 'Criminal Procedure Code'
        }

        # Regex patterns for section references
        self.patterns = [
            # "Section 365 of the Penal Code"
            r'section\s+(\d+[a-z]?)\s+of\s+(?:the\s+)?([a-z\s]+(?:code|ordinance|act|constitution))',
            # "Penal Code Section 365"
            r'([a-z\s]+(?:code|ordinance|act|constitution))\s+section\s+(\d+[a-z]?)',
            # "Section 365, Penal Code"
            r'section\s+(\d+[a-z]?),\s*([a-z\s]+(?:code|ordinance|act))',
            # "S. 365 of PC"
            r's\.?\s*(\d+[a-z]?)\s+of\s+([a-z\.]+)',
            # "Article 13 of the Constitution"
            r'article\s+(\d+[a-z]?)\s+of\s+(?:the\s+)?([a-z\s]+constitution)',
            # Just "Section 365" (less confident)
            r'\bsection\s+(\d+[a-z]?)\b'
        ]

    def extract(self, query: str, normalized_query: str) -> List[Dict]:
        """
        Extract statute and section references from query.

        Args:
            query: Original query
            normalized_query: Normalized query

        Returns:
            List of extracted references with metadata
        """
        logger.info("Extracting statute references")

        references = []
        text = normalized_query.lower()

        for pattern in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                groups = match.groups()

                if len(groups) == 2:
                    # Pattern with both section and statute
                    section, statute = groups

                    # Normalize statute name
                    statute_normalized = self._normalize_statute_name(statute.strip())

                    references.append({
                        'section': section.strip(),
                        'statute': statute_normalized,
                        'raw_text': match.group(0),
                        'confidence': 0.9
                    })

                elif len(groups) == 1:
                    # Just section number - try to infer statute from context
                    section = groups[0]
                    statute = self._infer_statute_from_context(text)

                    if statute:
                        references.append({
                            'section': section.strip(),
                            'statute': statute,
                            'raw_text': match.group(0),
                            'confidence': 0.5  # Lower confidence
                        })

        # Remove duplicates
        references = self._deduplicate_references(references)

        logger.info(f"Extracted {len(references)} statute references")
        return references

    def _normalize_statute_name(self, statute: str) -> str:
        """Normalize statute name using mappings."""
        statute_lower = statute.lower().strip()

        # Check exact match
        if statute_lower in self.statute_mappings:
            return self.statute_mappings[statute_lower]

        # Check partial match
        for abbr, full_name in self.statute_mappings.items():
            if abbr in statute_lower:
                return full_name

        # Capitalize each word
        return ' '.join(word.capitalize() for word in statute_lower.split())

    def _infer_statute_from_context(self, text: str) -> str:
        """Infer statute from query context."""
        # Check for statute mentions in the text
        for abbr, full_name in self.statute_mappings.items():
            if abbr in text:
                return full_name

        # Check domain keywords to infer statute
        if any(word in text for word in ['theft', 'murder', 'assault', 'criminal']):
            return 'Penal Code'

        if 'evidence' in text:
            return 'Evidence Ordinance'

        if 'civil' in text or 'suit' in text:
            return 'Civil Procedure Code'

        return None

    def _deduplicate_references(self, references: List[Dict]) -> List[Dict]:
        """Remove duplicate references, keeping highest confidence."""
        seen = {}

        for ref in references:
            key = (ref['section'], ref['statute'])

            if key not in seen or ref['confidence'] > seen[key]['confidence']:
                seen[key] = ref

        return list(seen.values())

    def extract_all_sections(self, text: str) -> List[str]:
        """Extract just section numbers from text."""
        sections = []

        # Pattern for section numbers
        pattern = r'\b(?:section|s\.?|sec\.?)\s*(\d+[a-z]?)\b'

        matches = re.finditer(pattern, text.lower())
        for match in matches:
            sections.append(match.group(1))

        return list(set(sections))  # Remove duplicates
