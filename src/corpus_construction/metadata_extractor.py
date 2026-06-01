"""
Metadata extraction module for legal documents.
Extracts temporal metadata, amendments, cross-references, and hierarchical relationships.
"""

import re
from typing import List, Dict, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict

from src.utils import get_logger, validate_date

logger = get_logger(__name__)


@dataclass
class Amendment:
    """Represents an amendment to a legal provision."""
    amendment_act: str
    amendment_number: str
    amendment_year: int
    effective_date: Optional[datetime]
    description: str
    section_affected: str


@dataclass
class CrossReference:
    """Represents a cross-reference to another legal provision."""
    source_section: str
    target_act: Optional[str]
    target_section: str
    reference_type: str  # 'defined_in', 'see', 'subject_to', etc.


class MetadataExtractor:
    """
    Extracts metadata from legal documents including temporal information,
    amendments, and cross-references.
    """

    def __init__(self):
        """Initialize metadata extractor with regex patterns."""

        # Patterns for amendments
        self.amendment_patterns = [
            re.compile(
                r'amended by Act No\.\s*(\d+)\s+of\s+(\d{4})',
                re.IGNORECASE
            ),
            re.compile(
                r'substituted by.*?Act No\.\s*(\d+)\s+of\s+(\d{4})',
                re.IGNORECASE
            ),
            re.compile(
                r'inserted by.*?Act No\.\s*(\d+)\s+of\s+(\d{4})',
                re.IGNORECASE
            ),
            re.compile(
                r'repealed by.*?Act No\.\s*(\d+)\s+of\s+(\d{4})',
                re.IGNORECASE
            )
        ]

        # Patterns for cross-references
        self.xref_patterns = {
            'section_ref': re.compile(
                r'[Ss]ection\s+(\d+[A-Z]?(?:\([a-z0-9]+\))?)',
                re.IGNORECASE
            ),
            'act_ref': re.compile(
                r'([A-Z][^(]+?)\s*\(Act No\.\s*(\d+)\s+of\s+(\d{4})\)'
            ),
            'defined_in': re.compile(
                r'defined in [Ss]ection\s+(\d+[A-Z]?)',
                re.IGNORECASE
            ),
            'subject_to': re.compile(
                r'subject to [Ss]ection\s+(\d+[A-Z]?)',
                re.IGNORECASE
            )
        }

        # Patterns for dates
        self.date_patterns = [
            re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'),
            re.compile(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})')
        ]

    def extract_act_metadata(
        self,
        text: str,
        act_name: str,
        act_number: str,
        year: int
    ) -> Dict:
        """
        Extract comprehensive metadata from an Act.

        Args:
            text: Full Act text
            act_name: Name of the Act
            act_number: Act number
            year: Year of enactment

        Returns:
            Dictionary containing metadata
        """
        logger.info(f"Extracting metadata from {act_name}")

        metadata = {
            'act_name': act_name,
            'act_number': act_number,
            'year': year,
            'enactment_date': self._extract_enactment_date(text),
            'amendments': self._extract_amendments(text),
            'repealed': self._check_repeal_status(text),
            'cross_references': self._extract_cross_references(text),
            'definitions_section': self._find_definitions_section(text)
        }

        return metadata

    def _extract_enactment_date(self, text: str) -> Optional[str]:
        """Extract enactment date from Act preamble."""
        # Look for date in first 1000 characters
        preamble = text[:1000]

        # Try various date patterns
        for pattern in self.date_patterns:
            match = pattern.search(preamble)
            if match:
                date_str = match.group(0)
                parsed_date = validate_date(date_str)
                if parsed_date:
                    return parsed_date.isoformat()

        return None

    def _extract_amendments(self, text: str) -> List[Dict]:
        """Extract amendment information from document."""
        amendments = []
        seen = set()

        for pattern in self.amendment_patterns:
            for match in pattern.finditer(text):
                amendment_number = match.group(1)
                amendment_year = int(match.group(2))

                key = f"{amendment_number}_{amendment_year}"
                if key in seen:
                    continue
                seen.add(key)

                # Extract context around match
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]

                amendment = {
                    'amendment_number': amendment_number,
                    'amendment_year': amendment_year,
                    'description': match.group(0),
                    'context': context
                }
                amendments.append(amendment)

        logger.info(f"Found {len(amendments)} amendments")
        return amendments

    def _check_repeal_status(self, text: str) -> Optional[Dict]:
        """Check if Act has been repealed."""
        repeal_pattern = re.compile(
            r'repealed by.*?Act No\.\s*(\d+)\s+of\s+(\d{4})',
            re.IGNORECASE
        )

        match = repeal_pattern.search(text[:2000])  # Check beginning
        if match:
            return {
                'repealed': True,
                'repeal_act_number': match.group(1),
                'repeal_year': int(match.group(2))
            }

        return {'repealed': False}

    def _extract_cross_references(self, text: str) -> List[Dict]:
        """Extract cross-references to other sections and Acts."""
        cross_refs = []

        # Find section references
        for match in self.xref_patterns['section_ref'].finditer(text):
            section_num = match.group(1)

            # Determine reference type from context
            context_start = max(0, match.start() - 20)
            context = text[context_start:match.start()].lower()

            ref_type = 'general'
            if 'defined in' in context:
                ref_type = 'definition'
            elif 'subject to' in context:
                ref_type = 'condition'
            elif 'see' in context:
                ref_type = 'see_also'

            cross_refs.append({
                'target_section': section_num,
                'reference_type': ref_type,
                'target_act': None  # Same Act
            })

        # Find Act references
        for match in self.xref_patterns['act_ref'].finditer(text):
            act_name = match.group(1).strip()
            act_number = match.group(2)
            act_year = int(match.group(3))

            cross_refs.append({
                'target_act': act_name,
                'target_act_number': act_number,
                'target_year': act_year,
                'reference_type': 'external_act'
            })

        return cross_refs

    def _find_definitions_section(self, text: str) -> Optional[str]:
        """Find the definitions section number."""
        # Definitions usually in Section 2 or 3
        patterns = [
            r'(\d+)\.\s+Definitions',
            r'(\d+)\.\s+Interpretation'
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def extract_section_metadata(
        self,
        section_text: str,
        section_number: str
    ) -> Dict:
        """
        Extract metadata specific to a section.

        Args:
            section_text: Text of the section
            section_number: Section number

        Returns:
            Section-specific metadata
        """
        metadata = {
            'section_number': section_number,
            'has_subsections': self._has_subsections(section_text),
            'cross_references': self._extract_cross_references(section_text),
            'penalties': self._extract_penalties(section_text),
            'conditions': self._extract_conditions(section_text)
        }

        return metadata

    def _has_subsections(self, text: str) -> bool:
        """Check if section has subsections."""
        subsection_pattern = re.compile(r'\(\d+[a-z]?\)')
        return bool(subsection_pattern.search(text))

    def _extract_penalties(self, text: str) -> List[str]:
        """Extract penalty clauses from section."""
        penalties = []

        penalty_keywords = [
            'fine', 'imprisonment', 'penalty', 'shall be liable',
            'shall be guilty', 'offense', 'offence'
        ]

        text_lower = text.lower()
        for keyword in penalty_keywords:
            if keyword in text_lower:
                # Extract sentence containing keyword
                sentences = re.split(r'[.!?]', text)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        penalties.append(sentence.strip())
                        break

        return penalties

    def _extract_conditions(self, text: str) -> List[str]:
        """Extract conditional clauses."""
        conditions = []

        condition_patterns = [
            r'if\s+.+?(?=\.|,|\n)',
            r'unless\s+.+?(?=\.|,|\n)',
            r'subject to\s+.+?(?=\.|,|\n)',
            r'provided that\s+.+?(?=\.|,|\n)'
        ]

        for pattern in condition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                conditions.append(match.group(0).strip())

        return conditions
