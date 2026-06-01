"""
Document segmentation module for parsing legal documents into structured units.
Implements hierarchical section extraction with metadata preservation.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils import get_logger, extract_section_number

logger = get_logger(__name__)


class SectionLevel(Enum):
    """Hierarchical levels in legal documents."""
    ACT = "act"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"


@dataclass
class LegalSection:
    """Represents a structured legal section."""
    section_id: str
    level: SectionLevel
    title: str
    text: str
    parent_id: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentSegmenter:
    """
    Segments legal documents into hierarchical sections.
    Implements rule-based parsing as described in methodology Section 3.4.4.
    """

    def __init__(self):
        """Initialize document segmenter with regex patterns."""

        # Section patterns for Sri Lankan legislation.
        # Sri Lankan legal PDFs use a two-column layout: marginal notes on the
        # left, section numbers + text on the right.  OCR linearises this so
        # section numbers (e.g. "10.") are NOT at the start of a line — they
        # appear after the marginal note text on the same logical line.
        # The patterns below therefore do NOT anchor on ^ and instead use a
        # negative lookbehind (?<!\d) to avoid matching page/year numbers.
        self.patterns = {
            # Only Roman numerals or digits are valid part/chapter identifiers.
            # Removing [A-Z] prevents matching single letters from running text
            # (e.g. "PART of the record" producing PART_o with empty text).
            'part': re.compile(
                r'PART\s+([IVXLCDM]+|\d+)\s*[:-]?\s*([^\n]*)',
                re.IGNORECASE
            ),
            'chapter': re.compile(
                r'CHAPTER\s+([IVXLCDM]+|\d+)\s*[:-]?\s*([^\n]*)',
                re.IGNORECASE
            ),
            # Matches "N." or "N ." (OCR bold numbers often have a space before
            # the period) anywhere in text, not preceded by a digit.
            # Followed by content starting uppercase or opening bracket.
            # Stops at the next such marker or end of text.
            # Handles: "1 . This Ordinance", "10. Every person", "18A. The word"
            'section': re.compile(
                r'(?<!\d)(\d{1,3}[A-Z]?)\s?\.\s{0,6}(?=[A-Z\("\']).{15,}?'
                r'(?=(?<!\d)\d{1,3}[A-Z]?\s?\.\s{0,6}[A-Z\("\'"]|\Z)',
                re.DOTALL
            ),
            'subsection': re.compile(
                r'^\((\d+[a-z]?)\)\s+(.+?)(?=\n\(|\n\n|\Z)',
                re.MULTILINE | re.DOTALL
            ),
            # Used post-chunking to extract the first visible section number.
            # Handles "1 ." (space before period) and "1." (no space).
            'leading_section': re.compile(
                r'(?<!\d)(\d{1,3}[A-Z]?)\s?\.\s{0,3}[A-Z\("\'"]'
            ),
        }

    def segment_document(
        self,
        text: str,
        act_name: str,
        act_number: str,
        year: int
    ) -> List[LegalSection]:
        """
        Segment a legal document into structured sections.

        Args:
            text: Full document text
            act_name: Name of the Act
            act_number: Act number
            year: Year of enactment

        Returns:
            List of LegalSection objects
        """
        logger.info(f"Segmenting document: {act_name}")

        sections = []

        # Create root Act section
        act_section = LegalSection(
            section_id=f"ACT_{act_number}_{year}",
            level=SectionLevel.ACT,
            title=act_name,
            text="",
            metadata={
                'act_number': act_number,
                'year': year,
                'name': act_name
            }
        )
        sections.append(act_section)

        # Extract hierarchical sections
        sections.extend(self._extract_parts(text, act_section.section_id))
        sections.extend(self._extract_sections(text, act_section.section_id))

        logger.info(f"Extracted {len(sections)} sections from {act_name}")

        return sections

    def _extract_parts(self, text: str, parent_id: str) -> List[LegalSection]:
        """Extract Parts from document."""
        parts = []
        matches = list(self.patterns['part'].finditer(text))

        for i, match in enumerate(matches):
            part_num = match.group(1)
            part_title = match.group(2).strip()

            # Extract part text (until next part or end)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            part_text = text[start:end].strip()

            part = LegalSection(
                section_id=f"{parent_id}_PART_{part_num}",
                level=SectionLevel.PART,
                title=f"Part {part_num}: {part_title}",
                text=part_text,
                parent_id=parent_id,
                metadata={'part_number': part_num}
            )
            parts.append(part)

        return parts

    def _extract_sections(self, text: str, parent_id: str) -> List[LegalSection]:
        """Extract numbered sections from document.

        Handles the Sri Lankan two-column PDF format where OCR linearises
        marginal notes before section numbers, so numbers are not at line
        starts.  The updated 'section' pattern finds them anywhere in text.
        """
        sections = []
        matches = list(self.patterns['section'].finditer(text))

        for match in matches:
            section_num = match.group(1)
            # Full matched text; skip past "N ." or "N." to get section content
            full_match = match.group(0)
            dot_pos = full_match.index('.')
            section_text = full_match[dot_pos + 1:].strip()

            # Extract title (first line) and body
            lines = section_text.split('\n', 1)
            title = lines[0].strip()[:120]  # Cap title length
            body = lines[1].strip() if len(lines) > 1 else section_text

            section = LegalSection(
                section_id=f"{parent_id}_SEC_{section_num}",
                level=SectionLevel.SECTION,
                title=title,
                text=section_text,
                parent_id=parent_id,
                metadata={'section_number': section_num}
            )
            sections.append(section)

            # Extract subsections
            subsections = self._extract_subsections(body, section.section_id)
            sections.extend(subsections)

        return sections

    def _extract_subsections(
        self,
        text: str,
        parent_id: str
    ) -> List[LegalSection]:
        """Extract subsections from section text."""
        subsections = []
        matches = list(self.patterns['subsection'].finditer(text))

        for match in matches:
            subsection_num = match.group(1)
            subsection_text = match.group(2).strip()

            subsection = LegalSection(
                section_id=f"{parent_id}_SUBSEC_{subsection_num}",
                level=SectionLevel.SUBSECTION,
                title=f"Subsection {subsection_num}",
                text=subsection_text,
                parent_id=parent_id,
                metadata={'subsection_number': subsection_num}
            )
            subsections.append(subsection)

        return subsections

    def create_passages(
        self,
        sections: List[LegalSection],
        max_length: int = 1500,
        fallback_text: Optional[str] = None
    ) -> List[Dict]:
        """
        Create passage-level segments suitable for retrieval.

        Args:
            sections: List of structured sections
            max_length: Maximum passage length in characters
            fallback_text: If no sections found, chunk this text instead

        Returns:
            List of passage dictionaries
        """
        passages = []

        # Count non-ACT sections
        non_act_sections = [s for s in sections if s.level != SectionLevel.ACT]
        
        # If no sections found, use fallback text chunking.
        # Even in this path, scan each chunk for a leading section/article
        # number so that citation generation can reference specific provisions.
        if not non_act_sections and fallback_text:
            logger.warning("No sections detected, using fallback text chunking")
            chunks = self._chunk_section(fallback_text, max_length)
            for i, chunk in enumerate(chunks):
                # Extract the first section/article number visible in this chunk
                sec_match = self.patterns['leading_section'].search(chunk)
                section_num = sec_match.group(1) if sec_match else None
                passage = {
                    'passage_id': f'PASSAGE_{i:04d}',
                    'text': chunk,
                    'title': (f'Section {section_num}' if section_num
                              else 'Document Passage'),
                    'level': 'passage',
                    'parent_id': None,
                    'metadata': {'section_number': section_num}
                }
                passages.append(passage)
            return passages

        for section in sections:
            # Skip Act-level (just metadata)
            if section.level == SectionLevel.ACT:
                continue

            # Create passage entry
            passage = {
                'passage_id': section.section_id,
                'text': section.text,
                'title': section.title,
                'level': section.level.value,
                'parent_id': section.parent_id,
                'metadata': section.metadata
            }

            # Split long sections into chunks if needed
            if len(section.text) > max_length:
                chunks = self._chunk_section(section.text, max_length)
                for i, chunk in enumerate(chunks):
                    chunk_passage = passage.copy()
                    chunk_passage['passage_id'] = f"{section.section_id}_CHUNK_{i}"
                    chunk_passage['text'] = chunk
                    passages.append(chunk_passage)
            else:
                passages.append(passage)

        return passages

    def _chunk_section(self, text: str, max_length: int) -> List[str]:
        """Chunk long section text preserving sentence boundaries."""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
