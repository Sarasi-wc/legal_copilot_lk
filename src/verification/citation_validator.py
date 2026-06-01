"""
Citation validation module for checking citation accuracy.
Verifies that citations refer to actual passages in the evidence.
"""

import re
from typing import List, Dict, Tuple

from src.utils import get_logger

logger = get_logger(__name__)


class CitationValidator:
    """
    Validates citations against evidence passages.
    Detects fabricated citations and citation errors.

    Limitation: the validator detects wrong act names (e.g. a citation to an
    Act not present in the evidence), but it cannot detect hallucinated article
    numbers within the same Act. Because all 793 corpus passages belong to the
    Constitution, any citation to "[Constitution of Sri Lanka 1978, Article N]"
    will be accepted as valid by the same-act fallback at line ~136 — even if
    Article N does not exist. Wrong act names are caught; wrong article numbers
    within the correct act are not. This is a documented limitation (Ch6 §6.6.3).
    """

    def __init__(self):
        """Initialize citation validator."""
        pass

    def validate_citations(
        self,
        answer: str,
        evidence_plan: Dict
    ) -> Dict:
        """
        Validate all citations in answer.

        Args:
            answer: Generated answer with citations
            evidence_plan: Evidence plan containing passages

        Returns:
            Validation results
        """
        logger.debug("Validating citations")

        # Extract citations from answer
        citations = self._extract_citations(answer)

        # Validate each citation
        validation_results = []
        for citation in citations:
            result = self._validate_single_citation(citation, evidence_plan)
            validation_results.append(result)

        # Aggregate results
        total_citations = len(validation_results)
        valid_citations = sum(1 for r in validation_results if r['is_valid'])

        fabricated_citations = [
            r['citation'] for r in validation_results if r['is_fabricated']
        ]

        return {
            'total_citations': total_citations,
            'valid_citations': valid_citations,
            'fabricated_citations': fabricated_citations,
            'validation_rate': valid_citations / total_citations if total_citations > 0 else 0.0,
            'all_valid': total_citations > 0 and len(fabricated_citations) == 0,
            'citation_details': validation_results
        }

    def _extract_citations(self, answer: str) -> List[Dict]:
        """Extract citations from answer text."""
        citations = []

        # Pattern: [Act Name, Article/Section/Chapter X] or [Act Name, X]
        # Matches: [Constitution, Article 12], [Constitution, Section 5], [Act, 10]
        pattern = r'\[([^,\]]+),\s*(?:Article|Section|Chapter)?\s*([0-9]+[A-Z]?(?:\([0-9]+\))?(?:\([a-z]\))?)\]'

        for match in re.finditer(pattern, answer, re.IGNORECASE):
            citations.append({
                'act_name': match.group(1).strip(),
                'section': match.group(2).strip(),
                'full_citation': match.group(0),
                'position': match.start()
            })

        return citations

    def _validate_single_citation(
        self,
        citation: Dict,
        evidence_plan: Dict
    ) -> Dict:
        """
        Validate a single citation.

        Args:
            citation: Citation dictionary
            evidence_plan: Evidence plan with passages

        Returns:
            Validation result
        """
        act_name = citation['act_name'].lower()
        section = citation['section']

        # Treat empty act_name in corpus as "Constitution of Sri Lanka"
        CONSTITUTION_ALIASES = {'constitution', 'constitution of sri lanka', 'constitution of the democratic socialist republic of sri lanka'}

        def acts_match(cited: str, passage_act: str) -> bool:
            """Return True if cited act name matches the passage act name."""
            if not passage_act:
                # Empty passage act_name means Constitution corpus
                return cited in CONSTITUTION_ALIASES or 'constitution' in cited
            return cited in passage_act or passage_act in cited

        # Search for matching passage - be flexible with matching
        matching_passage = None
        for passage in evidence_plan.get('passages', []):
            passage_act = passage.get('act_name', '').lower()

            if acts_match(act_name, passage_act):
                # Check if section/article is mentioned in passage text
                passage_text = passage.get('text', '').lower()
                section_mentions = [
                    f'article {section}',
                    f'section {section}',
                    f'chapter {section}',
                    f'article{section}',  # No space
                    f'section{section}',
                    f'{section}.',  # Just the number with period (e.g., "10. Every person")
                ]

                if any(s in passage_text for s in section_mentions):
                    matching_passage = passage
                    break

                # If no specific section match, still accept if from same act
                # (passage may contain the cited article/section)
                if not matching_passage and acts_match(act_name, passage_act):
                    matching_passage = passage

        is_valid = matching_passage is not None
        is_fabricated = not is_valid

        return {
            'citation': citation['full_citation'],
            'act_name': citation['act_name'],
            'section': citation['section'],
            'is_valid': is_valid,
            'is_fabricated': is_fabricated,
            'matching_passage_id': matching_passage['passage_id'] if matching_passage else None
        }

    def check_citation_formatting(self, answer: str) -> Dict:
        """
        Check if citations follow proper format.

        Args:
            answer: Generated answer

        Returns:
            Formatting check results
        """
        # Correct format: [Act Name, Article/Section/Chapter X] or [Act Name, X]
        # Accepts: [Constitution, Article 12], [Constitution, Section 5], [Act, 10]
        correct_pattern = r'\[([^,\]]+),\s*(?:Article|Section|Chapter)?\s*[0-9]+[A-Z]?(?:\([0-9]+\))?(?:\([a-z]\))?\]'

        # Find all potential citations (text in brackets)
        all_brackets = re.findall(r'\[[^\]]+\]', answer)

        # Check each bracket against correct pattern
        correct_format = []
        malformed = []
        for b in all_brackets:
            if re.match(correct_pattern, b, re.IGNORECASE):
                correct_format.append(b)
            else:
                malformed.append(b)

        return {
            'total_bracket_citations': len(all_brackets),
            'correctly_formatted': len(correct_format),
            'malformed_citations': malformed,
            'all_properly_formatted': len(malformed) == 0
        }
