"""
Assumption Extractor Module

Identifies and extracts assumptions made during legal analysis.
Makes explicit what the system assumes about facts, law interpretation,
and applicability conditions.
"""

from typing import Dict, List, Optional
import re
from dataclasses import dataclass, asdict
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Assumption:
    """Single assumption in legal analysis."""
    assumption_text: str
    assumption_type: str
    impact_if_incorrect: str
    verification_method: str
    confidence: str  # high, medium, low
    source: str  # Where assumption was detected


class AssumptionExtractor:
    """
    Extract and document assumptions from legal analysis.

    Identifies implicit and explicit assumptions about:
    - Facts stated in query
    - Interpretation of legal provisions
    - Applicability of laws
    - Completeness of information
    - Temporal factors (current law)
    """

    # Assumption indicator keywords
    ASSUMPTION_INDICATORS = [
        'assuming',
        'if',
        'provided that',
        'where',
        'given that',
        'suppose',
        'presuming',
        'considering',
        'based on',
        'taking into account'
    ]

    # Conditional language
    CONDITIONAL_INDICATORS = [
        'may',
        'might',
        'could',
        'would',
        'should',
        'unless',
        'except',
        'subject to'
    ]

    def __init__(
        self,
        extract_implicit: bool = True,
        include_standard_assumptions: bool = True
    ):
        """
        Initialize assumption extractor.

        Args:
            extract_implicit: Extract implicit assumptions
            include_standard_assumptions: Include standard legal assumptions
        """
        self.extract_implicit = extract_implicit
        self.include_standard_assumptions = include_standard_assumptions
        logger.info("AssumptionExtractor initialized")

    def extract_assumptions(
        self,
        query: str,
        answer: str,
        evidence_plan: Optional[Dict] = None,
        inference_result: Optional[Dict] = None
    ) -> Dict:
        """
        Extract all assumptions from legal analysis.

        Args:
            query: User query
            answer: Generated answer
            evidence_plan: Evidence plan from planner
            inference_result: Inference engine results

        Returns:
            Dictionary with extracted assumptions
        """
        logger.info("Extracting assumptions from legal analysis")

        assumptions = []

        # 1. Extract explicit assumptions from answer
        explicit_assumptions = self._extract_explicit_assumptions(answer)
        assumptions.extend(explicit_assumptions)

        # 2. Extract implicit assumptions (if enabled)
        if self.extract_implicit:
            implicit_assumptions = self._extract_implicit_assumptions(
                query,
                answer,
                evidence_plan
            )
            assumptions.extend(implicit_assumptions)

        # 3. Add standard legal assumptions (if enabled)
        if self.include_standard_assumptions:
            standard_assumptions = self._get_standard_assumptions()
            assumptions.extend(standard_assumptions)

        # 4. Extract assumptions from inference results
        if inference_result:
            inference_assumptions = self._extract_inference_assumptions(
                inference_result
            )
            assumptions.extend(inference_assumptions)

        # 5. Identify factual assumptions from query
        factual_assumptions = self._extract_factual_assumptions(query, answer)
        assumptions.extend(factual_assumptions)

        # Remove duplicates
        unique_assumptions = self._deduplicate_assumptions(assumptions)

        # Sort by confidence and impact
        sorted_assumptions = self._sort_assumptions(unique_assumptions)

        result = {
            'assumptions': [asdict(a) for a in sorted_assumptions],
            'total_assumptions': len(sorted_assumptions),
            'assumption_types': self._count_assumption_types(sorted_assumptions),
            'high_impact_count': len([
                a for a in sorted_assumptions
                if 'different' in a.impact_if_incorrect.lower() or
                   'may not apply' in a.impact_if_incorrect.lower()
            ])
        }

        logger.info(f"Extracted {len(sorted_assumptions)} assumptions")

        return result

    def _extract_explicit_assumptions(self, answer: str) -> List[Assumption]:
        """
        Extract explicitly stated assumptions from answer.

        Args:
            answer: Answer text

        Returns:
            List of Assumption objects
        """
        assumptions = []

        # Look for assumption indicator keywords
        for indicator in self.ASSUMPTION_INDICATORS:
            pattern = rf'({indicator}[^.!?]+[.!?])'
            matches = re.finditer(pattern, answer, re.IGNORECASE)

            for match in matches:
                assumption_text = match.group(1).strip()

                assumptions.append(Assumption(
                    assumption_text=assumption_text,
                    assumption_type='explicit',
                    impact_if_incorrect='Analysis conclusions may not be applicable',
                    verification_method='Verify the stated conditions with actual facts',
                    confidence='medium',
                    source='answer_text'
                ))

        return assumptions

    def _extract_implicit_assumptions(
        self,
        query: str,
        answer: str,
        evidence_plan: Optional[Dict]
    ) -> List[Assumption]:
        """
        Extract implicit assumptions.

        Args:
            query: User query
            answer: Answer text
            evidence_plan: Evidence plan

        Returns:
            List of implicit assumptions
        """
        assumptions = []

        # 1. Assumption about query completeness
        if len(query.split()) < 20:  # Short query
            assumptions.append(Assumption(
                assumption_text='The query provides complete and accurate information',
                assumption_type='implicit_factual',
                impact_if_incorrect='Different legal provisions may apply if facts are different',
                verification_method='Provide complete details to legal practitioner',
                confidence='low',
                source='query_analysis'
            ))

        # 2. Assumption about evidence coverage
        if evidence_plan:
            num_passages = len(evidence_plan.get('passages', []))
            if num_passages < 5:
                assumptions.append(Assumption(
                    assumption_text='Available legal provisions cover all relevant law',
                    assumption_type='implicit_coverage',
                    impact_if_incorrect='Additional legal provisions may exist that affect the outcome',
                    verification_method='Comprehensive legal research with official databases',
                    confidence='medium',
                    source='evidence_coverage'
                ))

        # 3. Assumption about jurisdiction
        if 'sri lanka' in answer.lower() or 'sri lanka' in query.lower():
            assumptions.append(Assumption(
                assumption_text='Analysis applies to Sri Lankan jurisdiction only',
                assumption_type='jurisdictional',
                impact_if_incorrect='Different laws apply in other jurisdictions',
                verification_method='Confirm jurisdiction with legal authority',
                confidence='high',
                source='jurisdiction_analysis'
            ))

        return assumptions

    def _get_standard_assumptions(self) -> List[Assumption]:
        """
        Get standard assumptions that apply to all legal analyses.

        Returns:
            List of standard assumptions
        """
        return [
            Assumption(
                assumption_text='Facts stated in query are accurate and complete',
                assumption_type='standard_factual',
                impact_if_incorrect='Legal analysis may be entirely inapplicable',
                verification_method='Verify all facts with supporting documents',
                confidence='low',
                source='standard'
            ),
            Assumption(
                assumption_text='Legal provisions cited are current and not amended',
                assumption_type='standard_temporal',
                impact_if_incorrect='Analysis based on outdated law may be incorrect',
                verification_method='Check official legal databases for latest amendments',
                confidence='medium',
                source='standard'
            ),
            Assumption(
                assumption_text='No exceptional circumstances or special conditions apply',
                assumption_type='standard_applicability',
                impact_if_incorrect='Exceptions to general rule may change the outcome',
                verification_method='Consult legal expert about potential exceptions',
                confidence='medium',
                source='standard'
            ),
            Assumption(
                assumption_text='System knowledge base is comprehensive and up-to-date',
                assumption_type='standard_system',
                impact_if_incorrect='Relevant legal provisions may be missing from analysis',
                verification_method='Cross-reference with official legal sources',
                confidence='medium',
                source='standard'
            )
        ]

    def _extract_inference_assumptions(
        self,
        inference_result: Dict
    ) -> List[Assumption]:
        """
        Extract assumptions from inference engine results.

        Args:
            inference_result: Inference results

        Returns:
            List of assumptions
        """
        assumptions = []

        # Check for conflict-related assumptions
        if inference_result.get('has_conflicts'):
            assumptions.append(Assumption(
                assumption_text='Conflicting provisions can be resolved through legal interpretation',
                assumption_type='inference_conflict',
                impact_if_incorrect='Outcome may be uncertain and require court determination',
                verification_method='Seek legal opinion on conflict resolution',
                confidence='low',
                source='inference_engine'
            ))

        # Check for rule matching assumptions
        if 'matched_rules' in inference_result:
            num_matches = len(inference_result.get('matched_rules', []))
            if num_matches > 0:
                assumptions.append(Assumption(
                    assumption_text=f'The {num_matches} identified rule(s) are the only applicable provisions',
                    assumption_type='inference_completeness',
                    impact_if_incorrect='Other legal provisions may also apply',
                    verification_method='Comprehensive legal research and expert review',
                    confidence='medium',
                    source='inference_engine'
                ))

        return assumptions

    def _extract_factual_assumptions(
        self,
        query: str,
        answer: str
    ) -> List[Assumption]:
        """
        Extract assumptions about factual matters.

        Args:
            query: User query
            answer: Answer text

        Returns:
            List of factual assumptions
        """
        assumptions = []

        # Look for factual references in answer that aren't in query
        fact_keywords = ['fact', 'situation', 'case', 'circumstance', 'event']

        answer_lower = answer.lower()
        query_lower = query.lower()

        for keyword in fact_keywords:
            if keyword in answer_lower and keyword not in query_lower:
                assumptions.append(Assumption(
                    assumption_text=f'Interpretation of factual {keyword} is correct',
                    assumption_type='factual_interpretation',
                    impact_if_incorrect='Different interpretation may lead to different legal conclusion',
                    verification_method='Clarify factual details with legal advisor',
                    confidence='medium',
                    source='fact_analysis'
                ))
                break  # Only add once

        return assumptions

    def _deduplicate_assumptions(
        self,
        assumptions: List[Assumption]
    ) -> List[Assumption]:
        """
        Remove duplicate assumptions.

        Args:
            assumptions: List of assumptions

        Returns:
            Deduplicated list
        """
        seen_texts = set()
        unique = []

        for assumption in assumptions:
            # Normalize text for comparison
            normalized = assumption.assumption_text.lower().strip()

            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique.append(assumption)

        return unique

    def _sort_assumptions(
        self,
        assumptions: List[Assumption]
    ) -> List[Assumption]:
        """
        Sort assumptions by importance (confidence and impact).

        Args:
            assumptions: List of assumptions

        Returns:
            Sorted list
        """
        def assumption_priority(a: Assumption) -> int:
            # Higher priority = lower number (sorts first)
            score = 0

            # Confidence priority
            if a.confidence == 'low':
                score += 10  # High priority if low confidence
            elif a.confidence == 'medium':
                score += 5
            else:  # high confidence
                score += 1

            # Impact priority
            if 'entirely' in a.impact_if_incorrect.lower():
                score += 10
            elif 'different' in a.impact_if_incorrect.lower():
                score += 5

            return -score  # Negate so higher scores come first

        return sorted(assumptions, key=assumption_priority)

    def _count_assumption_types(
        self,
        assumptions: List[Assumption]
    ) -> Dict[str, int]:
        """
        Count assumptions by type.

        Args:
            assumptions: List of assumptions

        Returns:
            Dictionary of type counts
        """
        counts = {}

        for assumption in assumptions:
            assumption_type = assumption.assumption_type
            counts[assumption_type] = counts.get(assumption_type, 0) + 1

        return counts

    def format_assumptions(self, extraction_result: Dict) -> str:
        """
        Format assumptions for display.

        Args:
            extraction_result: Result from extract_assumptions()

        Returns:
            Formatted assumptions text
        """
        assumptions = extraction_result.get('assumptions', [])

        if not assumptions:
            return "No specific assumptions identified."

        formatted = "ASSUMPTIONS IN LEGAL ANALYSIS\n"
        formatted += "=" * 60 + "\n\n"

        formatted += f"Total assumptions: {len(assumptions)}\n\n"

        for i, assumption in enumerate(assumptions, 1):
            formatted += f"{i}. {assumption['assumption_text']}\n"
            formatted += f"   Type: {assumption['assumption_type']}\n"
            formatted += f"   Impact if incorrect: {assumption['impact_if_incorrect']}\n"
            formatted += f"   How to verify: {assumption['verification_method']}\n"
            formatted += f"   Confidence: {assumption['confidence']}\n\n"

        formatted += "=" * 60 + "\n"
        formatted += "IMPORTANT: Review these assumptions with a legal professional.\n"
        formatted += "If any assumption is incorrect, the analysis may not apply.\n"

        return formatted

    def get_critical_assumptions(self, extraction_result: Dict) -> List[Dict]:
        """
        Get only critical assumptions (low confidence or high impact).

        Args:
            extraction_result: Result from extract_assumptions()

        Returns:
            List of critical assumptions
        """
        assumptions = extraction_result.get('assumptions', [])

        critical = [
            a for a in assumptions
            if a['confidence'] == 'low' or
               'entirely' in a['impact_if_incorrect'].lower() or
               'may not apply' in a['impact_if_incorrect'].lower()
        ]

        return critical

    def create_assumption_checklist(self, extraction_result: Dict) -> str:
        """
        Create checklist for users to verify assumptions.

        Args:
            extraction_result: Result from extract_assumptions()

        Returns:
            Checklist text
        """
        assumptions = extraction_result.get('assumptions', [])

        checklist = "ASSUMPTION VERIFICATION CHECKLIST\n"
        checklist += "=" * 60 + "\n\n"
        checklist += "Please verify the following assumptions before relying on this analysis:\n\n"

        for i, assumption in enumerate(assumptions, 1):
            checklist += f"[ ] {i}. {assumption['assumption_text']}\n"
            checklist += f"    Verification: {assumption['verification_method']}\n\n"

        checklist += "=" * 60 + "\n"
        checklist += "Consult a legal professional if you cannot verify these assumptions.\n"

        return checklist
