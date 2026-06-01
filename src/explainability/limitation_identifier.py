"""
Limitation Identifier Module

Identifies and documents limitations of the legal analysis.
Makes clear what the system cannot do or what factors limit
the reliability of the guidance.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Limitation:
    """Single limitation of legal analysis."""
    limitation_text: str
    limitation_type: str
    reason: str
    recommendation: str
    severity: str  # critical, high, medium, low
    source: str  # Where limitation was detected


class LimitationIdentifier:
    """
    Identify and document limitations of legal analysis.

    Identifies limitations from:
    - Evidence coverage gaps
    - Verification failures
    - Query ambiguity
    - System knowledge boundaries
    - Temporal factors
    - Jurisdictional scope
    """

    def __init__(
        self,
        include_standard_limitations: bool = True,
        severity_threshold: str = 'low'
    ):
        """
        Initialize limitation identifier.

        Args:
            include_standard_limitations: Include standard system limitations
            severity_threshold: Minimum severity to include (low, medium, high, critical)
        """
        self.include_standard_limitations = include_standard_limitations
        self.severity_threshold = severity_threshold
        logger.info("LimitationIdentifier initialized")

    def identify_limitations(
        self,
        query: str,
        answer: str,
        evidence_plan: Optional[Dict] = None,
        verification_result: Optional[Dict] = None,
        confidence_result: Optional[Dict] = None,
        inference_result: Optional[Dict] = None
    ) -> Dict:
        """
        Identify all limitations of legal analysis.

        Args:
            query: User query
            answer: Generated answer
            evidence_plan: Evidence plan from planner
            verification_result: Verification results
            confidence_result: Confidence scoring results
            inference_result: Inference engine results

        Returns:
            Dictionary with identified limitations
        """
        logger.info("Identifying limitations of legal analysis")

        limitations = []

        # 1. Evidence-based limitations
        if evidence_plan:
            evidence_limitations = self._identify_evidence_limitations(
                evidence_plan
            )
            limitations.extend(evidence_limitations)

        # 2. Verification-based limitations
        if verification_result:
            verification_limitations = self._identify_verification_limitations(
                verification_result
            )
            limitations.extend(verification_limitations)

        # 3. Confidence-based limitations
        if confidence_result:
            confidence_limitations = self._identify_confidence_limitations(
                confidence_result
            )
            limitations.extend(confidence_limitations)

        # 4. Inference-based limitations
        if inference_result:
            inference_limitations = self._identify_inference_limitations(
                inference_result
            )
            limitations.extend(inference_limitations)

        # 5. Query-based limitations
        query_limitations = self._identify_query_limitations(query)
        limitations.extend(query_limitations)

        # 6. System limitations
        if self.include_standard_limitations:
            system_limitations = self._get_standard_limitations()
            limitations.extend(system_limitations)

        # Filter by severity threshold
        filtered_limitations = self._filter_by_severity(limitations)

        # Sort by severity
        sorted_limitations = self._sort_by_severity(filtered_limitations)

        result = {
            'limitations': [asdict(l) for l in sorted_limitations],
            'total_limitations': len(sorted_limitations),
            'limitation_types': self._count_limitation_types(sorted_limitations),
            'critical_count': len([
                l for l in sorted_limitations if l.severity == 'critical'
            ]),
            'has_critical_limitations': any(
                l.severity == 'critical' for l in sorted_limitations
            )
        }

        logger.info(
            f"Identified {len(sorted_limitations)} limitations "
            f"({result['critical_count']} critical)"
        )

        return result

    def _identify_evidence_limitations(
        self,
        evidence_plan: Dict
    ) -> List[Limitation]:
        """
        Identify limitations from evidence coverage.

        Args:
            evidence_plan: Evidence plan

        Returns:
            List of limitations
        """
        limitations = []

        # Check number of passages
        num_passages = len(evidence_plan.get('passages', []))

        if num_passages == 0:
            limitations.append(Limitation(
                limitation_text='No relevant legal passages found in knowledge base',
                limitation_type='evidence_coverage',
                reason='Query may be outside system knowledge or too specific',
                recommendation='Consult legal databases and qualified legal practitioner',
                severity='critical',
                source='evidence_plan'
            ))
        elif num_passages < 3:
            limitations.append(Limitation(
                limitation_text='Limited legal passages retrieved (fewer than 3)',
                limitation_type='evidence_coverage',
                reason='Knowledge base may have incomplete coverage of this topic',
                recommendation='Verify with comprehensive legal research',
                severity='high',
                source='evidence_plan'
            ))

        # Check evidence sufficiency
        is_sufficient = evidence_plan.get('is_sufficient', True)
        if not is_sufficient:
            limitations.append(Limitation(
                limitation_text='Retrieved evidence deemed insufficient for complete analysis',
                limitation_type='evidence_sufficiency',
                reason='Evidence planner flagged insufficient supporting passages',
                recommendation='Additional legal research required for comprehensive analysis',
                severity='high',
                source='evidence_plan'
            ))

        # Check for evidence gaps
        has_gaps = evidence_plan.get('has_gaps', False)
        if has_gaps:
            limitations.append(Limitation(
                limitation_text='Gaps identified in available evidence',
                limitation_type='evidence_gaps',
                reason='Some aspects of query may not be fully addressed',
                recommendation='Seek additional legal sources and expert consultation',
                severity='medium',
                source='evidence_plan'
            ))

        return limitations

    def _identify_verification_limitations(
        self,
        verification_result: Dict
    ) -> List[Limitation]:
        """
        Identify limitations from verification results.

        Args:
            verification_result: Verification results

        Returns:
            List of limitations
        """
        limitations = []

        # Check faithfulness
        faithfulness = verification_result.get('faithfulness', {})
        faith_score = faithfulness.get('faithfulness_score', 1.0)

        if faith_score < 0.7:
            severity = 'critical' if faith_score < 0.5 else 'high'
            limitations.append(Limitation(
                limitation_text=f'Lower confidence in answer faithfulness ({faith_score:.1%})',
                limitation_type='verification_faithfulness',
                reason='Some answer claims may not be fully supported by evidence',
                recommendation='Verify all claims with original legal sources',
                severity=severity,
                source='verification'
            ))

        # Check for hallucinations
        hallucinations = faithfulness.get('hallucinations', [])
        if hallucinations:
            limitations.append(Limitation(
                limitation_text=f'{len(hallucinations)} potential unsupported claim(s) detected',
                limitation_type='verification_hallucination',
                reason='Answer may contain claims not grounded in retrieved evidence',
                recommendation='Critical: Verify with legal expert before relying on answer',
                severity='critical',
                source='verification'
            ))

        # Check citations
        citations = verification_result.get('citations', {})
        if not citations.get('all_valid', True):
            fabricated = citations.get('fabricated_citations', [])
            limitations.append(Limitation(
                limitation_text=f'{len(fabricated)} citation(s) could not be validated',
                limitation_type='verification_citations',
                reason='Some citations may not correspond to actual legal provisions',
                recommendation='Verify all citations with official legal sources',
                severity='critical',
                source='verification'
            ))

        return limitations

    def _identify_confidence_limitations(
        self,
        confidence_result: Dict
    ) -> List[Limitation]:
        """
        Identify limitations from confidence scores.

        Args:
            confidence_result: Confidence results

        Returns:
            List of limitations
        """
        limitations = []

        confidence = confidence_result.get('aggregate_confidence', 1.0)
        level = confidence_result.get('confidence_level', 'high')

        if confidence < 0.6:
            severity = 'critical' if confidence < 0.4 else 'high'
            limitations.append(Limitation(
                limitation_text=f'Overall confidence is {level} ({confidence:.1%})',
                limitation_type='confidence_score',
                reason='Multiple quality indicators show concerns about answer reliability',
                recommendation='Treat as preliminary guidance only; seek professional verification',
                severity=severity,
                source='confidence'
            ))

        # Check component scores
        components = confidence_result.get('component_scores', {})

        if components.get('retrieval_score', 1.0) < 0.5:
            limitations.append(Limitation(
                limitation_text='Low retrieval quality score',
                limitation_type='retrieval_quality',
                reason='Retrieved passages may not be optimally relevant',
                recommendation='Expand search with different query formulations',
                severity='medium',
                source='confidence'
            ))

        return limitations

    def _identify_inference_limitations(
        self,
        inference_result: Dict
    ) -> List[Limitation]:
        """
        Identify limitations from inference engine.

        Args:
            inference_result: Inference results

        Returns:
            List of limitations
        """
        limitations = []

        # Check for conflicts
        has_conflicts = inference_result.get('has_conflicts', False)
        if has_conflicts:
            conflicts = inference_result.get('conflicts', [])
            limitations.append(Limitation(
                limitation_text=f'{len(conflicts)} legal conflict(s) detected between provisions',
                limitation_type='legal_conflicts',
                reason='Multiple provisions may apply with potentially different outcomes',
                recommendation='Legal expert required to resolve conflicts and determine precedence',
                severity='critical',
                source='inference'
            ))

        # Check for ambiguity
        if inference_result.get('has_ambiguity', False):
            limitations.append(Limitation(
                limitation_text='Ambiguity in application of legal rules',
                limitation_type='legal_ambiguity',
                reason='Legal provisions may be interpreted in multiple ways',
                recommendation='Professional legal interpretation required',
                severity='high',
                source='inference'
            ))

        return limitations

    def _identify_query_limitations(self, query: str) -> List[Limitation]:
        """
        Identify limitations from query characteristics.

        Args:
            query: User query

        Returns:
            List of limitations
        """
        limitations = []

        # Check query length
        word_count = len(query.split())

        if word_count < 10:
            limitations.append(Limitation(
                limitation_text='Query is very brief (less than 10 words)',
                limitation_type='query_completeness',
                reason='Limited information may result in generic or incomplete analysis',
                recommendation='Provide more detailed factual context for better analysis',
                severity='medium',
                source='query'
            ))

        # Check for vague language
        vague_indicators = ['maybe', 'might', 'possibly', 'unclear', 'unsure', 'think']
        if any(indicator in query.lower() for indicator in vague_indicators):
            limitations.append(Limitation(
                limitation_text='Query contains uncertainty or vague language',
                limitation_type='query_clarity',
                reason='Uncertain facts may limit precision of legal analysis',
                recommendation='Clarify factual details before relying on analysis',
                severity='medium',
                source='query'
            ))

        return limitations

    def _get_standard_limitations(self) -> List[Limitation]:
        """
        Get standard limitations that apply to all analyses.

        Returns:
            List of standard limitations
        """
        return [
            Limitation(
                limitation_text='This is legal information, not personalized legal advice',
                limitation_type='system_scope',
                reason='System cannot provide case-specific professional legal advice',
                recommendation='Consult qualified legal practitioner for advice on your situation',
                severity='critical',
                source='standard'
            ),
            Limitation(
                limitation_text='Knowledge base may not include latest amendments or case law',
                limitation_type='system_currency',
                reason='Legal corpus has fixed update date; laws may have changed',
                recommendation='Verify currency of law with official legal databases',
                severity='high',
                source='standard'
            ),
            Limitation(
                limitation_text='Analysis based on AI interpretation of legal texts',
                limitation_type='system_nature',
                reason='AI systems may misinterpret complex legal reasoning',
                recommendation='Human legal expert review essential for important matters',
                severity='high',
                source='standard'
            ),
            Limitation(
                limitation_text='System has no knowledge of unpublished cases or practical implementation',
                limitation_type='system_coverage',
                reason='Knowledge limited to documented legal sources in corpus',
                recommendation='Legal practitioners have additional practical knowledge',
                severity='medium',
                source='standard'
            ),
            Limitation(
                limitation_text='Analysis assumes Sri Lankan jurisdiction only',
                limitation_type='jurisdictional',
                reason='System designed for Sri Lankan law; other jurisdictions not covered',
                recommendation='Consult local legal experts for other jurisdictions',
                severity='medium',
                source='standard'
            )
        ]

    def _filter_by_severity(self, limitations: List[Limitation]) -> List[Limitation]:
        """
        Filter limitations by severity threshold.

        Args:
            limitations: List of limitations

        Returns:
            Filtered list
        """
        severity_order = ['low', 'medium', 'high', 'critical']
        threshold_index = severity_order.index(self.severity_threshold)

        filtered = [
            l for l in limitations
            if severity_order.index(l.severity) >= threshold_index
        ]

        return filtered

    def _sort_by_severity(self, limitations: List[Limitation]) -> List[Limitation]:
        """
        Sort limitations by severity (critical first).

        Args:
            limitations: List of limitations

        Returns:
            Sorted list
        """
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

        return sorted(
            limitations,
            key=lambda l: severity_order.get(l.severity, 999)
        )

    def _count_limitation_types(
        self,
        limitations: List[Limitation]
    ) -> Dict[str, int]:
        """
        Count limitations by type.

        Args:
            limitations: List of limitations

        Returns:
            Dictionary of type counts
        """
        counts = {}

        for limitation in limitations:
            limitation_type = limitation.limitation_type
            counts[limitation_type] = counts.get(limitation_type, 0) + 1

        return counts

    def format_limitations(self, identification_result: Dict) -> str:
        """
        Format limitations for display.

        Args:
            identification_result: Result from identify_limitations()

        Returns:
            Formatted limitations text
        """
        limitations = identification_result.get('limitations', [])

        if not limitations:
            return "No specific limitations identified beyond standard system limitations."

        formatted = "LIMITATIONS OF LEGAL ANALYSIS\n"
        formatted += "=" * 60 + "\n\n"

        formatted += f"Total limitations: {len(limitations)}\n"

        critical_count = identification_result.get('critical_count', 0)
        if critical_count > 0:
            formatted += f"⚠️  CRITICAL LIMITATIONS: {critical_count}\n"

        formatted += "\n"

        # Group by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_limitations = [
                l for l in limitations if l['severity'] == severity
            ]

            if severity_limitations:
                formatted += f"\n{severity.upper()} SEVERITY:\n"
                formatted += "-" * 60 + "\n"

                for limitation in severity_limitations:
                    formatted += f"\n• {limitation['limitation_text']}\n"
                    formatted += f"  Reason: {limitation['reason']}\n"
                    formatted += f"  Recommendation: {limitation['recommendation']}\n"

        formatted += "\n" + "=" * 60 + "\n"
        formatted += "IMPORTANT: These limitations may affect the reliability of the analysis.\n"
        formatted += "Always consult a qualified legal professional for important legal matters.\n"

        return formatted

    def get_critical_limitations(self, identification_result: Dict) -> List[Dict]:
        """
        Get only critical limitations.

        Args:
            identification_result: Result from identify_limitations()

        Returns:
            List of critical limitations
        """
        limitations = identification_result.get('limitations', [])

        critical = [l for l in limitations if l['severity'] == 'critical']

        return critical

    def create_limitation_summary(self, identification_result: Dict) -> str:
        """
        Create brief summary of limitations.

        Args:
            identification_result: Result from identify_limitations()

        Returns:
            Summary text
        """
        total = identification_result.get('total_limitations', 0)
        critical = identification_result.get('critical_count', 0)

        if total == 0:
            return "No specific limitations identified."

        summary = f"{total} limitation(s) identified"

        if critical > 0:
            summary += f", including {critical} CRITICAL limitation(s)"

        summary += ". "

        if critical > 0:
            summary += "⚠️  Critical limitations may significantly affect analysis reliability. "

        summary += "Review all limitations before relying on this analysis."

        return summary

    def has_blocking_limitations(self, identification_result: Dict) -> bool:
        """
        Check if there are limitations that should block answer delivery.

        Args:
            identification_result: Result from identify_limitations()

        Returns:
            True if blocking limitations exist
        """
        limitations = identification_result.get('limitations', [])

        # Check for specific blocking limitation types
        blocking_types = [
            'verification_hallucination',
            'verification_citations'
        ]

        for limitation in limitations:
            if limitation['limitation_type'] in blocking_types:
                if limitation['severity'] == 'critical':
                    return True

        return False
