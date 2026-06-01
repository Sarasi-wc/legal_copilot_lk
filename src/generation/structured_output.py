"""
Structured output formatter for legal answers.
Implements structured legal advice format as specified in Diagram 1.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from src.utils import get_logger

logger = get_logger(__name__)


class ActionType(Enum):
    """Types of legal actions."""
    FILE_DOCUMENT = "file_document"
    CONSULT_LAWYER = "consult_lawyer"
    GATHER_EVIDENCE = "gather_evidence"
    ATTEND_HEARING = "attend_hearing"
    SUBMIT_APPLICATION = "submit_application"
    APPEAL = "appeal"


@dataclass
class Citation:
    """Legal citation."""
    act_name: str
    section_number: str
    text_excerpt: str
    relevance: str


@dataclass
class ActionableStep:
    """Single actionable step for user."""
    step_number: int
    action_type: ActionType
    description: str
    deadline: Optional[str]
    required_documents: List[str]
    estimated_cost: Optional[str]
    priority: str  # high, medium, low


@dataclass
class Assumption:
    """Assumption made in legal analysis."""
    assumption: str
    impact_if_incorrect: str
    verification_method: str


@dataclass
class Limitation:
    """Limitation of the legal guidance."""
    limitation: str
    reason: str
    recommendation: str


@dataclass
class StructuredLegalOutput:
    """Complete structured legal output."""
    # Summary
    query: str
    summary: str
    applicable_law: str

    # Reasoning
    reasoning_steps: List[str]
    relevant_provisions: List[Citation]

    # Actionable guidance
    next_steps: List[ActionableStep]
    forum_guidance: Optional[Dict]  # Which court/tribunal

    # Metadata
    assumptions: List[Assumption]
    limitations: List[Limitation]
    warnings: List[str]
    confidence_score: float

    # Source attribution
    sources_consulted: List[str]
    last_updated: str


class StructuredOutputFormatter:
    """
    Formats legal answers into structured output.
    Implements structured legal advice format from Diagram 1.
    """

    def __init__(self):
        """Initialize formatter."""
        pass

    def format_answer(
        self,
        query: str,
        answer_text: str,
        citations: List[Dict],
        evidence_plan: Dict,
        inference_result: Optional[Dict] = None,
        verification: Optional[Dict] = None
    ) -> Dict:
        """
        Format answer into structured output.

        Args:
            query: User query
            answer_text: Generated answer text
            citations: List of citations
            evidence_plan: Evidence plan from planner
            inference_result: Results from inference engine
            verification: Verification results

        Returns:
            Structured output dictionary
        """
        logger.info("Formatting structured output")

        # Extract summary (first 2 sentences)
        summary = self._extract_summary(answer_text)

        # Identify applicable law
        applicable_law = self._identify_applicable_law(citations)

        # Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(
            answer_text,
            inference_result
        )

        # Format citations
        relevant_provisions = self._format_citations(citations, evidence_plan)

        # Generate next steps
        next_steps = self._generate_next_steps(answer_text, citations)

        # Generate forum guidance
        forum_guidance = self._generate_forum_guidance(citations)

        # Extract assumptions
        assumptions = self._extract_assumptions(answer_text)

        # Identify limitations
        limitations = self._identify_limitations(
            evidence_plan,
            verification
        )

        # Generate warnings
        warnings = self._generate_warnings(
            inference_result,
            verification
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            evidence_plan,
            verification,
            inference_result
        )

        # Format sources
        sources = self._format_sources(citations)

        structured = StructuredLegalOutput(
            query=query,
            summary=summary,
            applicable_law=applicable_law,
            reasoning_steps=reasoning_steps,
            relevant_provisions=relevant_provisions,
            next_steps=next_steps,
            forum_guidance=forum_guidance,
            assumptions=assumptions,
            limitations=limitations,
            warnings=warnings,
            confidence_score=confidence,
            sources_consulted=sources,
            last_updated=self._get_timestamp()
        )

        return asdict(structured)

    def _extract_summary(self, answer_text: str) -> str:
        """Extract 1-2 sentence summary."""
        sentences = answer_text.split('.')
        if len(sentences) >= 2:
            return '. '.join(sentences[:2]) + '.'
        return answer_text[:200] + '...'

    def _identify_applicable_law(self, citations: List[Dict]) -> str:
        """Identify primary applicable law."""
        if citations:
            return f"{citations[0]['act_name']}, Section {citations[0]['section']}"
        return "General legal principles"

    def _extract_reasoning_steps(
        self,
        answer_text: str,
        inference_result: Optional[Dict]
    ) -> List[str]:
        """Extract step-by-step reasoning."""
        steps = []

        # Use inference engine reasoning if available
        if inference_result and 'reasoning_chain' in inference_result:
            return inference_result['reasoning_chain']

        # Otherwise extract from answer text
        # Look for numbered steps or paragraphs
        paragraphs = answer_text.split('\n\n')
        for i, para in enumerate(paragraphs[:5], 1):
            if len(para.strip()) > 20:
                steps.append(f"Step {i}: {para.strip()[:150]}...")

        if not steps:
            steps.append("The legal analysis is based on the cited provisions.")

        return steps

    def _format_citations(
        self,
        citations: List[Dict],
        evidence_plan: Dict
    ) -> List[Citation]:
        """Format citations with excerpts."""
        formatted = []

        for citation in citations:
            # Find matching passage
            passage = None
            for p in evidence_plan.get('passages', []):
                if (p.get('act_name') == citation['act_name'] and
                    p['metadata'].get('section_number') == citation['section']):
                    passage = p
                    break

            excerpt = passage['text'][:200] + '...' if passage else "Full text in source"

            formatted.append(Citation(
                act_name=citation['act_name'],
                section_number=citation['section'],
                text_excerpt=excerpt,
                relevance="Directly applicable to query"
            ))

        return [asdict(c) for c in formatted]

    def _generate_next_steps(
        self,
        answer_text: str,
        citations: List[Dict]
    ) -> List[ActionableStep]:
        """Generate actionable next steps."""
        steps = []

        # Analyze answer for action keywords
        text_lower = answer_text.lower()

        step_num = 1

        # Check for filing requirements
        if 'file' in text_lower or 'plaint' in text_lower or 'petition' in text_lower:
            steps.append(ActionableStep(
                step_number=step_num,
                action_type=ActionType.FILE_DOCUMENT,
                description="Prepare and file required legal documents",
                deadline="Within statutory limitation period",
                required_documents=["Plaint/Petition", "Supporting affidavits", "Relevant documents"],
                estimated_cost="Court filing fees applicable",
                priority="high"
            ))
            step_num += 1

        # Always recommend consulting lawyer for complex matters
        if len(citations) > 2:
            steps.append(ActionableStep(
                step_number=step_num,
                action_type=ActionType.CONSULT_LAWYER,
                description="Consult qualified legal practitioner for case-specific advice",
                deadline="As soon as possible",
                required_documents=["All relevant documents", "Timeline of events"],
                estimated_cost="Varies by practitioner",
                priority="high"
            ))
            step_num += 1

        # Evidence gathering
        if 'evidence' in text_lower or 'proof' in text_lower:
            steps.append(ActionableStep(
                step_number=step_num,
                action_type=ActionType.GATHER_EVIDENCE,
                description="Collect and preserve relevant evidence",
                deadline="Before filing suit",
                required_documents=["Documents", "Witness statements", "Physical evidence"],
                estimated_cost="Minimal",
                priority="high"
            ))
            step_num += 1

        # If no specific steps, provide general guidance
        if not steps:
            steps.append(ActionableStep(
                step_number=1,
                action_type=ActionType.CONSULT_LAWYER,
                description="Seek legal advice for your specific situation",
                deadline="No specific deadline",
                required_documents=["Query details"],
                estimated_cost="Consultation fee",
                priority="medium"
            ))

        return [asdict(s) for s in steps]

    def _generate_forum_guidance(self, citations: List[Dict]) -> Optional[Dict]:
        """Generate guidance on which forum/court to approach."""
        if not citations:
            return None

        # Simple heuristics based on cited Acts
        act_name = citations[0]['act_name'].lower()

        if 'penal' in act_name or 'criminal' in act_name:
            return {
                'forum': 'Magistrate Court / High Court',
                'jurisdiction': 'Criminal',
                'reason': 'Criminal matters under Penal Code',
                'additional_info': 'Severity of offense determines court level'
            }
        elif 'civil' in act_name:
            return {
                'forum': 'District Court / High Court',
                'jurisdiction': 'Civil',
                'reason': 'Civil matters under Civil Procedure Code',
                'additional_info': 'Monetary value determines court level'
            }
        else:
            return {
                'forum': 'Consult lawyer for appropriate forum',
                'jurisdiction': 'To be determined',
                'reason': 'Forum depends on specific case details',
                'additional_info': 'Multiple forums may have jurisdiction'
            }

    def _extract_assumptions(self, answer_text: str) -> List[Assumption]:
        """Extract assumptions from answer."""
        assumptions = []

        # Look for assumption keywords
        assumption_keywords = ['assuming', 'if', 'provided that', 'where']

        text_lower = answer_text.lower()
        for keyword in assumption_keywords:
            if keyword in text_lower:
                assumptions.append(Assumption(
                    assumption=f"Analysis assumes certain facts based on query interpretation",
                    impact_if_incorrect="Different legal provisions may apply",
                    verification_method="Consult lawyer with complete case details"
                ))
                break  # Only add once

        # Always include general assumption
        if not assumptions:
            assumptions.append(Assumption(
                assumption="Query accurately represents the factual situation",
                impact_if_incorrect="Legal analysis may not be applicable",
                verification_method="Verify facts with legal documents"
            ))

        return [asdict(a) for a in assumptions]

    def _identify_limitations(
        self,
        evidence_plan: Dict,
        verification: Optional[Dict]
    ) -> List[Limitation]:
        """Identify limitations of the guidance."""
        limitations = []

        # Evidence-based limitations
        if evidence_plan.get('filtered_passages', 0) < 3:
            limitations.append(Limitation(
                limitation="Limited statutory passages retrieved",
                reason="Query may be too broad or specific provisions not in corpus",
                recommendation="Consult official legal databases for comprehensive research"
            ))

        # Verification-based limitations
        if verification and verification.get('faithfulness', {}).get('faithfulness_score', 1.0) < 0.8:
            limitations.append(Limitation(
                limitation="Lower confidence in answer faithfulness",
                reason="Some answer claims may not be fully supported by evidence",
                recommendation="Verify with original statutes and legal expert"
            ))

        # General limitation
        limitations.append(Limitation(
            limitation="This is legal information, not legal advice",
            reason="System cannot provide personalized legal advice",
            recommendation="Consult qualified legal practitioner for case-specific guidance"
        ))

        return [asdict(l) for l in limitations]

    def _generate_warnings(
        self,
        inference_result: Optional[Dict],
        verification: Optional[Dict]
    ) -> List[str]:
        """Generate warnings for user."""
        warnings = []

        # Conflict warnings
        if inference_result and inference_result.get('has_conflicts'):
            warnings.append(
                "⚠️ CONFLICT DETECTED: Multiple provisions may apply with different outcomes. "
                "Seek legal advice immediately."
            )

        # Verification warnings
        if verification and not verification.get('verification_passed'):
            warnings.append(
                "⚠️ VERIFICATION ISSUES: Some answer claims could not be fully verified. "
                "Cross-check with official sources."
            )

        # Always include disclaimer
        warnings.append(
            "⚠️ DISCLAIMER: This information is for educational purposes only. "
            "It does not constitute legal advice. Consult a qualified legal practitioner."
        )

        return warnings

    def _calculate_confidence(
        self,
        evidence_plan: Dict,
        verification: Optional[Dict],
        inference_result: Optional[Dict]
    ) -> float:
        """Calculate overall confidence score."""
        scores = []

        # Evidence sufficiency
        if evidence_plan.get('is_sufficient'):
            scores.append(0.8)
        else:
            scores.append(0.3)

        # Verification score
        if verification:
            faith_score = verification.get('faithfulness', {}).get('faithfulness_score', 0.5)
            scores.append(faith_score)

        # Inference confidence (no conflicts = higher confidence)
        if inference_result:
            if inference_result.get('has_conflicts'):
                scores.append(0.4)
            else:
                scores.append(0.9)

        return sum(scores) / len(scores) if scores else 0.5

    def _format_sources(self, citations: List[Dict]) -> List[str]:
        """Format source list."""
        sources = []
        seen = set()

        for citation in citations:
            source = f"{citation['act_name']}, Section {citation['section']}"
            if source not in seen:
                sources.append(source)
                seen.add(source)

        return sources

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
