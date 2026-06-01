"""
Review Trigger Module

Flags cases that require human review based on:
- Low confidence scores
- Detected conflicts in law
- Verification failures
- Complex legal scenarios
- High-stakes queries
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
from src.utils import get_logger

logger = get_logger(__name__)


class ReviewPriority(Enum):
    """Priority levels for human review."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewReason(Enum):
    """Reasons for triggering human review."""
    LOW_CONFIDENCE = "low_confidence"
    LEGAL_CONFLICT = "legal_conflict"
    VERIFICATION_FAILURE = "verification_failure"
    COMPLEX_QUERY = "complex_query"
    HIGH_STAKES = "high_stakes"
    HALLUCINATION_DETECTED = "hallucination_detected"
    CITATION_ERRORS = "citation_errors"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AMBIGUOUS_LAW = "ambiguous_law"


class ReviewTrigger:
    """
    Flag cases for human review based on various criteria.

    Implements a rule-based system to identify outputs that require
    expert human review before being presented to users.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        faithfulness_threshold: float = 0.7,
        citation_threshold: float = 0.8,
        enable_auto_flagging: bool = True
    ):
        """
        Initialize review trigger.

        Args:
            confidence_threshold: Minimum confidence to avoid review
            faithfulness_threshold: Minimum faithfulness to avoid review
            citation_threshold: Minimum citation validity to avoid review
            enable_auto_flagging: Enable automatic flagging
        """
        self.confidence_threshold = confidence_threshold
        self.faithfulness_threshold = faithfulness_threshold
        self.citation_threshold = citation_threshold
        self.enable_auto_flagging = enable_auto_flagging

        logger.info("ReviewTrigger initialized")

    def check_review_needed(
        self,
        query: str,
        answer: str,
        confidence_result: Optional[Dict] = None,
        verification_result: Optional[Dict] = None,
        inference_result: Optional[Dict] = None,
        evidence_plan: Optional[Dict] = None
    ) -> Dict:
        """
        Check if human review is needed.

        Args:
            query: User query
            answer: Generated answer
            confidence_result: Confidence scoring results
            verification_result: Verification results
            inference_result: Inference engine results
            evidence_plan: Evidence planning results

        Returns:
            Review decision dictionary
        """
        if not self.enable_auto_flagging:
            return {
                'review_needed': False,
                'reason': 'Auto-flagging disabled',
                'priority': None,
                'triggers': []
            }

        logger.info("Checking if human review is needed")

        # Collect all review triggers
        triggers = []

        # 1. Check confidence score
        confidence_trigger = self._check_confidence(confidence_result)
        if confidence_trigger:
            triggers.append(confidence_trigger)

        # 2. Check verification results
        verification_trigger = self._check_verification(verification_result)
        if verification_trigger:
            triggers.append(verification_trigger)

        # 3. Check for legal conflicts
        conflict_trigger = self._check_conflicts(inference_result)
        if conflict_trigger:
            triggers.append(conflict_trigger)

        # 4. Check evidence sufficiency
        evidence_trigger = self._check_evidence(evidence_plan)
        if evidence_trigger:
            triggers.append(evidence_trigger)

        # 5. Check query complexity
        complexity_trigger = self._check_query_complexity(query)
        if complexity_trigger:
            triggers.append(complexity_trigger)

        # 6. Check for high-stakes indicators
        stakes_trigger = self._check_high_stakes(query, answer)
        if stakes_trigger:
            triggers.append(stakes_trigger)

        # Determine if review is needed
        review_needed = len(triggers) > 0

        # Determine priority (highest among triggers)
        priority = self._determine_priority(triggers) if review_needed else None

        # Generate review recommendation
        recommendation = self._generate_recommendation(triggers)

        result = {
            'review_needed': review_needed,
            'priority': priority.value if priority else None,
            'num_triggers': len(triggers),
            'triggers': triggers,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'auto_flagged': self.enable_auto_flagging
        }

        if review_needed:
            logger.warning(
                f"Human review flagged: {len(triggers)} triggers, "
                f"priority={priority.value if priority else 'none'}"
            )
        else:
            logger.info("No human review needed")

        return result

    def _check_confidence(self, confidence_result: Optional[Dict]) -> Optional[Dict]:
        """Check confidence score trigger."""
        if not confidence_result:
            return None

        confidence = confidence_result.get('aggregate_confidence', 1.0)

        if confidence < self.confidence_threshold:
            severity = 'critical' if confidence < 0.4 else 'high'

            return {
                'reason': ReviewReason.LOW_CONFIDENCE.value,
                'priority': ReviewPriority.CRITICAL.value if confidence < 0.4 else ReviewPriority.HIGH.value,
                'description': f"Confidence score ({confidence:.2%}) below threshold ({self.confidence_threshold:.2%})",
                'severity': severity,
                'confidence_score': confidence,
                'threshold': self.confidence_threshold
            }

        return None

    def _check_verification(self, verification_result: Optional[Dict]) -> Optional[Dict]:
        """Check verification results trigger."""
        if not verification_result:
            return None

        triggers = []

        # Check faithfulness
        faithfulness = verification_result.get('faithfulness', {})
        faith_score = faithfulness.get('faithfulness_score', 1.0)

        if faith_score < self.faithfulness_threshold:
            triggers.append({
                'reason': ReviewReason.HALLUCINATION_DETECTED.value,
                'priority': ReviewPriority.CRITICAL.value,
                'description': f"Low faithfulness score ({faith_score:.2%}), possible hallucinations",
                'severity': 'critical',
                'faithfulness_score': faith_score,
                'hallucinations': faithfulness.get('hallucinations', [])
            })

        # Check citations
        citations = verification_result.get('citations', {})
        citation_rate = citations.get('validation_rate', 1.0)

        if citation_rate < self.citation_threshold:
            triggers.append({
                'reason': ReviewReason.CITATION_ERRORS.value,
                'priority': ReviewPriority.HIGH.value,
                'description': f"Citation validation rate ({citation_rate:.2%}) below threshold",
                'severity': 'high',
                'citation_rate': citation_rate,
                'fabricated_citations': citations.get('fabricated_citations', [])
            })

        # Return first trigger (if any)
        return triggers[0] if triggers else None

    def _check_conflicts(self, inference_result: Optional[Dict]) -> Optional[Dict]:
        """Check for legal conflicts trigger."""
        if not inference_result:
            return None

        has_conflicts = inference_result.get('has_conflicts', False)

        if has_conflicts:
            conflicts = inference_result.get('conflicts', [])
            num_conflicts = len(conflicts)

            return {
                'reason': ReviewReason.LEGAL_CONFLICT.value,
                'priority': ReviewPriority.CRITICAL.value,
                'description': f"{num_conflicts} legal conflict(s) detected",
                'severity': 'critical',
                'num_conflicts': num_conflicts,
                'conflicts': conflicts
            }

        return None

    def _check_evidence(self, evidence_plan: Optional[Dict]) -> Optional[Dict]:
        """Check evidence sufficiency trigger."""
        if not evidence_plan:
            return None

        is_sufficient = evidence_plan.get('is_sufficient', True)
        num_passages = len(evidence_plan.get('passages', []))

        if not is_sufficient or num_passages < 2:
            return {
                'reason': ReviewReason.INSUFFICIENT_EVIDENCE.value,
                'priority': ReviewPriority.HIGH.value,
                'description': f"Insufficient evidence: only {num_passages} passage(s) found",
                'severity': 'high',
                'num_passages': num_passages,
                'is_sufficient': is_sufficient
            }

        return None

    def _check_query_complexity(self, query: str) -> Optional[Dict]:
        """Check query complexity trigger."""
        # Complex query indicators
        complexity_indicators = [
            'multiple',
            'conflict',
            'exception',
            'both',
            'either',
            'however',
            'unless',
            'notwithstanding',
            'complex',
            'ambiguous'
        ]

        query_lower = query.lower()
        matches = [ind for ind in complexity_indicators if ind in query_lower]

        # Also check query length
        word_count = len(query.split())

        if len(matches) >= 2 or word_count > 100:
            return {
                'reason': ReviewReason.COMPLEX_QUERY.value,
                'priority': ReviewPriority.MEDIUM.value,
                'description': f"Complex query detected: {len(matches)} complexity indicators, {word_count} words",
                'severity': 'medium',
                'complexity_indicators': matches,
                'word_count': word_count
            }

        return None

    def _check_high_stakes(self, query: str, answer: str) -> Optional[Dict]:
        """Check for high-stakes scenario trigger."""
        # High-stakes indicators
        high_stakes_keywords = [
            'criminal',
            'imprisonment',
            'jail',
            'death penalty',
            'life sentence',
            'property',
            'inheritance',
            'custody',
            'divorce',
            'deportation',
            'asylum',
            'refugee',
            'fundamental rights',
            'constitutional'
        ]

        combined_text = (query + " " + answer).lower()
        matches = [kw for kw in high_stakes_keywords if kw in combined_text]

        if matches:
            return {
                'reason': ReviewReason.HIGH_STAKES.value,
                'priority': ReviewPriority.HIGH.value,
                'description': f"High-stakes legal matter detected: {', '.join(matches[:3])}",
                'severity': 'high',
                'high_stakes_indicators': matches
            }

        return None

    def _determine_priority(self, triggers: List[Dict]) -> ReviewPriority:
        """
        Determine overall priority from triggers.

        Args:
            triggers: List of trigger dictionaries

        Returns:
            Overall priority level
        """
        if not triggers:
            return ReviewPriority.LOW

        priorities = [ReviewPriority(t['priority']) for t in triggers]

        # Return highest priority
        if ReviewPriority.CRITICAL in priorities:
            return ReviewPriority.CRITICAL
        elif ReviewPriority.HIGH in priorities:
            return ReviewPriority.HIGH
        elif ReviewPriority.MEDIUM in priorities:
            return ReviewPriority.MEDIUM
        else:
            return ReviewPriority.LOW

    def _generate_recommendation(self, triggers: List[Dict]) -> str:
        """
        Generate human-readable recommendation.

        Args:
            triggers: List of trigger dictionaries

        Returns:
            Recommendation text
        """
        if not triggers:
            return "No human review required. Output meets quality standards."

        recommendation = "HUMAN REVIEW RECOMMENDED\n\n"
        recommendation += f"Number of issues detected: {len(triggers)}\n\n"
        recommendation += "Issues:\n"

        for i, trigger in enumerate(triggers, 1):
            reason = trigger['reason']
            desc = trigger['description']
            priority = trigger['priority']

            recommendation += f"{i}. [{priority.upper()}] {reason}: {desc}\n"

        recommendation += "\nAction: Review by qualified legal expert before presenting to user."

        return recommendation

    def create_review_task(self, review_result: Dict, query: str, answer: str) -> Dict:
        """
        Create a review task for human reviewers.

        Args:
            review_result: Result from check_review_needed()
            query: User query
            answer: Generated answer

        Returns:
            Review task dictionary
        """
        if not review_result['review_needed']:
            return {}

        task = {
            'task_id': self._generate_task_id(),
            'created_at': datetime.now().isoformat(),
            'priority': review_result['priority'],
            'status': 'pending',
            'query': query,
            'answer': answer,
            'triggers': review_result['triggers'],
            'recommendation': review_result['recommendation'],
            'assigned_to': None,
            'reviewed_at': None,
            'reviewer_notes': None,
            'decision': None  # approve, reject, modify
        }

        logger.info(f"Review task created: {task['task_id']}")

        return task

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        import uuid
        return f"REVIEW-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

    def get_review_statistics(self, review_results: List[Dict]) -> Dict:
        """
        Calculate statistics from review results.

        Args:
            review_results: List of review result dictionaries

        Returns:
            Statistics dictionary
        """
        total = len(review_results)
        flagged = sum(1 for r in review_results if r['review_needed'])

        if total == 0:
            return {
                'total_checks': 0,
                'flagged_count': 0,
                'flagged_rate': 0.0,
                'priority_breakdown': {}
            }

        # Priority breakdown
        priority_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }

        for result in review_results:
            if result['review_needed']:
                priority = result.get('priority', 'low')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Reason breakdown
        reason_counts = {}
        for result in review_results:
            for trigger in result.get('triggers', []):
                reason = trigger['reason']
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            'total_checks': total,
            'flagged_count': flagged,
            'flagged_rate': flagged / total,
            'priority_breakdown': priority_counts,
            'reason_breakdown': reason_counts,
            'most_common_reason': max(reason_counts.items(), key=lambda x: x[1])[0] if reason_counts else None
        }
