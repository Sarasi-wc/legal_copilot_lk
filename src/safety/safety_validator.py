"""
Safety Validator - Orchestration Module

Orchestrates all safety checks and validations:
- Faithfulness checking (via verification module)
- Citation validation (via verification module)
- Confidence scoring
- Disclaimer enforcement
- Human review triggering

This is the main entry point for safety & validation layer.
"""

from typing import Dict, List, Optional
from src.utils import get_logger
from src.safety.confidence_scorer import ConfidenceScorer
from src.safety.disclaimer_enforcer import DisclaimerEnforcer
from src.safety.review_trigger import ReviewTrigger

logger = get_logger(__name__)


class SafetyValidator:
    """
    Orchestrate all safety checks and validations.

    Main safety layer orchestrator that coordinates:
    - Confidence scoring
    - Disclaimer enforcement
    - Human review triggering
    - Safety validation reporting
    """

    def __init__(
        self,
        faithfulness_checker=None,  # Will use from verification module
        citation_validator=None,    # Will use from verification module
        confidence_scorer: Optional[ConfidenceScorer] = None,
        disclaimer_enforcer: Optional[DisclaimerEnforcer] = None,
        review_trigger: Optional[ReviewTrigger] = None,
        enable_strict_mode: bool = True
    ):
        """
        Initialize safety validator.

        Args:
            faithfulness_checker: FaithfulnessChecker instance (from verification)
            citation_validator: CitationValidator instance (from verification)
            confidence_scorer: ConfidenceScorer instance
            disclaimer_enforcer: DisclaimerEnforcer instance
            review_trigger: ReviewTrigger instance
            enable_strict_mode: Enable strict validation (reject on failures)
        """
        self.faithfulness_checker = faithfulness_checker
        self.citation_validator = citation_validator
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.disclaimer_enforcer = disclaimer_enforcer or DisclaimerEnforcer()
        self.review_trigger = review_trigger or ReviewTrigger()
        self.enable_strict_mode = enable_strict_mode

        logger.info("SafetyValidator initialized")

    def validate_answer(
        self,
        query: str,
        answer: str,
        evidence_plan: Dict,
        inference_result: Optional[Dict] = None,
        retrieval_results: Optional[Dict] = None
    ) -> Dict:
        """
        Perform complete safety validation on answer.

        This is the main orchestration method that runs all safety checks.

        Args:
            query: User query
            answer: Generated answer
            evidence_plan: Evidence plan from planner
            inference_result: Results from inference engine
            retrieval_results: Results from retrieval system

        Returns:
            Complete validation results dictionary
        """
        logger.info("Starting comprehensive safety validation")

        validation_results = {
            'query': query,
            'validation_timestamp': self._get_timestamp(),
            'strict_mode': self.enable_strict_mode
        }

        # Step 1: Faithfulness checking
        logger.info("Step 1: Checking answer faithfulness")
        faithfulness_result = self._check_faithfulness(answer, evidence_plan)
        validation_results['faithfulness'] = faithfulness_result

        # Step 2: Citation validation
        logger.info("Step 2: Validating citations")
        citation_result = self._validate_citations(answer, evidence_plan)
        validation_results['citations'] = citation_result

        # Step 3: Confidence scoring
        logger.info("Step 3: Calculating confidence scores")
        confidence_result = self.confidence_scorer.calculate_confidence(
            retrieval_results=retrieval_results,
            faithfulness_results=faithfulness_result,
            citation_results=citation_result,
            evidence_plan=evidence_plan,
            inference_results=inference_result
        )
        validation_results['confidence'] = confidence_result

        # Step 4: Check if should abstain
        logger.info("Step 4: Checking abstention criteria")
        should_abstain = self.confidence_scorer.should_abstain(confidence_result)
        validation_results['should_abstain'] = should_abstain

        if should_abstain:
            logger.warning("Abstention recommended - confidence too low")
            validation_results['abstention_reason'] = self._get_abstention_reason(
                confidence_result
            )

        # Step 5: Disclaimer enforcement
        logger.info("Step 5: Enforcing disclaimers")
        disclaimer_result = self.disclaimer_enforcer.enforce_disclaimers(
            answer=answer,
            confidence_result=confidence_result,
            inference_result=inference_result,
            verification_result={
                'faithfulness': faithfulness_result,
                'citations': citation_result
            }
        )
        validation_results['disclaimers'] = disclaimer_result

        # Step 6: Human review triggering
        logger.info("Step 6: Checking human review triggers")
        review_result = self.review_trigger.check_review_needed(
            query=query,
            answer=answer,
            confidence_result=confidence_result,
            verification_result={
                'faithfulness': faithfulness_result,
                'citations': citation_result
            },
            inference_result=inference_result,
            evidence_plan=evidence_plan
        )
        validation_results['review'] = review_result

        # Step 7: Overall validation decision
        logger.info("Step 7: Making overall validation decision")
        validation_passed, validation_issues = self._make_validation_decision(
            faithfulness_result,
            citation_result,
            confidence_result,
            should_abstain,
            review_result
        )

        validation_results['validation_passed'] = validation_passed
        validation_results['validation_issues'] = validation_issues
        validation_results['final_answer'] = disclaimer_result['answer_with_disclaimers']

        # Log final decision
        if validation_passed:
            logger.info("✓ Safety validation PASSED")
        else:
            logger.warning(f"✗ Safety validation FAILED: {len(validation_issues)} issues")

        return validation_results

    def _check_faithfulness(
        self,
        answer: str,
        evidence_plan: Dict
    ) -> Dict:
        """
        Check answer faithfulness using NLI.

        Args:
            answer: Generated answer
            evidence_plan: Evidence plan with passages

        Returns:
            Faithfulness check results
        """
        if self.faithfulness_checker:
            try:
                result = self.faithfulness_checker.check_faithfulness(
                    answer=answer,
                    evidence_passages=evidence_plan.get('passages', [])
                )
                logger.info(
                    f"Faithfulness score: {result.get('faithfulness_score', 0):.2%}"
                )
                return result
            except Exception as e:
                logger.error(f"Faithfulness check failed: {e}")
                return {
                    'is_faithful': False,
                    'faithfulness_score': 0.0,
                    'error': str(e)
                }
        else:
            logger.warning("Faithfulness checker not available")
            return {
                'is_faithful': True,
                'faithfulness_score': 1.0,
                'note': 'Faithfulness checker not initialized'
            }

    def _validate_citations(
        self,
        answer: str,
        evidence_plan: Dict
    ) -> Dict:
        """
        Validate citations in answer.

        Args:
            answer: Generated answer
            evidence_plan: Evidence plan with passages

        Returns:
            Citation validation results
        """
        if self.citation_validator:
            try:
                result = self.citation_validator.validate_citations(
                    answer=answer,
                    evidence_plan=evidence_plan
                )
                logger.info(
                    f"Citation validation rate: {result.get('validation_rate', 0):.2%}"
                )
                return result
            except Exception as e:
                logger.error(f"Citation validation failed: {e}")
                return {
                    'all_valid': False,
                    'validation_rate': 0.0,
                    'error': str(e)
                }
        else:
            logger.warning("Citation validator not available")
            return {
                'all_valid': True,
                'validation_rate': 1.0,
                'note': 'Citation validator not initialized'
            }

    def _make_validation_decision(
        self,
        faithfulness_result: Dict,
        citation_result: Dict,
        confidence_result: Dict,
        should_abstain: bool,
        review_result: Dict
    ) -> tuple:
        """
        Make overall validation decision.

        Args:
            faithfulness_result: Faithfulness check results
            citation_result: Citation validation results
            confidence_result: Confidence scoring results
            should_abstain: Whether to abstain
            review_result: Review trigger results

        Returns:
            Tuple of (passed: bool, issues: List[str])
        """
        issues = []

        # Check faithfulness
        if not faithfulness_result.get('is_faithful', True):
            issues.append(
                f"Low faithfulness score: {faithfulness_result.get('faithfulness_score', 0):.2%}"
            )

        # Check citations
        if not citation_result.get('all_valid', True):
            fabricated = citation_result.get('fabricated_citations', [])
            if fabricated:
                issues.append(
                    f"Fabricated citations detected: {len(fabricated)} citations"
                )

        # Check confidence
        if not confidence_result.get('meets_minimum_threshold', True):
            issues.append(
                f"Confidence below threshold: {confidence_result.get('aggregate_confidence', 0):.2%}"
            )

        # Check abstention
        if should_abstain:
            issues.append("Abstention recommended due to low confidence")

        # Check review needs
        if review_result.get('review_needed', False):
            priority = review_result.get('priority', 'unknown')
            issues.append(f"Human review required (priority: {priority})")

        # Determine pass/fail
        if self.enable_strict_mode:
            # In strict mode, any issue is a failure
            passed = len(issues) == 0
        else:
            # In permissive mode, only critical issues fail
            critical_keywords = ['fabricated', 'abstention']
            critical_issues = [
                i for i in issues
                if any(kw in i.lower() for kw in critical_keywords)
            ]
            passed = len(critical_issues) == 0

        return passed, issues

    def _get_abstention_reason(self, confidence_result: Dict) -> str:
        """
        Get human-readable abstention reason.

        Args:
            confidence_result: Confidence scoring results

        Returns:
            Abstention reason string
        """
        reasons = []

        scores = confidence_result.get('component_scores', {})

        if scores.get('faithfulness_score', 1.0) < 0.4:
            reasons.append("very low answer faithfulness")

        if scores.get('citation_score', 1.0) < 0.3:
            reasons.append("citation validation failures")

        if scores.get('evidence_score', 1.0) < 0.4:
            reasons.append("insufficient evidence")

        if scores.get('retrieval_score', 1.0) < 0.3:
            reasons.append("poor retrieval quality")

        if not reasons:
            reasons.append("overall confidence below threshold")

        return "Abstaining due to: " + ", ".join(reasons)

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_validation_summary(self, validation_result: Dict) -> str:
        """
        Generate human-readable validation summary.

        Args:
            validation_result: Result from validate_answer()

        Returns:
            Summary string
        """
        summary = "SAFETY VALIDATION SUMMARY\n"
        summary += "=" * 60 + "\n\n"

        # Overall status
        passed = validation_result.get('validation_passed', False)
        status = "✓ PASSED" if passed else "✗ FAILED"
        summary += f"Status: {status}\n\n"

        # Confidence
        confidence = validation_result.get('confidence', {})
        conf_score = confidence.get('aggregate_confidence', 0)
        conf_level = confidence.get('confidence_level', 'unknown')
        summary += f"Confidence: {conf_score:.2%} ({conf_level})\n\n"

        # Component scores
        summary += "Component Scores:\n"
        comp_scores = confidence.get('component_scores', {})
        for component, score in comp_scores.items():
            summary += f"  - {component}: {score:.2%}\n"

        # Issues
        issues = validation_result.get('validation_issues', [])
        if issues:
            summary += f"\nIssues ({len(issues)}):\n"
            for i, issue in enumerate(issues, 1):
                summary += f"  {i}. {issue}\n"

        # Review status
        review = validation_result.get('review', {})
        if review.get('review_needed', False):
            priority = review.get('priority', 'unknown')
            num_triggers = review.get('num_triggers', 0)
            summary += f"\nHuman Review: REQUIRED (Priority: {priority.upper()})\n"
            summary += f"Review Triggers: {num_triggers}\n"

        # Abstention
        if validation_result.get('should_abstain', False):
            reason = validation_result.get('abstention_reason', 'Unknown')
            summary += f"\nAbstention: RECOMMENDED\n"
            summary += f"Reason: {reason}\n"

        summary += "\n" + "=" * 60

        return summary

    def get_safe_answer(self, validation_result: Dict) -> Optional[str]:
        """
        Get safe answer with disclaimers, or None if should abstain.

        Args:
            validation_result: Result from validate_answer()

        Returns:
            Safe answer string or None
        """
        # Check if should abstain
        if validation_result.get('should_abstain', False):
            logger.warning("Returning None - abstention recommended")
            return None

        # Get answer with disclaimers
        final_answer = validation_result.get('final_answer')

        if not final_answer:
            logger.error("No final answer in validation result")
            return None

        return final_answer

    def create_abstention_response(self, validation_result: Dict) -> str:
        """
        Create abstention response when system cannot provide answer.

        Args:
            validation_result: Result from validate_answer()

        Returns:
            Abstention response string
        """
        response = "UNABLE TO PROVIDE ANSWER\n\n"
        response += "I apologize, but I cannot provide a confident answer to your question "
        response += "due to quality and safety concerns.\n\n"

        # Include reason
        reason = validation_result.get('abstention_reason', 'Insufficient confidence')
        response += f"Reason: {reason}\n\n"

        # Recommendations
        response += "Recommendations:\n"
        response += "1. Consult a qualified legal practitioner for professional advice\n"
        response += "2. Rephrase your question to be more specific\n"
        response += "3. Provide additional context or details\n"
        response += "4. Consult official legal sources and databases\n\n"

        # Disclaimer
        response += self.disclaimer_enforcer.get_minimal_disclaimer()

        return response

    def set_strict_mode(self, enable: bool) -> None:
        """
        Enable or disable strict validation mode.

        Args:
            enable: True to enable strict mode, False to disable
        """
        self.enable_strict_mode = enable
        logger.info(f"Strict mode {'enabled' if enable else 'disabled'}")

    def update_thresholds(
        self,
        confidence_threshold: Optional[float] = None,
        faithfulness_threshold: Optional[float] = None,
        citation_threshold: Optional[float] = None
    ) -> None:
        """
        Update validation thresholds.

        Args:
            confidence_threshold: New confidence threshold
            faithfulness_threshold: New faithfulness threshold
            citation_threshold: New citation threshold
        """
        if confidence_threshold is not None:
            self.review_trigger.confidence_threshold = confidence_threshold
            logger.info(f"Confidence threshold updated to {confidence_threshold}")

        if faithfulness_threshold is not None:
            self.review_trigger.faithfulness_threshold = faithfulness_threshold
            logger.info(f"Faithfulness threshold updated to {faithfulness_threshold}")

        if citation_threshold is not None:
            self.review_trigger.citation_threshold = citation_threshold
            logger.info(f"Citation threshold updated to {citation_threshold}")
