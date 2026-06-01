"""
Complete verification module integrating faithfulness and citation validation.
Implements multi-layered verification as described in methodology.
"""

from typing import Dict, List

from src.verification.faithfulness_checker import FaithfulnessChecker
from src.verification.citation_validator import CitationValidator
from src.utils import get_logger

logger = get_logger(__name__)


class Verifier:
    """
    Complete verifier integrating faithfulness and citation checking.
    Implements verification as described in RO3.
    """

    def __init__(
        self,
        nli_model: str = "roberta-large-mnli",
        entailment_threshold: float = 0.30,  # Empirically calibrated (see faithfulness_checker.py)
    ):
        """
        Initialize verifier.

        Args:
            nli_model: NLI model for faithfulness checking
            entailment_threshold: Per-claim NLI entailment threshold (default 0.30,
                empirically calibrated on 22 constitutional pairs). The aggregate
                faithfulness gate (≥ 0.80 of claims must be entailed) is hardcoded
                in FaithfulnessChecker.check_faithfulness(). should_abstain() uses
                a separate hard-abstention gate of 0.50.
        """
        self.faithfulness_checker = FaithfulnessChecker(
            model_name=nli_model,
            entailment_threshold=entailment_threshold
        )
        self.citation_validator = CitationValidator()

    def verify_answer(
        self,
        answer_result: Dict,
        evidence_plan: Dict
    ) -> Dict:
        """
        Perform complete verification of generated answer.

        Args:
            answer_result: Answer result from generator
            evidence_plan: Evidence plan used for generation

        Returns:
            Complete verification result
        """
        logger.info("Performing answer verification")

        answer = answer_result['answer']

        # Skip verification if abstained
        if answer_result.get('abstained', False):
            return {
                'verified': False,
                'abstained': True,
                'verification_passed': False
            }

        # Step 1: Check faithfulness
        faithfulness_result = self.faithfulness_checker.check_faithfulness(
            answer=answer,
            evidence_passages=evidence_plan.get('passages', [])
        )

        # Step 2: Validate citations
        citation_validation = self.citation_validator.validate_citations(
            answer=answer,
            evidence_plan=evidence_plan
        )

        # Step 3: Check citation coverage
        citation_coverage = self.faithfulness_checker.check_citation_coverage(
            answer=answer,
            evidence_plan=evidence_plan
        )

        # Step 4: Check citation formatting
        citation_formatting = self.citation_validator.check_citation_formatting(
            answer=answer
        )

        # Step 5: Determine if verification passed
        verification_passed = self._determine_pass(
            faithfulness_result,
            citation_validation,
            citation_coverage,
            citation_formatting
        )

        # Compile verification report
        verification_report = {
            'verified': True,
            'verification_passed': verification_passed,
            'faithfulness': faithfulness_result,
            'citation_validation': citation_validation,
            'citation_coverage': citation_coverage,
            'citation_formatting': citation_formatting,
            'issues': self._identify_issues(
                faithfulness_result,
                citation_validation,
                citation_coverage,
                citation_formatting
            )
        }

        logger.info(f"Verification complete. Passed: {verification_passed}")

        return verification_report

    def _determine_pass(
        self,
        faithfulness: Dict,
        citation_validation: Dict,
        citation_coverage: Dict,
        citation_formatting: Dict
    ) -> bool:
        """
        Determine if answer passes verification.

        Criteria:
            - Faithfulness score >= threshold
            - All citations valid (no fabrications)
            - Adequate citation coverage
            - Proper citation formatting
        """
        checks = [
            faithfulness['is_faithful'],
            citation_validation['all_valid'],
            citation_coverage['adequate_coverage'],
            citation_formatting['all_properly_formatted']
        ]

        return all(checks)

    def _identify_issues(
        self,
        faithfulness: Dict,
        citation_validation: Dict,
        citation_coverage: Dict,
        citation_formatting: Dict
    ) -> List[str]:
        """Identify specific verification issues."""
        issues = []

        if not faithfulness['is_faithful']:
            issues.append(
                f"Low faithfulness score: {faithfulness['faithfulness_score']:.2%}"
            )

        if faithfulness['hallucinations']:
            issues.append(
                f"Hallucinations detected: {len(faithfulness['hallucinations'])} claims"
            )

        if not citation_validation['all_valid']:
            num_fabricated = len(citation_validation['fabricated_citations'])
            issues.append(f"Fabricated citations: {num_fabricated}")

        if not citation_coverage['adequate_coverage']:
            density = citation_coverage['citation_density']
            issues.append(f"Low citation coverage: {density:.2%}")

        if not citation_formatting['all_properly_formatted']:
            num_malformed = len(citation_formatting['malformed_citations'])
            issues.append(f"Malformed citations: {num_malformed}")

        return issues

    def should_abstain(self, verification_report: Dict) -> bool:
        """
        Determine if system should abstain based on verification.

        Only abstains on critical issues:
        - Fabricated citations (wrong source attributed)
        - Critically low faithfulness (< 0.5)

        Does NOT abstain on cosmetic issues like citation density or formatting,
        which would cause too many false negatives on valid answers.

        Args:
            verification_report: Verification result

        Returns:
            True if system should abstain
        """
        # Abstain if fabricated citations detected (wrong source is a critical error)
        citation_validation = verification_report.get('citation_validation', {})
        if len(citation_validation.get('fabricated_citations', [])) > 0:
            return True

        # Abstain if faithfulness is critically low
        faithfulness = verification_report.get('faithfulness', {})
        if faithfulness.get('faithfulness_score', 1.0) < 0.5:
            return True

        return False
