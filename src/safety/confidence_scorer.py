"""
Confidence Scorer Module

Calculates aggregate confidence scores from multiple sources including:
- Retrieval quality (passage relevance, coverage)
- Faithfulness scores (NLI-based verification)
- Citation validation results
- Evidence sufficiency
- Conflict detection results
"""

from typing import Dict, List, Optional
from src.utils import get_logger

logger = get_logger(__name__)


class ConfidenceScorer:
    """
    Calculate aggregate confidence scores for legal answers.

    Combines multiple signals to produce an overall confidence metric
    that indicates the reliability of the generated answer.
    """

    def __init__(
        self,
        retrieval_weight: float = 0.25,
        faithfulness_weight: float = 0.30,
        citation_weight: float = 0.20,
        evidence_weight: float = 0.15,
        conflict_weight: float = 0.10
    ):
        """
        Initialize confidence scorer.

        Args:
            retrieval_weight: Weight for retrieval quality
            faithfulness_weight: Weight for faithfulness score
            citation_weight: Weight for citation validation
            evidence_weight: Weight for evidence sufficiency
            conflict_weight: Weight for conflict detection
        """
        self.retrieval_weight = retrieval_weight
        self.faithfulness_weight = faithfulness_weight
        self.citation_weight = citation_weight
        self.evidence_weight = evidence_weight
        self.conflict_weight = conflict_weight

        # Validate weights sum to 1.0
        total_weight = sum([
            retrieval_weight,
            faithfulness_weight,
            citation_weight,
            evidence_weight,
            conflict_weight
        ])

        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {total_weight}, not 1.0. "
                "Normalizing weights."
            )
            # Normalize weights
            self.retrieval_weight /= total_weight
            self.faithfulness_weight /= total_weight
            self.citation_weight /= total_weight
            self.evidence_weight /= total_weight
            self.conflict_weight /= total_weight

        logger.info("ConfidenceScorer initialized")

    def calculate_confidence(
        self,
        retrieval_results: Optional[Dict] = None,
        faithfulness_results: Optional[Dict] = None,
        citation_results: Optional[Dict] = None,
        evidence_plan: Optional[Dict] = None,
        inference_results: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate aggregate confidence score.

        Args:
            retrieval_results: Results from retrieval system
            faithfulness_results: Faithfulness check results
            citation_results: Citation validation results
            evidence_plan: Evidence planning results
            inference_results: Inference engine results

        Returns:
            Dictionary with confidence scores and breakdown
        """
        logger.info("Calculating aggregate confidence score")

        scores = {}

        # 1. Retrieval quality score
        retrieval_score = self._calculate_retrieval_score(retrieval_results)
        scores['retrieval_score'] = retrieval_score

        # 2. Faithfulness score
        faithfulness_score = self._calculate_faithfulness_score(faithfulness_results)
        scores['faithfulness_score'] = faithfulness_score

        # 3. Citation validation score
        citation_score = self._calculate_citation_score(citation_results)
        scores['citation_score'] = citation_score

        # 4. Evidence sufficiency score
        evidence_score = self._calculate_evidence_score(evidence_plan)
        scores['evidence_score'] = evidence_score

        # 5. Conflict detection score (inverse - no conflicts is good)
        conflict_score = self._calculate_conflict_score(inference_results)
        scores['conflict_score'] = conflict_score

        # Calculate weighted aggregate
        aggregate_score = (
            retrieval_score * self.retrieval_weight +
            faithfulness_score * self.faithfulness_weight +
            citation_score * self.citation_weight +
            evidence_score * self.evidence_weight +
            conflict_score * self.conflict_weight
        )

        # Determine confidence level
        confidence_level = self._determine_confidence_level(aggregate_score)

        # Check if meets minimum threshold
        meets_threshold = aggregate_score >= 0.6  # 60% minimum

        result = {
            'aggregate_confidence': aggregate_score,
            'confidence_level': confidence_level,
            'meets_minimum_threshold': meets_threshold,
            'component_scores': scores,
            'score_breakdown': {
                'retrieval': retrieval_score * self.retrieval_weight,
                'faithfulness': faithfulness_score * self.faithfulness_weight,
                'citation': citation_score * self.citation_weight,
                'evidence': evidence_score * self.evidence_weight,
                'conflict': conflict_score * self.conflict_weight
            },
            'weights': {
                'retrieval': self.retrieval_weight,
                'faithfulness': self.faithfulness_weight,
                'citation': self.citation_weight,
                'evidence': self.evidence_weight,
                'conflict': self.conflict_weight
            }
        }

        logger.info(
            f"Confidence calculated: {aggregate_score:.3f} ({confidence_level})"
        )

        return result

    def _calculate_retrieval_score(
        self,
        retrieval_results: Optional[Dict]
    ) -> float:
        """
        Calculate retrieval quality score.

        Based on:
        - Number of retrieved passages
        - Relevance scores of top passages
        - Diversity of sources

        Args:
            retrieval_results: Retrieval results dictionary

        Returns:
            Score between 0.0 and 1.0
        """
        if not retrieval_results:
            return 0.5  # Neutral default

        score_components = []

        # Check number of passages
        num_passages = len(retrieval_results.get('passages', []))
        if num_passages >= 5:
            score_components.append(1.0)
        elif num_passages >= 3:
            score_components.append(0.8)
        elif num_passages >= 1:
            score_components.append(0.5)
        else:
            score_components.append(0.2)

        # Check average relevance score
        passages = retrieval_results.get('passages', [])
        if passages:
            relevance_scores = [
                p.get('rerank_score', p.get('score', 0.5))
                for p in passages
            ]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            score_components.append(avg_relevance)

        # Check source diversity (unique acts)
        if passages:
            unique_acts = len(set(
                p.get('act_name', '') for p in passages
            ))
            diversity_score = min(unique_acts / 3.0, 1.0)  # Max at 3 acts
            score_components.append(diversity_score)

        return sum(score_components) / len(score_components) if score_components else 0.5

    def _calculate_faithfulness_score(
        self,
        faithfulness_results: Optional[Dict]
    ) -> float:
        """
        Calculate faithfulness score.

        Args:
            faithfulness_results: Faithfulness check results

        Returns:
            Score between 0.0 and 1.0
        """
        if not faithfulness_results:
            return 0.5  # Neutral default

        # Use the faithfulness score directly
        score = faithfulness_results.get('faithfulness_score', 0.5)

        # Penalize if hallucinations detected
        if faithfulness_results.get('hallucinations'):
            num_hallucinations = len(faithfulness_results['hallucinations'])
            penalty = min(num_hallucinations * 0.1, 0.3)  # Max 30% penalty
            score = max(score - penalty, 0.0)

        return score

    def _calculate_citation_score(
        self,
        citation_results: Optional[Dict]
    ) -> float:
        """
        Calculate citation validation score.

        Args:
            citation_results: Citation validation results

        Returns:
            Score between 0.0 and 1.0
        """
        if not citation_results:
            return 0.5  # Neutral default

        # Use validation rate directly
        validation_rate = citation_results.get('validation_rate', 0.5)

        # Check if all citations are valid
        all_valid = citation_results.get('all_valid', False)
        if all_valid:
            return 1.0

        # Penalize fabricated citations heavily
        fabricated = citation_results.get('fabricated_citations', [])
        if fabricated:
            penalty = min(len(fabricated) * 0.15, 0.5)  # Max 50% penalty
            validation_rate = max(validation_rate - penalty, 0.0)

        return validation_rate

    def _calculate_evidence_score(
        self,
        evidence_plan: Optional[Dict]
    ) -> float:
        """
        Calculate evidence sufficiency score.

        Args:
            evidence_plan: Evidence plan from planner

        Returns:
            Score between 0.0 and 1.0
        """
        if not evidence_plan:
            return 0.5  # Neutral default

        # Check if evidence is sufficient
        is_sufficient = evidence_plan.get('is_sufficient', False)

        if is_sufficient:
            base_score = 0.9
        else:
            base_score = 0.4

        # Adjust based on number of passages
        num_passages = len(evidence_plan.get('passages', []))
        if num_passages >= 5:
            base_score = min(base_score + 0.1, 1.0)
        elif num_passages < 2:
            base_score = max(base_score - 0.2, 0.0)

        # Check for gaps in evidence
        has_gaps = evidence_plan.get('has_gaps', False)
        if has_gaps:
            base_score = max(base_score - 0.15, 0.0)

        return base_score

    def _calculate_conflict_score(
        self,
        inference_results: Optional[Dict]
    ) -> float:
        """
        Calculate conflict detection score.

        No conflicts = high score
        Conflicts present = low score

        Args:
            inference_results: Inference engine results

        Returns:
            Score between 0.0 and 1.0
        """
        if not inference_results:
            return 0.5  # Neutral default

        # Check for conflicts
        has_conflicts = inference_results.get('has_conflicts', False)

        if not has_conflicts:
            return 1.0  # Perfect - no conflicts

        # If conflicts exist, score based on severity
        conflicts = inference_results.get('conflicts', [])
        num_conflicts = len(conflicts)

        if num_conflicts == 0:
            return 1.0
        elif num_conflicts == 1:
            return 0.6  # Single conflict - moderate concern
        elif num_conflicts == 2:
            return 0.3  # Multiple conflicts - significant concern
        else:
            return 0.1  # Many conflicts - critical concern

    def _determine_confidence_level(self, score: float) -> str:
        """
        Determine confidence level category.

        Args:
            score: Aggregate confidence score

        Returns:
            Confidence level string
        """
        if score >= 0.85:
            return "very_high"
        elif score >= 0.70:
            return "high"
        elif score >= 0.60:
            return "medium"
        elif score >= 0.40:
            return "low"
        else:
            return "very_low"

    def should_abstain(self, confidence_result: Dict) -> bool:
        """
        Determine if system should abstain from answering.

        Args:
            confidence_result: Result from calculate_confidence()

        Returns:
            True if should abstain, False otherwise
        """
        aggregate = confidence_result['aggregate_confidence']

        # Abstain if confidence is very low
        if aggregate < 0.3:
            logger.warning(
                f"Abstention recommended: confidence too low ({aggregate:.3f})"
            )
            return True

        # Abstain if faithfulness is critically low
        faithfulness = confidence_result['component_scores'].get('faithfulness_score', 1.0)
        if faithfulness < 0.4:
            logger.warning(
                f"Abstention recommended: faithfulness too low ({faithfulness:.3f})"
            )
            return True

        # Abstain if citation validation failed badly
        citation = confidence_result['component_scores'].get('citation_score', 1.0)
        if citation < 0.3:
            logger.warning(
                f"Abstention recommended: citation validation failed ({citation:.3f})"
            )
            return True

        return False

    def get_confidence_explanation(self, confidence_result: Dict) -> str:
        """
        Generate human-readable explanation of confidence score.

        Args:
            confidence_result: Result from calculate_confidence()

        Returns:
            Explanation string
        """
        score = confidence_result['aggregate_confidence']
        level = confidence_result['confidence_level']
        components = confidence_result['component_scores']

        explanation = f"Overall confidence: {score:.2%} ({level})\n\n"
        explanation += "Score breakdown:\n"
        explanation += f"  - Retrieval quality: {components['retrieval_score']:.2%}\n"
        explanation += f"  - Answer faithfulness: {components['faithfulness_score']:.2%}\n"
        explanation += f"  - Citation validity: {components['citation_score']:.2%}\n"
        explanation += f"  - Evidence sufficiency: {components['evidence_score']:.2%}\n"
        explanation += f"  - Conflict detection: {components['conflict_score']:.2%}\n"

        # Add interpretation
        if level == "very_high" or level == "high":
            explanation += "\nInterpretation: High confidence in answer quality and reliability."
        elif level == "medium":
            explanation += "\nInterpretation: Moderate confidence. Verify with additional sources."
        else:
            explanation += "\nInterpretation: Low confidence. Consult legal professional for verification."

        return explanation
