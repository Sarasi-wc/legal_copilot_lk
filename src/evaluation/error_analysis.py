"""
Automated failure mode categorisation for RO6.

Classifies per-item results from run_experiment.py into failure mode buckets,
enabling systematic error pattern analysis and boundary condition documentation.

Failure modes:
    RETRIEVAL_MISS        Gold passage not in top-k retrieved
    CITATION_HALLUCINATION  Answer cites a passage not in evidence
    FAITHFULNESS_FAILURE  Claim not entailed by retrieved evidence
    WRONG_ABSTENTION      System abstained when evidence was sufficient
    MISSED_ABSTENTION     System answered when it should have abstained
    COVERAGE_FAILURE      Cross-ref query returned only single passage
    PARTIAL_RETRIEVAL     Gold passage retrieved but not in top-5
    NO_FAILURE            Item passed all checks
"""

from typing import List, Dict, Optional
from collections import Counter
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)

# Failure mode labels
RETRIEVAL_MISS = "retrieval_miss"
CITATION_HALLUCINATION = "citation_hallucination"
FAITHFULNESS_FAILURE = "faithfulness_failure"
WRONG_ABSTENTION = "wrong_abstention"
MISSED_ABSTENTION = "missed_abstention"
COVERAGE_FAILURE = "coverage_failure"
PARTIAL_RETRIEVAL = "partial_retrieval"
NO_FAILURE = "no_failure"


class FailureModeAnalyser:
    """
    Classifies per-query results into failure mode categories.
    Produces a structured report for RO6 error analysis.
    """

    def __init__(self, top_k_threshold: int = 5):
        """
        Args:
            top_k_threshold: k at which a retrieval miss is declared
                (passage not in top-k despite being gold-relevant)
        """
        self.top_k_threshold = top_k_threshold

    def classify_item(
        self,
        item_id: str,
        question: str,
        query_type: str,
        gold_passage_ids: List[str],
        retrieved_passage_ids: List[str],
        answer_result: Dict,
        should_abstain: bool
    ) -> Dict:
        """
        Classify a single Q-A-C item into failure mode(s).

        Args:
            item_id: Dataset item identifier
            question: Original question
            query_type: factual / procedural / interpretive / cross_referenced
            gold_passage_ids: Ground-truth relevant passage IDs
            retrieved_passage_ids: Passage IDs returned by retrieval (ranked)
            answer_result: Full answer dict from RAGPipeline.answer_question()
            should_abstain: Whether the item has no answerable gold evidence

        Returns:
            Dict with 'item_id', 'failure_modes', 'severity', and detail fields.
        """
        failure_modes = []
        details = {}

        gold = set(gold_passage_ids)
        retrieved_set = set(retrieved_passage_ids)
        top_k_set = set(retrieved_passage_ids[:self.top_k_threshold])

        abstained = answer_result.get('abstained', False)
        verification = answer_result.get('verification_report') or {}
        faithfulness = verification.get('faithfulness', {})
        citation_val = verification.get('citation_validation', {})

        # --- Retrieval failures ---
        if gold and not (gold & retrieved_set):
            failure_modes.append(RETRIEVAL_MISS)
            details['retrieval_miss'] = {
                'gold_ids': list(gold),
                'retrieved_ids': retrieved_passage_ids[:10]
            }
        elif gold and not (gold & top_k_set):
            failure_modes.append(PARTIAL_RETRIEVAL)
            details['partial_retrieval'] = {
                'gold_in_retrieved_at': self._first_hit_rank(
                    retrieved_passage_ids, gold
                )
            }

        # --- Coverage failure (cross-referenced queries) ---
        if query_type == 'cross_referenced' and len(retrieved_passage_ids) < 2:
            failure_modes.append(COVERAGE_FAILURE)
            details['coverage_failure'] = {
                'retrieved_count': len(retrieved_passage_ids),
                'required': 2
            }

        # --- Abstention failures ---
        if abstained and not should_abstain:
            failure_modes.append(WRONG_ABSTENTION)
            details['wrong_abstention'] = {
                'reason': answer_result.get('abstention_reason')
            }
        elif not abstained and should_abstain:
            failure_modes.append(MISSED_ABSTENTION)
            details['missed_abstention'] = {
                'answer_snippet': answer_result.get('answer', '')[:100]
            }

        # --- Generation failures (only when not abstained) ---
        if not abstained:
            fabricated = citation_val.get('fabricated_citations', [])
            if fabricated:
                failure_modes.append(CITATION_HALLUCINATION)
                details['citation_hallucination'] = {
                    'fabricated': fabricated
                }

            faithfulness_score = faithfulness.get('faithfulness_score', 1.0)
            hallucinations = faithfulness.get('hallucinations', [])
            if not faithfulness.get('is_faithful', True) or hallucinations:
                failure_modes.append(FAITHFULNESS_FAILURE)
                details['faithfulness_failure'] = {
                    'score': faithfulness_score,
                    'hallucinated_claims': hallucinations[:3]
                }

        if not failure_modes:
            failure_modes.append(NO_FAILURE)

        # Severity: critical > high > medium > none
        severity = self._compute_severity(failure_modes)

        return {
            'item_id': item_id,
            'question': question,
            'query_type': query_type,
            'failure_modes': failure_modes,
            'severity': severity,
            'details': details
        }

    def analyse_batch(
        self,
        classified_items: List[Dict]
    ) -> Dict:
        """
        Aggregate failure mode classifications across all items.

        Args:
            classified_items: Output of classify_item() for each Q-A-C item.

        Returns:
            Structured report with frequency counts, severity distribution,
            and per-mode examples for qualitative error analysis.
        """
        all_modes = []
        for item in classified_items:
            all_modes.extend(item['failure_modes'])

        mode_counts = Counter(all_modes)
        severity_counts = Counter(item['severity'] for item in classified_items)

        n = len(classified_items)
        mode_rates = {
            mode: round(count / n, 4)
            for mode, count in mode_counts.items()
        }

        # Per-mode examples (up to 3 per mode for qualitative analysis)
        examples = {}
        for mode in mode_counts:
            if mode == NO_FAILURE:
                continue
            examples[mode] = [
                {
                    'item_id': item['item_id'],
                    'question': item['question'],
                    'details': item['details'].get(mode, {})
                }
                for item in classified_items
                if mode in item['failure_modes']
            ][:3]

        # Cross-tabulation by query type
        by_query_type = {}
        for item in classified_items:
            qt = item['query_type']
            if qt not in by_query_type:
                by_query_type[qt] = Counter()
            by_query_type[qt].update(item['failure_modes'])

        report = {
            'total_items': n,
            'items_with_no_failure': mode_counts.get(NO_FAILURE, 0),
            'failure_rate': round(1 - mode_counts.get(NO_FAILURE, 0) / n, 4),
            'mode_counts': dict(mode_counts),
            'mode_rates': mode_rates,
            'severity_distribution': dict(severity_counts),
            'by_query_type': {
                qt: dict(counts)
                for qt, counts in by_query_type.items()
            },
            'examples_per_mode': examples
        }

        # Log summary
        logger.info(f"Error analysis: {n} items, failure rate={report['failure_rate']:.1%}")
        for mode, rate in sorted(mode_rates.items(), key=lambda x: -x[1]):
            if mode != NO_FAILURE:
                logger.info(f"  {mode:<30} {rate:.1%}  (n={mode_counts[mode]})")

        return report

    @staticmethod
    def _first_hit_rank(retrieved: List[str], gold: set) -> Optional[int]:
        """Return the rank (1-indexed) of the first gold passage in retrieved list."""
        for i, pid in enumerate(retrieved, 1):
            if pid in gold:
                return i
        return None

    @staticmethod
    def _compute_severity(modes: List[str]) -> str:
        """Map failure modes to an overall severity level."""
        if CITATION_HALLUCINATION in modes or FAITHFULNESS_FAILURE in modes:
            return 'critical'
        if RETRIEVAL_MISS in modes or MISSED_ABSTENTION in modes:
            return 'high'
        if WRONG_ABSTENTION in modes or COVERAGE_FAILURE in modes:
            return 'medium'
        if PARTIAL_RETRIEVAL in modes:
            return 'low'
        return 'none'
