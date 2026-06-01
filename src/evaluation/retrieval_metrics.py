"""
Retrieval evaluation metrics.
Implements Recall@k, MRR, and NDCG for retrieval evaluation.
"""

from typing import List, Dict, Set
import numpy as np
from sklearn.metrics import ndcg_score

from src.utils import get_logger

logger = get_logger(__name__)


class RetrievalMetrics:
    """
    Compute retrieval effectiveness metrics.
    Implements metrics described in methodology Section 3.4.
    """

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Compute Recall@k.

        Args:
            retrieved: List of retrieved passage IDs (ordered)
            relevant: Set of relevant passage IDs
            k: Cutoff rank

        Returns:
            Recall@k score
        """
        if not relevant:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        hits = len(retrieved_at_k.intersection(relevant))

        return hits / len(relevant)

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Compute Precision@k.

        Args:
            retrieved: List of retrieved passage IDs
            relevant: Set of relevant passage IDs
            k: Cutoff rank

        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        hits = len(retrieved_at_k.intersection(relevant))

        return hits / k

    @staticmethod
    def mean_reciprocal_rank(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        Args:
            retrieved: List of retrieved passage IDs
            relevant: Set of relevant passage IDs

        Returns:
            MRR score (reciprocal rank of first relevant item)
        """
        for i, passage_id in enumerate(retrieved, 1):
            if passage_id in relevant:
                return 1.0 / i

        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevant_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG@k).

        Args:
            retrieved: List of retrieved passage IDs
            relevant_scores: Dict mapping passage IDs to relevance scores
            k: Cutoff rank

        Returns:
            NDCG@k score
        """
        if not relevant_scores:
            return 0.0

        # Get scores for retrieved items
        retrieved_at_k = retrieved[:k]
        scores = [
            relevant_scores.get(pid, 0.0)
            for pid in retrieved_at_k
        ]

        # Pad to k if needed
        while len(scores) < k:
            scores.append(0.0)

        # Ideal scores (sorted)
        ideal_scores = sorted(relevant_scores.values(), reverse=True)[:k]
        while len(ideal_scores) < k:
            ideal_scores.append(0.0)

        # Compute NDCG using sklearn
        try:
            ndcg = ndcg_score([ideal_scores], [scores])
            return ndcg
        except Exception as e:
            logger.warning(f"NDCG computation error: {e}")
            return 0.0

    @staticmethod
    def evaluate_retrieval(
        retrieved_passages: List[str],
        gold_passages: List[str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict:
        """
        Comprehensive retrieval evaluation.

        Two evaluation protocols (see Ch6):
        - Article-level (N=334, authoritative): retrieved IDs are normalised by
          stripping _CHUNK_i and _SUBSEC_i suffixes before matching, so any
          chunk of the correct article counts as a hit. Callers pre-normalise IDs
          before calling (see run_experiment.py, run_ablation.py).
        - Exact passage-ID: no normalisation; retrieved ID must exactly match
          the gold passage ID. Callers are responsible for passing verbatim IDs.

        Args:
            retrieved_passages: List of retrieved passage IDs (ranked)
            gold_passages: List of relevant passage IDs
            k_values: List of k values for evaluation

        Returns:
            Dictionary of metrics
        """
        relevant_set = set(gold_passages)

        metrics = {}

        # Recall@k for each k
        for k in k_values:
            metrics[f'recall@{k}'] = RetrievalMetrics.recall_at_k(
                retrieved_passages,
                relevant_set,
                k
            )

        # Precision@k for each k
        for k in k_values:
            metrics[f'precision@{k}'] = RetrievalMetrics.precision_at_k(
                retrieved_passages,
                relevant_set,
                k
            )

        # MRR
        metrics['mrr'] = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_passages,
            relevant_set
        )

        # Binary relevance for NDCG
        relevant_scores = {pid: 1.0 for pid in gold_passages}

        # NDCG@k for each k
        for k in k_values:
            metrics[f'ndcg@{k}'] = RetrievalMetrics.ndcg_at_k(
                retrieved_passages,
                relevant_scores,
                k
            )

        return metrics

    @staticmethod
    def aggregate_metrics(metric_list: List[Dict]) -> Dict:
        """
        Aggregate metrics across multiple queries.

        Args:
            metric_list: List of metric dictionaries

        Returns:
            Aggregated metrics (mean and std)
        """
        if not metric_list:
            return {}

        # Get all metric keys
        keys = metric_list[0].keys()

        aggregated = {}
        for key in keys:
            values = [m[key] for m in metric_list]
            # Only aggregate numeric values; skip string metadata fields
            if not values or not isinstance(values[0], (int, float)):
                continue
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

        return aggregated

    @staticmethod
    def abstention_metrics(
        system_abstained: List[bool],
        should_abstain: List[bool]
    ) -> Dict:
        """
        Compute abstention precision, recall, and F1 (RO5).

        A true positive is when the system correctly abstains on a query
        that has no sufficient gold evidence.

        Args:
            system_abstained: Boolean list — True if the system abstained
            should_abstain: Boolean list — True if abstention was correct
                (e.g. the Q-A-C item has no answerable gold passage)

        Returns:
            Dictionary with abstention_precision, abstention_recall,
            abstention_f1, and raw counts
        """
        if len(system_abstained) != len(should_abstain):
            raise ValueError("Prediction and label lists must have equal length")

        tp = sum(1 for s, g in zip(system_abstained, should_abstain) if s and g)
        fp = sum(1 for s, g in zip(system_abstained, should_abstain) if s and not g)
        fn = sum(1 for s, g in zip(system_abstained, should_abstain) if not s and g)
        tn = sum(1 for s, g in zip(system_abstained, should_abstain) if not s and not g)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return {
            'abstention_precision': precision,
            'abstention_recall': recall,
            'abstention_f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'total': len(system_abstained)
        }

    @staticmethod
    def attribution_metrics(
        cited_passage_ids: List[str],
        gold_passage_ids: List[str]
    ) -> Dict:
        """
        Compute attribution precision, recall, and F1 (RO5).

        Attribution precision: fraction of cited passages that are gold-relevant.
        Attribution recall: fraction of gold passages that were cited.

        Args:
            cited_passage_ids: Passage IDs extracted from the generated answer
            gold_passage_ids: Gold-standard relevant passage IDs from Q-A-C item

        Returns:
            Dictionary with attribution_precision, attribution_recall, attribution_f1
        """
        cited = set(cited_passage_ids)
        gold = set(gold_passage_ids)

        tp = len(cited & gold)

        precision = tp / len(cited) if cited else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return {
            'attribution_precision': precision,
            'attribution_recall': recall,
            'attribution_f1': f1,
            'cited_count': len(cited),
            'gold_count': len(gold),
            'correctly_cited': tp
        }

    @staticmethod
    def aggregate_attribution_metrics(metric_list: List[Dict]) -> Dict:
        """Aggregate attribution metrics across multiple Q-A-C items."""
        if not metric_list:
            return {}

        keys = ['attribution_precision', 'attribution_recall', 'attribution_f1']
        aggregated = {}
        for key in keys:
            values = [m[key] for m in metric_list]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        return aggregated
