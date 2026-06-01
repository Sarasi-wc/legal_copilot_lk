"""
Cross-encoder reranker for improving top-k precision.
Implements reranking as described in methodology.
"""

from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

from src.utils import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker for refining retrieval results.
    Uses cross-encoder models for fine-grained relevance scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize reranker.

        Args:
            model_name: Name of cross-encoder model
        """
        self.model_name = model_name
        self.model = None

    def load_model(self) -> None:
        """Load cross-encoder model."""
        logger.info(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info("Reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        passages: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank passages using cross-encoder with query-aware boosting.

        Args:
            query: Query string
            passages: List of passage dictionaries
            top_k: Number of top results to return

        Returns:
            Reranked list of passages with updated scores
        """
        if self.model is None:
            self.load_model()

        if not passages:
            return []

        logger.debug(f"Reranking {len(passages)} passages")

        # Prepare query-passage pairs
        pairs = [[query, p['text']] for p in passages]

        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores to passages and apply query-aware adjustments
        for i, passage in enumerate(passages):
            base_score = float(scores[i])
            passage['rerank_score'] = base_score
            passage['original_score'] = passage.get('score', 0.0)

            # Apply query-intent based boosting
            adjusted_score = self._apply_query_intent_boost(
                query, passage['text'], base_score
            )
            passage['rerank_score'] = adjusted_score
            # Update score so downstream filtering (EvidencePlanner) uses rerank score
            passage['score'] = adjusted_score

        # Sort by rerank score
        reranked = sorted(
            passages,
            key=lambda x: x['rerank_score'],
            reverse=True
        )

        # Return top-k
        return reranked[:top_k]

    def _apply_query_intent_boost(
        self,
        query: str,
        passage_text: str,
        base_score: float
    ) -> float:
        """
        Apply query-intent based score adjustments.

        For definitional queries ("What is/are X?"), boost passages with
        definitions and penalize passages about restrictions.
        """
        query_lower = query.lower()
        passage_lower = passage_text.lower()

        # Detect definitional queries
        is_definitional = any(pattern in query_lower for pattern in [
            'what is', 'what are', 'what does', 'define',
            'meaning of', 'explain'
        ])

        if is_definitional:
            # Penalty signals — passages primarily about restrictions/exceptions,
            # not definitions. Applicable across all Acts, not constitution-specific.
            restriction_signals = [
                'shall be subject to', 'restrictions', 'prescribed by law',
                'subject to such', 'may be restricted', 'limitations',
                'shall not apply', 'except', 'provided that',
                'notwithstanding', 'nothing in this section',
                'shall not be construed', 'does not apply'
            ]

            # Boost signals — passages containing substantive definitions/rights.
            # General statutory language, not tied to any specific Act or chapter.
            content_signals = [
                'means', 'defined as', 'includes', 'shall mean',
                'entitled to', 'guarantees', 'recognizes',
                'every person', 'any person', 'person who',
                '"', "'"  # Definitions frequently use quoted terms
            ]

            # Count restriction vs content signals
            restriction_count = sum(1 for sig in restriction_signals if sig in passage_lower)
            content_count = sum(1 for sig in content_signals if sig in passage_lower)

            # If passage is primarily about restrictions, penalize
            if restriction_count > content_count and restriction_count >= 2:
                return base_score * 0.7  # 30% penalty

            # If passage contains good content signals, boost
            elif content_count >= 2 and content_count > restriction_count:
                return base_score * 1.15  # 15% boost

        return base_score

    def rerank_with_fusion(
        self,
        query: str,
        passages: List[Dict],
        alpha: float = 0.5,
        top_k: int = 5
    ) -> List[Dict]:
        """
        UNUSED — superseded by the 70/30 cross-encoder/hybrid blending in
        HybridRetriever._rerank_candidates(), which normalises both score
        scales before fusion. This method fuses raw cross-encoder logits
        (range ~-6 to +6) with [0,1]-normalised hybrid scores directly,
        making the fusion numerically meaningless. Do not call in production.

        Args:
            query: Query string
            passages: List of passage dictionaries
            alpha: Fusion weight (0=original only, 1=rerank only)
            top_k: Number of results to return

        Returns:
            Reranked passages with fused scores
        """
        reranked = self.rerank(query, passages, top_k=len(passages))

        # Fuse scores
        for passage in reranked:
            original = passage.get('original_score', 0.0)
            rerank = passage.get('rerank_score', 0.0)

            # Normalize scores to [0, 1] if needed
            passage['fused_score'] = (1 - alpha) * original + alpha * rerank

        # Re-sort by fused score
        reranked = sorted(
            reranked,
            key=lambda x: x['fused_score'],
            reverse=True
        )

        return reranked[:top_k]
