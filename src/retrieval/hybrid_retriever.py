"""
Hybrid retrieval system combining BM25, dense retrieval, and reranking.
Implements the complete retrieval pipeline for RO2.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.article_boosted_retriever import ArticleBoostedRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import Reranker
from src.utils import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval system combining multiple retrieval methods.
    Implements BM25 + Dense + Reranking pipeline.
    Now uses ArticleBoostedRetriever for improved article-specific queries.
    """

    def __init__(
        self,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fusion_alpha: float = 0.5,
        article_boost_factor: float = 5.0
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            embedding_model: Dense retrieval model name. Defaults to
                sentence-transformers/all-mpnet-base-v2, a contrastive
                bi-encoder trained for semantic similarity (RO2).
                For ablation baseline use "sentence-transformers/all-MiniLM-L6-v2".
            reranker_model: Reranker model name
            fusion_alpha: Score fusion weight (0=BM25 only, 1=dense only)
            article_boost_factor: Boost factor for article number matches
        """
        # Use ArticleBoostedRetriever instead of plain BM25Retriever
        self.bm25_retriever = ArticleBoostedRetriever(boost_factor=article_boost_factor)
        self.dense_retriever = DenseRetriever(model_name=embedding_model)
        self.reranker = Reranker(model_name=reranker_model)
        self.fusion_alpha = fusion_alpha

        # Track whether dense index is available
        self.dense_index_available = False

        # Cross-reference pattern for recursive retrieval (RO2)
        self._xref_pattern = re.compile(
            r'[Ss]ection\s+(\d+[A-Z]?(?:\([a-z0-9]+\))?)',
            re.IGNORECASE
        )

    def build_indices(self, passages: List[Dict]) -> None:
        """
        Build all retrieval indices.

        Args:
            passages: List of passage dictionaries
        """
        logger.info("Building hybrid retrieval indices")

        # Build BM25 index
        self.bm25_retriever.build_index(passages)

        # Build dense index
        self.dense_retriever.build_index(passages)
        self.dense_index_available = True

        logger.info("All indices built successfully")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
        effective_year: Optional[int] = None,
        exclude_repealed: bool = True
    ) -> List[Dict]:
        """
        Retrieve passages using specified method.

        Args:
            query: Query string
            top_k: Number of results to return
            method: Retrieval method ('bm25', 'dense', 'hybrid', 'hybrid_rerank')
            effective_year: If set, exclude passages from Acts enacted after
                this year, enabling temporal queries ("what was the law in 2005?")
            exclude_repealed: If True, exclude passages from repealed Acts
                (default True — only return current law)

        Returns:
            List of retrieved passages with scores
        """
        # Fall back to BM25 if dense index not available and method requires it
        if not self.dense_index_available and method in ['dense', 'hybrid', 'hybrid_rerank']:
            logger.warning(f"Dense index not available. Falling back from '{method}' to 'bm25'")
            method = "bm25"

        if method == "bm25":
            results = self._retrieve_bm25(query, top_k)

        elif method == "dense":
            results = self._retrieve_dense(query, top_k)

        elif method == "hybrid":
            results = self._retrieve_hybrid(query, top_k)

        elif method == "hybrid_rerank":
            results = self._retrieve_hybrid_rerank(query, top_k)

        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        # Act filtering: when query targets a specific act (e.g. "Article X" → Constitution),
        # remove passages from other acts before expansion/reranking so they cannot
        # sneak in through the dense retrieval path.
        results = self._filter_by_act(query, results)

        # Post-retrieval expansions (RO2)
        results = self._expand_with_parent_context(results)
        results = self._expand_with_cross_references(results, top_k)

        # Temporal filtering (RO1/RO2 — enables "what was the law as of year X?" queries)
        results = self._filter_by_temporal(results, effective_year, exclude_repealed)

        return results[:top_k]

    def _deduplicate_by_article(self, passages: List[Dict]) -> List[Dict]:
        """
        Deduplicate retrieved passages by parent article, keeping highest-scored.

        When a chunked article produces multiple chunks (e.g. ACT__1978_SEC_13_CHUNK_0
        and ACT__1978_SEC_13_CHUNK_1), only the first (highest-ranked) is kept. This
        ensures each article occupies at most one slot in the top-k result, which
        both improves coverage across articles and matches article-level evaluation.

        NOTE: This method is not called in the default retrieve() path. The RQ1
        evaluation (N=334) used article-level ID normalisation in the evaluation
        scripts (stripping _CHUNK_N suffixes before matching), so multi-chunk
        articles did not inflate reported metrics. Calling this method in retrieve()
        would improve live-system article coverage without affecting reported results.
        """
        seen_articles: set = set()
        deduped: List[Dict] = []
        for p in passages:
            pid = p['passage_id']
            # Normalize to article level: strip _CHUNK_N, _SUBSEC_N suffixes
            article_id = re.sub(r'_(CHUNK|SUBSEC)_.*$', '', pid)
            if article_id not in seen_articles:
                seen_articles.add(article_id)
                deduped.append(p)
        return deduped

    def _build_passage_map(self) -> Dict:
        """Build a passage_id -> passage dict from the BM25 retriever's passages list."""
        passages_list = getattr(self.bm25_retriever, 'passages', None)
        if not passages_list:
            return {}
        return {p['passage_id']: p for p in passages_list}

    def _expand_with_parent_context(self, passages: List[Dict]) -> List[Dict]:
        """
        Hierarchical context expansion (RO2).

        For each retrieved passage, prepend its parent section title as context
        so the generator understands where in the Act hierarchy the passage sits
        (e.g. Part II, Section 14, Subsection (1)).
        """
        passage_map = self._build_passage_map()
        expanded = []
        for passage in passages:
            parent_id = passage.get('parent_id')
            if parent_id and parent_id in passage_map:
                parent = passage_map[parent_id]
                parent_title = parent.get('title', '')
                if parent_title and parent_title not in passage.get('text', '')[:50]:
                    p = passage.copy()
                    p['text'] = f"[{parent_title}]\n{p['text']}"
                    p['parent_context_added'] = True
                    expanded.append(p)
                    continue
            expanded.append(passage)
        return expanded

    def _expand_with_cross_references(
        self,
        passages: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Cross-reference resolution (RO2).

        Parses section references from the top retrieved passages and appends
        the target passages (from the same Act) so the generator has the
        referenced provisions available without requiring a second query.
        At most 2 additional passages are added to avoid noise.
        """
        passage_map = self._build_passage_map()
        if not passage_map:
            return passages

        existing_ids = {p['passage_id'] for p in passages}
        additional = []

        for passage in passages[:3]:  # Only expand from top-3 to limit noise
            refs = self._xref_pattern.findall(passage.get('text', ''))
            passage_act = passage.get('act_name', '')

            for ref_section in refs[:3]:
                if len(additional) >= 2:
                    break
                for pid, p in passage_map.items():
                    if pid in existing_ids:
                        continue
                    sec = (
                        p.get('metadata', {}).get('section_number') or
                        p.get('metadata', {}).get('article_number', '')
                    )
                    if sec == ref_section and p.get('act_name', '') == passage_act:
                        ref_p = p.copy()
                        ref_p['score'] = passage.get('score', 0) * 0.6
                        ref_p['retrieval_method'] = 'cross_reference'
                        additional.append(ref_p)
                        existing_ids.add(pid)
                        break

        return passages + additional

    def _filter_by_act(self, query: str, passages: List[Dict]) -> List[Dict]:
        """
        Post-retrieval act filtering.

        Delegates to ArticleBoostedRetriever's act filter logic so that
        dense retrieval results are restricted to the same target act as BM25.
        Only filters when the query clearly targets a specific act.
        """
        act_filter = self.bm25_retriever.extract_act_filter(query)
        if act_filter is None:
            return passages  # No filtering needed

        if act_filter == '':
            filtered = [
                p for p in passages
                if not p.get('act_name')
                or 'constitution' in p.get('act_name', '').lower()
            ]
        else:
            filtered = [
                p for p in passages
                if act_filter.lower() in p.get('act_name', '').lower()
            ]

        if len(filtered) < len(passages):
            logger.info(f"Act filter removed {len(passages) - len(filtered)} passages "
                        f"(act_filter={act_filter!r}, kept {len(filtered)})")

        # When an act filter was explicitly requested, do NOT fall back to
        # unfiltered results — returning wrong-act passages is worse than
        # returning nothing (which triggers a clean "insufficient_evidence" abstention).
        return filtered

    def _filter_by_temporal(
        self,
        passages: List[Dict],
        effective_year: Optional[int],
        exclude_repealed: bool
    ) -> List[Dict]:
        """
        Temporal filtering for RO2 — "what was the law as of year X?" queries.

        Removes:
          - Repealed Acts (when exclude_repealed=True, the default)
          - Acts enacted after effective_year (when effective_year is set)

        Passages without temporal metadata are kept to avoid silent data loss.
        """
        if not effective_year and not exclude_repealed:
            return passages

        filtered = []
        removed = 0
        for p in passages:
            meta = p.get('metadata', {})

            # Drop passages from repealed Acts
            if exclude_repealed and meta.get('is_repealed', False):
                removed += 1
                continue

            # Drop passages from Acts enacted after effective_year
            if effective_year:
                enactment_date = meta.get('enactment_date')
                if enactment_date:
                    try:
                        enact_year = int(str(enactment_date)[:4])
                        if enact_year > effective_year:
                            removed += 1
                            continue
                    except (ValueError, TypeError):
                        pass  # Keep passage if date cannot be parsed

            filtered.append(p)

        if removed:
            logger.info(
                f"Temporal filter removed {removed} passages "
                f"(effective_year={effective_year}, exclude_repealed={exclude_repealed})"
            )
        return filtered

    def _retrieve_bm25(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using BM25 only."""
        return self.bm25_retriever.retrieve_with_passages(query, top_k)

    def _retrieve_dense(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using dense retrieval only."""
        return self.dense_retriever.retrieve_with_passages(query, top_k)

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        retrieval_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve using hybrid fusion of BM25 and dense.

        Args:
            query: Query string
            top_k: Number of final results
            retrieval_k: Number to retrieve from each method before fusion

        Returns:
            Fused and ranked passages
        """
        logger.debug(f"Hybrid retrieval for query: {query[:50]}...")

        # Retrieve from both methods
        bm25_results = self.bm25_retriever.retrieve_with_passages(query, retrieval_k)
        dense_results = self.dense_retriever.retrieve_with_passages(query, retrieval_k)

        # Mirror BM25 act filtering on dense results so they draw from the same
        # target act. Without this, dense passages from other acts dominate the
        # fusion pool and push act-filtered BM25 results out of top-k.
        act_filter = self.bm25_retriever.extract_act_filter(query)
        if act_filter is not None:
            if act_filter == '':
                dense_filtered = [
                    r for r in dense_results
                    if not r.get('act_name')
                    or 'constitution' in r.get('act_name', '').lower()
                ]
            else:
                dense_filtered = [
                    r for r in dense_results
                    if act_filter.lower() in r.get('act_name', '').lower()
                ]
            # Always apply the filter — even if dense_filtered is empty.
            # If dense has no passages from the target act, use an empty list so
            # BM25-only results (already act-filtered) carry the fusion score.
            # Do NOT fall back to unfiltered dense as that lets other acts dominate.
            logger.info(f"Dense act-filter kept {len(dense_filtered)}/{len(dense_results)} "
                        f"passages (act_filter={act_filter!r})")
            dense_results = dense_filtered

        # Fuse results
        fused = self._fuse_results(bm25_results, dense_results, self.fusion_alpha)

        return fused[:top_k]

    def _retrieve_hybrid_rerank(
        self,
        query: str,
        top_k: int,
        retrieval_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve using hybrid fusion followed by reranking.

        Exact matches (injected by ArticleBoostedRetriever for explicit article/
        section references) are kept at the top and not re-scored by the general-
        purpose cross-encoder reranker. The cross-encoder was trained on web-search
        data and consistently demotes short authoritative provisions in favour of
        longer passages that merely cross-reference them. Preserving exact matches
        gives the generator direct access to the definitively relevant passage.

        Args:
            query: Query string
            top_k: Number of final results
            retrieval_k: Number to retrieve before reranking

        Returns:
            Reranked passages with exact matches preserved at the top
        """
        # Get hybrid results
        hybrid_results = self._retrieve_hybrid(query, retrieval_k)

        # Separate exact matches (explicit article/section hits) from regular results.
        # Exact matches carry the `exact_match=True` flag set by ArticleBoostedRetriever.
        exact = [p for p in hybrid_results if p.get('exact_match', False)]
        regular = [p for p in hybrid_results if not p.get('exact_match', False)]

        # Rerank non-exact passages using score fusion between hybrid signal and
        # cross-encoder reranker. Pure reranker (ms-marco-MiniLM trained on web
        # search) tends to demote short authoritative legal provisions in favour
        # of longer passages. Blending with the hybrid fusion score (70% reranker,
        # 30% hybrid) retains the reranker's cross-attention benefit while
        # preventing it from fully overriding strong BM25+dense agreement signals.
        if regular:
            # Normalise hybrid scores to [0,1] within the regular candidate pool
            hybrid_s = [r.get('score', 0.0) for r in regular]
            h_min, h_max = min(hybrid_s), max(hybrid_s)
            h_range = h_max - h_min if h_max > h_min else 1.0
            for r in regular:
                r['hybrid_norm'] = (r.get('score', 0.0) - h_min) / h_range

            # Get cross-encoder scores (adds 'rerank_score' to each passage)
            self.reranker.rerank(query, regular, top_k=len(regular))

            # Normalise reranker scores to [0,1]
            rr_s = [r.get('rerank_score', 0.0) for r in regular]
            rr_min, rr_max = min(rr_s), max(rr_s)
            rr_range = rr_max - rr_min if rr_max > rr_min else 1.0
            for r in regular:
                rr_norm = (r.get('rerank_score', 0.0) - rr_min) / rr_range
                r['score'] = 0.7 * rr_norm + 0.3 * r['hybrid_norm']

            reranked_regular = sorted(regular, key=lambda x: x['score'], reverse=True)[:top_k]
        else:
            reranked_regular = []

        # Exact matches go first; fill remaining slots with reranked regular passages
        exact_ids = {p['passage_id'] for p in exact}
        combined = exact + [p for p in reranked_regular if p['passage_id'] not in exact_ids]

        logger.debug(f"hybrid_rerank: {len(exact)} exact + {len(reranked_regular)} reranked → {len(combined)} total")

        return combined[:top_k]

    def _fuse_results(
        self,
        bm25_results: List[Dict],
        dense_results: List[Dict],
        alpha: float
    ) -> List[Dict]:
        """
        Fuse BM25 and dense retrieval results using min-max normalised score fusion.

        Each method's scores are independently normalised to [0, 1] with min-max
        scaling, then combined as: fused = (1-alpha)*bm25 + alpha*dense.
        ArticleBoostedRetriever assigns exact-match passages a score of 1000,
        which after normalisation maps to 1.0 — preserving their priority over
        all other candidates regardless of the raw score magnitude.

        Args:
            bm25_results: BM25 retrieval results (ranked)
            dense_results: Dense retrieval results (ranked)
            alpha: Fusion weight; 0=BM25 only, 1=dense only (default 0.5)

        Returns:
            Fused and sorted results
        """
        # Normalize scores using min-max per method
        bm25_scores = self._normalize_scores([r['score'] for r in bm25_results])
        dense_scores = self._normalize_scores([r['score'] for r in dense_results])

        # Create score lookup — keep the MAXIMUM score when the same passage_id
        # appears multiple times (corpus can have duplicate IDs from short articles
        # that appear in multiple schedule sections; always use the highest-ranked hit).
        score_map = {}

        for i, result in enumerate(bm25_results):
            passage_id = result['passage_id']
            if passage_id not in score_map:
                score_map[passage_id] = {
                    'passage': result,
                    'bm25_score': bm25_scores[i],
                    'dense_score': 0.0
                }
            elif bm25_scores[i] > score_map[passage_id]['bm25_score']:
                score_map[passage_id]['bm25_score'] = bm25_scores[i]
                score_map[passage_id]['passage'] = result

        for i, result in enumerate(dense_results):
            passage_id = result['passage_id']
            if passage_id not in score_map:
                score_map[passage_id] = {
                    'passage': result,
                    'bm25_score': 0.0,
                    'dense_score': dense_scores[i]
                }
            elif dense_scores[i] > score_map[passage_id].get('dense_score', 0.0):
                score_map[passage_id]['dense_score'] = dense_scores[i]

        # Compute fused scores
        fused_results = []
        for passage_id, data in score_map.items():
            passage = data['passage'].copy()
            fused_score = (1 - alpha) * data['bm25_score'] + alpha * data['dense_score']
            passage['score'] = fused_score
            passage['bm25_score'] = data['bm25_score']
            passage['dense_score'] = data['dense_score']
            passage['retrieval_method'] = 'hybrid'
            fused_results.append(passage)

        fused_results.sort(key=lambda x: x['score'], reverse=True)
        return fused_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if not scores:
            return []

        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score < 1e-6:
            return [1.0] * len(scores)

        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()

    def save_indices(self, index_dir: Path) -> None:
        """Save all indices to disk."""
        logger.info(f"Saving indices to {index_dir}")

        index_dir.mkdir(parents=True, exist_ok=True)

        # Save BM25 index
        self.bm25_retriever.save_index(index_dir / 'bm25_index.pkl')

        # Save dense index
        self.dense_retriever.save_index(
            index_dir / 'faiss_index.bin',
            index_dir / 'dense_metadata.pkl'
        )

        logger.info("All indices saved successfully")

    def load_indices(self, index_dir: Path) -> None:
        """Load all indices from disk."""
        logger.info(f"Loading indices from {index_dir}")

        # Load BM25 index (required)
        self.bm25_retriever.load_index(index_dir / 'bm25_index.pkl')
        logger.info("BM25 index loaded successfully")

        # Try to load dense index (optional - gracefully handle if missing)
        try:
            faiss_path = index_dir / 'faiss_index.bin'
            metadata_path = index_dir / 'dense_metadata.pkl'

            if faiss_path.exists() and metadata_path.exists():
                self.dense_retriever.load_index(faiss_path, metadata_path)
                self.dense_index_available = True
                logger.info("Dense index loaded successfully")
            else:
                logger.warning(f"Dense index not found at {faiss_path}. Using BM25 only.")
                self.dense_index_available = False
        except Exception as e:
            logger.warning(f"Failed to load dense index: {e}. Using BM25 only.")
            self.dense_index_available = False

        if self.dense_index_available:
            logger.info("All indices loaded successfully")
        else:
            logger.info("BM25 index loaded. Dense retrieval unavailable - hybrid methods will fall back to BM25.")
