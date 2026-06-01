"""
Article-aware BM25 retriever with pattern matching boost.
Boosts passages that start with the article number mentioned in the query.
"""

import re
from typing import List, Dict, Tuple
from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever
from src.utils import get_logger

logger = get_logger(__name__)


class ArticleBoostedRetriever(BM25Retriever):
    """
    BM25 retriever with article number pattern matching.
    Boosts passages that start with "X." when query mentions "Article X".
    """

    def __init__(self, boost_factor: float = 3.0):
        """
        Initialize article-boosted retriever.

        Args:
            boost_factor: Multiplier for passages matching article pattern
        """
        super().__init__()
        self.boost_factor = boost_factor

        # Pattern to extract article number from query (Constitution style)
        self.article_query_pattern = re.compile(
            r'Article\s+(\d+[A-Z]?)',
            re.IGNORECASE
        )

        # Pattern to extract section number from query (Penal Code / CPC style)
        self.section_query_pattern = re.compile(
            r'[Ss]ection\s+(\d+[A-Z]?)',
            re.IGNORECASE
        )

        # Known act name patterns for source filtering
        self._act_patterns = [
            (re.compile(r'\bconstitution\b', re.IGNORECASE), ''),          # Constitution has empty act_name
            (re.compile(r'\bpenal\s+code\b', re.IGNORECASE), 'Penal Code'),
            (re.compile(r'\bcivil\s+procedure\s+code\b', re.IGNORECASE), 'Civil Procedure Code'),
            (re.compile(r'\bevidence\s+ordinance\b', re.IGNORECASE), 'Evidence Ordinance'),
            (re.compile(r'\bcriminal\s+procedure\s+code\b', re.IGNORECASE), 'Criminal Procedure Code'),
        ]

    def extract_article_number(self, query: str) -> str:
        """
        Extract article number from query.

        Args:
            query: Query string

        Returns:
            Article number or None if not found
        """
        match = self.article_query_pattern.search(query)
        if match:
            return match.group(1)
        return None

    def passage_starts_with_article(self, passage_text: str, article_num: str) -> bool:
        """
        Check if passage starts with given article number.

        Args:
            passage_text: Passage text
            article_num: Article number to check

        Returns:
            True if passage starts with "X." or "Chapter...X."
        """
        # Pattern: "X." at start or after chapter heading
        pattern = re.compile(
            rf'(?:^|Chapter[^0-9]+)\s*{re.escape(article_num)}\.\s',
            re.IGNORECASE | re.MULTILINE
        )
        return bool(pattern.search(passage_text[:200]))  # Check first 200 chars

    def extract_act_filter(self, query: str) -> str:
        """
        Extract act name filter from query text.

        Returns the expected act_name value (empty string for Constitution),
        or None if no specific act is mentioned.

        Defaults to Constitution ('') when the query uses "Article X" without
        naming a specific act, because in Sri Lankan legal convention "Article X"
        refers to the Constitution while statutes use "Section X".
        """
        for pattern, act_name in self._act_patterns:
            if pattern.search(query):
                return act_name  # May be '' for Constitution

        # Default: "Article X" without an explicit act → treat as Constitution
        if self.article_query_pattern.search(query):
            # Only default to Constitution if query doesn't mention another act by keyword
            section_only = re.search(r'\bsection\b', query, re.IGNORECASE)
            if not section_only:
                logger.debug("No explicit act mentioned with 'Article X' query — defaulting to Constitution")
                return ''  # Constitution

        return None

    def find_exact_article_matches(self, article_num: str, act_filter=None) -> List[Dict]:
        """
        Find passages with exact article number match in metadata or passage text.

        First checks metadata.article_number (fast, precise).
        Falls back to text-based matching using passage_starts_with_article when
        metadata is absent (e.g. Constitution passages with metadata={}).

        Args:
            article_num: Article number to find
            act_filter: If provided, only return passages from this act.
                        Empty string '' means Constitution (empty act_name).
                        None means no filter (all acts).

        Returns:
            List of matching passages with artificial high scores
        """
        def _passes_act_filter(passage: Dict) -> bool:
            if act_filter is None:
                return True
            passage_act = passage.get('act_name', '')
            if act_filter == '':
                return not passage_act or 'constitution' in passage_act.lower()
            return act_filter.lower() in passage_act.lower()

        matches = []
        for passage in self.passages:
            metadata = passage.get('metadata', {})
            # Check article_number first; fall back to section_number when absent.
            # In the Constitution corpus the segmenter stores article numbers under
            # section_number (OCR does not distinguish article vs section headings),
            # so we treat section_number as article_number for Constitution passages.
            art_num_meta = str(metadata.get('article_number') or '').strip()
            if not art_num_meta:
                art_num_meta = str(metadata.get('section_number') or '').strip()
            if art_num_meta == article_num:
                if _passes_act_filter(passage):
                    match = passage.copy()
                    match['score'] = 1000.0
                    match['exact_match'] = True
                    matches.append(match)

        # Text-based fallback: scan passage text for "X. " pattern.
        if not matches:
            for passage in self.passages:
                if not _passes_act_filter(passage):
                    continue
                if self.passage_starts_with_article(passage.get('text', ''), article_num):
                    match = passage.copy()
                    match['score'] = 900.0
                    match['exact_match'] = True
                    match['text_matched'] = True
                    matches.append(match)

        logger.info(f"Found {len(matches)} exact matches for Article {article_num}"
                    + (f" (act_filter={act_filter!r})" if act_filter is not None else ""))
        return matches

    def find_exact_section_matches(self, section_num: str, act_filter=None) -> List[Dict]:
        """
        Find passages whose metadata.section_number matches section_num.
        Used to boost "Section X of the Penal Code" style queries.
        """
        def _passes_act_filter(passage: Dict) -> bool:
            if act_filter is None:
                return True
            passage_act = passage.get('act_name', '')
            if act_filter == '':
                return not passage_act or 'constitution' in passage_act.lower()
            return act_filter.lower() in passage_act.lower()

        matches = []
        for passage in self.passages:
            metadata = passage.get('metadata', {})
            if str(metadata.get('section_number', '')).strip() == section_num:
                if _passes_act_filter(passage):
                    match = passage.copy()
                    match['score'] = 950.0
                    match['exact_match'] = True
                    matches.append(match)

        logger.info(f"Found {len(matches)} exact matches for Section {section_num}"
                    + (f" (act_filter={act_filter!r})" if act_filter is not None else ""))
        return matches

    def retrieve_with_passages(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve passages with article/section number boosting.

        For article-specific queries, first finds exact metadata matches for
        ALL article numbers mentioned in the query, then supplements with BM25.
        Supports multi-article queries such as
        "Article 10 Article 11 Article 12 Article 13 Article 14" generated by
        the query expansion layer for enumeration questions.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of passage dictionaries with boosted scores
        """
        # Extract ALL article numbers from query (supports multi-article expansions)
        all_article_nums = list(dict.fromkeys(self.article_query_pattern.findall(query)))

        if all_article_nums:
            logger.info(f"Detected article number query: Articles {all_article_nums}")

            # Extract act filter from query so we don't mix up Articles across acts
            act_filter = self.extract_act_filter(query)

            # Collect exact matches for ALL mentioned article numbers
            exact_matches = []
            seen_exact_ids: set = set()
            for art_num in all_article_nums:
                for m in self.find_exact_article_matches(art_num, act_filter=act_filter):
                    if m['passage_id'] not in seen_exact_ids:
                        seen_exact_ids.add(m['passage_id'])
                        exact_matches.append(m)

            # Then get BM25 results for supplementary context
            bm25_results = super().retrieve_with_passages(query, top_k=top_k * 3)

            # Filter out duplicates (passages already in exact matches)
            bm25_results = [r for r in bm25_results if r['passage_id'] not in seen_exact_ids]

            # When act_filter is set, restrict BM25 results to the target act.
            # This is critical when metadata is absent (e.g., Constitution passages
            # have metadata={}) so exact_matches is empty but BM25 must still be
            # filtered to avoid returning passages from other acts.
            if act_filter is not None:
                if act_filter == '':
                    bm25_results = [
                        r for r in bm25_results
                        if not r.get('act_name')
                        or 'constitution' in r.get('act_name', '').lower()
                    ]
                else:
                    bm25_results = [
                        r for r in bm25_results
                        if act_filter.lower() in r.get('act_name', '').lower()
                    ]
                logger.info(f"Act-filtered BM25 to {len(bm25_results)} passages "
                            f"(act_filter={act_filter!r})")

            # Combine: exact matches first, then BM25 results
            all_results = exact_matches + bm25_results

            # Mark BM25 results as not exact
            for result in bm25_results:
                result['exact_match'] = False

            # Return top_k
            return all_results[:top_k]

        else:
            # No article number — check for explicit section number (Penal Code / CPC)
            section_match = self.section_query_pattern.search(query)
            act_filter = self.extract_act_filter(query)

            if section_match:
                section_num = section_match.group(1)
                logger.info(f"Detected section number query: Section {section_num}")
                exact_matches = self.find_exact_section_matches(section_num, act_filter=act_filter)
                bm25_results = super().retrieve_with_passages(query, top_k=top_k * 3)
                exact_ids = {m['passage_id'] for m in exact_matches}
                bm25_results = [r for r in bm25_results if r['passage_id'] not in exact_ids]
                for r in bm25_results:
                    r['exact_match'] = False
                return (exact_matches + bm25_results)[:top_k]

            # No article or section number — check if query targets a specific act
            # Fetch a larger pool so act filtering still yields top_k results
            pool = super().retrieve_with_passages(query, top_k=top_k * 5)

            if act_filter is not None:
                if act_filter == '':
                    pool = [
                        r for r in pool
                        if not r.get('act_name')
                        or 'constitution' in r.get('act_name', '').lower()
                    ]
                else:
                    pool = [
                        r for r in pool
                        if act_filter.lower() in r.get('act_name', '').lower()
                    ]
                logger.info(f"Act-filtered BM25 to {len(pool)} passages "
                            f"(act_filter={act_filter!r})")

            return pool[:top_k]
