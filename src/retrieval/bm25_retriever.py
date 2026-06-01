"""
BM25 (sparse) retrieval implementation for lexical matching.
Implements BM25 retrieval as described in methodology Section 3.4.
"""

import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """
    BM25-based lexical retriever for legal passages.
    Uses BM25Okapi algorithm with configurable parameters.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.passages = None
        self.passage_ids = None

    def build_index(self, passages: List[Dict]) -> None:
        """
        Build BM25 index from passages.

        Args:
            passages: List of passage dictionaries with 'text' field
        """
        logger.info(f"Building BM25 index for {len(passages)} passages")

        self.passages = passages
        self.passage_ids = [p['passage_id'] for p in passages]

        # Tokenize passages
        tokenized_corpus = [
            self._tokenize(p['text'])
            for p in passages
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

        logger.info("BM25 index built successfully")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        Lowercases and strips punctuation while preserving legal section
        references such as '14(1)(a)', '12A', and hyphenated terms.
        """
        text = text.lower()
        # Split on whitespace first, then strip leading/trailing punctuation
        # from each token (but preserve internal structure like parentheses
        # in section references and hyphens in compound terms)
        raw_tokens = text.split()
        tokens = []
        for tok in raw_tokens:
            tok = tok.strip('.,;:!?\'"')
            if tok:
                tokens.append(tok)
        return tokens

    def retrieve(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k passages for query using BM25.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (passage_id, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return passage IDs and scores
        results = [
            (self.passage_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]

        return results

    def retrieve_with_passages(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve passages with full metadata.

        Uses direct positional lookup into self.passages so that the correct
        act-specific passage is returned even when passage_ids are not unique
        across acts (e.g. PASSAGE_0038 exists in Constitution, Penal Code, and
        Civil Procedure Code with different texts).

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of passage dictionaries with scores
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved_passages = []
        for idx in top_indices:
            passage = self.passages[idx]
            result = passage.copy()
            result['score'] = float(scores[idx])
            result['retrieval_method'] = 'bm25'
            retrieved_passages.append(result)

        return retrieved_passages

    def save_index(self, path: Path) -> None:
        """Save BM25 index to disk."""
        logger.info(f"Saving BM25 index to {path}")

        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'passages': self.passages,
                'passage_ids': self.passage_ids,
                'k1': self.k1,
                'b': self.b
            }, f)

    def load_index(self, path: Path) -> None:
        """Load BM25 index from disk."""
        logger.info(f"Loading BM25 index from {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.bm25 = data['bm25']
        self.passages = data['passages']
        self.passage_ids = data['passage_ids']
        self.k1 = data['k1']
        self.b = data['b']

        logger.info(f"Loaded BM25 index with {len(self.passages)} passages")
