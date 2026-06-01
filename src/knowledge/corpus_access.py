"""
Corpus Access
Unified interface to access the legal corpus.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from src.utils import get_logger

logger = get_logger(__name__)


class CorpusAccess:
    """
    Provides structured access to the legal corpus.
    """

    def __init__(self, corpus_path: Optional[Path] = None):
        """
        Initialize corpus access.

        Args:
            corpus_path: Path to processed corpus file
        """
        self.corpus_path = corpus_path or Path("./data/processed/corpus.jsonl")
        self.passages = []
        self.acts_index = {}
        self.loaded = False

    def load(self):
        """Load corpus into memory."""
        if self.loaded:
            return

        logger.info(f"Loading corpus from {self.corpus_path}")

        if not self.corpus_path.exists():
            logger.warning(f"Corpus file not found: {self.corpus_path}")
            return

        self.passages = []
        with open(self.corpus_path, 'r') as f:
            for line in f:
                passage = json.loads(line)
                self.passages.append(passage)

        # Build acts index
        self._build_acts_index()

        self.loaded = True
        logger.info(f"Loaded {len(self.passages)} passages from corpus")

    def _build_acts_index(self):
        """Build index of acts for fast lookup."""
        for passage in self.passages:
            act_name = passage.get('act_name')
            if act_name:
                if act_name not in self.acts_index:
                    self.acts_index[act_name] = []
                self.acts_index[act_name].append(passage)

    def get_act_by_name(self, act_name: str) -> List[Dict]:
        """
        Get all passages for a specific act.

        Args:
            act_name: Name of the act

        Returns:
            List of passages from that act
        """
        if not self.loaded:
            self.load()

        return self.acts_index.get(act_name, [])

    def get_section(self, act_name: str, section: str) -> Optional[Dict]:
        """
        Get specific section from an act.

        Args:
            act_name: Name of the act
            section: Section number

        Returns:
            Passage dict if found, None otherwise
        """
        if not self.loaded:
            self.load()

        passages = self.get_act_by_name(act_name)

        for passage in passages:
            section_num = passage.get('metadata', {}).get('section_number')
            if section_num == section:
                return passage

        return None

    def search_by_domain(self, domain: str) -> List[Dict]:
        """
        Get all passages for a legal domain.

        Args:
            domain: Legal domain (criminal, civil, etc.)

        Returns:
            List of passages in that domain
        """
        if not self.loaded:
            self.load()

        return [p for p in self.passages if p.get('domain') == domain]

    def get_all_acts(self) -> List[str]:
        """
        Get list of all acts in corpus.

        Returns:
            List of act names
        """
        if not self.loaded:
            self.load()

        return list(self.acts_index.keys())

    def get_corpus_statistics(self) -> Dict:
        """
        Get corpus statistics.

        Returns:
            Dictionary with corpus stats
        """
        if not self.loaded:
            self.load()

        domains = {}
        for passage in self.passages:
            domain = passage.get('domain', 'unknown')
            domains[domain] = domains.get(domain, 0) + 1

        return {
            'total_passages': len(self.passages),
            'total_acts': len(self.acts_index),
            'domains': domains,
            'acts': list(self.acts_index.keys())
        }
