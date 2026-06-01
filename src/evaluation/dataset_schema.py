"""
Q-A-C dataset schema and structures.
Defines the structure for question-answer-citation evaluation dataset.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    """Single citation reference."""
    act_name: str
    act_number: str
    act_year: int
    section_number: str
    passage_id: str
    passage_text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


@dataclass
class GoldAnswer:
    """Gold standard answer with citations."""
    answer_text: str
    citations: List[Citation]
    annotator_id: str
    annotation_date: str


@dataclass
class QACItem:
    """Single Q-A-C dataset item."""
    item_id: str
    question: str
    query_type: str  # factual, procedural, interpretive, cross_referenced
    legal_domain: str  # criminal, contract, labor, administrative
    difficulty: str  # easy, medium, hard
    gold_answers: List[GoldAnswer]  # Multiple annotators
    relevant_passages: List[str]  # Passage IDs
    metadata: Dict


class QACDataset:
    """
    Q-A-C dataset for evaluation.
    Implements dataset structure from RO4.
    """

    def __init__(self):
        """Initialize dataset."""
        self.items: List[QACItem] = []

    def add_item(self, item: QACItem) -> None:
        """Add item to dataset."""
        self.items.append(item)

    def get_item(self, item_id: str) -> Optional[QACItem]:
        """Get item by ID."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None

    def filter_by_domain(self, domain: str) -> List[QACItem]:
        """Filter items by legal domain."""
        return [
            item for item in self.items
            if item.legal_domain == domain
        ]

    def filter_by_query_type(self, query_type: str) -> List[QACItem]:
        """Filter items by query type."""
        return [
            item for item in self.items
            if item.query_type == query_type
        ]

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        from collections import Counter

        stats = {
            'total_items': len(self.items),
            'by_domain': dict(Counter(item.legal_domain for item in self.items)),
            'by_query_type': dict(Counter(item.query_type for item in self.items)),
            'by_difficulty': dict(Counter(item.difficulty for item in self.items)),
            'avg_citations_per_item': (
                sum(len(item.gold_answers[0].citations) for item in self.items if item.gold_answers)
                / len(self.items)
                if self.items else 0
            )
        }

        return stats

    def save_to_json(self, filepath: Path) -> None:
        """Save dataset to JSON file."""
        logger.info(f"Saving dataset to {filepath}")

        data = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'num_items': len(self.items),
            'items': [
                self._item_to_dict(item)
                for item in self.items
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Dataset saved: {len(self.items)} items")

    def load_from_json(self, filepath: Path) -> None:
        """Load dataset from JSON file."""
        logger.info(f"Loading dataset from {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.items = [
            self._dict_to_item(item_dict)
            for item_dict in data['items']
        ]

        logger.info(f"Dataset loaded: {len(self.items)} items")

    def _item_to_dict(self, item: QACItem) -> Dict:
        """Convert QACItem to dictionary."""
        return {
            'item_id': item.item_id,
            'question': item.question,
            'query_type': item.query_type,
            'legal_domain': item.legal_domain,
            'difficulty': item.difficulty,
            'gold_answers': [
                {
                    'answer_text': ans.answer_text,
                    'citations': [asdict(cit) for cit in ans.citations],
                    'annotator_id': ans.annotator_id,
                    'annotation_date': ans.annotation_date
                }
                for ans in item.gold_answers
            ],
            'relevant_passages': item.relevant_passages,
            'metadata': item.metadata
        }

    def _dict_to_item(self, d: Dict) -> QACItem:
        """Convert dictionary to QACItem."""
        gold_answers = [
            GoldAnswer(
                answer_text=ans['answer_text'],
                citations=[Citation(**cit) for cit in ans['citations']],
                annotator_id=ans['annotator_id'],
                annotation_date=ans['annotation_date']
            )
            for ans in d['gold_answers']
        ]

        return QACItem(
            item_id=d['item_id'],
            question=d['question'],
            query_type=d['query_type'],
            legal_domain=d['legal_domain'],
            difficulty=d['difficulty'],
            gold_answers=gold_answers,
            relevant_passages=d['relevant_passages'],
            metadata=d['metadata']
        )


def create_annotation_template() -> Dict:
    """
    Create template for annotators.

    Returns:
        Annotation template dictionary
    """
    template = {
        'item_id': '',
        'question': '',
        'query_type': '',  # factual, procedural, interpretive, cross_referenced
        'legal_domain': '',  # criminal, contract, labor, administrative
        'difficulty': '',  # easy, medium, hard
        'answer_text': '',
        'citations': [
            {
                'act_name': '',
                'act_number': '',
                'act_year': 0,
                'section_number': '',
                'passage_id': '',
                'passage_text': ''
            }
        ],
        'relevant_passages': [],
        'notes': ''
    }

    return template
