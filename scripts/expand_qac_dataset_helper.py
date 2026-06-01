#!/usr/bin/env python3
"""
Helper script to expand Q-A-C dataset.
Provides templates and utilities for creating new Q-A-C pairs.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset_schema import QACDataset, QACItem, GoldAnswer, Citation

def load_existing_dataset(dataset_path: Path) -> QACDataset:
    """Load existing Q-A-C dataset."""
    dataset = QACDataset()
    dataset.load_from_json(dataset_path)
    return dataset

def create_qac_item_template() -> Dict:
    """Create a template for a new Q-A-C item."""
    return {
        "item_id": "QAC_XXX",
        "question": "Your question here?",
        "query_type": "factual",  # factual, procedural, interpretive, cross_referenced
        "legal_domain": "administrative",  # administrative, criminal, contract, labor
        "difficulty": "easy",  # easy, medium, hard
        "gold_answers": [
            {
                "answer_text": "Your answer text here with citations [Act Name, Article X].",
                "citations": [
                    {
                        "act_name": "Constitution of the Democratic Socialist Republic of Sri Lanka",
                        "act_number": "",
                        "act_year": 1978,
                        "section_number": "X",
                        "passage_id": "PASSAGE_XXXX",
                        "passage_text": "Relevant passage text...",
                        "start_char": None,
                        "end_char": None
                    }
                ],
                "annotator_id": "LEGAL_EXPERT_1",
                "annotation_date": datetime.now().isoformat()
            }
        ],
        "relevant_passages": ["PASSAGE_XXXX", "PASSAGE_YYYY"],
        "metadata": {
            "chapter": "I",
            "articles": ["X", "Y"]
        }
    }

def find_relevant_passages_for_question(question: str, retriever, top_k: int = 10) -> List[Dict]:
    """Helper to find relevant passages for a question."""
    retrieved = retriever.retrieve(question, top_k=top_k, method="hybrid_rerank")
    return retrieved

def add_item_to_dataset(dataset: QACDataset, item_data: Dict) -> None:
    """Add a new item to the dataset."""
    from src.evaluation.dataset_schema import QACItem, GoldAnswer
    
    gold_answers = [
        GoldAnswer(
            answer_text=ans['answer_text'],
            citations=[Citation(**cit) for cit in ans.get('citations', [])],
            annotator_id=ans['annotator_id'],
            annotation_date=ans['annotation_date']
        )
        for ans in item_data['gold_answers']
    ]
    
    item = QACItem(
        item_id=item_data['item_id'],
        question=item_data['question'],
        query_type=item_data['query_type'],
        legal_domain=item_data['legal_domain'],
        difficulty=item_data['difficulty'],
        gold_answers=gold_answers,
        relevant_passages=item_data['relevant_passages'],
        metadata=item_data.get('metadata', {})
    )
    
    dataset.items.append(item)
    dataset.metadata['num_items'] = len(dataset.items)
    dataset.metadata['updated'] = datetime.now().isoformat()

def save_dataset(dataset: QACDataset, output_path: Path) -> None:
    """Save dataset to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)

def print_template():
    """Print a template for creating new Q-A-C items."""
    template = create_qac_item_template()
    print("=" * 70)
    print("Q-A-C Item Template")
    print("=" * 70)
    print(json.dumps(template, indent=2, ensure_ascii=False))
    print("\nInstructions:")
    print("1. Copy this template")
    print("2. Fill in the question, answer, and citations")
    print("3. Find relevant passage IDs using retrieval system")
    print("4. Use add_item_to_dataset() to add to dataset")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Helper for expanding Q-A-C dataset')
    parser.add_argument('--template', action='store_true', help='Print template')
    parser.add_argument('--dataset', type=Path, default=Path('data/evaluation/qac_dataset.json'),
                       help='Path to Q-A-C dataset')
    parser.add_argument('--add-item', type=Path, help='Path to JSON file with new item to add')
    
    args = parser.parse_args()
    
    if args.template:
        print_template()
    elif args.add_item:
        dataset = load_existing_dataset(args.dataset)
        with open(args.add_item) as f:
            item_data = json.load(f)
        add_item_to_dataset(dataset, item_data)
        save_dataset(dataset, args.dataset)
        print(f"✅ Added item {item_data['item_id']} to dataset")
        print(f"   Total items: {len(dataset.items)}")
    else:
        print("Usage:")
        print("  python scripts/expand_qac_dataset_helper.py --template")
        print("  python scripts/expand_qac_dataset_helper.py --add-item new_item.json")
