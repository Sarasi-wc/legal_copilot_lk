"""
Simplified Q-A-C Dataset Creation (Works Without Indices)

This version creates the Q-A-C dataset structure even if retrieval indices
are not available. Citations can be added later manually or when indices are built.
"""

import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    from src.evaluation.dataset_schema import (
        QACDataset, QACItem, GoldAnswer, Citation
    )
    from src.query_processing.query_classifier import QueryClassifier
    from src.query_processing.domain_identifier import DomainIdentifier
    from src.utils import get_logger
    
    logger = get_logger(__name__)
except ImportError as e:
    print(f"Error: {e}")
    print("Install: pip install datasets")
    sys.exit(1)


def create_qac_without_indices(
    hf_dataset_name: str = "Shifaur/sri_lanka_constitutional_law_qa",
    split: str = "train",
    max_items: int = None,
    output_path: Path = None
) -> QACDataset:
    """
    Create Q-A-C dataset without requiring retrieval indices.
    Citations will be empty and can be added later.
    """
    
    logger.info(f"Loading dataset: {hf_dataset_name}")
    
    # Load dataset
    try:
        hf_dataset = load_dataset(hf_dataset_name)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Get split
    if split not in hf_dataset:
        split = list(hf_dataset.keys())[0]
        logger.info(f"Using split: {split}")
    
    dataset_split = hf_dataset[split]
    total_items = len(dataset_split)
    
    if max_items:
        total_items = min(total_items, max_items)
    
    logger.info(f"Processing {total_items} items")
    
    # Initialize classifiers
    query_classifier = QueryClassifier()
    domain_identifier = DomainIdentifier()
    
    # Create QAC dataset
    qac_dataset = QACDataset()
    
    processed = 0
    skipped = 0
    
    for i, hf_item in enumerate(dataset_split):
        if i >= total_items:
            break
        
        try:
            # Extract question and answer
            question = hf_item.get('question', hf_item.get('Question', ''))
            answer = hf_item.get('answer', hf_item.get('Answer', hf_item.get('answer_text', '')))
            
            if not question or not answer:
                skipped += 1
                continue
            
            # Classify
            query_type_result = query_classifier.classify(question)
            query_type = query_type_result.get('query_type', 'factual')
            
            domain_result = domain_identifier.identify(question)
            legal_domain = domain_result.get('domain', 'constitutional')
            
            # Determine difficulty
            difficulty = 'medium'
            if len(question.split()) < 10:
                difficulty = 'easy'
            elif len(question.split()) > 25:
                difficulty = 'hard'
            
            # Create gold answer (without citations for now)
            gold_answer = GoldAnswer(
                answer_text=answer,
                citations=[],  # Empty - to be added later
                annotator_id='hf_dataset_creator',
                annotation_date=datetime.now().isoformat()
            )
            
            # Create QAC item
            item_id = f"constitutional_{i:04d}"
            
            qac_item = QACItem(
                item_id=item_id,
                question=question,
                query_type=query_type,
                legal_domain='constitutional',
                difficulty=difficulty,
                gold_answers=[gold_answer],
                relevant_passages=[],  # Empty - to be added when indices available
                metadata={
                    'source_dataset': hf_dataset_name,
                    'source_item_id': hf_item.get('id', str(i)),
                    'original_index': i,
                    'citations_auto_generated': False,
                    'citations_to_be_added': True,
                    'created_date': datetime.now().isoformat()
                }
            )
            
            qac_dataset.add_item(qac_item)
            processed += 1
            
            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{total_items}...")
                
        except Exception as e:
            logger.warning(f"Error processing item {i}: {e}")
            skipped += 1
            continue
    
    logger.info(f"Created {processed} items, skipped {skipped}")
    
    # Statistics
    stats = qac_dataset.get_statistics()
    logger.info(f"Statistics: {stats}")
    
    # Save
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "qac_dataset_constitutional.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qac_dataset.save_to_json(output_path)
    logger.info(f"Saved to: {output_path}")
    
    return qac_dataset


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Q-A-C dataset (simple version)")
    parser.add_argument('--max-items', type=int, default=50, help="Max items to process")
    parser.add_argument('--output', type=str, default=None, help="Output path")
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    
    print("="*80)
    print("Q-A-C DATASET CREATION (Simple Version)")
    print("="*80)
    print("\nThis version works WITHOUT retrieval indices.")
    print("Citations will be empty and can be added later.")
    print("="*80 + "\n")
    
    qac_dataset = create_qac_without_indices(
        max_items=args.max_items,
        output_path=output_path
    )
    
    print("\n" + "="*80)
    print("✅ DATASET CREATED!")
    print("="*80)
    print(f"\nTotal items: {len(qac_dataset.items)}")
    print(f"Location: {output_path or 'data/qac_dataset_constitutional.json'}")
    print("\n📝 NEXT STEPS:")
    print("  1. Review the dataset")
    print("  2. Add citations when indices are available:")
    print("     python examples/create_qac_from_hf_dataset.py")
    print("  3. Or add citations manually")
    print("  4. Validate: python examples/validate_and_enhance_qac_dataset.py")
    print("="*80)


if __name__ == "__main__":
    main()
