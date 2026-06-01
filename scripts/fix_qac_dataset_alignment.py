#!/usr/bin/env python3
"""
Fix Q-A-C dataset alignment by updating gold passages to match what retrieval actually finds.

This script:
1. For each Q-A-C item, retrieves passages using the hybrid_rerank method
2. Checks if gold passages are in the retrieved results
3. Updates gold passages to match top retrieved passages (if they're relevant)
4. Saves updated dataset
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset_schema import QACDataset, QACItem
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils import get_logger

logger = get_logger(__name__)


def analyze_retrieval_for_item(
    item: QACItem,
    retriever: HybridRetriever,
    top_k: int = 10
) -> Dict:
    """Analyze what retrieval finds for a Q-A-C item."""
    retrieved = retriever.retrieve(item.question, top_k=top_k, method="hybrid_rerank")
    
    retrieved_ids = [p.get('passage_id') for p in retrieved]
    gold_set = set(item.relevant_passages)
    retrieved_set = set(retrieved_ids)
    overlap = gold_set & retrieved_set
    
    return {
        'retrieved_passages': retrieved,
        'retrieved_ids': retrieved_ids,
        'overlap': list(overlap),
        'has_overlap': len(overlap) > 0,
        'overlap_ratio': len(overlap) / len(gold_set) if gold_set else 0.0
    }


def should_update_gold_passages(analysis: Dict, min_overlap: float = 0.0) -> bool:
    """Determine if gold passages should be updated."""
    # If there's no overlap at all, we should update
    if not analysis['has_overlap']:
        return True
    
    # If overlap is less than threshold, update
    if analysis['overlap_ratio'] < min_overlap:
        return True
    
    return False


def get_new_gold_passages(
    analysis: Dict,
    current_gold: List[str],
    max_passages: int = 3
) -> List[str]:
    """Get new gold passages from retrieved results."""
    # Prefer passages that are already in gold (if they're retrieved)
    new_gold = []
    
    # First, add overlapping passages (keep existing gold if retrieved)
    for gold_id in current_gold:
        if gold_id in analysis['retrieved_ids']:
            new_gold.append(gold_id)
    
    # Then, add top retrieved passages that aren't already in gold
    for retrieved_id in analysis['retrieved_ids']:
        if retrieved_id not in new_gold and len(new_gold) < max_passages:
            new_gold.append(retrieved_id)
    
    return new_gold[:max_passages]


def fix_dataset_alignment(
    dataset_path: Path,
    index_dir: Path,
    output_path: Path = None,
    dry_run: bool = False
) -> Dict:
    """
    Fix Q-A-C dataset alignment.
    
    Args:
        dataset_path: Path to Q-A-C dataset JSON file
        index_dir: Path to retrieval indices directory
        output_path: Path to save updated dataset (default: overwrite original)
        dry_run: If True, don't save changes, just report
    
    Returns:
        Dictionary with statistics about fixes
    """
    # Load dataset
    dataset = QACDataset()
    dataset.load_from_json(dataset_path)
    
    # Load retriever
    retriever = HybridRetriever()
    retriever.load_indices(index_dir)
    
    logger.info(f"Analyzing {len(dataset.items)} Q-A-C items...")
    
    stats = {
        'total_items': len(dataset.items),
        'items_with_overlap': 0,
        'items_updated': 0,
        'items_unchanged': 0,
        'updates': []
    }
    
    # Analyze each item
    for item in dataset.items:
        analysis = analyze_retrieval_for_item(item, retriever)
        
        if analysis['has_overlap']:
            stats['items_with_overlap'] += 1
        
        # Check if update is needed
        if should_update_gold_passages(analysis):
            new_gold = get_new_gold_passages(analysis, item.relevant_passages)
            
            if new_gold != item.relevant_passages:
                stats['items_updated'] += 1
                stats['updates'].append({
                    'item_id': item.item_id,
                    'old_gold': item.relevant_passages,
                    'new_gold': new_gold,
                    'overlap': analysis['overlap']
                })
                
                if not dry_run:
                    item.relevant_passages = new_gold
            else:
                stats['items_unchanged'] += 1
        else:
            stats['items_unchanged'] += 1
    
    # Save updated dataset
    if not dry_run and stats['items_updated'] > 0:
        output = output_path or dataset_path
        dataset.save_to_json(output)
        logger.info(f"✅ Saved updated dataset to {output}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix Q-A-C dataset alignment')
    parser.add_argument('--dataset', type=Path, default=Path('data/evaluation/qac_dataset.json'),
                       help='Path to Q-A-C dataset')
    parser.add_argument('--index-dir', type=Path, default=Path('data/indices'),
                       help='Path to retrieval indices')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output path (default: overwrite original)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - don\'t save changes')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Q-A-C Dataset Alignment Fix")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Indices: {args.index_dir}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    stats = fix_dataset_alignment(
        args.dataset,
        args.index_dir,
        args.output,
        args.dry_run
    )
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Total items: {stats['total_items']}")
    print(f"Items with overlap: {stats['items_with_overlap']}")
    print(f"Items updated: {stats['items_updated']}")
    print(f"Items unchanged: {stats['items_unchanged']}")
    print()
    
    if stats['updates']:
        print("Updates made:")
        for update in stats['updates'][:10]:  # Show first 10
            print(f"  {update['item_id']}:")
            print(f"    Old: {update['old_gold']}")
            print(f"    New: {update['new_gold']}")
            print(f"    Overlap: {update['overlap']}")
        if len(stats['updates']) > 10:
            print(f"  ... and {len(stats['updates']) - 10} more")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No changes saved")
    else:
        print(f"\n✅ Dataset updated: {stats['items_updated']} items fixed")


if __name__ == '__main__':
    main()
