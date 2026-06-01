"""
Validate and Enhance Q-A-C Dataset

This script:
1. Validates the Q-A-C dataset structure
2. Checks citation accuracy
3. Identifies items needing manual review
4. Provides statistics and quality metrics

Usage:
    python examples/validate_and_enhance_qac_dataset.py --dataset data/qac_dataset_constitutional.json
"""

import sys
from pathlib import Path
from typing import List, Dict
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.dataset_schema import QACDataset
from src.utils import get_logger

logger = get_logger(__name__)


def validate_dataset(dataset_path: Path) -> Dict:
    """
    Validate Q-A-C dataset.
    
    Args:
        dataset_path: Path to dataset JSON file
        
    Returns:
        Validation report dictionary
    """
    logger.info(f"Validating dataset: {dataset_path}")
    
    # Load dataset
    qac_dataset = QACDataset()
    qac_dataset.load_from_json(dataset_path)
    
    report = {
        'total_items': len(qac_dataset.items),
        'valid_items': 0,
        'invalid_items': 0,
        'items_without_citations': 0,
        'items_without_answers': 0,
        'items_needing_review': [],
        'errors': [],
        'warnings': []
    }
    
    # Validate each item
    for item in qac_dataset.items:
        is_valid = True
        needs_review = False
        item_issues = []
        
        # Check required fields
        if not item.question:
            report['errors'].append(f"Item {item.item_id}: Missing question")
            is_valid = False
        
        if not item.gold_answers:
            report['items_without_answers'] += 1
            report['errors'].append(f"Item {item.item_id}: Missing gold answers")
            is_valid = False
        
        # Check citations
        if item.gold_answers:
            for gold_answer in item.gold_answers:
                if not gold_answer.citations:
                    report['items_without_citations'] += 1
                    item_issues.append("No citations")
                    needs_review = True
                else:
                    # Check citation quality
                    for citation in gold_answer.citations:
                        if not citation.act_name:
                            item_issues.append("Citation missing act_name")
                            needs_review = True
                        if not citation.section_number:
                            item_issues.append("Citation missing section_number")
                            needs_review = True
                        if not citation.passage_text:
                            item_issues.append("Citation missing passage_text")
                            needs_review = True
        
        # Check metadata
        if item.metadata.get('citations_auto_generated'):
            needs_review = True
            item_issues.append("Citations auto-generated - needs manual verification")
        
        if is_valid:
            report['valid_items'] += 1
        else:
            report['invalid_items'] += 1
        
        if needs_review:
            report['items_needing_review'].append({
                'item_id': item.item_id,
                'question': item.question[:100],
                'issues': item_issues
            })
    
    # Statistics
    stats = qac_dataset.get_statistics()
    report['statistics'] = stats
    
    return report


def print_validation_report(report: Dict):
    """Print validation report."""
    print("\n" + "="*80)
    print("Q-A-C DATASET VALIDATION REPORT")
    print("="*80)
    
    print(f"\n📊 OVERVIEW:")
    print(f"  Total items: {report['total_items']}")
    print(f"  Valid items: {report['valid_items']}")
    print(f"  Invalid items: {report['invalid_items']}")
    print(f"  Items without citations: {report['items_without_citations']}")
    print(f"  Items without answers: {report['items_without_answers']}")
    print(f"  Items needing review: {len(report['items_needing_review'])}")
    
    print(f"\n📈 STATISTICS:")
    stats = report['statistics']
    print(f"  By domain: {stats['by_domain']}")
    print(f"  By query type: {stats['by_query_type']}")
    print(f"  By difficulty: {stats['by_difficulty']}")
    print(f"  Avg citations per item: {stats['avg_citations_per_item']:.2f}")
    
    if report['errors']:
        print(f"\n❌ ERRORS ({len(report['errors'])}):")
        for error in report['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(report['errors']) > 10:
            print(f"  ... and {len(report['errors']) - 10} more errors")
    
    if report['items_needing_review']:
        print(f"\n⚠️  ITEMS NEEDING REVIEW ({len(report['items_needing_review'])}):")
        for item in report['items_needing_review'][:10]:  # Show first 10
            print(f"\n  Item ID: {item['item_id']}")
            print(f"  Question: {item['question']}...")
            print(f"  Issues: {', '.join(item['issues'])}")
        if len(report['items_needing_review']) > 10:
            print(f"\n  ... and {len(report['items_needing_review']) - 10} more items")
    
    print("\n" + "="*80)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if report['items_without_citations'] > 0:
        print(f"  - {report['items_without_citations']} items need citations added")
        print("    → Use retrieval system or manual annotation")
    
    if report['items_needing_review']:
        print(f"  - {len(report['items_needing_review'])} items need manual review")
        print("    → Verify citations are accurate")
        print("    → Check answers are correct")
    
    if report['invalid_items'] > 0:
        print(f"  - {report['invalid_items']} items have errors")
        print("    → Fix errors before using dataset")
    
    if report['valid_items'] == report['total_items']:
        print("  ✅ All items are valid!")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate and enhance Q-A-C dataset"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Path to Q-A-C dataset JSON file"
    )
    parser.add_argument(
        '--output-report',
        type=str,
        default=None,
        help="Path to save validation report (optional)"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    # Validate dataset
    report = validate_dataset(dataset_path)
    
    # Print report
    print_validation_report(report)
    
    # Save report if requested
    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to: {output_path}")


if __name__ == "__main__":
    main()
