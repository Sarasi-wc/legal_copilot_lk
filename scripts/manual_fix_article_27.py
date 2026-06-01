#!/usr/bin/env python3
"""
Manual fix for Article 27 (Directive Principles / social welfare) passage.
This passage contains correct content but has wrong metadata (labeled as Article 5).
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


def fix_article_27_passage():
    """Manually fix the Article 27 passage metadata."""
    logger.info("="*70)
    logger.info("MANUAL FIX: Article 27 (Directive Principles) Passage")
    logger.info("="*70)

    # Use the corpus_fixed.jsonl which had better overall structure
    corpus_path = Path(__file__).parent.parent / 'data' / 'processed' / 'corpus_fixed.jsonl'

    # Load corpus
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_doc = json.loads(f.readline())

    passages = corpus_doc['passages']
    logger.info(f"Loaded {len(passages)} passages")

    # Find passages containing social welfare content
    # These should be Article 27 (Directive Principles) but are mislabeled as Article 5
    fixes_applied = 0

    for passage in passages:
        text = passage.get('text', '')
        current_article = passage.get('metadata', {}).get('article_number')

        # Check if this passage contains Directive Principles content
        # Key phrases: "social security and welfare", "eliminate economic and social privilege"
        is_directive_principles = (
            'social security and welfare' in text.lower() or
            ('eliminate economic and social privilege' in text.lower() and
             'State shall ensure' in text)
        )

        if is_directive_principles and current_article != '27':
            logger.info(f"Fixing {passage['passage_id']}: {current_article} -> 27")
            logger.info(f"  Text preview: {text[:150]}...")

            # Update metadata
            passage['metadata']['article_number'] = '27'
            passage['metadata']['chapter'] = 'Chapter VI: DIRECTIVE PRINCIPLES OF STATE POLICY AND FUNDAMENTAL DUTIES'

            # Update title
            if 'Article 5' in passage.get('title', ''):
                passage['title'] = passage['title'].replace('Article 5', 'Article 27')
            elif not passage.get('title', '').startswith('Article 27'):
                passage['title'] = f"Article 27: Directive Principles of State Policy - {passage.get('title', 'Part')}"

            fixes_applied += 1

    logger.info(f"\nApplied {fixes_applied} fixes")

    # Save fixed corpus
    output_path = corpus_path.parent / 'corpus.jsonl'
    logger.info(f"Writing fixed corpus to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(corpus_doc, ensure_ascii=False) + '\n')

    # Verify
    logger.info("\nVerifying fix...")
    article_27_passages = [p for p in passages if p['metadata']['article_number'] == '27']
    logger.info(f"Article 27 now has {len(article_27_passages)} passages")

    for i, p in enumerate(article_27_passages, 1):
        logger.info(f"  {i}. {p['passage_id']}: {p['text'][:100]}...")

    logger.info("\n" + "="*70)
    logger.info("Manual fix complete!")
    logger.info("="*70)

    return True


if __name__ == '__main__':
    success = fix_article_27_passage()
    sys.exit(0 if success else 1)
