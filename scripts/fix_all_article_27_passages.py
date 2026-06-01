#!/usr/bin/env python3
"""
Comprehensive fix for all mislabeled Article 27 passages.
Fixes PASSAGE_0014_CHUNK_2, 3, 4, 6 from Article 5 to Article 27.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


def fix_all_article_27_passages():
    """Fix all mislabeled Article 27 passages."""
    logger.info("="*70)
    logger.info("COMPREHENSIVE FIX: All Article 27 Passages")
    logger.info("="*70)

    corpus_path = Path(__file__).parent.parent / 'data' / 'processed' / 'corpus.jsonl'

    # Load corpus
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_doc = json.loads(f.readline())

    passages = corpus_doc['passages']
    logger.info(f"Loaded {len(passages)} passages")

    # Passages to fix (currently labeled as Article 5 but contain Article 27 content)
    passages_to_fix = [
        'PASSAGE_0014_CHUNK_2',  # Contains 27(1)
        'PASSAGE_0014_CHUNK_3',  # Contains 27(2) with lettered sub-clauses
        'PASSAGE_0014_CHUNK_4',  # Contains 27(3) and (4)
        'PASSAGE_0014_CHUNK_6',  # Contains 27(14) and (15)
    ]

    fixes_applied = 0

    for passage in passages:
        if passage['passage_id'] in passages_to_fix:
            current_article = passage['metadata'].get('article_number')

            if current_article != '27':
                logger.info(f"\nFixing {passage['passage_id']}: Article {current_article} -> 27")
                logger.info(f"  Text preview: {passage['text'][:100]}...")

                # Update metadata
                passage['metadata']['article_number'] = '27'
                passage['metadata']['chapter'] = 'Chapter VI: DIRECTIVE PRINCIPLES OF STATE POLICY AND FUNDAMENTAL DUTIES'

                # Update title
                if 'Article 5' in passage.get('title', ''):
                    passage['title'] = passage['title'].replace('Article 5', 'Article 27')
                elif not passage.get('title', '').startswith('Article 27'):
                    passage['title'] = f"Article 27: {passage.get('title', 'Part')}"

                fixes_applied += 1

    logger.info(f"\n\nApplied {fixes_applied} fixes")

    # Save fixed corpus
    output_path = corpus_path
    logger.info(f"Writing fixed corpus to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(corpus_doc, ensure_ascii=False) + '\n')

    # Verify
    logger.info("\nVerifying fixes...")
    article_27_passages = [p for p in passages if p['metadata']['article_number'] == '27']
    article_5_passages = [p for p in passages if p['metadata']['article_number'] == '5']

    logger.info(f"\nArticle 27 now has {len(article_27_passages)} passages")
    logger.info(f"Article 5 now has {len(article_5_passages)} passages")

    # Check for any remaining Article 5 passages with welfare keywords
    logger.info("\nChecking Article 5 passages for welfare content...")
    for p in article_5_passages:
        text_lower = p['text'].lower()
        if any(kw in text_lower for kw in ['democratic socialist society', 'welfare of the people', 'adequate standard of living']):
            logger.warning(f"  STILL MISLABELED: {p['passage_id']}")
            logger.warning(f"    Preview: {p['text'][:150]}")

    logger.info("\n" + "="*70)
    logger.info("Comprehensive fix complete!")
    logger.info("="*70)

    return True


if __name__ == '__main__':
    success = fix_all_article_27_passages()
    sys.exit(0 if success else 1)
