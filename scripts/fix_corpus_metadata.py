#!/usr/bin/env python3
"""
Fix corpus metadata by correctly detecting article numbers from passage text.
Much simpler approach: read existing passages and extract correct article numbers.
"""

import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


def extract_article_number_from_text(text: str, title: str = "") -> str:
    """
    Extract article number from passage text and title.

    Args:
        text: Passage text
        title: Passage title

    Returns:
        Article number or None
    """
    # Pattern 1: Title contains "Article X:"
    title_article = re.search(r'\bArticle\s+(\d+[A-Z]?)', title, re.IGNORECASE)
    if title_article:
        return title_article.group(1)

    # Pattern 2: Text mentions "Article X" or "Chapter X" early
    article_mention = re.search(r'\bArticle\s+(\d+[A-Z]?)', text[:300], re.IGNORECASE)
    if article_mention:
        return article_mention.group(1)

    # Pattern 3: Text starts with "CHAPTER ... \n27. (1)" format
    chapter_article = re.search(r'CHAPTER\s+[IVXLCDM]+[^\\n]*\\n(\d+[A-Z]?)\\.\s+', text[:500], re.IGNORECASE)
    if chapter_article:
        return chapter_article.group(1)

    # Pattern 4: Text starts with number and period: "9. The Republic..."
    number_start = re.match(r'^(?:Chapter[^\\n]*\\n)?(\d+[A-Z]?)\\.\s+', text)
    if number_start:
        return number_start.group(1)

    # Pattern 5: Text contains "27. (1) The Directive Principles" - extract 27
    numbered_subclause = re.search(r'(\d+[A-Z]?)\.\s+\(\d+[a-z]?\)', text[:500])
    if numbered_subclause:
        return numbered_subclause.group(1)
    
    # Pattern 6: Text contains "14. (1) (e)" format - extract 14
    article_with_subclause = re.search(r'(\d+[A-Z]?)\.\s+\(\d+\)\s*\([a-z]\)', text[:500])
    if article_with_subclause:
        return article_with_subclause.group(1)

    return None


def fix_corpus_metadata():
    """Fix corpus metadata for all passages."""
    logger.info("Starting corpus metadata fix")

    # Paths
    data_path = Path(__file__).parent.parent / 'data'
    corpus_path = data_path / 'processed' / 'corpus.jsonl'
    output_path = data_path / 'processed' / 'corpus_fixed.jsonl'
    backup_path = data_path / 'processed' / 'corpus_backup.jsonl'

    # Load existing corpus
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_doc = json.loads(f.readline())

    passages = corpus_doc['passages']
    logger.info(f"Loaded {len(passages)} passages")

    # Fix each passage
    fixed = 0
    errors = 0

    for passage in passages:
        text = passage.get('text', '')
        title = passage.get('title', '')
        current_article = passage.get('metadata', {}).get('article_number')

        # Try to extract correct article number
        detected_article = extract_article_number_from_text(text, title)

        if detected_article and detected_article != current_article:
            logger.debug(f"Fixing {passage['passage_id']}: {current_article} -> {detected_article}")
            passage['metadata']['article_number'] = detected_article

            # Update title to reflect correct article
            if 'Article' not in passage.get('title', ''):
                passage['title'] = f"Article {detected_article}: {passage.get('title', '')}"

            fixed += 1
        elif not detected_article and current_article:
            # Couldn't detect article number from text, might be continuation/chunk
            # Keep existing if it seems reasonable
            logger.debug(f"Keeping {passage['passage_id']} with article {current_article} (no detection)")
        elif not detected_article and not current_article:
            logger.warning(f"No article number for {passage['passage_id']}")
            errors += 1

    logger.info(f"Fixed {fixed} passages")
    logger.info(f"Errors: {errors} passages without article numbers")

    # Backup original
    logger.info(f"Backing up original to {backup_path}")
    import shutil
    shutil.copy(corpus_path, backup_path)

    # Update corpus document
    corpus_doc['passages'] = passages

    # Write fixed corpus
    logger.info(f"Writing fixed corpus to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(corpus_doc, ensure_ascii=False) + '\n')

    # Show statistics
    article_counts = {}
    for passage in passages:
        art_num = passage.get('metadata', {}).get('article_number')
        if art_num:
            article_counts[art_num] = article_counts.get(art_num, 0) + 1

    logger.info(f"\nArticle distribution ({len(article_counts)} unique articles):")
    for art_num in sorted(article_counts.keys(), key=lambda x: int(re.match(r'\\d+', x).group()) if re.match(r'\\d+', x) else 999):
        count = article_counts[art_num]
        if count > 10:  # Show articles with many passages (potential issues)
            logger.info(f"  Article {art_num}: {count} passages ⚠️")
        else:
            logger.info(f"  Article {art_num}: {count} passages")

    logger.info("\nCorpus metadata fix complete!")


if __name__ == '__main__':
    fix_corpus_metadata()
