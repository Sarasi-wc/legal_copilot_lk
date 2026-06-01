#!/usr/bin/env python3
"""
Rebuild Constitution corpus with proper article segmentation.
Uses Constitution-specific segmenter for accurate article boundary detection.
"""

import json
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_construction.ocr_processor import OCRProcessor
from src.corpus_construction.constitution_segmenter import ConstitutionSegmenter
from src.corpus_construction.metadata_extractor import MetadataExtractor
from src.utils import get_logger

logger = get_logger(__name__)


def rebuild_constitution_corpus():
    """Rebuild Constitution corpus with proper segmentation."""
    logger.info("="*70)
    logger.info("REBUILDING CONSTITUTION CORPUS WITH PROPER SEGMENTATION")
    logger.info("="*70)

    # Paths
    data_path = Path(__file__).parent.parent / 'data'
    pdf_path = data_path / 'raw' / 'acts' / 'constitution_1978.pdf'
    corpus_path = data_path / 'processed' / 'corpus.jsonl'
    backup_path = data_path / 'processed' / 'corpus_old.jsonl'

    # Validate PDF exists
    if not pdf_path.exists():
        logger.error(f"Constitution PDF not found at {pdf_path}")
        return False

    # Step 1: Backup existing corpus
    if corpus_path.exists():
        logger.info(f"Backing up existing corpus to {backup_path}")
        shutil.copy(corpus_path, backup_path)

    # Step 2: Extract text from PDF
    logger.info(f"Extracting text from PDF: {pdf_path}")
    ocr = OCRProcessor()
    text, ocr_metadata = ocr.extract_text_from_pdf(pdf_path, use_ocr=False)
    logger.info(f"Extracted {len(text)} characters")

    # Step 3: Segment Constitution into articles
    logger.info("Segmenting Constitution into articles...")
    segmenter = ConstitutionSegmenter()
    articles = segmenter.segment_constitution(text)
    logger.info(f"Segmented into {len(articles)} articles")

    # Show article distribution
    logger.info("\nArticle numbers detected:")
    for i, article in enumerate(articles[:20]):
        logger.info(f"  {i+1}. Article {article.article_number}: {article.title[:60]}...")
    if len(articles) > 20:
        logger.info(f"  ... and {len(articles) - 20} more articles")

    # Step 4: Create retrieval passages
    logger.info("\nCreating retrieval passages...")
    passages = segmenter.create_passages(articles, max_length=1000)
    logger.info(f"Created {len(passages)} passages")

    # Step 5: Extract metadata
    logger.info("\nExtracting metadata...")
    metadata_extractor = MetadataExtractor()
    act_metadata = metadata_extractor.extract_act_metadata(
        text=text,
        act_name="Constitution of the Democratic Socialist Republic of Sri Lanka",
        act_number="",
        year=1978
    )

    # Step 6: Validate segmentation quality
    logger.info("\nValidating segmentation quality...")
    validation_results = validate_segmentation(articles, passages)

    if not validation_results['passed']:
        logger.warning("Segmentation validation issues detected:")
        for issue in validation_results['issues']:
            logger.warning(f"  - {issue}")

    # Step 7: Create corpus document
    logger.info("\nCreating corpus document...")
    corpus_doc = {
        'act_info': {
            'name': 'Constitution of the Democratic Socialist Republic of Sri Lanka',
            'number': '',
            'year': 1978
        },
        'act_metadata': act_metadata,
        'ocr_metadata': ocr_metadata,
        'quality_score': 1.0,
        'num_sections': len(articles),
        'num_passages': len(passages),
        'passages': passages
    }

    # Step 8: Write corpus
    logger.info(f"Writing corpus to {corpus_path}")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(corpus_doc, ensure_ascii=False) + '\n')

    # Step 9: Show statistics
    logger.info("\n" + "="*70)
    logger.info("CORPUS REBUILD COMPLETE")
    logger.info("="*70)
    logger.info(f"Articles detected: {len(articles)}")
    logger.info(f"Passages created: {len(passages)}")

    # Article distribution
    article_counts = {}
    for passage in passages:
        art_num = passage['metadata']['article_number']
        article_counts[art_num] = article_counts.get(art_num, 0) + 1

    logger.info(f"Unique articles: {len(article_counts)}")

    # Show articles with multiple passages
    multi_passage_articles = {k: v for k, v in article_counts.items() if v > 1}
    logger.info(f"\nArticles with multiple passages ({len(multi_passage_articles)}):")
    for art_num, count in sorted(multi_passage_articles.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)[:10]:
        logger.info(f"  Article {art_num}: {count} passages")

    # Check specific important articles
    logger.info("\nVerifying key articles:")
    key_articles = ['5', '9', '10', '14', '27']
    for art_num in key_articles:
        count = article_counts.get(art_num, 0)
        status = "✓" if count > 0 else "✗"
        logger.info(f"  {status} Article {art_num}: {count} passage(s)")

    return True


def validate_segmentation(articles: list, passages: list) -> dict:
    """
    Validate segmentation quality.

    Args:
        articles: List of articles
        passages: List of passages

    Returns:
        Dictionary with validation results
    """
    issues = []

    # Check 1: Ensure Article 5 doesn't have too many passages
    article_5_count = sum(1 for p in passages if p['metadata']['article_number'] == '5')
    if article_5_count > 5:
        issues.append(f"Article 5 has {article_5_count} passages (expected 1-2) - may include other articles")

    # Check 2: Ensure Article 27 exists
    article_27_count = sum(1 for p in passages if p['metadata']['article_number'] == '27')
    if article_27_count == 0:
        issues.append("Article 27 (Directive Principles) not found")
    elif article_27_count < 3:
        issues.append(f"Article 27 has only {article_27_count} passages (expected 5-10) - may be incomplete")

    # Check 3: Ensure Article 9 exists
    article_9_count = sum(1 for p in passages if p['metadata']['article_number'] == '9')
    if article_9_count == 0:
        issues.append("Article 9 (Buddhism) not found")

    # Check 4: Check for reasonable article count
    unique_articles = len(set(p['metadata']['article_number'] for p in passages))
    if unique_articles < 100:
        issues.append(f"Only {unique_articles} unique articles detected (Constitution has ~170+)")

    passed = len(issues) == 0

    return {
        'passed': passed,
        'issues': issues,
        'article_5_passages': article_5_count,
        'article_27_passages': article_27_count,
        'article_9_passages': article_9_count,
        'unique_articles': unique_articles
    }


if __name__ == '__main__':
    success = rebuild_constitution_corpus()
    sys.exit(0 if success else 1)
