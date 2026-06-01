#!/usr/bin/env python3
"""
Fix Constitution corpus segmentation by properly detecting articles.
The original segmenter failed to detect article boundaries correctly.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


def extract_constitution_articles(text: str) -> List[Dict]:
    """
    Extract articles from Constitution text with proper boundary detection.

    Args:
        text: Full Constitution text

    Returns:
        List of article dictionaries with proper metadata
    """
    articles = []

    # Skip the table of contents - find where actual content starts
    # Look for "Chapter I:" followed by article text
    content_start_pattern = re.compile(r'CHAPTER\s+I\s*\n', re.IGNORECASE)
    content_match = content_start_pattern.search(text)

    if content_match:
        # Start from the first chapter in actual content
        text = text[content_match.start():]
        logger.info(f"Starting extraction from position {content_match.start()}")

    # Pattern to match article numbers in Constitution
    # Matches: "9. The Republic..." or "27. (1) The Directive..."
    # This pattern looks for: number + period + space + text
    article_pattern = re.compile(
        r'\n(\d+[A-Z]?)\.\s+([^\n]+)',
        re.MULTILINE
    )

    # Pattern to match chapter headers
    # Format: "CHAPTER II" or "CHAPTER III" with optional title on same/next line
    chapter_pattern = re.compile(
        r'\nCHAPTER\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([A-Z\s]+)?',
        re.IGNORECASE
    )

    # Find all chapter positions first
    chapters = []
    for match in chapter_pattern.finditer(text):
        chapters.append({
            'number': match.group(1),
            'title': match.group(2).strip(),
            'start': match.start(),
            'end': match.end()
        })

    # Find all article matches
    matches = list(article_pattern.finditer(text))

    current_chapter = None
    current_chapter_idx = 0

    for i, match in enumerate(matches):
        article_num = match.group(1)
        article_title_line = match.group(2).strip()
        article_start = match.start()

        # Find which chapter this article belongs to
        while (current_chapter_idx < len(chapters) and
               chapters[current_chapter_idx]['start'] < article_start):
            current_chapter = chapters[current_chapter_idx]
            current_chapter_idx += 1

        # Determine article end (next article or end of text)
        if i + 1 < len(matches):
            article_end = matches[i + 1].start()
        else:
            article_end = len(text)

        # Extract full article text
        article_text = text[match.end():article_end].strip()

        # Create article entry
        article = {
            'article_number': article_num,
            'title': f"Article {article_num}: {article_title_line}",
            'chapter': current_chapter['title'] if current_chapter else "Unknown",
            'text': article_text,
            'full_text': text[article_start:article_end].strip()
        }

        articles.append(article)

    logger.info(f"Extracted {len(articles)} articles from Constitution")
    return articles


def create_constitution_passages(articles: List[Dict], max_length: int = 500) -> List[Dict]:
    """
    Create passages from articles, chunking long ones if needed.

    Args:
        articles: List of article dictionaries
        max_length: Maximum passage length

    Returns:
        List of passage dictionaries
    """
    passages = []
    passage_id = 0

    for article in articles:
        article_num = article['article_number']
        article_text = article['text']

        # If article is short enough, create single passage
        if len(article_text) <= max_length:
            passage = {
                'passage_id': f'PASSAGE_{passage_id:04d}',
                'text': article['full_text'],
                'title': article['title'],
                'level': 'article',
                'metadata': {
                    'article_number': article_num,
                    'chapter': article['chapter'],
                    'act_name': 'Constitution of the Democratic Socialist Republic of Sri Lanka'
                }
            }
            passages.append(passage)
            passage_id += 1
        else:
            # Chunk long articles
            chunks = chunk_article(article_text, max_length)
            for i, chunk in enumerate(chunks, 1):
                passage = {
                    'passage_id': f'PASSAGE_{passage_id:04d}',
                    'text': chunk,
                    'title': f"{article['title']} (Part {i})",
                    'level': 'article',
                    'metadata': {
                        'article_number': article_num,
                        'chapter': article['chapter'],
                        'act_name': 'Constitution of the Democratic Socialist Republic of Sri Lanka',
                        'chunk': i,
                        'total_chunks': len(chunks)
                    }
                }
                passages.append(passage)
                passage_id += 1

    logger.info(f"Created {len(passages)} passages from {len(articles)} articles")
    return passages


def chunk_article(text: str, max_length: int) -> List[str]:
    """Chunk article text preserving sub-clause boundaries."""
    # Try to split by sub-clauses first (1), (2), etc.
    subclause_pattern = re.compile(r'(?=\(\d+[a-z]?\))')
    parts = subclause_pattern.split(text)

    if len(parts) <= 1:
        # No sub-clauses, split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return chunk_by_sentences(sentences, max_length)

    # Chunk by sub-clauses
    chunks = []
    current_chunk = []
    current_length = 0

    for part in parts:
        part_length = len(part)

        if current_length + part_length > max_length and current_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = [part]
            current_length = part_length
        else:
            current_chunk.append(part)
            current_length += part_length

    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks


def chunk_by_sentences(sentences: List[str], max_length: int) -> List[str]:
    """Chunk by sentences."""
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def main():
    """Main function to fix Constitution corpus."""
    logger.info("Starting Constitution corpus fix")

    # Paths
    data_path = Path(__file__).parent.parent / 'data'
    raw_corpus_path = data_path / 'processed' / 'corpus.jsonl'
    output_path = data_path / 'processed' / 'corpus_fixed.jsonl'
    backup_path = data_path / 'processed' / 'corpus_backup.jsonl'

    # Load existing corpus
    logger.info(f"Loading corpus from {raw_corpus_path}")
    with open(raw_corpus_path, 'r', encoding='utf-8') as f:
        corpus_doc = json.loads(f.readline())

    # Get the raw Constitution text from OCR metadata or reconstruct from passages
    # For now, read from PDF directly
    pdf_path = data_path / 'raw' / 'acts' / 'constitution_1978.pdf'

    if not pdf_path.exists():
        logger.error(f"Constitution PDF not found at {pdf_path}")
        return

    # Extract text from PDF
    from src.corpus_construction.ocr_processor import OCRProcessor
    ocr = OCRProcessor()
    text, ocr_metadata = ocr.extract_text_from_pdf(pdf_path, use_ocr=False)

    # Extract articles properly
    articles = extract_constitution_articles(text)

    logger.info(f"Found {len(articles)} distinct articles")
    logger.info(f"Sample articles: {[a['article_number'] for a in articles[:10]]}")

    # Create passages
    passages = create_constitution_passages(articles, max_length=1000)

    # Verify no duplicate article numbers in single passages
    article_counts = {}
    for passage in passages:
        art_num = passage['metadata']['article_number']
        article_counts[art_num] = article_counts.get(art_num, 0) + 1

    logger.info(f"Article distribution:")
    for art_num in sorted(article_counts.keys(), key=lambda x: int(re.match(r'\d+', x).group())):
        logger.info(f"  Article {art_num}: {article_counts[art_num]} passages")

    # Create new corpus document
    fixed_corpus = {
        'act_info': corpus_doc['act_info'],
        'act_metadata': corpus_doc['act_metadata'],
        'ocr_metadata': ocr_metadata,
        'quality_score': corpus_doc['quality_score'],
        'num_sections': len(articles),
        'num_passages': len(passages),
        'passages': passages
    }

    # Backup original
    logger.info(f"Backing up original corpus to {backup_path}")
    import shutil
    shutil.copy(raw_corpus_path, backup_path)

    # Write fixed corpus
    logger.info(f"Writing fixed corpus to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(fixed_corpus, ensure_ascii=False) + '\n')

    logger.info("Constitution corpus fix complete!")
    logger.info(f"  Original: {corpus_doc['num_passages']} passages")
    logger.info(f"  Fixed: {len(passages)} passages")
    logger.info(f"  Detected articles: {len(articles)}")


if __name__ == '__main__':
    main()
