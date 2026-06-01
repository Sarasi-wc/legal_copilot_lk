"""
Article-aware document segmenter that respects article boundaries.
Improved version that handles Constitution articles properly.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Article:
    """Represents a legal article."""
    number: str  # e.g., "9", "10", "12A"
    title: str
    text: str
    chapter: Optional[str] = None
    metadata: Dict = None


class ArticleAwareSegmenter:
    """
    Segments legal documents respecting article boundaries.
    Designed for Constitution where each article should be its own passage.
    """

    def __init__(self):
        """Initialize with patterns for article detection."""

        # Pattern for articles: matches "9. Text..." even if not at line start
        # This handles cases like "BUDDHISM 9. The Republic..."
        self.article_pattern = re.compile(
            r'(?:^|\s)(\d+[A-Z]?)\.\s+([A-Z][^.]+(?:\.|$).+?)(?=(?:^|\s)\d+[A-Z]?\.\s+[A-Z]|\Z)',
            re.MULTILINE | re.DOTALL
        )

        # Chapter pattern
        self.chapter_pattern = re.compile(
            r'CHAPTER\s+([IVXLCDM]+|[A-Z]|\d+)\s*[:-]?\s*(.+?)(?=\d+\.|$)',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )

        # OCR artifact patterns
        self.ocr_patterns = {
            # Page footers: "Sri Lanka 9 law" or "Sri Lanka 5 9" or "Sri Lanka 9 The"
            'page_footer': re.compile(
                r'(The Constitution.*?Sri Lanka)\s+\d+(?:\s+\d+)?(?:\s+(law|section|The|Constitution))?',
                re.IGNORECASE
            ),
            # Embedded page numbers: "9 The Constitution" or "9 law"
            'page_number': re.compile(
                r'\s(\d{1,3})\s+(The Constitution|law|section|Constitution)',
                re.IGNORECASE
            ),
            # Headers and page markers
            'header': re.compile(r'(Cap\.|Vol\.|Page)\s+\d+', re.IGNORECASE),
            # Standalone "The Constitution...Sri Lanka" followed by numbers
            'constitution_page': re.compile(
                r'The Constitution of the Democratic Socialist Republic of Sri Lanka\s+\d+(?:\s+\d+)?',
                re.IGNORECASE
            ),
        }

    def clean_text(self, text: str) -> str:
        """
        Remove OCR artifacts from text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        cleaned = text

        # Remove standalone "Constitution...Sri Lanka [page numbers]"
        cleaned = self.ocr_patterns['constitution_page'].sub('', cleaned)

        # Remove page footers like "The Constitution of... Sri Lanka 9 law"
        cleaned = self.ocr_patterns['page_footer'].sub(r'\1', cleaned)

        # Remove embedded page numbers
        cleaned = self.ocr_patterns['page_number'].sub(r' \2', cleaned)

        # Remove headers
        cleaned = self.ocr_patterns['header'].sub('', cleaned)

        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)

        return cleaned.strip()

    def extract_articles(
        self,
        text: str,
        act_name: str,
        clean_artifacts: bool = True
    ) -> List[Article]:
        """
        Extract articles from document text.

        Args:
            text: Document text
            act_name: Name of the act
            clean_artifacts: Whether to clean OCR artifacts

        Returns:
            List of Article objects
        """
        logger.info(f"Extracting articles from {act_name}")

        # Clean text first if requested
        if clean_artifacts:
            text = self.clean_text(text)

        articles = []
        current_chapter = None

        # Extract chapters first
        chapters = {}
        for match in self.chapter_pattern.finditer(text):
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            chapters[match.start()] = {
                'number': chapter_num,
                'title': chapter_title
            }

        # Extract articles
        matches = list(self.article_pattern.finditer(text))

        logger.info(f"Found {len(matches)} article matches")

        for match in matches:
            article_num = match.group(1)
            article_content = match.group(2).strip()

            # Check if this article falls under a chapter
            article_pos = match.start()
            for chapter_pos, chapter_info in sorted(chapters.items(), reverse=True):
                if chapter_pos < article_pos:
                    current_chapter = f"Chapter {chapter_info['number']}: {chapter_info['title']}"
                    break

            # Extract title (first line or first sentence)
            lines = article_content.split('\n', 1)
            if len(lines) > 1:
                title = lines[0].strip()
                body = lines[1].strip()
            else:
                # If single line, try to extract title from first sentence
                sentences = re.split(r'(?<=[.!?])\s+', article_content, maxsplit=1)
                title = sentences[0] if len(sentences) > 0 else article_content[:100]
                body = sentences[1] if len(sentences) > 1 else article_content

            article = Article(
                number=article_num,
                title=title,
                text=body if body else article_content,
                chapter=current_chapter,
                metadata={
                    'article_number': article_num,
                    'chapter': current_chapter,
                    'act_name': act_name
                }
            )
            articles.append(article)

        logger.info(f"Extracted {len(articles)} articles")
        return articles

    def create_passages_from_articles(
        self,
        articles: List[Article],
        max_length: int = 1000
    ) -> List[Dict]:
        """
        Create retrieval passages from articles.

        Each article becomes one passage (unless too long, then chunked).

        Args:
            articles: List of Article objects
            max_length: Maximum passage length

        Returns:
            List of passage dictionaries
        """
        passages = []

        for i, article in enumerate(articles):
            # Combine title and text for full article content
            full_text = f"{article.title}\n{article.text}".strip()

            # Add chapter context if available
            if article.chapter:
                full_text = f"{article.chapter}\n{article.number}. {full_text}"
            else:
                full_text = f"{article.number}. {full_text}"

            # If article is short enough, create single passage
            if len(full_text) <= max_length:
                passage = {
                    'passage_id': f'PASSAGE_{i:04d}',
                    'text': full_text,
                    'title': f"Article {article.number}: {article.title}",
                    'level': 'article',
                    'metadata': {
                        'article_number': article.number,
                        'chapter': article.chapter,
                        **article.metadata
                    }
                }
                passages.append(passage)
            else:
                # Split long articles at sentence boundaries
                chunks = self._chunk_text(full_text, max_length)
                for chunk_idx, chunk in enumerate(chunks):
                    passage = {
                        'passage_id': f'PASSAGE_{i:04d}_CHUNK_{chunk_idx}',
                        'text': chunk,
                        'title': f"Article {article.number}: {article.title} (Part {chunk_idx + 1})",
                        'level': 'article_chunk',
                        'metadata': {
                            'article_number': article.number,
                            'chapter': article.chapter,
                            'chunk_index': chunk_idx,
                            **article.metadata
                        }
                    }
                    passages.append(passage)

        logger.info(f"Created {len(passages)} passages from {len(articles)} articles")
        return passages

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """
        Chunk long text at sentence boundaries.

        Args:
            text: Text to chunk
            max_length: Maximum chunk length

        Returns:
            List of text chunks
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

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
                current_length += sentence_length + 1  # +1 for space

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_constitution(
        self,
        text: str,
        act_name: str = "Constitution of the Democratic Socialist Republic of Sri Lanka"
    ) -> List[Dict]:
        """
        Process Constitution document with article-aware segmentation.

        Args:
            text: Full Constitution text
            act_name: Name of the Constitution

        Returns:
            List of passage dictionaries
        """
        logger.info("Processing Constitution with article-aware segmentation")

        # Extract articles
        articles = self.extract_articles(text, act_name, clean_artifacts=True)

        if not articles:
            logger.warning("No articles detected, falling back to simple chunking")
            # Fallback to simple chunking if no articles found
            chunks = self._chunk_text(text, max_length=1000)
            passages = []
            for i, chunk in enumerate(chunks):
                passage = {
                    'passage_id': f'PASSAGE_{i:04d}',
                    'text': chunk,
                    'title': f'Passage {i + 1}',
                    'level': 'chunk',
                    'metadata': {}
                }
                passages.append(passage)
            return passages

        # Create passages from articles
        passages = self.create_passages_from_articles(articles, max_length=1000)

        return passages
