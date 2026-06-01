"""
Constitution-specific document segmenter.
Properly handles the unique structure of constitutional documents.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ConstitutionArticle:
    """Represents a constitutional article."""
    article_number: str
    chapter: str
    title: str
    text: str
    full_text: str
    start_pos: int
    end_pos: int


class ConstitutionSegmenter:
    """
    Segments constitutional documents into articles.
    Designed specifically for Sri Lankan Constitution structure.
    """

    def __init__(self):
        """Initialize Constitution segmenter."""
        self.logger = get_logger(__name__)

    def segment_constitution(self, text: str) -> List[ConstitutionArticle]:
        """
        Segment Constitution text into articles.

        Args:
            text: Full Constitution text

        Returns:
            List of ConstitutionArticle objects
        """
        self.logger.info("Segmenting Constitution document")

        # Step 1: Find where actual content starts (skip TOC)
        content_start = self._find_content_start(text)
        if content_start > 0:
            text = text[content_start:]
            self.logger.info(f"Skipped {content_start} characters of front matter/TOC")

        # Step 2: Extract all chapters
        chapters = self._extract_chapters(text)
        self.logger.info(f"Found {len(chapters)} chapters")

        # Step 3: Extract all articles with their boundaries
        articles = self._extract_articles(text, chapters)
        self.logger.info(f"Extracted {len(articles)} articles")

        return articles

    def _find_content_start(self, text: str) -> int:
        """
        Find where actual Constitution content starts (skip TOC).

        Args:
            text: Full text

        Returns:
            Position where content starts
        """
        # Strategy: Find a unique marker that appears after TOC
        # Look for "The Constitution of the Democratic Socialist Republic of Sri Lanka"
        # followed by a page number, then actual content

        # Find all occurrences of the Constitution title
        title = "The Constitution of the Democratic Socialist Republic of Sri Lanka"
        title_positions = []
        start_search = 0
        while True:
            pos = text.find(title, start_search)
            if pos == -1:
                break
            title_positions.append(pos)
            start_search = pos + 1

        if len(title_positions) >= 2:
            # The actual content usually starts after the 2nd or 3rd occurrence
            # Look for "CHAPTER I" after this point
            search_start = title_positions[1]
            chapter_pattern = re.compile(r'CHAPTER\s+I\s+THE PEOPLE', re.IGNORECASE)
            match = chapter_pattern.search(text, search_start)
            if match:
                self.logger.info(f"Found content start at position {match.start()}")
                return match.start()

        # Fallback: Look for "1. Sri Lanka" which is the actual Article 1
        article_1_pattern = re.compile(r'1\.\s+Sri Lanka \(Ceylon\) is a Free')
        match = article_1_pattern.search(text)
        if match:
            # Back up to find the chapter heading
            chapter_start = text.rfind('CHAPTER', 0, match.start())
            if chapter_start != -1 and match.start() - chapter_start < 200:
                self.logger.info(f"Found content start at position {chapter_start}")
                return chapter_start

        self.logger.warning("Could not find clear content start, using position 0")
        return 0

    def _extract_chapters(self, text: str) -> List[Dict]:
        """
        Extract chapter information.

        Args:
            text: Constitution text

        Returns:
            List of chapter dictionaries
        """
        chapters = []

        # Pattern for chapter headers: "CHAPTER II BUDDHISM" or "CHAPTER III FUNDAMENTAL RIGHTS"
        # Format: CHAPTER <number> <title>
        pattern = re.compile(
            r'CHAPTER\s+([IVXLCDM]+[A-Z]?)\s+([A-Z][A-Z\s,\-\']+?)(?=\s+\d+\.)',
            re.IGNORECASE
        )

        for match in pattern.finditer(text):
            chapter_num = match.group(1).strip()
            chapter_title = match.group(2).strip()

            chapter = {
                'number': chapter_num,
                'title': chapter_title,
                'start': match.start(),
                'end': match.end()
            }
            chapters.append(chapter)

        return chapters

    def _extract_articles(
        self,
        text: str,
        chapters: List[Dict]
    ) -> List[ConstitutionArticle]:
        """
        Extract articles from Constitution text.

        Args:
            text: Constitution text
            chapters: List of chapter information

        Returns:
            List of ConstitutionArticle objects
        """
        articles = []

        # Pattern to match article numbers
        # Matches: "9. The Republic..." or "14A. Right to..." or "154G. Provincial..."
        # Can appear after chapter title or after previous article
        article_pattern = re.compile(
            r'(?:^|[A-Z\s]|\n)(\d+[A-Z]?)\.\s+(?=[A-Z])',
            re.MULTILINE
        )

        # Find all article matches
        matches = list(article_pattern.finditer(text))

        self.logger.info(f"Found {len(matches)} article number patterns")

        # Track current chapter
        current_chapter = None
        current_chapter_idx = 0

        for i, match in enumerate(matches):
            article_num = match.group(1)
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

            # Extract article text
            # Full text includes article number
            article_full = text[article_start:article_end].strip()

            # Article text without the number prefix
            article_text = text[match.end():article_end].strip()

            # Extract article title (first line or sentence)
            title = self._extract_article_title(article_text, article_num)

            # Get chapter info
            chapter_name = ""
            if current_chapter:
                chapter_num = current_chapter['number']
                chapter_title = current_chapter['title']
                chapter_name = f"Chapter {chapter_num}: {chapter_title}" if chapter_title else f"Chapter {chapter_num}"

            article = ConstitutionArticle(
                article_number=article_num,
                chapter=chapter_name,
                title=title,
                text=article_text,
                full_text=article_full,
                start_pos=article_start,
                end_pos=article_end
            )

            articles.append(article)

        return articles

    def _extract_article_title(self, text: str, article_num: str) -> str:
        """
        Extract title for an article.

        Args:
            text: Article text
            article_num: Article number

        Returns:
            Article title
        """
        # Get first line or first sentence (up to period or newline)
        lines = text.split('\n', 1)
        first_line = lines[0].strip()

        # If first line is very short, it might be a marginal note
        # Try to get more context
        if len(first_line) < 20 and len(lines) > 1:
            first_line = (first_line + ' ' + lines[1].split('\n')[0]).strip()

        # Truncate if too long
        if len(first_line) > 200:
            first_line = first_line[:200] + '...'

        return f"Article {article_num}: {first_line}"

    def create_passages(
        self,
        articles: List[ConstitutionArticle],
        max_length: int = 1000
    ) -> List[Dict]:
        """
        Create retrieval passages from articles.

        Args:
            articles: List of ConstitutionArticle objects
            max_length: Maximum passage length

        Returns:
            List of passage dictionaries
        """
        passages = []
        passage_id = 0

        for article in articles:
            # If article is short enough, create single passage
            if len(article.full_text) <= max_length:
                passage = {
                    'passage_id': f'PASSAGE_{passage_id:04d}',
                    'text': article.full_text,
                    'title': article.title,
                    'level': 'article',
                    'metadata': {
                        'article_number': article.article_number,
                        'chapter': article.chapter,
                        'act_name': 'Constitution of the Democratic Socialist Republic of Sri Lanka'
                    }
                }
                passages.append(passage)
                passage_id += 1
            else:
                # Chunk long articles by sub-clauses
                chunks = self._chunk_article_by_subclauses(article, max_length)

                for i, chunk in enumerate(chunks, 1):
                    passage = {
                        'passage_id': f'PASSAGE_{passage_id:04d}',
                        'text': chunk,
                        'title': f"{article.title} (Part {i}/{len(chunks)})",
                        'level': 'article',
                        'metadata': {
                            'article_number': article.article_number,
                            'chapter': article.chapter,
                            'act_name': 'Constitution of the Democratic Socialist Republic of Sri Lanka',
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }
                    }
                    passages.append(passage)
                    passage_id += 1

        self.logger.info(f"Created {len(passages)} passages from {len(articles)} articles")
        return passages

    def _chunk_article_by_subclauses(
        self,
        article: ConstitutionArticle,
        max_length: int
    ) -> List[str]:
        """
        Chunk article text by sub-clauses (1), (2), etc.

        Args:
            article: ConstitutionArticle object
            max_length: Maximum chunk length

        Returns:
            List of text chunks
        """
        text = article.full_text

        # Find all sub-clause boundaries: (1), (2), (a), etc.
        subclause_pattern = re.compile(r'\n\((\d+[a-z]?)\)')
        matches = list(subclause_pattern.finditer(text))

        if not matches:
            # No sub-clauses, split by sentences
            return self._chunk_by_sentences(text, max_length)

        # Split by sub-clauses
        chunks = []
        current_chunk = text[:matches[0].start()]

        for i, match in enumerate(matches):
            subclause_start = match.start()

            # Determine subclause end
            if i + 1 < len(matches):
                subclause_end = matches[i + 1].start()
            else:
                subclause_end = len(text)

            subclause_text = text[subclause_start:subclause_end]

            # Check if adding this subclause exceeds max length
            if len(current_chunk) + len(subclause_text) > max_length and current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = subclause_text
            else:
                current_chunk += subclause_text

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_by_sentences(self, text: str, max_length: int) -> List[str]:
        """
        Chunk text by sentences when no sub-clauses exist.

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
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
