"""
Utility functions for the Legal Copilot system.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_text(text: str) -> str:
    """Clean and normalize legal text."""
    # Remove excessive newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Normalize whitespace
    text = normalize_whitespace(text)

    # Remove common OCR artifacts
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII

    return text


def extract_section_number(text: str) -> Optional[str]:
    """Extract section number from text."""
    # Matches patterns like "Section 123", "Sec. 123A", "123(1)(a)"
    patterns = [
        r'Section\s+(\d+[A-Z]?(?:\(\d+\)(?:\([a-z]\))?)?)',
        r'Sec\.\s+(\d+[A-Z]?(?:\(\d+\)(?:\([a-z]\))?)?)',
        r'^(\d+[A-Z]?(?:\(\d+\)(?:\([a-z]\))?)?)\.'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def parse_act_reference(text: str) -> Optional[Dict[str, str]]:
    """Parse Act reference from text."""
    # Match patterns like "Penal Code (Act No. 2 of 1883)"
    pattern = r'([^(]+)\s*\(Act\s+No\.\s*(\d+)\s+of\s+(\d{4})\)'
    match = re.search(pattern, text)

    if match:
        return {
            'name': match.group(1).strip(),
            'number': match.group(2),
            'year': match.group(3)
        }

    return None


def save_json(data: Any, filepath: Path) -> None:
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def chunk_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunk text into smaller segments with overlap.
    Preserves sentence boundaries where possible.
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_length and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))

            # Start new chunk with overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def format_citation(act_name: str, section: str, year: Optional[int] = None) -> str:
    """Format a legal citation."""
    citation = f"{act_name}, Section {section}"
    if year:
        citation += f" ({year})"
    return citation


def validate_date(date_str: str) -> Optional[datetime]:
    """Validate and parse date string."""
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None
