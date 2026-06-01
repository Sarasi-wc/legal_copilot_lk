#!/usr/bin/env python3
"""
Extract section numbers for CPC and Penal Code passages.
Improves metadata coverage from 13% to higher coverage.
"""

import json
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


def extract_section_number_from_text(text: str, title: str = "") -> Optional[str]:
    """
    Extract section number from passage text.
    
    Handles patterns like:
    - "Section 123"
    - "Sec. 123"
    - "123. Text starts with number"
    - "Section 123A"
    - "Section 123(1)"
    
    Args:
        text: Passage text
        title: Passage title
        
    Returns:
        Section number or None
    """
    # Pattern 1: Title contains "Section X"
    title_section = re.search(r'\bSection\s+(\d+[A-Z]?(?:\(\d+\)(?:\([a-z]\))?)?)', title, re.IGNORECASE)
    if title_section:
        return title_section.group(1)
    
    # Pattern 2: Text starts with "Section 123" or "Sec. 123"
    section_mention = re.search(r'\b(?:Section|Sec\.?)\s+(\d+[A-Z]?(?:\(\d+\)(?:\([a-z]\))?)?)', text[:300], re.IGNORECASE)
    if section_mention:
        return section_mention.group(1)
    
    # Pattern 3: Text starts with number and period: "123. The text..."
    # For CPC/Penal Code, sections are typically 3-4 digits (100-9999)
    # For Constitution, articles are typically 1-2 digits (1-200)
    number_start = re.match(r'^(\d{1,4}[A-Z]?)\.\s+[A-Z]', text)
    if number_start:
        section_num = number_start.group(1)
        num_match = re.match(r'\d+', section_num)
        if num_match:
            num_value = int(num_match.group())
            # If it's a high number (>200), likely a section, not article
            if num_value > 200:
                return section_num
            # For lower numbers, check context to distinguish section vs article
            text_lower = text[:300].lower()
            # If context mentions "section" or act names, it's likely a section
            if ('section' in text_lower or 
                'penal code' in text_lower or 
                'civil procedure' in text_lower or
                'cpc' in text_lower):
                return section_num
            # If it's Constitution context, it's an article (handled separately)
            if 'constitution' in text_lower and 'article' in text_lower:
                return None  # This is an article, not a section
    
    # Pattern 4: Text contains "Section 123" anywhere in first 500 chars
    section_anywhere = re.search(r'\bSection\s+(\d+[A-Z]?(?:\(\d+\)(?:\([a-z]\))?)?)', text[:500], re.IGNORECASE)
    if section_anywhere:
        return section_anywhere.group(1)
    
    return None


def identify_act_type(passage: Dict) -> Optional[str]:
    """
    Identify act type from passage content.
    
    Returns:
        'CPC', 'Penal Code', 'Constitution', or None
    """
    text = passage.get('text', '').lower()
    title = passage.get('title', '').lower()
    passage_id = passage.get('passage_id', '').lower()
    
    # Check passage ID first
    if 'cpc' in passage_id or 'civil_procedure' in passage_id:
        return 'Civil Procedure Code'
    if 'penal' in passage_id or 'penal_code' in passage_id:
        return 'Penal Code'
    if 'constitution' in passage_id or 'article' in passage_id:
        return 'Constitution'
    
    # Check text content
    if 'civil procedure code' in text[:500] or 'cpc' in text[:500]:
        return 'Civil Procedure Code'
    if 'penal code' in text[:500]:
        return 'Penal Code'
    if 'constitution' in text[:500] or 'article' in text[:500]:
        return 'Constitution'
    
    # Check title
    if 'civil procedure' in title or 'cpc' in title:
        return 'Civil Procedure Code'
    if 'penal code' in title:
        return 'Penal Code'
    
    return None


def extract_metadata_for_passage(passage: Dict) -> Dict:
    """
    Extract metadata for a passage.
    
    Returns:
        Updated metadata dictionary
    """
    metadata = passage.get('metadata', {}).copy()
    text = passage.get('text', '')
    title = passage.get('title', '')
    
    # Identify act type
    act_type = identify_act_type(passage)
    if act_type:
        # Update act_name if empty or unknown
        if not passage.get('act_name') or passage.get('act_name') == 'Unknown':
            passage['act_name'] = act_type
    
    # Extract section number for CPC and Penal Code
    if act_type in ['Civil Procedure Code', 'Penal Code']:
        section_num = extract_section_number_from_text(text, title)
        if section_num:
            metadata['section_number'] = section_num
            logger.debug(f"Extracted section {section_num} for {passage.get('passage_id')}")
    
    # Extract article number for Constitution (if not already present)
    if act_type == 'Constitution' and not metadata.get('article_number'):
        # Use existing article extraction logic
        from scripts.fix_corpus_metadata import extract_article_number_from_text
        article_num = extract_article_number_from_text(text, title)
        if article_num:
            metadata['article_number'] = article_num
    
    return metadata


def fix_non_constitution_metadata():
    """Fix metadata for non-Constitution passages."""
    logger.info("Starting non-Constitution metadata extraction")
    
    # Paths
    data_path = Path(__file__).parent.parent / 'data'
    corpus_path = data_path / 'processed' / 'corpus.jsonl'
    backup_path = data_path / 'processed' / f'corpus_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    
    # Load corpus
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_doc = json.loads(f.readline())
    
    passages = corpus_doc['passages']
    logger.info(f"Loaded {len(passages)} passages")
    
    # Create backup
    logger.info(f"Creating backup: {backup_path}")
    shutil.copy2(corpus_path, backup_path)
    
    # Process each passage
    fixed_sections = 0
    fixed_acts = 0
    fixed_articles = 0
    
    for passage in passages:
        original_metadata = passage.get('metadata', {}).copy()
        original_act = passage.get('act_name', '')
        
        # Extract metadata
        new_metadata = extract_metadata_for_passage(passage)
        
        # Update passage
        passage['metadata'] = new_metadata
        
        # Count fixes
        if new_metadata.get('section_number') and not original_metadata.get('section_number'):
            fixed_sections += 1
        if passage.get('act_name') and passage.get('act_name') != original_act and original_act in ['', 'Unknown']:
            fixed_acts += 1
        if new_metadata.get('article_number') and not original_metadata.get('article_number'):
            fixed_articles += 1
    
    # Save updated corpus
    logger.info(f"Saving updated corpus to {corpus_path}")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(corpus_doc, ensure_ascii=False) + '\n')
    
    # Statistics
    has_section = sum(1 for p in passages if p.get('metadata', {}).get('section_number'))
    has_article = sum(1 for p in passages if p.get('metadata', {}).get('article_number'))
    has_act = sum(1 for p in passages if p.get('act_name') and p.get('act_name') != 'Unknown')
    empty_metadata = sum(1 for p in passages if not p.get('metadata') or p.get('metadata') == {})
    
    logger.info("\n" + "=" * 70)
    logger.info("Metadata Extraction Results")
    logger.info("=" * 70)
    logger.info(f"Fixed section numbers: {fixed_sections}")
    logger.info(f"Fixed act names: {fixed_acts}")
    logger.info(f"Fixed article numbers: {fixed_articles}")
    logger.info("")
    logger.info(f"Final Statistics:")
    logger.info(f"  Has section_number: {has_section} ({has_section/len(passages)*100:.1f}%)")
    logger.info(f"  Has article_number: {has_article} ({has_article/len(passages)*100:.1f}%)")
    logger.info(f"  Has act_name: {has_act} ({has_act/len(passages)*100:.1f}%)")
    logger.info(f"  Empty metadata: {empty_metadata} ({empty_metadata/len(passages)*100:.1f}%)")
    logger.info("")
    logger.info("✅ Metadata extraction complete!")
    logger.info(f"📁 Backup saved to: {backup_path}")


if __name__ == '__main__':
    fix_non_constitution_metadata()
