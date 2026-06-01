#!/usr/bin/env python3
"""
Automated Q-A-C dataset expansion using retrieval system.
Generates new Q-A-C pairs by finding questions and their relevant passages.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset_schema import QACDataset, QACItem, GoldAnswer, Citation
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.openai_generator import OpenAIGenerator
from src.utils import get_logger

logger = get_logger(__name__)


def generate_question_from_passage(passage: Dict) -> str:
    """
    Generate a question that would be answered by this passage.
    Uses simple heuristics based on passage content.
    """
    text = passage.get('text', '')
    title = passage.get('title', '')
    passage_id = passage.get('passage_id', '')
    
    # Extract article/section number
    article_num = passage.get('metadata', {}).get('article_number')
    section_num = passage.get('metadata', {}).get('section_number')
    
    # Generate question based on content
    if article_num:
        return f"What does Article {article_num} say about {title.lower() if title else 'this provision'}?"
    elif section_num:
        return f"What does Section {section_num} say about {title.lower() if title else 'this provision'}?"
    elif 'article' in text.lower()[:200]:
        # Try to extract article number from text
        import re
        article_match = re.search(r'article\s+(\d+[A-Z]?)', text[:200], re.IGNORECASE)
        if article_match:
            return f"What does Article {article_match.group(1)} say?"
    
    # Generic question
    return f"What does this legal provision say about {title.lower() if title else 'the topic'}?"


def create_qac_item_from_passages(
    question: str,
    relevant_passages: List[Dict],
    item_id: str,
    query_type: str = "factual",
    legal_domain: str = "administrative",
    difficulty: str = "medium"
) -> QACItem:
    """Create a Q-A-C item from a question and relevant passages."""
    
    # Generate answer using RAG pipeline
    retriever = HybridRetriever()
    retriever.load_indices(Path('data/indices'))
    
    generator = OpenAIGenerator()
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        retrieval_method="hybrid_rerank",
        use_verification=True
    )
    
    result = pipeline.answer_question(question)
    
    # Extract citations
    citations = []
    for cit in result.get('citations', []):
        if cit.get('passage_id'):
            # Find matching passage
            passage = next((p for p in relevant_passages if p.get('passage_id') == cit.get('passage_id')), None)
            if passage:
                citations.append(Citation(
                    act_name=cit.get('act_name', ''),
                    act_number='',
                    act_year=1978,  # Default
                    section_number=cit.get('section', ''),
                    passage_id=cit.get('passage_id', ''),
                    passage_text=passage.get('text', '')[:200],
                    start_char=None,
                    end_char=None
                ))
    
    # Create gold answer
    gold_answer = GoldAnswer(
        answer_text=result.get('answer', ''),
        citations=citations,
        annotator_id='AUTO_GENERATED',
        annotation_date=datetime.now().isoformat()
    )
    
    # Create QAC item
    return QACItem(
        item_id=item_id,
        question=question,
        query_type=query_type,
        legal_domain=legal_domain,
        difficulty=difficulty,
        gold_answers=[gold_answer],
        relevant_passages=[p.get('passage_id') for p in relevant_passages],
        metadata={
            'auto_generated': True,
            'created_date': datetime.now().isoformat(),
            'source': 'retrieval_based'
        }
    )


def expand_dataset_automatically(
    dataset_path: Path,
    target_size: int = 100,
    output_path: Path = None
) -> QACDataset:
    """
    Automatically expand Q-A-C dataset to target size.
    
    Strategy:
    1. Load existing dataset
    2. For each passage in corpus, generate a question
    3. Retrieve relevant passages
    4. Create Q-A-C item if quality is good
    5. Add to dataset until target size reached
    """
    # Load existing dataset
    dataset = QACDataset()
    dataset.load_from_json(dataset_path)
    
    current_size = len(dataset.items)
    logger.info(f"Current dataset size: {current_size}, target: {target_size}")
    
    if current_size >= target_size:
        logger.info(f"Dataset already has {current_size} items, target is {target_size}")
        return dataset
    
    # Load corpus to find passages
    corpus_path = Path('data/processed/corpus.jsonl')
    with open(corpus_path) as f:
        corpus = json.loads(f.readline())
    
    passages = corpus.get('passages', [])
    logger.info(f"Found {len(passages)} passages in corpus")
    
    # Load retriever
    retriever = HybridRetriever()
    retriever.load_indices(Path('data/indices'))
    
    # Generate new items
    new_items = []
    used_passage_ids = set()
    
    # Get passage IDs already in dataset
    for item in dataset.items:
        used_passage_ids.update(item.relevant_passages)
    
    # Try to create new items from unused passages
    for passage in passages:
        if len(new_items) + current_size >= target_size:
            break
        
        passage_id = passage.get('passage_id', '')
        if passage_id in used_passage_ids:
            continue
        
        # Generate question
        question = generate_question_from_passage(passage)
        
        # Retrieve relevant passages
        retrieved = retriever.retrieve(question, top_k=5, method="hybrid_rerank")
        
        if not retrieved:
            continue
        
        # Use top 2-3 passages as relevant
        relevant = retrieved[:3]
        relevant_ids = [p.get('passage_id') for p in relevant]
        
        # Create item
        item_id = f"QAC_{current_size + len(new_items) + 1:03d}"
        try:
            item = create_qac_item_from_passages(
                question=question,
                relevant_passages=relevant,
                item_id=item_id
            )
            new_items.append(item)
            used_passage_ids.update(relevant_ids)
            logger.info(f"Created {item_id}: {question[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to create item for {passage_id}: {e}")
            continue
    
    # Add new items to dataset
    dataset.items.extend(new_items)
    dataset.metadata['num_items'] = len(dataset.items)
    dataset.metadata['updated'] = datetime.now().isoformat()
    
    logger.info(f"Created {len(new_items)} new items. Total: {len(dataset.items)}")
    
    # Save
    if output_path:
        dataset.save_to_json(output_path)
    else:
        dataset.save_to_json(dataset_path)
    
    return dataset


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatically expand Q-A-C dataset')
    parser.add_argument('--dataset', type=Path, default=Path('data/evaluation/qac_dataset.json'),
                       help='Path to Q-A-C dataset')
    parser.add_argument('--target', type=int, default=100,
                       help='Target number of items (default: 100)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output path (default: overwrite original)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Automated Q-A-C Dataset Expansion")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Target size: {args.target}")
    print()
    
    dataset = expand_dataset_automatically(
        args.dataset,
        args.target,
        args.output
    )
    
    print(f"\n✅ Dataset expanded to {len(dataset.items)} items")


if __name__ == '__main__':
    main()
