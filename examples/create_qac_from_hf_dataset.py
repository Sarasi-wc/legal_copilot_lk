"""
Complete Q-A-C Dataset Creation from Hugging Face Dataset

This script:
1. Loads the Sri Lanka Constitutional Law QA dataset from Hugging Face
2. Uses your retrieval system to find relevant passages and citations
3. Converts to your Q-A-C schema format
4. Classifies query types and domains
5. Creates a complete evaluation dataset

Usage:
    python examples/create_qac_from_hf_dataset.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    from src.evaluation.dataset_schema import (
        QACDataset, QACItem, GoldAnswer, Citation
    )
    from src.generation.rag_pipeline import RAGPipeline
    from src.query_processing.query_classifier import QueryClassifier
    from src.query_processing.domain_identifier import DomainIdentifier
    from config.settings import settings
    from src.utils import get_logger
    
    logger = get_logger(__name__)
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("\nPlease install required packages:")
    print("  pip install datasets")
    sys.exit(1)


class QACDatasetCreator:
    """Create Q-A-C dataset from Hugging Face dataset."""
    
    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize dataset creator.
        
        Args:
            index_path: Path to retrieval indices (if None, uses settings)
        """
        self.index_path = index_path or settings.index_path
        
        # Initialize components
        logger.info("Initializing Q-A-C dataset creator...")
        
        # Initialize retrieval pipeline
        self.pipeline = RAGPipeline(
            top_k=10,  # Retrieve more passages for citation finding
            retrieval_method='hybrid_rerank'
        )
        
        # Load indices if they exist
        if self.index_path.exists():
            logger.info(f"Loading indices from {self.index_path}")
            try:
                self.pipeline.load_indices(self.index_path)
                self.indices_loaded = True
            except Exception as e:
                logger.warning(f"Could not load indices: {e}")
                logger.warning("Will create dataset without citations (manual annotation needed)")
                self.indices_loaded = False
        else:
            logger.warning(f"Indices not found at {self.index_path}")
            logger.warning("Run 'python main.py build-indices' first to enable citation finding")
            self.indices_loaded = False
        
        # Initialize classifiers
        self.query_classifier = QueryClassifier()
        self.domain_identifier = DomainIdentifier()
        
        logger.info("Q-A-C dataset creator initialized")
    
    def extract_citations_from_passages(
        self,
        retrieved_passages: List[Dict],
        top_k: int = 5
    ) -> List[Citation]:
        """
        Extract citations from retrieved passages.
        
        Args:
            retrieved_passages: List of retrieved passage dictionaries
            top_k: Number of top passages to use for citations
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        # Use top-k passages
        top_passages = retrieved_passages[:top_k]
        
        for passage in top_passages:
            # Extract metadata
            metadata = passage.get('metadata', {})
            act_name = metadata.get('act_name', passage.get('act_name', 'Unknown Act'))
            section_number = metadata.get('section_number', passage.get('section', ''))
            passage_id = passage.get('passage_id', '')
            passage_text = passage.get('text', passage.get('content', ''))
            
            # Try to extract act information
            act_number = metadata.get('act_number', '')
            act_year = metadata.get('act_year', 0)
            
            # Create citation
            citation = Citation(
                act_name=act_name,
                act_number=act_number,
                act_year=act_year if act_year else 0,
                section_number=section_number,
                passage_id=passage_id,
                passage_text=passage_text[:500]  # Limit text length
            )
            
            citations.append(citation)
        
        return citations
    
    def classify_item(
        self,
        question: str,
        answer: str
    ) -> Dict:
        """
        Classify query type and domain.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Dictionary with query_type, legal_domain, and difficulty
        """
        # Normalize query
        normalized_question = question.lower().strip()
        
        # Classify query type (returns tuple: query_type, confidence)
        try:
            query_type, confidence = self.query_classifier.classify(question, normalized_question)
        except Exception as e:
            logger.warning(f"Classification error: {e}, using default")
            query_type = 'factual'
        
        # Identify domain (returns dict)
        try:
            domain_result = self.domain_identifier.identify(question, normalized_question)
            legal_domain = domain_result.get('domain', 'constitutional')
        except Exception as e:
            logger.warning(f"Domain identification error: {e}, using default")
            legal_domain = 'constitutional'
        
        # Determine difficulty (simple heuristic)
        difficulty = 'medium'
        if len(question.split()) < 10:
            difficulty = 'easy'
        elif len(question.split()) > 25 or 'interpret' in question.lower():
            difficulty = 'hard'
        
        return {
            'query_type': query_type,
            'legal_domain': legal_domain,
            'difficulty': difficulty
        }
    
    def find_relevant_passages(
        self,
        question: str
    ) -> List[Dict]:
        """
        Find relevant passages for a question using retrieval system.
        
        Args:
            question: Question text
            
        Returns:
            List of retrieved passage dictionaries
        """
        if not self.indices_loaded:
            return []
        
        try:
            # Retrieve passages
            retrieved = self.pipeline.retriever.retrieve(
                query=question,
                top_k=10,
                method='hybrid_rerank'
            )
            
            return retrieved
        except Exception as e:
            logger.warning(f"Error retrieving passages: {e}")
            return []
    
    def convert_hf_item_to_qac(
        self,
        hf_item: Dict,
        item_index: int,
        source_dataset: str = "Shifaur/sri_lanka_constitutional_law_qa"
    ) -> QACItem:
        """
        Convert Hugging Face dataset item to QACItem.
        
        Args:
            hf_item: Item from Hugging Face dataset
            item_index: Index of item in dataset
            source_dataset: Name of source dataset
            
        Returns:
            QACItem object
        """
        # Extract question and answer
        question = hf_item.get('question', hf_item.get('Question', ''))
        answer = hf_item.get('answer', hf_item.get('Answer', hf_item.get('answer_text', '')))
        
        if not question or not answer:
            logger.warning(f"Item {item_index} missing question or answer, skipping")
            return None
        
        # Classify
        classification = self.classify_item(question, answer)
        
        # Find relevant passages
        retrieved_passages = self.find_relevant_passages(question)
        
        # Extract citations
        citations = self.extract_citations_from_passages(retrieved_passages)
        
        # Create gold answer
        gold_answer = GoldAnswer(
            answer_text=answer,
            citations=citations,
            annotator_id='hf_dataset_creator',
            annotation_date=datetime.now().isoformat()
        )
        
        # Create QAC item
        item_id = f"constitutional_{item_index:04d}"
        
        qac_item = QACItem(
            item_id=item_id,
            question=question,
            query_type=classification['query_type'],
            legal_domain='constitutional',  # This dataset is constitutional law
            difficulty=classification['difficulty'],
            gold_answers=[gold_answer],
            relevant_passages=[p.get('passage_id', '') for p in retrieved_passages[:5]],
            metadata={
                'source_dataset': source_dataset,
                'source_item_id': hf_item.get('id', str(item_index)),
                'original_index': item_index,
                'citations_auto_generated': self.indices_loaded,
                'num_retrieved_passages': len(retrieved_passages),
                'created_date': datetime.now().isoformat()
            }
        )
        
        return qac_item
    
    def create_dataset(
        self,
        hf_dataset_name: str = "Shifaur/sri_lanka_constitutional_law_qa",
        split: str = "train",
        max_items: Optional[int] = None,
        output_path: Optional[Path] = None
    ) -> QACDataset:
        """
        Create Q-A-C dataset from Hugging Face dataset.
        
        Args:
            hf_dataset_name: Name of Hugging Face dataset
            split: Dataset split to use
            max_items: Maximum number of items to process (None for all)
            output_path: Path to save dataset (optional)
            
        Returns:
            QACDataset object
        """
        logger.info(f"Loading dataset: {hf_dataset_name}")
        
        # Load Hugging Face dataset
        try:
            hf_dataset = load_dataset(hf_dataset_name)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        
        # Get the specified split
        if split not in hf_dataset:
            available_splits = list(hf_dataset.keys())
            logger.warning(f"Split '{split}' not found. Available: {available_splits}")
            split = available_splits[0]
            logger.info(f"Using split: {split}")
        
        dataset_split = hf_dataset[split]
        total_items = len(dataset_split)
        
        logger.info(f"Processing {total_items} items from {split} split")
        
        if max_items:
            total_items = min(total_items, max_items)
            logger.info(f"Limiting to {max_items} items")
        
        # Create QAC dataset
        qac_dataset = QACDataset()
        
        # Process each item
        processed = 0
        skipped = 0
        
        for i, hf_item in enumerate(dataset_split):
            if max_items and i >= max_items:
                break
            
            try:
                qac_item = self.convert_hf_item_to_qac(
                    hf_item,
                    item_index=i,
                    source_dataset=hf_dataset_name
                )
                
                if qac_item:
                    qac_dataset.add_item(qac_item)
                    processed += 1
                    
                    if (processed + skipped) % 10 == 0:
                        logger.info(f"Processed {processed + skipped}/{total_items} items...")
                else:
                    skipped += 1
                    
            except Exception as e:
                logger.warning(f"Error processing item {i}: {e}")
                skipped += 1
                continue
        
        logger.info(f"Dataset creation complete!")
        logger.info(f"  Processed: {processed}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Total: {len(qac_dataset.items)}")
        
        # Get statistics
        stats = qac_dataset.get_statistics()
        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Total items: {stats['total_items']}")
        logger.info(f"  By domain: {stats['by_domain']}")
        logger.info(f"  By query type: {stats['by_query_type']}")
        logger.info(f"  By difficulty: {stats['by_difficulty']}")
        logger.info(f"  Avg citations per item: {stats['avg_citations_per_item']:.2f}")
        
        # Save dataset
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            qac_dataset.save_to_json(output_path)
            logger.info(f"\nDataset saved to: {output_path}")
        else:
            # Default output path
            output_path = Path(__file__).parent.parent / "data" / "qac_dataset_constitutional.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            qac_dataset.save_to_json(output_path)
            logger.info(f"\nDataset saved to: {output_path}")
        
        return qac_dataset


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create Q-A-C dataset from Hugging Face dataset"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="Shifaur/sri_lanka_constitutional_law_qa",
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        '--split',
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help="Maximum number of items to process"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output file path (default: data/qac_dataset_constitutional.json)"
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default=None,
        help="Path to retrieval indices (default: from settings)"
    )
    
    args = parser.parse_args()
    
    # Create output path
    output_path = Path(args.output) if args.output else None
    
    # Create index path
    index_path = Path(args.index_path) if args.index_path else None
    
    # Create dataset creator
    creator = QACDatasetCreator(index_path=index_path)
    
    # Create dataset
    qac_dataset = creator.create_dataset(
        hf_dataset_name=args.dataset,
        split=args.split,
        max_items=args.max_items,
        output_path=output_path
    )
    
    print("\n" + "="*80)
    print("Q-A-C DATASET CREATION COMPLETE!")
    print("="*80)
    print(f"\nDataset contains {len(qac_dataset.items)} items")
    print(f"\nNext steps:")
    print("  1. Review the dataset: Check data/qac_dataset_constitutional.json")
    print("  2. Validate citations: Ensure citations are accurate")
    print("  3. Manual review: Have legal experts review answers and citations")
    print("  4. Expand dataset: Add more items from other domains")
    print("  5. Use for evaluation: Integrate into evaluation framework")
    print("\nFor evaluation, see:")
    print("  - examples/run_evaluation.py")
    print("  - src/evaluation/retrieval_metrics.py")
    print("  - src/evaluation/answer_quality_metrics.py")


if __name__ == "__main__":
    main()
