"""
Example script for running system evaluation.
Demonstrates how to evaluate retrieval and generation quality.
"""

import sys
sys.path.append('..')

from pathlib import Path
from src.corpus_construction.corpus_builder import CorpusBuilder
from src.generation.rag_pipeline import RAGPipeline
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.dataset_schema import QACDataset
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_retrieval(pipeline: RAGPipeline, dataset: QACDataset):
    """Evaluate retrieval performance."""
    logger.info("Evaluating retrieval...")

    all_metrics = []

    for item in dataset.items:
        # Retrieve passages
        retrieved = pipeline.retriever.retrieve(
            query=item.question,
            top_k=20,
            method='hybrid_rerank'
        )

        retrieved_ids = [p['passage_id'] for p in retrieved]

        # Compute metrics
        metrics = RetrievalMetrics.evaluate_retrieval(
            retrieved_passages=retrieved_ids,
            gold_passages=item.relevant_passages,
            k_values=[5, 10, 20]
        )

        all_metrics.append(metrics)

        logger.info(f"Item {item.item_id}: Recall@5={metrics['recall@5']:.3f}")

    # Aggregate
    aggregated = RetrievalMetrics.aggregate_metrics(all_metrics)

    logger.info("\nAggregated Retrieval Metrics:")
    for metric, value in aggregated.items():
        logger.info(f"  {metric}: {value:.3f}")

    return aggregated


def evaluate_generation(pipeline: RAGPipeline, dataset: QACDataset):
    """Evaluate generation quality."""
    logger.info("Evaluating generation...")

    results = []

    for item in dataset.items:
        # Generate answer
        result = pipeline.answer_question(item.question)

        results.append({
            'item_id': item.item_id,
            'question': item.question,
            'generated_answer': result['answer'],
            'gold_answer': item.gold_answers[0].answer_text if item.gold_answers else None,
            'abstained': result['abstained'],
            'num_citations': len(result['citations'])
        })

        logger.info(f"Item {item.item_id}: Abstained={result['abstained']}")

    # Calculate statistics
    total = len(results)
    abstained = sum(1 for r in results if r['abstained'])

    logger.info(f"\nGeneration Statistics:")
    logger.info(f"  Total questions: {total}")
    logger.info(f"  Abstained: {abstained} ({abstained/total*100:.1f}%)")
    logger.info(f"  Answered: {total-abstained} ({(total-abstained)/total*100:.1f}%)")

    return results


def main():
    """Run evaluation."""
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = RAGPipeline(top_k=5, retrieval_method='hybrid_rerank')
    pipeline.load_indices(Path('../data/indices'))

    # Load evaluation dataset
    logger.info("Loading evaluation dataset...")
    dataset = QACDataset()
    dataset.load_from_json(Path('../data/qac_dataset.json'))

    logger.info(f"Loaded {len(dataset.items)} evaluation items")

    # Run evaluations
    retrieval_metrics = evaluate_retrieval(pipeline, dataset)
    generation_results = evaluate_generation(pipeline, dataset)

    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()
