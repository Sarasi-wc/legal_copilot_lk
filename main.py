"""
Main entry point for the Legal Copilot system.
Provides command-line interface for building indices and answering questions.
"""

import argparse
from pathlib import Path
import sys

from config.settings import settings
from src.corpus_construction.corpus_builder import CorpusBuilder
from src.generation.rag_pipeline import RAGPipeline
from src.generation.answer_generator import AnswerGenerator
from src.generation.openai_generator import OpenAIGenerator
from src.utils import get_logger, save_json

logger = get_logger(__name__)


def build_corpus(args):
    """Build corpus from PDF documents."""
    logger.info("Building corpus...")

    # Load manifest
    import json
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    # Build corpus
    builder = CorpusBuilder(
        data_path=settings.data_path,
        use_ocr=args.use_ocr
    )

    builder.build_corpus(
        act_manifest=manifest['acts'],
        output_file='corpus.jsonl'
    )

    logger.info("Corpus building complete!")


def build_indices(args):
    """Build retrieval indices from corpus."""
    logger.info("Building retrieval indices...")

    # Load corpus
    builder = CorpusBuilder(data_path=settings.data_path)
    passages = builder.extract_all_passages('corpus.jsonl')

    # Build indices
    pipeline = RAGPipeline(
        top_k=settings.top_k_rerank,
        retrieval_method='hybrid_rerank'
    )

    pipeline.build_indices(passages)

    # Save indices
    pipeline.retriever.save_indices(settings.index_path)

    logger.info(f"Indices built and saved to {settings.index_path}")


def _get_generator():
    """Use OpenAI when USE_OPENAI_GENERATOR is set and API key present; else local Mistral."""
    if getattr(settings, "use_openai_generator", False) and settings.openai_api_key:
        return OpenAIGenerator(
            model_name=settings.openai_model,
            max_tokens=settings.openai_max_tokens,
            temperature=settings.openai_temperature,
        )
    return AnswerGenerator(max_tokens=settings.max_tokens, temperature=settings.temperature)


def answer_question(args):
    """Answer a single question."""
    logger.info("Initializing system...")

    pipeline = RAGPipeline(
        generator=_get_generator(),
        top_k=settings.top_k_rerank,
        retrieval_method=args.method
    )

    # Load indices
    pipeline.load_indices(settings.index_path)

    # Answer question (pipeline runs retrieval, generation, and verification internally)
    logger.info(f"Question: {args.question}")
    result = pipeline.answer_question(args.question)

    # Print result
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(result['answer'])
    print("\n" + "="*80)
    print("METADATA:")
    print("="*80)
    print(f"Abstained: {result['abstained']}")
    print(f"Citations: {len(result['citations'])}")
    print(f"Evidence passages used: {result['evidence_used']}")

    verification_report = result.get('verification_report')
    if verification_report:
        print(f"Verification passed: {verification_report.get('verification_passed', 'N/A')}")

    if args.output:
        save_json(result, Path(args.output))
        logger.info(f"Result saved to {args.output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Legal Copilot for Sri Lankan Law'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Build corpus command
    corpus_parser = subparsers.add_parser('build-corpus', help='Build corpus from PDFs')
    corpus_parser.add_argument('--manifest', required=True, help='Path to manifest JSON')
    corpus_parser.add_argument('--use-ocr', action='store_true', help='Force OCR')

    # Build indices command
    index_parser = subparsers.add_parser('build-indices', help='Build retrieval indices')

    # Answer question command
    answer_parser = subparsers.add_parser('answer', help='Answer a question')
    answer_parser.add_argument('--question', required=True, help='Question to answer')
    answer_parser.add_argument('--method', default='hybrid_rerank',
                               choices=['bm25', 'dense', 'hybrid', 'hybrid_rerank'],
                               help='Retrieval method')
    answer_parser.add_argument('--output', help='Output file for result JSON')

    args = parser.parse_args()

    if args.command == 'build-corpus':
        build_corpus(args)
    elif args.command == 'build-indices':
        build_indices(args)
    elif args.command == 'answer':
        answer_question(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
