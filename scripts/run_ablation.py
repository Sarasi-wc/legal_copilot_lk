"""
Ablation study runner (RO5).

Systematically isolates the contribution of each architectural component by
running the system with one component removed or swapped at a time, then
comparing retrieval and generation metrics against the full system.

Ablation conditions:
  A1  BM25 only              (vs full hybrid_rerank)
  A2  Dense only             (vs full hybrid_rerank)
  A3  Hybrid, no reranking   (vs full hybrid_rerank)
  A4  Full system, no verifier  (ablate verification layer)
  A5  General embedding model instead of LEGAL-BERT

Usage:
    python scripts/run_ablation.py \
        --dataset data/evaluation/qac_dataset.json \
        --index_dir data/indices \
        --output_dir results/ablation
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.openai_generator import OpenAIGenerator
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.dataset_schema import QACDataset
from src.utils import get_logger, save_json

logger = get_logger(__name__)

K_VALUES = [5, 10, 20]

ABLATION_CONDITIONS = {
    'A0_full_system': {
        'retrieval_method': 'hybrid_rerank',
        'use_verification': True,
        'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
        'description': 'Full system (hybrid+rerank, all-mpnet bi-encoder)'
    },
    'A1_bm25_only': {
        'retrieval_method': 'bm25',
        'use_verification': True,
        'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
        'description': 'BM25 only — ablates dense retrieval'
    },
    'A2_dense_only': {
        'retrieval_method': 'dense',
        'use_verification': True,
        'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
        'description': 'Dense only — ablates BM25 retrieval'
    },
    'A3_hybrid_no_rerank': {
        'retrieval_method': 'hybrid',
        'use_verification': True,
        'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
        'description': 'Hybrid without reranking — ablates cross-encoder'
    },
    'A4_no_verification': {
        'retrieval_method': 'hybrid_rerank',
        'use_verification': False,
        'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
        'description': 'Full retrieval, no verifier — ablates verification layer'
    },
    'A5_mini_embedding': {
        'retrieval_method': 'hybrid_rerank',
        'use_verification': True,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'description': 'Smaller bi-encoder (MiniLM) — ablates embedding capacity'
    }
}


def _article_id(passage_id: str) -> str:
    """Strip chunk/subsection suffixes for article-level matching — mirrors run_experiment.py normalize_passage_id().

    Examples:
        ACT__1978_SEC_17_CHUNK_2         → ACT__1978_SEC_17
        ACT__1978_SEC_41A_SUBSEC_1_CHUNK_0 → ACT__1978_SEC_41A
        ACT__1978_SEC_79_SUBSEC_3        → ACT__1978_SEC_79
    """
    import re
    return re.sub(r'_(CHUNK|SUBSEC)_.*$|_SUBSEC_\d+.*$', '', passage_id)


def run_ablation_condition(
    condition_name: str,
    condition: Dict,
    dataset: QACDataset,
    index_dir: Path,
    output_dir: Path,
    retrieval_only: bool = False,
    article_level: bool = True,
) -> Dict:
    """Run a single ablation condition and return aggregated metrics."""
    logger.info(f"\n--- Ablation: {condition_name} ---")
    logger.info(f"    {condition['description']}")
    if retrieval_only:
        logger.info(f"    [retrieval-only mode; article_level={article_level}]")

    retriever = HybridRetriever(
        embedding_model=condition['embedding_model']
    )
    retriever.load_indices(index_dir)

    pipeline = None
    if not retrieval_only:
        generator = OpenAIGenerator()
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            retrieval_method=condition['retrieval_method'],
            use_verification=condition['use_verification']
        )

    retrieval_metrics_list = []
    abstention_pred, abstention_gold = [], []
    attribution_list = []

    for item in dataset.items:
        gold_passages = set(item.relevant_passages)
        if article_level:
            gold_passages = {_article_id(p) for p in gold_passages}
        should_abstain = len(item.relevant_passages) == 0

        # Retrieval metrics
        try:
            retrieved = retriever.retrieve(
                query=item.question,
                top_k=max(K_VALUES),
                method=condition['retrieval_method']
            )
            retrieved_ids = [r['passage_id'] for r in retrieved]
            if article_level:
                # Deduplicate after normalisation to avoid inflated recall
                seen, deduped = set(), []
                for pid in retrieved_ids:
                    norm = _article_id(pid)
                    if norm not in seen:
                        seen.add(norm)
                        deduped.append(norm)
                retrieved_ids = deduped
            metrics = RetrievalMetrics.evaluate_retrieval(
                retrieved_passages=retrieved_ids,
                gold_passages=list(gold_passages),
                k_values=K_VALUES
            )
            retrieval_metrics_list.append(metrics)
        except Exception as e:
            logger.warning(f"[{condition_name}] Retrieval failed item={item.item_id}: {e}")

        if retrieval_only:
            continue

        # Generation + attribution + abstention
        try:
            answer = pipeline.answer_question(item.question)
            cited_ids = [
                c.get('passage_id') for c in answer.get('citations', [])
                if c.get('passage_id')
            ]
            if article_level:
                cited_ids = [_article_id(p) for p in cited_ids]
            attr = RetrievalMetrics.attribution_metrics(cited_ids, list(gold_passages))
            attribution_list.append(attr)
            abstention_pred.append(answer.get('abstained', False))
            abstention_gold.append(should_abstain)
        except Exception as e:
            logger.warning(f"[{condition_name}] Generation failed item={item.item_id}: {e}")

    agg_retrieval = RetrievalMetrics.aggregate_metrics(retrieval_metrics_list)
    agg_attribution = RetrievalMetrics.aggregate_attribution_metrics(attribution_list) if attribution_list else {}
    abstention = (
        RetrievalMetrics.abstention_metrics(abstention_pred, abstention_gold)
        if abstention_pred else {}
    )

    result = {
        'condition': condition_name,
        'description': condition['description'],
        'n_items': len(retrieval_metrics_list),
        'retrieval_only': retrieval_only,
        'article_level': article_level,
        'retrieval': {k: v for k, v in agg_retrieval.items()
                      if 'mrr' in k or 'recall@10' in k or 'ndcg@10' in k},
        'attribution': agg_attribution,
        'abstention': abstention
    }

    logger.info(f"    MRR={agg_retrieval.get('mrr_mean', 0):.4f}  "
                f"R@10={agg_retrieval.get('recall@10_mean', 0):.4f}  "
                f"NDCG@10={agg_retrieval.get('ndcg@10_mean', 0):.4f}")

    save_json(result, output_dir / f'{condition_name}.json')
    return result


def print_ablation_table(results: List[Dict]) -> None:
    """Print a comparison table of all ablation conditions."""
    header = (
        f"{'Condition':<30}  {'MRR':>6}  {'R@10':>6}  "
        f"{'NDCG@10':>8}  {'Attr-F1':>8}  {'Abs-F1':>7}"
    )
    print("\n=== Ablation Study Results ===")
    print(header)
    print("-" * len(header))

    for r in results:
        ret = r.get('retrieval', {})
        attr = r.get('attribution', {})
        abst = r.get('abstention', {})
        print(
            f"{r['condition']:<30}  "
            f"{ret.get('mrr_mean', 0):>6.4f}  "
            f"{ret.get('recall@10_mean', 0):>6.4f}  "
            f"{ret.get('ndcg@10_mean', 0):>8.4f}  "
            f"{attr.get('attribution_f1_mean', 0):>8.4f}  "
            f"{abst.get('abstention_f1', 0):>7.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--index_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('results/ablation'))
    parser.add_argument(
        '--conditions', nargs='+',
        choices=list(ABLATION_CONDITIONS.keys()),
        default=list(ABLATION_CONDITIONS.keys()),
        help='Subset of ablation conditions to run (default: all)'
    )
    parser.add_argument(
        '--retrieval-only', action='store_true',
        help='Skip generation (no OpenAI calls); compute retrieval metrics only'
    )
    parser.add_argument(
        '--no-article-level', action='store_true',
        help='Use exact passage-ID matching instead of article-level (strip _CHUNK_N)'
    )
    args = parser.parse_args()

    retrieval_only = args.retrieval_only
    article_level = not args.no_article_level

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_json({
        'conditions': args.conditions,
        'retrieval_only': retrieval_only,
        'article_level': article_level,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }, args.output_dir / 'ablation_config.json')

    # Load dataset
    dataset = QACDataset()
    dataset.load_from_json(args.dataset)
    logger.info(f"Loaded {len(dataset.items)} Q-A-C items")
    if retrieval_only:
        logger.info("Retrieval-only mode: skipping generation (no OpenAI API calls)")

    results = []
    for name in args.conditions:
        condition = ABLATION_CONDITIONS[name]
        result = run_ablation_condition(
            name, condition, dataset, args.index_dir, args.output_dir,
            retrieval_only=retrieval_only,
            article_level=article_level,
        )
        results.append(result)

    # Save combined results
    save_json(results, args.output_dir / 'ablation_summary.json')
    print_ablation_table(results)
    logger.info(f"\nAblation results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
