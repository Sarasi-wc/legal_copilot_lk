"""
Baseline comparison experiment runner.

Answers RQ1 and RQ2 by running all retrieval baselines and the no-RAG
baseline against the Q-A-C evaluation dataset and computing all metrics
defined in RO5.

Usage:
    python scripts/run_experiment.py \
        --dataset data/evaluation/qac_dataset.json \
        --index_dir data/indices \
        --output_dir results/experiment_1

Baselines evaluated (RQ1):
    - bm25          BM25 lexical retrieval only
    - dense         Dense semantic retrieval only
    - hybrid        BM25 + dense fusion (no reranking)
    - hybrid_rerank BM25 + dense + cross-encoder reranking  (full system)

Baselines evaluated (RQ2):
    - no_rag        General-purpose LLM with no retrieval (parametric only)
    - hybrid_rerank Full RAG system (same as above, reused)

Metrics computed (RO5):
    Retrieval:  Recall@k (k=5,10,20), MRR, NDCG@k
    Generation: correctness*, completeness*, clarity* (* = manual rubric scores)
    Faithfulness: attribution precision, attribution recall, attribution F1
    Abstention: abstention precision, recall, F1
    Statistical: mean ± std; Bonferroni-corrected p-values for pairwise comparisons
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats as scipy_stats
import numpy as np

# wandb is optional — tracking is enabled only when --wandb_project is passed
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.openai_generator import OpenAIGenerator
from src.evaluation.baselines import NoRAGBaseline
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.dataset_schema import QACDataset
from src.utils import get_logger, save_json
from config.settings import settings

logger = get_logger(__name__)

RETRIEVAL_METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]
K_VALUES = [5, 10, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_qac_dataset(path: Path) -> QACDataset:
    dataset = QACDataset()
    dataset.load_from_json(path)
    logger.info(f"Loaded {len(dataset.items)} Q-A-C items")
    return dataset


def extract_cited_passage_ids(answer_result: Dict) -> List[str]:
    """Pull passage IDs from citations returned by the generator."""
    cited = []
    for cit in answer_result.get('citations', []):
        pid = cit.get('passage_id')
        if pid:
            cited.append(pid)
    return cited


def bonferroni_pairwise(metric_values: Dict[str, List[float]]) -> Dict:
    """
    Pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    Args:
        metric_values: {baseline_name: [per-query metric scores]}

    Returns:
        Dictionary of (baseline_a, baseline_b) -> corrected p-value
    """
    names = list(metric_values.keys())
    num_comparisons = len(names) * (len(names) - 1) // 2
    pairwise = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            vals_a = metric_values[a]
            vals_b = metric_values[b]
            if len(vals_a) == len(vals_b) and len(vals_a) > 1:
                _, p = scipy_stats.wilcoxon(vals_a, vals_b, zero_method='wilcox')
                corrected_p = min(p * num_comparisons, 1.0)  # Bonferroni
            else:
                corrected_p = None
            pairwise[f"{a}_vs_{b}"] = corrected_p

    return pairwise


# ---------------------------------------------------------------------------
# RQ1: Retrieval evaluation
# ---------------------------------------------------------------------------

import re as _re

def normalize_passage_id(pid: str) -> str:
    """
    Normalize a passage ID to article level by stripping chunk/subsection suffixes.

    Examples:
        ACT__1978_SEC_13_CHUNK_0          → ACT__1978_SEC_13
        ACT__1978_SEC_41A_SUBSEC_1_CHUNK_0 → ACT__1978_SEC_41A
        ACT__1978_SEC_79_SUBSEC_3          → ACT__1978_SEC_79
        ACT__1978_SEC_13                   → ACT__1978_SEC_13  (unchanged)
    """
    return _re.sub(r'_(CHUNK|SUBSEC)_.*$|_SUBSEC_\d+.*$', '', pid)


def normalize_retrieved_for_eval(retrieved_ids: list, gold_ids: set) -> list:
    """
    Normalize retrieved passage IDs to article level and deduplicate,
    preserving rank order of first occurrence of each normalized ID.

    This allows any chunk of a gold article to count as a retrieval hit,
    which is the correct evaluation semantics for a chunked corpus.
    """
    seen = set()
    normalized = []
    for pid in retrieved_ids:
        norm = normalize_passage_id(pid)
        if norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


def normalize_gold_ids(gold_ids: set) -> set:
    """Normalize gold IDs to article level (strip chunk/subsection suffixes)."""
    return {normalize_passage_id(gid) for gid in gold_ids}


def run_retrieval_evaluation(
    dataset: QACDataset,
    retriever: HybridRetriever,
    output_dir: Path,
    index_dir: Path = None
) -> Dict:
    """
    Evaluate all four retrieval baselines on the Q-A-C dataset (RQ1).

    Returns per-baseline aggregated retrieval metrics.
    """
    logger.info("=== RQ1: Retrieval Evaluation ===")
    results = {method: [] for method in RETRIEVAL_METHODS}

    # Build full corpus ID set for gold passage expansion
    import pickle
    all_corpus_ids = set()
    if index_dir:
        try:
            with open(index_dir / 'dense_metadata.pkl', 'rb') as f:
                meta = pickle.load(f)
            all_corpus_ids = set(p['passage_id'] for p in meta['passages'])
            logger.info(f"Loaded {len(all_corpus_ids)} corpus IDs for gold expansion")
        except Exception as e:
            logger.warning(f"Could not load corpus IDs for expansion: {e}")

    for item in dataset.items:
        gold_passages_raw = set(item.relevant_passages)
        if not gold_passages_raw:
            continue

        # Normalize gold IDs to article level (strip _CHUNK_N, _SUBSEC_N)
        gold_passages = normalize_gold_ids(gold_passages_raw)

        for method in RETRIEVAL_METHODS:
            try:
                retrieved = retriever.retrieve(
                    query=item.question,
                    top_k=max(K_VALUES),
                    method=method
                )
                raw_ids = [r['passage_id'] for r in retrieved]
                # Normalize retrieved IDs to article level and deduplicate
                retrieved_ids = normalize_retrieved_for_eval(raw_ids, gold_passages)

                metrics = RetrievalMetrics.evaluate_retrieval(
                    retrieved_passages=retrieved_ids,
                    gold_passages=list(gold_passages),
                    k_values=K_VALUES
                )
                metrics['item_id'] = item.item_id
                metrics['query_type'] = item.query_type
                metrics['legal_domain'] = item.legal_domain
                results[method].append(metrics)

            except Exception as e:
                logger.warning(f"Retrieval failed [{method}] item={item.item_id}: {e}")

    # Aggregate and compute statistical tests
    aggregated = {}
    mrr_per_method = {}

    for method, metric_list in results.items():
        if not metric_list:
            continue
        agg = RetrievalMetrics.aggregate_metrics(metric_list)
        aggregated[method] = agg
        mrr_per_method[method] = [m['mrr'] for m in metric_list]
        logger.info(f"[{method}] MRR={agg.get('mrr_mean', 0):.4f}  "
                    f"Recall@10={agg.get('recall@10_mean', 0):.4f}  "
                    f"NDCG@10={agg.get('ndcg@10_mean', 0):.4f}")

    # Bonferroni-corrected pairwise significance tests on MRR
    significance = bonferroni_pairwise(mrr_per_method)

    rq1_results = {
        'per_baseline': aggregated,
        'significance_tests_mrr': significance,
        'per_item_detail': {m: results[m] for m in results}
    }

    save_json(rq1_results, output_dir / 'rq1_retrieval_results.json')
    logger.info(f"RQ1 results saved to {output_dir / 'rq1_retrieval_results.json'}")
    return rq1_results


# ---------------------------------------------------------------------------
# RQ2: Generation evaluation (RAG vs no-RAG)
# ---------------------------------------------------------------------------

def run_generation_evaluation(
    dataset: QACDataset,
    rag_pipeline: RAGPipeline,
    no_rag_baseline: NoRAGBaseline,
    output_dir: Path
) -> Dict:
    """
    Evaluate RAG system vs no-RAG baseline on the Q-A-C dataset (RQ2).

    Computes attribution metrics and abstention F1 automatically.
    Correctness/completeness/clarity placeholders are left for manual expert scoring.
    """
    logger.info("=== RQ2: Generation Evaluation (RAG vs No-RAG) ===")

    rag_results = []
    no_rag_results = []
    abstention_rag_pred = []
    abstention_rag_gold = []
    attribution_rag = []
    faithfulness_scores = []

    for item in dataset.items:
        gold_passages = item.relevant_passages
        # Items with no relevant passages should trigger abstention
        should_abstain = len(gold_passages) == 0

        # --- Full RAG system ---
        try:
            rag_answer = rag_pipeline.answer_question(item.question)
            cited_ids = extract_cited_passage_ids(rag_answer)

            attr = RetrievalMetrics.attribution_metrics(cited_ids, gold_passages)
            attribution_rag.append(attr)

            abstention_rag_pred.append(rag_answer.get('abstained', False))
            abstention_rag_gold.append(should_abstain)

            # Extract faithfulness score from verification report (RO5)
            verification_report = rag_answer.get('verification_report') or {}
            faithfulness_info = verification_report.get('faithfulness') or {}
            faithfulness_score = faithfulness_info.get('faithfulness_score')
            if faithfulness_score is not None:
                faithfulness_scores.append(faithfulness_score)

            rag_results.append({
                'item_id': item.item_id,
                'question': item.question,
                'query_type': item.query_type,
                'legal_domain': item.legal_domain,
                'abstained': rag_answer.get('abstained', False),
                'abstention_reason': rag_answer.get('abstention_reason'),
                'num_retrieved': rag_answer.get('num_retrieved', 0),
                'attribution': attr,
                'faithfulness_score': faithfulness_score,
                'verification_passed': verification_report.get('verification_passed'),
                'answer': rag_answer['answer'],
                # Expert scoring fields — to be filled by annotators
                'expert_correctness': None,
                'expert_completeness': None,
                'expert_clarity': None
            })

        except Exception as e:
            logger.warning(f"RAG failed item={item.item_id}: {e}")

        # --- No-RAG baseline ---
        try:
            no_rag_answer = no_rag_baseline.answer_question(item.question)
            no_rag_results.append({
                'item_id': item.item_id,
                'question': item.question,
                'query_type': item.query_type,
                'legal_domain': item.legal_domain,
                'abstained': no_rag_answer.get('abstained', False),
                'answer': no_rag_answer['answer'],
                # Expert scoring fields — to be filled by annotators
                'expert_correctness': None,
                'expert_completeness': None,
                'expert_clarity': None
            })
        except Exception as e:
            logger.warning(f"No-RAG baseline failed item={item.item_id}: {e}")

    # Aggregate abstention F1 for RAG
    abstention = (
        RetrievalMetrics.abstention_metrics(abstention_rag_pred, abstention_rag_gold)
        if abstention_rag_pred else {}
    )

    # Aggregate attribution metrics
    attribution_agg = (
        RetrievalMetrics.aggregate_attribution_metrics(attribution_rag)
        if attribution_rag else {}
    )

    # Aggregate faithfulness score distribution (RO5 — hallucination detection)
    faithfulness_agg = {}
    if faithfulness_scores:
        arr = np.array(faithfulness_scores)
        faithfulness_agg = {
            'faithfulness_score_mean': round(float(arr.mean()), 4),
            'faithfulness_score_std': round(float(arr.std()), 4),
            'faithfulness_score_min': round(float(arr.min()), 4),
            'faithfulness_score_max': round(float(arr.max()), 4),
            'faithfulness_score_median': round(float(np.median(arr)), 4),
            'n_items_with_score': len(faithfulness_scores)
        }

    logger.info(f"RAG abstention F1:          {abstention.get('abstention_f1', 0):.4f}")
    logger.info(f"RAG attribution precision:  {attribution_agg.get('attribution_precision_mean', 0):.4f}")
    logger.info(f"RAG attribution recall:     {attribution_agg.get('attribution_recall_mean', 0):.4f}")
    logger.info(f"RAG attribution F1:         {attribution_agg.get('attribution_f1_mean', 0):.4f}")
    if faithfulness_agg:
        logger.info(
            f"RAG faithfulness score:     "
            f"mean={faithfulness_agg['faithfulness_score_mean']:.4f}  "
            f"std={faithfulness_agg['faithfulness_score_std']:.4f}"
        )

    rq2_results = {
        'rag': {
            'per_item': rag_results,
            'abstention_metrics': abstention,
            'attribution_metrics': attribution_agg,
            'faithfulness_distribution': faithfulness_agg
        },
        'no_rag': {
            'per_item': no_rag_results
        },
        'note': (
            'expert_correctness / expert_completeness / expert_clarity fields '
            'require manual annotation using the rubric in src/evaluation/answer_quality_metrics.py'
        )
    }

    save_json(rq2_results, output_dir / 'rq2_generation_results.json')

    # Export per-item answers to a flat file for annotators
    annotation_export = []
    for rag_item, no_rag_item in zip(rag_results, no_rag_results):
        annotation_export.append({
            'item_id': rag_item['item_id'],
            'question': rag_item['question'],
            'query_type': rag_item['query_type'],
            'legal_domain': rag_item['legal_domain'],
            'rag_answer': rag_item['answer'],
            'no_rag_answer': no_rag_item['answer'],
            'rag_abstained': rag_item['abstained'],
            'rag_attribution_f1': rag_item['attribution'].get('attribution_f1'),
            # Annotator fills these:
            'rag_correctness': None,
            'rag_completeness': None,
            'rag_clarity': None,
            'no_rag_correctness': None,
            'no_rag_completeness': None,
            'no_rag_clarity': None
        })

    annotation_path = output_dir / 'annotation_sheet.json'
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_export, f, indent=2, ensure_ascii=False)
    logger.info(f"Annotation sheet saved to {annotation_path}")

    return rq2_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison experiments")
    parser.add_argument(
        '--dataset', type=Path, required=True,
        help='Path to Q-A-C dataset JSON file'
    )
    parser.add_argument(
        '--index_dir', type=Path, required=True,
        help='Directory containing pre-built BM25 and FAISS indices'
    )
    parser.add_argument(
        '--output_dir', type=Path, default=Path('results/experiment'),
        help='Directory to write results'
    )
    parser.add_argument(
        '--rq1_only', action='store_true',
        help='Run retrieval evaluation only (skip generation)'
    )
    parser.add_argument(
        '--rq2_only', action='store_true',
        help='Run generation evaluation only (skip retrieval)'
    )
    parser.add_argument(
        '--embedding_model', type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Embedding model for dense retrieval (use all-MiniLM-L6-v2 for ablation A5 only)'
    )
    parser.add_argument(
        '--no_verify', action='store_true',
        help='Disable verification in RAG pipeline (for ablation study)'
    )
    parser.add_argument(
        '--wandb_project', type=str, default=None,
        help='wandb project name to enable experiment tracking (optional)'
    )
    parser.add_argument(
        '--wandb_run_name', type=str, default=None,
        help='wandb run name (defaults to timestamp)'
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        'dataset': str(args.dataset),
        'index_dir': str(args.index_dir),
        'embedding_model': args.embedding_model,
        'use_verification': not args.no_verify,
        'k_values': K_VALUES,
        'retrieval_methods': RETRIEVAL_METHODS,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    save_json(config, args.output_dir / 'experiment_config.json')

    # Initialise wandb if requested
    wandb_run = None
    if args.wandb_project:
        if not _WANDB_AVAILABLE:
            logger.warning("wandb not installed. Run: pip install wandb. Skipping tracking.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or time.strftime('run_%Y%m%d_%H%M%S'),
                config=config
            )
            logger.info(f"wandb tracking enabled: {wandb_run.url}")

    # Load dataset
    dataset = load_qac_dataset(args.dataset)

    # Build retriever (shared across all baselines)
    logger.info("Loading retrieval indices...")
    retriever = HybridRetriever(embedding_model=args.embedding_model)
    retriever.load_indices(args.index_dir)

    results_summary = {}

    # RQ1 — retrieval baselines
    if not args.rq2_only:
        rq1 = run_retrieval_evaluation(dataset, retriever, args.output_dir, index_dir=args.index_dir)
        results_summary['rq1'] = {
            method: {
                k: v for k, v in agg.items()
                if k in ['mrr_mean', 'recall@10_mean', 'ndcg@10_mean']
            }
            for method, agg in rq1['per_baseline'].items()
        }

    # RQ2 — generation baselines
    if not args.rq1_only:
        generator = OpenAIGenerator()
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            retrieval_method="hybrid_rerank",
            use_verification=not args.no_verify
        )
        no_rag_baseline = NoRAGBaseline(api_key=settings.openai_api_key)
        rq2 = run_generation_evaluation(dataset, rag_pipeline, no_rag_baseline, args.output_dir)
        results_summary['rq2'] = {
            'rag_abstention_f1': rq2['rag']['abstention_metrics'].get('abstention_f1'),
            'rag_attribution_precision': rq2['rag']['attribution_metrics'].get('attribution_precision_mean'),
            'rag_attribution_recall': rq2['rag']['attribution_metrics'].get('attribution_recall_mean'),
            'rag_attribution_f1': rq2['rag']['attribution_metrics'].get('attribution_f1_mean'),
            'rag_faithfulness_mean': rq2['rag']['faithfulness_distribution'].get('faithfulness_score_mean'),
            'rag_faithfulness_std': rq2['rag']['faithfulness_distribution'].get('faithfulness_score_std')
        }

    save_json(results_summary, args.output_dir / 'results_summary.json')

    # Log final metrics to wandb
    if wandb_run:
        flat_metrics = {}
        for rq, metrics in results_summary.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        for mk, mv in v.items():
                            flat_metrics[f"{rq}/{k}/{mk}"] = mv
                    elif v is not None:
                        flat_metrics[f"{rq}/{k}"] = v
        wandb.log(flat_metrics)
        wandb_run.finish()
        logger.info("wandb run finished")

    logger.info("=== Experiment complete ===")
    logger.info(f"Results written to {args.output_dir}")

    # Print summary table
    print("\n=== Results Summary ===")
    if 'rq1' in results_summary:
        print("\nRQ1 — Retrieval (MRR | Recall@10 | NDCG@10):")
        for method, m in results_summary['rq1'].items():
            print(f"  {method:<20}  MRR={m.get('mrr_mean', 0):.4f}  "
                  f"R@10={m.get('recall@10_mean', 0):.4f}  "
                  f"NDCG@10={m.get('ndcg@10_mean', 0):.4f}")
    if 'rq2' in results_summary:
        print("\nRQ2 — Generation (RAG system):")
        m = results_summary['rq2']
        print(f"  Abstention F1:         {m.get('rag_abstention_f1', 0):.4f}")
        print(f"  Attribution Precision: {m.get('rag_attribution_precision', 0):.4f}")
        print(f"  Attribution Recall:    {m.get('rag_attribution_recall', 0):.4f}")
        print(f"  Attribution F1:        {m.get('rag_attribution_f1', 0):.4f}")
        if m.get('rag_faithfulness_mean') is not None:
            print(f"  Faithfulness (mean):   {m['rag_faithfulness_mean']:.4f}  "
                  f"(std={m.get('rag_faithfulness_std', 0):.4f})")
        print("  (Correctness/Completeness/Clarity require manual annotation — "
              f"see {args.output_dir}/annotation_sheet.json)")


if __name__ == '__main__':
    main()
