"""
Abstention quantitative evaluation for the Sri Lankan legal copilot.

Evaluates the system's ability to:
  - Correctly abstain on out-of-corpus queries (no constitutional basis)
  - Correctly answer in-corpus constitutional queries (should NOT abstain)

Test set: data/evaluation/abstention_test_set.json (20 items)
  - 15 out-of-corpus items (expected_abstain=True, relevant_passages=[])
  - 5  in-corpus controls (expected_abstain=False, relevant_passages non-empty)
    ABS_016: Art.89 — eligibility to be elected President
    ABS_017: Art.30 — term of office of the President (question narrowed May 2026)
    ABS_018: Art.30 SUBSEC_1 — President as head of state
    ABS_019: Art.35 — immunity of the President
    ABS_020: Arts.43–44 — Cabinet of Ministers

Gold label: expected_abstain = (len(item['relevant_passages']) == 0)

Results saved to: results/abstention_eval/
Last confirmed result (2026-05-26): Precision=1.000, Recall=1.000, F1=1.000 (N=20)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.generation.rag_pipeline import RAGPipeline
from src.generation.openai_generator import OpenAIGenerator
from src.retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "abstention_eval"
TEST_SET_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "abstention_test_set.json"


def load_test_set() -> List[Dict]:
    with open(TEST_SET_PATH) as f:
        data = json.load(f)
    return data["items"]


def run_query(pipeline: RAGPipeline, question: str) -> Dict[str, Any]:
    """Run a single query through the full RAG pipeline and return the response."""
    try:
        result = pipeline.answer_question(question)
        return result
    except Exception as e:
        logger.error(f"Pipeline error for query '{question[:60]}...': {e}")
        return {"abstained": True, "error": str(e), "answer": ""}


def evaluate(items: List[Dict], pipeline: RAGPipeline) -> Dict[str, Any]:
    """Run all test items and compute abstention metrics."""
    item_results = []
    system_abstained_list = []
    should_abstain_list = []

    out_results = []  # out-of-corpus rows
    in_results = []   # in-corpus control rows

    for i, item in enumerate(items):
        item_id = item["item_id"]
        question = item["question"]
        expected_abstain = len(item["relevant_passages"]) == 0

        print(f"  [{i+1:2d}/{len(items)}] {item_id}  expected_abstain={expected_abstain}", end="", flush=True)

        t0 = time.time()
        response = run_query(pipeline, question)
        elapsed = time.time() - t0

        system_abstained = response.get("abstained", False)

        correct = system_abstained == expected_abstain
        marker = "OK" if correct else "WRONG"

        print(f"  system_abstained={system_abstained}  [{marker}]  ({elapsed:.1f}s)")

        row = {
            "item_id": item_id,
            "question": question,
            "domain": item.get("domain", ""),
            "expected_abstain": expected_abstain,
            "system_abstained": system_abstained,
            "correct": correct,
            "elapsed_s": round(elapsed, 2),
            "answer_snippet": (response.get("answer", "") or "")[:120],
            "abstain_reason": response.get("abstention_reason", ""),
            "faithfulness": response.get("faithfulness_score"),
            "error": response.get("error", ""),
        }
        item_results.append(row)
        system_abstained_list.append(system_abstained)
        should_abstain_list.append(expected_abstain)

        if expected_abstain:
            out_results.append(row)
        else:
            in_results.append(row)

    metrics = RetrievalMetrics.abstention_metrics(system_abstained_list, should_abstain_list)

    # Per-group accuracy
    out_correct = sum(1 for r in out_results if r["correct"])
    in_correct = sum(1 for r in in_results if r["correct"])

    summary = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "test_set": str(TEST_SET_PATH),
        "n_total": len(items),
        "n_out_of_corpus": len(out_results),
        "n_in_corpus": len(in_results),
        "abstention_metrics": metrics,
        "accuracy_out_of_corpus": round(out_correct / len(out_results), 4) if out_results else None,
        "accuracy_in_corpus": round(in_correct / len(in_results), 4) if in_results else None,
        "overall_accuracy": round(sum(r["correct"] for r in item_results) / len(item_results), 4),
        "item_results": item_results,
    }

    return summary


def print_report(summary: Dict) -> None:
    m = summary["abstention_metrics"]
    print("\n" + "=" * 64)
    print("ABSTENTION EVALUATION RESULTS")
    print("=" * 64)
    print(f"Test set          : {summary['n_total']} items "
          f"({summary['n_out_of_corpus']} out-of-corpus, {summary['n_in_corpus']} in-corpus)")
    print()
    print("Confusion matrix (positive = should abstain):")
    print(f"  TP (correctly abstained)         : {m['true_positives']}")
    print(f"  FP (wrongly abstained on answer) : {m['false_positives']}")
    print(f"  FN (wrongly answered OOC query)  : {m['false_negatives']}")
    print(f"  TN (correctly answered)          : {m['true_negatives']}")
    print()
    print(f"Precision  : {m['abstention_precision']:.4f}")
    print(f"Recall     : {m['abstention_recall']:.4f}")
    print(f"F1         : {m['abstention_f1']:.4f}")
    print()
    print(f"Accuracy (out-of-corpus)  : {summary['accuracy_out_of_corpus']:.4f}  "
          f"({m['true_positives']}/{summary['n_out_of_corpus']} correctly abstained)")
    print(f"Accuracy (in-corpus ctrl) : {summary['accuracy_in_corpus']:.4f}  "
          f"({m['true_negatives']}/{summary['n_in_corpus']} correctly answered)")
    print(f"Overall accuracy          : {summary['overall_accuracy']:.4f}")
    print()
    errors = [r for r in summary["item_results"] if not r["correct"]]
    if errors:
        print("Misclassified items:")
        for r in errors:
            tag = "FALSE_NEGATIVE" if r["expected_abstain"] and not r["system_abstained"] else "FALSE_POSITIVE"
            print(f"  {r['item_id']} [{tag}]  {r['question'][:70]}")
    else:
        print("All items correctly classified.")
    print("=" * 64)


def main():
    print("Loading test set...")
    items = load_test_set()
    print(f"Loaded {len(items)} items "
          f"({sum(1 for i in items if not i['relevant_passages'])} out-of-corpus, "
          f"{sum(1 for i in items if i['relevant_passages'])} in-corpus)\n")

    print("Initialising RAG pipeline...")

    index_dir = Path(__file__).parent.parent / "data" / "indices"
    retriever = HybridRetriever(embedding_model=settings.embedding_model)
    retriever.load_indices(index_dir)

    generator = OpenAIGenerator(
        model_name=settings.openai_model,
        max_tokens=settings.openai_max_tokens,
        temperature=settings.openai_temperature,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        min_confidence=settings.min_confidence,
        top_k=settings.top_k_rerank,
        retrieval_method="hybrid_rerank",
    )

    print("Running evaluation queries...\n")
    summary = evaluate(items, pipeline)

    print_report(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "abstention_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
