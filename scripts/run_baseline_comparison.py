"""
RQ2 baseline comparison: RAG system vs. no-RAG general-purpose LLM.

Runs 10 representative constitutional questions through both:
  - RAGPipeline (hybrid_rerank, with evidence, verification, and abstention)
  - NoRAGBaseline (GPT-3.5-turbo, parametric knowledge only, no retrieval)

Outputs a side-by-side JSON file and prints a formatted comparison table.

Usage:
    # RAG mode (requires built indices):
    python scripts/run_baseline_comparison.py \
        --index_dir data/indices --output results/rq2_comparison.json

    # Baseline-only mode (no indices required):
    python scripts/run_baseline_comparison.py --baseline_only \
        --output results/rq2_baseline_only.json
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.baselines import NoRAGBaseline
from src.utils import get_logger
from config.settings import settings

logger = get_logger(__name__)

# 10 representative constitutional questions (spans different topics and difficulty levels)
TEST_QUESTIONS = [
    {
        "id": "Q01",
        "question": "What is the official religion of Sri Lanka?",
        "gold_article": "Article 9",
        "topic": "religion/state"
    },
    {
        "id": "Q02",
        "question": "What are the fundamental rights guaranteed under the Constitution of Sri Lanka?",
        "gold_article": "Articles 10-14",
        "topic": "fundamental rights"
    },
    {
        "id": "Q03",
        "question": "How can Parliament be dissolved before the end of its term?",
        "gold_article": "Article 70",
        "topic": "parliament/dissolution"
    },
    {
        "id": "Q04",
        "question": "What are the qualifications to become President of Sri Lanka?",
        "gold_article": "Article 91",
        "topic": "executive/president"
    },
    {
        "id": "Q05",
        "question": "How is the Constitution of Sri Lanka amended?",
        "gold_article": "Articles 82-83",
        "topic": "constitutional amendment"
    },
    {
        "id": "Q06",
        "question": "What are the official languages of Sri Lanka under the Constitution?",
        "gold_article": "Articles 18-19",
        "topic": "official language"
    },
    {
        "id": "Q07",
        "question": "On what grounds can the President of Sri Lanka be removed from office?",
        "gold_article": "Article 38",
        "topic": "executive/removal"
    },
    {
        "id": "Q08",
        "question": "What is the jurisdiction of the Supreme Court of Sri Lanka?",
        "gold_article": "Article 118",
        "topic": "judiciary"
    },
    {
        "id": "Q09",
        "question": "What is the Consolidated Fund under the Constitution of Sri Lanka?",
        "gold_article": "Article 148",
        "topic": "finance"
    },
    {
        "id": "Q10",
        "question": "Can the freedom of thought and conscience be restricted under the Constitution?",
        "gold_article": "Article 10",
        "topic": "fundamental rights"
    },
]


def run_baseline(questions: list) -> list:
    """Run all questions through NoRAGBaseline."""
    baseline = NoRAGBaseline(
        max_tokens=600,
        temperature=0.1,
        api_key=settings.openai_api_key or None
    )
    results = []
    for item in questions:
        logger.info(f"NoRAGBaseline: {item['id']} — {item['question'][:60]}...")
        result = baseline.answer_question(item["question"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold_article": item["gold_article"],
            "topic": item["topic"],
            "answer": result["answer"],
            "abstained": result["abstained"],
            "citations": result.get("citations", []),
            "num_retrieved": 0,
            "system": "no_rag_baseline"
        })
    return results


def run_rag(questions: list, index_dir: Path) -> list:
    """Run all questions through the full RAG pipeline."""
    from src.generation.rag_pipeline import RAGPipeline
    from src.generation.openai_generator import OpenAIGenerator
    from config.settings import settings

    generator = OpenAIGenerator()
    pipeline = RAGPipeline(
        generator=generator,
        top_k=settings.top_k_rerank,
        retrieval_method='hybrid_rerank',
        use_verification=True,
        use_inference=True,
        use_explainability=True
    )
    pipeline.load_indices(index_dir)

    results = []
    for item in questions:
        logger.info(f"RAGPipeline: {item['id']} — {item['question'][:60]}...")
        try:
            result = pipeline.answer_question(item["question"])
            results.append({
                "id": item["id"],
                "question": item["question"],
                "gold_article": item["gold_article"],
                "topic": item["topic"],
                "answer": result["answer"],
                "abstained": result.get("abstained", False),
                "abstention_reason": result.get("abstention_reason"),
                "citations": result.get("citations", []),
                "num_retrieved": result.get("num_retrieved", 0),
                "verification_report": result.get("verification_report"),
                "system": "rag_pipeline"
            })
        except Exception as e:
            logger.error(f"RAGPipeline failed for {item['id']}: {e}")
            results.append({
                "id": item["id"],
                "question": item["question"],
                "gold_article": item["gold_article"],
                "topic": item["topic"],
                "answer": f"[ERROR: {e}]",
                "abstained": True,
                "citations": [],
                "num_retrieved": 0,
                "system": "rag_pipeline"
            })
    return results


def print_comparison_table(baseline_results: list, rag_results: list) -> None:
    """Print formatted side-by-side comparison for RQ2."""
    print("\n" + "=" * 80)
    print("RQ2 COMPARISON: RAG Pipeline vs. No-RAG Baseline")
    print("=" * 80)

    rag_map = {r["id"]: r for r in rag_results}

    rag_citation_count = 0
    rag_abstain_count = 0
    baseline_abstain_count = 0

    for b in baseline_results:
        r = rag_map.get(b["id"])
        print(f"\n{'─' * 80}")
        print(f"[{b['id']}] {b['question']}")
        print(f"Expected: {b['gold_article']}  |  Topic: {b['topic']}")

        print(f"\n  NO-RAG BASELINE:")
        answer_wrapped = textwrap.fill(b["answer"], width=72, initial_indent="    ",
                                       subsequent_indent="    ")
        print(answer_wrapped)
        print(f"  Citations: none  |  Abstained: {b['abstained']}")
        if b["abstained"]:
            baseline_abstain_count += 1

        if r:
            print(f"\n  RAG PIPELINE:")
            answer_wrapped = textwrap.fill(r["answer"], width=72, initial_indent="    ",
                                           subsequent_indent="    ")
            print(answer_wrapped)
            cit_list = [f"{c.get('act_name','?')} {c.get('section','?')}"
                        for c in r.get("citations", [])]
            print(f"  Citations: {', '.join(cit_list) or 'none'}  |  "
                  f"Retrieved: {r['num_retrieved']}  |  "
                  f"Abstained: {r['abstained']}")
            if r.get("citations"):
                rag_citation_count += 1
            if r["abstained"]:
                rag_abstain_count += 1

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"  No-RAG Baseline: citations=0/{len(baseline_results)}, "
          f"abstentions={baseline_abstain_count}/{len(baseline_results)}")
    if rag_results:
        print(f"  RAG Pipeline:   citations={rag_citation_count}/{len(rag_results)}, "
              f"abstentions={rag_abstain_count}/{len(rag_results)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="RQ2 baseline comparison")
    parser.add_argument(
        '--index_dir', type=Path,
        default=Path('data/indices'),
        help='Directory with pre-built retrieval indices'
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('results/rq2_comparison.json'),
        help='Output JSON file path'
    )
    parser.add_argument(
        '--baseline_only', action='store_true',
        help='Skip RAG pipeline (for when indices are not yet built)'
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Always run baseline
    logger.info("Running NoRAG Baseline...")
    baseline_results = run_baseline(TEST_QUESTIONS)

    # Optionally run RAG
    rag_results = []
    if not args.baseline_only:
        logger.info("Running RAG Pipeline...")
        rag_results = run_rag(TEST_QUESTIONS, args.index_dir)

    # Save results
    output = {
        "questions": TEST_QUESTIONS,
        "baseline_results": baseline_results,
        "rag_results": rag_results,
        "summary": {
            "n_questions": len(TEST_QUESTIONS),
            "baseline_abstentions": sum(1 for r in baseline_results if r["abstained"]),
            "rag_abstentions": sum(1 for r in rag_results if r["abstained"]) if rag_results else None,
            "rag_with_citations": sum(1 for r in rag_results if r.get("citations")) if rag_results else None
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {args.output}")

    print_comparison_table(baseline_results, rag_results)


if __name__ == '__main__':
    main()
