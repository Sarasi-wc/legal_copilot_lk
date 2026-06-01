"""
Performance Benchmarking Script

Measures retrieval and generation latency, memory usage, and throughput.
"""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List
import statistics

sys.path.append(str(Path(__file__).parent.parent))

from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)


class PerformanceBenchmark:
    """Performance benchmarking for the system."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.results = {
            'retrieval': [],
            'generation': [],
            'end_to_end': [],
            'memory': []
        }
    
    def benchmark_retrieval(
        self,
        retriever: HybridRetriever,
        queries: List[str],
        method: str = 'hybrid_rerank'
    ) -> Dict:
        """Benchmark retrieval performance."""
        logger.info(f"Benchmarking retrieval ({method}) with {len(queries)} queries")
        
        times = []
        
        for query in queries:
            start = time.time()
            try:
                retriever.retrieve(query, top_k=5, method=method)
                elapsed = time.time() - start
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"Error on query '{query}': {e}")
        
        if not times:
            return {}
        
        return {
            'method': method,
            'num_queries': len(queries),
            'mean_latency_ms': statistics.mean(times) * 1000,
            'median_latency_ms': statistics.median(times) * 1000,
            'p95_latency_ms': sorted(times)[int(len(times) * 0.95)] * 1000,
            'p99_latency_ms': sorted(times)[int(len(times) * 0.99)] * 1000,
            'min_latency_ms': min(times) * 1000,
            'max_latency_ms': max(times) * 1000,
            'throughput_qps': len(times) / sum(times) if times else 0
        }
    
    def benchmark_generation(
        self,
        pipeline: RAGPipeline,
        queries: List[str]
    ) -> Dict:
        """Benchmark generation performance."""
        logger.info(f"Benchmarking generation with {len(queries)} queries")
        
        times = []
        
        for query in queries:
            start = time.time()
            try:
                pipeline.answer_question(query)
                elapsed = time.time() - start
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"Error on query '{query}': {e}")
        
        if not times:
            return {}
        
        return {
            'num_queries': len(queries),
            'mean_latency_ms': statistics.mean(times) * 1000,
            'median_latency_ms': statistics.median(times) * 1000,
            'p95_latency_ms': sorted(times)[int(len(times) * 0.95)] * 1000,
            'throughput_qps': len(times) / sum(times) if times else 0
        }
    
    def benchmark_memory(self, func, *args, **kwargs):
        """Benchmark memory usage."""
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'result': result
        }
    
    def run_full_benchmark(self, test_queries: List[str] = None):
        """Run complete benchmark suite."""
        if test_queries is None:
            test_queries = [
                "What is theft?",
                "What is the penalty for murder?",
                "How do I file a civil suit?",
                "What is the definition of contract?",
                "What are labor rights?"
            ]
        
        print("="*80)
        print("PERFORMANCE BENCHMARK")
        print("="*80)
        
        # Initialize pipeline
        pipeline = RAGPipeline()
        
        # Try to load indices
        index_path = settings.index_path
        if index_path.exists():
            logger.info(f"Loading indices from {index_path}")
            try:
                pipeline.load_indices(index_path)
                indices_loaded = True
            except Exception as e:
                logger.warning(f"Could not load indices: {e}")
                indices_loaded = False
        else:
            logger.warning("Indices not found. Some benchmarks will be skipped.")
            indices_loaded = False
        
        results = {}
        
        # Benchmark retrieval methods
        if indices_loaded:
            print("\n" + "="*80)
            print("RETRIEVAL BENCHMARKS")
            print("="*80)
            
            for method in ['bm25', 'dense', 'hybrid', 'hybrid_rerank']:
                try:
                    result = self.benchmark_retrieval(
                        pipeline.retriever,
                        test_queries,
                        method=method
                    )
                    if result:
                        results[f'retrieval_{method}'] = result
                        print(f"\n{method.upper()}:")
                        print(f"  Mean latency: {result['mean_latency_ms']:.2f} ms")
                        print(f"  P95 latency: {result['p95_latency_ms']:.2f} ms")
                        print(f"  Throughput: {result['throughput_qps']:.2f} queries/sec")
                except Exception as e:
                    logger.warning(f"Error benchmarking {method}: {e}")
        
        # Benchmark generation (if OpenAI key available)
        print("\n" + "="*80)
        print("GENERATION BENCHMARKS")
        print("="*80)
        
        if settings.openai_api_key:
            try:
                result = self.benchmark_generation(pipeline, test_queries[:3])  # Limit for cost
                if result:
                    results['generation'] = result
                    print(f"\nGeneration:")
                    print(f"  Mean latency: {result['mean_latency_ms']:.2f} ms")
                    print(f"  P95 latency: {result['p95_latency_ms']:.2f} ms")
                    print(f"  Throughput: {result['throughput_qps']:.2f} queries/sec")
            except Exception as e:
                logger.warning(f"Error benchmarking generation: {e}")
        else:
            print("  Skipped (OpenAI API key not configured)")
        
        # Memory benchmark
        print("\n" + "="*80)
        print("MEMORY BENCHMARKS")
        print("="*80)
        
        if indices_loaded:
            try:
                mem_result = self.benchmark_memory(
                    pipeline.retriever.retrieve,
                    test_queries[0],
                    top_k=5,
                    method='hybrid_rerank'
                )
                results['memory'] = mem_result
                print(f"\nMemory usage:")
                print(f"  Current: {mem_result['current_mb']:.2f} MB")
                print(f"  Peak: {mem_result['peak_mb']:.2f} MB")
            except Exception as e:
                logger.warning(f"Error benchmarking memory: {e}")
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "benchmark_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Benchmark results saved to: {output_path}")
        print("="*80)
        
        return results


def main():
    """Main function."""
    benchmark = PerformanceBenchmark()
    
    # Test queries
    test_queries = [
        "What is theft?",
        "What is the penalty for murder?",
        "How do I file a civil suit?",
        "What is the definition of contract?",
        "What are labor rights?",
        "What is evidence?",
        "What is the Constitution?",
        "What is administrative law?"
    ]
    
    benchmark.run_full_benchmark(test_queries)


if __name__ == "__main__":
    main()
