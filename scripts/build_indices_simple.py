#!/usr/bin/env python3
"""
Simple script to rebuild indices without full pipeline initialization.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_construction.corpus_builder import CorpusBuilder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever

def build_indices():
    """Build BM25 and dense indices."""

    print("🔄 Building Retrieval Indices")
    print("=" * 70)
    print()

    # Setup paths
    data_path = Path("data")
    indices_path = data_path / "indices"
    indices_path.mkdir(parents=True, exist_ok=True)

    # Load corpus
    print("1️⃣  Loading corpus...")
    corpus_builder = CorpusBuilder(data_path=data_path)
    corpus_builder.load_corpus()
    passages = corpus_builder.extract_all_passages()
    print(f"   ✅ Loaded {len(passages)} passages")
    print()

    # Build BM25 index
    print("2️⃣  Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.build_index(passages)
    bm25.save_index(indices_path / "bm25_index.pkl")
    print("   ✅ BM25 index saved")
    print()

    # Build dense index
    print("3️⃣  Building dense index...")
    print("   Loading sentence transformer model...")
    dense = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("   Encoding passages...")
    dense.build_index(passages)
    print("   Saving index...")
    dense.save_index(
        index_path=indices_path / "faiss_index.bin",
        metadata_path=indices_path / "dense_metadata.pkl"
    )
    print("   ✅ Dense index saved")
    print()

    print("=" * 70)
    print("✅ Indices built successfully!")
    print()
    print("Next step: Test Article 9 query")
    print("  python scripts/test_article9.py")
    print()

if __name__ == "__main__":
    build_indices()
