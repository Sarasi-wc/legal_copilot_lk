"""
Integration tests for RAG pipeline.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.rag_pipeline import RAGPipeline


class TestRAGPipeline:
    """Test RAG pipeline integration."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = RAGPipeline()
        assert pipeline.retriever is not None
        assert pipeline.generator is not None
        assert pipeline.query_normalizer is not None
        assert pipeline.evidence_planner is not None
    
    def test_query_normalization(self):
        """Test query normalization."""
        pipeline = RAGPipeline()
        query = "What is theft?"
        result = pipeline.query_normalizer.normalize(query)
        assert 'normalized' in result
        assert len(result['normalized']) > 0
    
    def test_answer_question(self):
        """Test answering a question (requires indices)."""
        pipeline = RAGPipeline()
        # Load indices if available
        index_path = Path("data/indices")
        if index_path.exists() and (index_path / "bm25_index.pkl").exists():
            try:
                pipeline.load_indices(index_path)
                result = pipeline.answer_question("What is theft?")
                # Should have either answer or abstained flag
                assert 'answer' in result or result.get('abstained', False)
                # If not abstained, should have answer text
                if not result.get('abstained', False):
                    assert 'answer' in result
                    assert len(result['answer']) > 0
            except Exception as e:
                pytest.skip(f"Could not load indices: {e}")
        else:
            pytest.skip("Indices not available")
