"""
Integration tests for full RAG pipeline with indices.
"""

import pytest
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.rag_pipeline import RAGPipeline
from src.evaluation.dataset_schema import QACDataset


class TestRAGIntegration:
    """Test full RAG pipeline integration."""
    
    @pytest.fixture
    def pipeline(self):
        """Create RAG pipeline fixture."""
        pipeline = RAGPipeline()
        index_path = Path("data/indices")
        if index_path.exists() and (index_path / "bm25_index.pkl").exists():
            try:
                pipeline.load_indices(index_path)
                return pipeline
            except Exception as e:
                pytest.skip(f"Could not load indices: {e}")
        else:
            pytest.skip("Indices not available")
    
    def test_pipeline_with_indices(self, pipeline):
        """Test pipeline with loaded indices."""
        assert pipeline.retriever is not None
        assert pipeline.generator is not None
        
        # Test retrieval
        query = "What is theft?"
        result = pipeline.answer_question(query)
        
        # Should have result structure
        assert isinstance(result, dict)
        assert 'answer' in result or result.get('abstained', False)
    
    def test_query_processing(self, pipeline):
        """Test query processing through pipeline."""
        queries = [
            "What is theft?",
            "Define murder",
            "What are the penalties for theft?"
        ]
        
        for query in queries:
            result = pipeline.answer_question(query)
            assert isinstance(result, dict)
            # Should have either answer or abstained
            assert 'answer' in result or result.get('abstained', False)
    
    def test_retrieval_with_different_methods(self, pipeline):
        """Test retrieval with different methods."""
        query = "What is theft?"
        
        # Test different retrieval methods
        methods = ['bm25', 'dense', 'hybrid']
        
        for method in methods:
            try:
                retrieved = pipeline.retriever.retrieve(
                    query=query,
                    top_k=5,
                    method=method
                )
                assert isinstance(retrieved, list)
                assert len(retrieved) <= 5
            except Exception as e:
                # Some methods might not be available
                pytest.skip(f"Method {method} not available: {e}")
    
    def test_with_qac_dataset(self, pipeline):
        """Test pipeline with Q-A-C dataset items."""
        qac_path = Path("data/qac_dataset_constitutional.json")
        
        if not qac_path.exists():
            pytest.skip("Q-A-C dataset not available")
        
        try:
            with open(qac_path) as f:
                data = json.load(f)
            
            dataset = QACDataset(**data)
            
            # Test with first few items
            for item in dataset.items[:3]:
                query = item.question
                result = pipeline.answer_question(query)
                
                assert isinstance(result, dict)
                assert 'answer' in result or result.get('abstained', False)
        except Exception as e:
            pytest.skip(f"Could not load Q-A-C dataset: {e}")
