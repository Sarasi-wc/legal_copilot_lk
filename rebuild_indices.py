"""Quick script to rebuild indices from corpus.jsonl"""
import sys
sys.path.insert(0, '.')
from config.settings import settings
from src.corpus_construction.corpus_builder import CorpusBuilder
from src.generation.rag_pipeline import RAGPipeline

builder = CorpusBuilder(data_path=settings.data_path)
passages = builder.extract_all_passages('corpus.jsonl')
print(f'Loaded {len(passages)} passages', flush=True)

pipeline = RAGPipeline(top_k=settings.top_k_rerank, retrieval_method='hybrid_rerank')
pipeline.build_indices(passages)
pipeline.retriever.save_indices(settings.index_path)
print('Index rebuild complete.', flush=True)
