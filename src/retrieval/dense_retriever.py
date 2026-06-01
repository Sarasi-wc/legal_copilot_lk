"""
Dense retrieval implementation using sentence transformers and FAISS.
Implements semantic search for legal passages.
"""

import pickle
import torch  # must be imported before faiss to avoid OpenMP conflict on macOS
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

from src.utils import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Dense retrieval using sentence transformers and FAISS.
    Implements semantic similarity search.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize dense retriever.

        Args:
            model_name: Name of sentence transformer model. Defaults to
                all-mpnet-base-v2 (768-dim bi-encoder; the model used in all
                evaluations). Use "sentence-transformers/all-MiniLM-L6-v2"
                for ablation A5 (embedding capacity comparison) only.
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.passages = None
        self.passage_ids = None
        self.embeddings = None

    def load_model(self) -> None:
        """Load sentence transformer model from local cache (no network calls)."""
        import os
        logger.info(f"Loading model: {self.model_name}")
        # Prevent huggingface_hub from making network calls when model is cached.
        # Must be set in the process environment before the library makes any calls.
        os.environ['HF_HUB_OFFLINE'] = '1'
        self.model = SentenceTransformer(self.model_name)
        logger.info("Model loaded successfully")

    def build_index(
        self,
        passages: List[Dict],
        batch_size: int = 32
    ) -> None:
        """
        Build FAISS index from passages.

        Args:
            passages: List of passage dictionaries
            batch_size: Batch size for encoding
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Building dense index for {len(passages)} passages")

        self.passages = passages
        self.passage_ids = [p['passage_id'] for p in passages]

        # Extract texts for embedding
        texts = [p['text'] for p in passages]

        # Generate embeddings
        logger.info("Generating embeddings...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Build FAISS index
        logger.info("Building FAISS index...")
        dimension = self.embeddings.shape[1]

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Add to index
        self.index.add(self.embeddings)

        logger.info(f"Dense index built: {self.index.ntotal} vectors, dim={dimension}")

    def retrieve(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k passages using dense retrieval.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (passage_id, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if self.model is None:
            self.load_model()

        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return results
        results = [
            (self.passage_ids[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
        ]

        return results

    def retrieve_with_passages(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve passages with full metadata.

        Uses direct positional lookup into self.passages so that the correct
        act-specific passage is returned even when passage_ids are not unique
        across acts (e.g. PASSAGE_0038 exists in Constitution, Penal Code, and
        Civil Procedure Code with different texts).

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of passage dictionaries with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if self.model is None:
            self.load_model()

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        retrieved_passages = []
        for idx, score in zip(indices[0], scores[0]):
            passage = self.passages[idx]
            result = passage.copy()
            result['score'] = float(score)
            result['retrieval_method'] = 'dense'
            retrieved_passages.append(result)

        return retrieved_passages

    def save_index(self, index_path: Path, metadata_path: Path) -> None:
        """Save FAISS index and metadata to disk."""
        logger.info(f"Saving dense index to {index_path}")

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'model_name': self.model_name,
                'passages': self.passages,
                'passage_ids': self.passage_ids,
                'embeddings': self.embeddings
            }, f)

        logger.info("Dense index saved successfully")

    def load_index(self, index_path: Path, metadata_path: Path) -> None:
        """Load FAISS index and metadata from disk."""
        logger.info(f"Loading dense index from {index_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)

        self.model_name = data['model_name']
        self.passages = data['passages']
        self.passage_ids = data['passage_ids']
        self.embeddings = data['embeddings']

        logger.info(f"Loaded dense index with {len(self.passages)} passages")
