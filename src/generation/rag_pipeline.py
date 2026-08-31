"""
Complete RAG pipeline integrating retrieval and generation.
Orchestrates the full question-answering workflow.
"""

from typing import Dict, Optional, List
from pathlib import Path

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.query_normalizer import QueryNormalizer
from src.generation.evidence_planner import EvidencePlanner
from src.generation.answer_generator import AnswerGenerator
from src.verification.verifier import Verifier
from src.inference.inference_engine import InferenceEngine
from src.explainability.explainability_engine import ExplainabilityEngine
from src.utils import get_logger
from config.settings import settings

logger = get_logger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for legal question answering.
    Integrates query normalization, retrieval, evidence planning, and generation.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        min_confidence: float = 0.3,
        top_k: int = 5,
        retrieval_method: str = "hybrid_rerank",
        use_verification: bool = True,
        use_inference: bool = True,
        use_explainability: bool = True
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever: HybridRetriever instance (created if None)
            generator: AnswerGenerator instance (created if None)
            min_confidence: Minimum confidence for evidence
            top_k: Number of passages to retrieve
            retrieval_method: Retrieval method to use
            use_verification: Whether to run verifier after generation (RO3)
            use_inference: Whether to run InferenceEngine for rule application
                and conflict-based abstention (RO3)
            use_explainability: Whether to run ExplainabilityEngine to produce
                reasoning steps, assumptions, and limitations (Gap 3, RO3)
        """
        self.retriever = retriever or HybridRetriever(
            embedding_model=settings.embedding_model
        )
        # Use settings for generator if not provided
        self.generator = generator or AnswerGenerator(
            max_tokens=settings.max_tokens,
            temperature=settings.temperature
        )

        self.query_normalizer = QueryNormalizer()
        self.evidence_planner = EvidencePlanner(min_confidence=min_confidence)

        self.use_verification = use_verification
        self.verifier = Verifier() if use_verification else None

        # InferenceEngine is a keyword-based conflict PROXY for abstention,
        # not a legal reasoning engine. See inference_engine.py for full
        # characterisation and limitations.
        self.use_inference = use_inference
        self.inference_engine = InferenceEngine() if use_inference else None

        self.use_explainability = use_explainability
        self.explainability_engine = ExplainabilityEngine() if use_explainability else None

        self.top_k = top_k
        self.retrieval_method = retrieval_method

    def build_indices(self, passages: List[Dict]) -> None:
        """
        Build retrieval indices.

        Args:
            passages: List of passages from corpus
        """
        logger.info("Building retrieval indices for RAG pipeline")
        self.retriever.build_indices(passages)

    def load_indices(self, index_dir: Path) -> None:
        """
        Load pre-built indices.

        Args:
            index_dir: Directory containing indices
        """
        logger.info("Loading retrieval indices for RAG pipeline")
        self.retriever.load_indices(index_dir)

    def answer_question(self, query: str) -> Dict:
        """
        Answer a legal question using RAG.

        Args:
            query: User query

        Returns:
            Complete answer dictionary with citations
        """
        logger.info(f"Processing query: {query}")

        # Step 1: Normalize query
        query_metadata = self.query_normalizer.normalize(query)
        normalized_query = query_metadata['normalized']
        # Use expanded query for retrieval (adds statutory synonyms for BM25/dense)
        # but keep the clean normalized query for the generation prompt.
        retrieval_query = query_metadata.get('retrieval_query', normalized_query)

        logger.debug(f"Query type: {query_metadata['query_type']}")

        # Step 2: Retrieve passages
        retrieved_passages = self.retriever.retrieve(
            query=retrieval_query,
            top_k=self.top_k,
            method=self.retrieval_method
        )

        logger.debug(f"Retrieved {len(retrieved_passages)} passages")

        # Compact retrieval summary for UI/diagnostic consumers (does not affect
        # evaluation — scores mirror what HybridRetriever already computed).
        retrieval_scores = [
            {
                'passage_id': p.get('passage_id'),
                'act_name': p.get('act_name'),
                'section_number': p.get('metadata', {}).get('section_number'),
                'score': p.get('score'),
                'exact_match': p.get('exact_match', False)
            }
            for p in retrieved_passages
        ]

        # Step 3: Inference — keyword-based conflict proxy for abstention (RO3)
        inference_result = None
        if self.use_inference and retrieved_passages:
            inference_result = self.inference_engine.apply_legal_rules(
                facts=[normalized_query],
                passages=retrieved_passages
            )
            reasoning_chain = self.inference_engine.generate_reasoning_chain(
                inference_result['applicable_rules'],
                inference_result['conflicts']
            )
            inference_result['reasoning_chain'] = reasoning_chain

            # Inference-level abstention: conflict between applicable provisions
            should_abstain_inference, abstain_reason = self.inference_engine.should_abstain(
                inference_result
            )
            if should_abstain_inference:
                logger.info(f"InferenceEngine triggered abstention: {abstain_reason}")
                result = {
                    'query': query,
                    'answer': (
                        "I cannot provide a reliable answer because conflicting "
                        "legal provisions were detected in the retrieved evidence. "
                        "The applicable provisions appear to contradict each other. "
                        "Please consult a qualified legal practitioner."
                    ),
                    'citations': [],
                    'evidence_used': 0,
                    'abstained': True,
                    'abstention_reason': abstain_reason,
                    'inference_result': inference_result,
                    'verification_report': None,
                    'explainability': None,
                    'query_metadata': query_metadata,
                    'retrieval_method': self.retrieval_method,
                    'num_retrieved': len(retrieved_passages),
                    'retrieval_scores': retrieval_scores,
                    'verification_enabled': self.use_verification
                }
                return result

        # Step 4: Plan evidence
        evidence_plan = self.evidence_planner.plan_evidence(
            query=normalized_query,
            retrieved_passages=retrieved_passages,
            query_metadata=query_metadata
        )

        logger.debug(f"Evidence sufficient: {evidence_plan['is_sufficient']}")

        # Step 5: Format evidence
        formatted_evidence = self.evidence_planner.format_evidence_for_prompt(
            evidence_plan
        )

        # Step 6: Generate answer
        answer_result = self.generator.generate_answer(
            query=normalized_query,
            evidence_plan=evidence_plan,
            formatted_evidence=formatted_evidence
        )

        # Step 7: Verify answer (RO3 — NLI faithfulness + citation validation)
        verification_report = None
        if self.use_verification and not answer_result.get('abstained', False):
            verification_report = self.verifier.verify_answer(answer_result, evidence_plan)

            if self.verifier.should_abstain(verification_report):
                answer_result['abstained'] = True
                answer_result['abstention_reason'] = 'verification_failed'
                answer_result['answer'] = (
                    "I cannot provide a verified answer to this question. "
                    "The generated answer could not be confirmed against the retrieved evidence. "
                    "Please consult a qualified legal practitioner."
                )
                answer_result['citations'] = []
            elif not verification_report.get('verification_passed', True):
                # Soft failure: verification criteria not fully met (e.g. low
                # citation density or formatting) but not severe enough to abstain.
                # Flag as low_confidence so callers can surface a warning to users.
                logger.warning(
                    "Verification did not pass (low citation density or formatting) "
                    "but abstention not triggered — answer returned with low_confidence=True"
                )
                answer_result['low_confidence'] = True

        answer_result['verification_report'] = verification_report
        answer_result['inference_result'] = inference_result

        # Step 8: Explainability — reasoning steps, assumptions, limitations (RO3, Gap 3)
        explainability = None
        if self.use_explainability and not answer_result.get('abstained', False):
            explainability = self.explainability_engine.generate_explainability(
                query=normalized_query,
                answer=answer_result['answer'],
                evidence_plan=evidence_plan,
                inference_result=inference_result,
                verification_result=verification_report,
                citations=answer_result.get('citations', [])
            )

        answer_result['explainability'] = explainability

        # Step 9: Add metadata
        answer_result['query_metadata'] = query_metadata
        answer_result['retrieval_method'] = self.retrieval_method
        answer_result['num_retrieved'] = len(retrieved_passages)
        answer_result['retrieval_scores'] = retrieval_scores
        answer_result['verification_enabled'] = self.use_verification
        answer_result['inference_enabled'] = self.use_inference
        answer_result['explainability_enabled'] = self.use_explainability

        logger.info(f"Answer generated. Abstained: {answer_result['abstained']}")

        return answer_result

    def batch_answer(self, queries: List[str]) -> List[Dict]:
        """
        Answer multiple queries in batch.

        Args:
            queries: List of queries

        Returns:
            List of answer dictionaries
        """
        logger.info(f"Processing batch of {len(queries)} queries")

        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.answer_question(query)
            results.append(result)

        return results
